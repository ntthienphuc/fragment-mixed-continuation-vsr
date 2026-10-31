"""Resumable timing derivation with bounded audio prefetch; never reads test labels."""
import argparse
import os
import sys
import time
import json
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import av
import torch
root = Path('/root/lip_phase05_20261014')
os.chdir(root)
sys.path.insert(0, str(root))
from phase5.common import digest, save_json, read_json
from align_engine import Aligner, decode_audio

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0)
args = parser.parse_args()
core = read_json(root/'configs/lrs2.json').get('core_frames', 125)
torch.set_num_threads(2)
torch.manual_seed(17)
warnings.filterwarnings('ignore', message='.*forced_align has been deprecated.*')
aligner = Aligner()
model_file = Path(torch.hub.get_dir())/'checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'
model_hash = digest(model_file)
identity = {'model': 'WAV2VEC2_ASR_BASE_960H', 'model_sha256': model_hash,
            'torchaudio': __import__('torchaudio').__version__, 'engine_sha256': digest(root/'server/align_engine.py'),
            'training_minimum_frames': core+32}
counts = {}
began = time.monotonic()


def load(job):
    split, identifier = job
    video = root/'data/lrs2/media/mvlrs_v1/main'/f'{identifier}.mp4'
    label = video.with_suffix('.txt')
    text = next(line.split(':', 1)[1].strip() for line in label.read_text().splitlines() if line.startswith('Text:'))
    sidecar = video.with_suffix('.alignment.json')
    base = {'clip_id': identifier, 'transcript': text, 'label_sha256': digest(label),
            'video_sha256': digest(video), 'identity': identity}
    if sidecar.exists():
        cached = read_json(sidecar)
        if all(cached.get(key) == value for key, value in base.items()):
            return sidecar, cached, None, None
    with av.open(str(video)) as container:
        length = container.streams.video[0].frames
    if split == 'train' and length and not core+32 <= length <= 400:
        return sidecar, {**base, 'status': 'excluded_by_training_duration', 'words': [],
                         'header_frames': length}, None, None
    try:
        waveform, offset = decode_audio(video)
        return sidecar, base, waveform, offset
    except Exception as error:
        return sidecar, {**base, 'status': 'error', 'words': [], 'error': str(error)}, None, None


with ThreadPoolExecutor(max_workers=4) as pool:
    for split in ['val', 'train']:
        ids = [line.split()[0] for line in (root/f'data/lrs2/{split}.txt').read_text().splitlines() if line.strip()]
        if args.limit:
            ids = ids[:args.limit]
        counts[split] = {'planned': len(ids), 'ok': 0, 'error': 0, 'excluded_by_training_duration': 0}
        iterator = iter((split, item) for item in ids)
        pending = deque()
        for _ in range(8):
            job = next(iterator, None)
            if job:
                pending.append(pool.submit(load, job))
        done = 0
        while pending:
            sidecar, result, waveform, offset = pending.popleft().result()
            if waveform is not None:
                try:
                    words = aligner.align(waveform, offset, result['transcript'])
                    result.update(status='ok', words=words, audio_seconds=len(waveform)/16000,
                                  audio_video_offset=offset)
                except Exception as error:
                    result.update(status='error', words=[], error=f'{type(error).__name__}: {error}')
            save_json(sidecar, result)
            counts[split][result['status']] += 1
            done += 1
            job = next(iterator, None)
            if job:
                pending.append(pool.submit(load, job))
            if done % 100 == 0 or done == len(ids):
                print(json.dumps({'split': split, 'done': done, 'counts': counts[split],
                                  'seconds': time.monotonic()-began}), flush=True)
                save_json(root/'validation/server/lrs2_alignment_progress.json',
                          {'identity': identity, 'counts': counts, 'seconds': time.monotonic()-began})
        if not args.limit:
            (root/f'server/lrs2_{split}_alignment_ready').touch()
save_json(root/'validation/server/lrs2_alignment.json',
          {'identity': identity, 'counts': counts, 'seconds': time.monotonic()-began,
           'scope': 'engineering_smoke' if args.limit else 'train_val',
           'timing_status': 'Automatic audio-derived labels; not human-reviewed word boundaries'})
