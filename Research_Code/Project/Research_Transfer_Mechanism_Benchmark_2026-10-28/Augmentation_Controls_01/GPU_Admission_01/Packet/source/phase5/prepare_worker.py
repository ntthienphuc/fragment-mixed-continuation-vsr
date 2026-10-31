"""One independent cache job per clip; each process owns its cropper and tokenizer."""

import hashlib
from functools import lru_cache
from pathlib import Path
import av
import numpy as np
from .common import resolve, digest, rank
from .media import read_video, MouthCropper
from .text import Tokenizer, required_frames, reference_text
from .views import window_start, target_text
from .frame_cache import save_frames


@lru_cache(maxsize=1)
def tokenizer(model, units, language):
    return Tokenizer(Path(model), Path(units), language)


@lru_cache(maxsize=1)
def cropper(source):
    return MouthCropper(Path(source))

def feasible(tokens, text, length):
    try:
        return required_frames(tokens.encode(text)) <= length
    except ValueError:
        return False


def process(job):
    config, split, destination, original = job
    try:
        path = Path(original['video'])
        media_hash = digest(path)
        expected = original.get('expected_video_blake2b_128')
        if expected and hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest() != expected:
            raise ValueError('VLR video differs from manifest')
        raw = config['dataset'] == 'lrs2'
        frames = read_video(path, raw=raw)
        tokens = tokenizer(str(resolve(config, 'tokenizer')), str(resolve(config, 'units')), config['language'])
        length = len(frames)
        core = config.get('core_frames', 125)
        minimum = core + 32
        if not isinstance(core, int) or core < 1:
            raise ValueError('Invalid target window length')
        words = original['words']
        timing_valid = bool(words) and all(w['end'] <= length / 25 + .04 for w in words)
        timing_valid = timing_valid and all(a['start'] <= b['start'] for a, b in zip(words, words[1:]))
        full_eligible = 1 <= length <= 400 and feasible(tokens, original['text'], length)
        eligible = minimum <= length <= 400 and full_eligible
        eligible = eligible and timing_valid
        start = window_start(length, original['clip_id'], core=core) if length >= minimum else None
        raw_reference = target_text(words, start, start + core) if start is not None else ''
        reference = reference_text(raw_reference, config['language'])
        window_ok = bool(reference) and timing_valid and feasible(tokens, reference, core)
        starts = []
        if split == 'train' and eligible:
            for begin in range(16, length - core - 15):
                text = target_text(words, begin, begin + core)
                if text and feasible(tokens, text, core):
                    starts.append(begin)
            eligible = bool(starts)
        cache_format = config.get('frame_cache_format', 'npy')
        if cache_format not in ['npy', 'npz']:
            raise ValueError('Unsupported lossless frame cache format')
        filename = 'frames/' + rank(original['clip_id'])[:24] + '.' + cache_format
        target = Path(destination) / filename
        retain = split != 'train' or eligible or (config.get('cache_full_training', False) and full_eligible)
        frame_hash = None
        if retain:
            if raw:
                frames = cropper(str(resolve(config, 'source')))(frames)
            target.parent.mkdir(exist_ok=True)
            save_frames(target, frames)
            frame_hash = digest(target)
        row = {**original, 'frame_path': filename if retain else None, 'frames_sha256': frame_hash,
               'video_sha256': media_hash, 'frames': length, 'train_eligible': eligible,
               'full_eligible': full_eligible,
               'train_starts': starts, 'window_eligible': window_ok, 'eval_start': start,
               'window_reference': reference, 'timing_valid': timing_valid,
               'window_reference_original': raw_reference,
               'core_frames': core, 'frame_cache_format': cache_format,
               'zero_duration_words': sum(w['start'] == w['end'] for w in words)}
        return {'row': row}
    except (OSError, ValueError, RuntimeError, OverflowError, av.error.FFmpegError) as error:
        return {'failure': {'clip_id': original['clip_id'], 'source_id': original['source_id'],
                            'text': original['text'], 'error': f'{type(error).__name__}: {error}'}}
