"""A separate fixed CTC diagnostic using the unchanged assessment runtime."""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT / 'Research_Short_Only_Control_2026-10-30/Assessment'
assert hasattr(queue, 'SimpleQueue')
sys.path.insert(0, str(FROZEN))
from data_tasks import open_data, positions
from runtime_bridge import create_runtime


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def collapse(path, blank=0, eos=2001):
    output = []
    previous = None
    for token in path:
        token = int(token)
        if token != previous and token not in (blank, eos):
            output.append(token)
        previous = token
    return output


def run(arguments):
    import numpy as np
    import torch

    assert collapse([0, 5, 5, 0, 5, 2001, 5, 5, 0]) == [5, 5, 5]
    assert collapse([0, 0, 2001]) == []
    lock = json.loads((HERE / 'Code_Lock.json').read_text(encoding='utf-8'))
    for relative, expected in lock['files'].items():
        assert sha(ROOT / relative) == expected, relative
    output = arguments.output.resolve()
    assert output.is_relative_to(HERE) and output != HERE
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    save(output / 'Start.json', {
        'utc': utc(), 'model': arguments.model, 'stage': arguments.stage, 'mode': arguments.mode,
        'pid': os.getpid(), 'deadline_unix': arguments.deadline,
        'protocol_sha256': sha(HERE / 'Protocol.md'), 'code_lock_sha256': sha(HERE / 'Code_Lock.json'),
        'admission_sha256': sha(HERE / 'Admission.json') if arguments.mode == 'population' else None,
        'decoder': 'greedy_ctc_zero_blank_bias', 'blank_id': 0, 'eos_id': 2001,
        'references_used_for_inference': False})
    try:
        if arguments.mode == 'population':
            admission = json.loads((HERE / 'Admission.json').read_text(encoding='utf-8'))
            assert arguments.stage in admission['stages'] and arguments.model in admission['models']
            assert arguments.deadline <= admission['deadline_unix']
        runtime = create_runtime(HERE / 'Bindings' / arguments.model, arguments.model, root=ROOT)
        save(output / 'Identity.json', runtime.identity)
        sys.path.insert(0, str(ROOT / 'Research_Server_Confirmation_2026-10-19'))
        from confirmation_common import edit_counts

        torch.cuda.reset_peak_memory_stats()
        with open_data(arguments.stage, ROOT / 'Destination_Stage_01/Payload', root=ROOT) as dataset:
            selected = (positions(dataset, arguments.stage + '_probe', 'probe', root=ROOT)
                        if arguments.mode == 'probe' else list(range(len(dataset))))
            frame_count = 0
            encode_seconds = 0.0
            with (output / 'Predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
                for ordinal, position in enumerate(selected, 1):
                    assert time.time() < arguments.deadline, 'Preserve incomplete attempt at deadline'
                    row, pixels = dataset.load(position)
                    encoded = runtime.encode(pixels)
                    assert encoded['metadata']['pixel_sha256'] == row['pixel_sha256']
                    logp = encoded['logp']
                    path = np.argmax(logp, axis=1).astype(int).tolist()
                    tokens = collapse(path, runtime.blank_id, runtime.eos_id)
                    prediction = runtime.text(tokens)
                    blank_prob = np.exp(logp[:, runtime.blank_id].astype(np.float64))
                    result = {
                        'position': position, **row, 'model': arguments.model, 'stage': arguments.stage,
                        'argmax_path': path, 'tokens': tokens, 'text': prediction,
                        'counts': edit_counts(row['reference'], prediction),
                        'blank_argmax_frames': path.count(runtime.blank_id),
                        'eos_argmax_frames': path.count(runtime.eos_id),
                        'blank_probability_sum': float(blank_prob.sum()), 'encoding': encoded['metadata']}
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    frame_count += len(path)
                    encode_seconds += encoded['metadata']['encode_seconds']
                    del encoded
                    if ordinal % 100 == 0 or ordinal == len(selected):
                        elapsed = time.time() - began
                        progress = {'utc': utc(), 'clips': ordinal, 'total': len(selected),
                                    'seconds': elapsed, 'eta_seconds': elapsed * (len(selected) - ordinal) / ordinal,
                                    'frames': frame_count, 'encode_seconds': encode_seconds}
                        save(output / 'Progress.json', progress)
                        print(json.dumps(progress), flush=True)
                os.fsync(stream.fileno())
        save(output / 'Complete.json', {
            'utc': utc(), 'status': 'complete_pending_independent_verification',
            'model': arguments.model, 'stage': arguments.stage, 'mode': arguments.mode,
            'clips': len(selected), 'frames': frame_count, 'seconds': time.time() - began,
            'encode_seconds': encode_seconds, 'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
            'predictions_sha256': sha(output / 'Predictions.jsonl'),
            'identity_sha256': sha(output / 'Identity.json'), 'start_sha256': sha(output / 'Start.json')})
    except BaseException:
        save(output / 'Failure.json', {'utc': utc(), 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['MATCHED_FC17', 'MATCHED_WW17', 'MATCHED_SO17'])
    parser.add_argument('--stage', required=True, choices=['source', 'assessment'])
    parser.add_argument('--mode', required=True, choices=['probe', 'population'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    run(parser.parse_args())
