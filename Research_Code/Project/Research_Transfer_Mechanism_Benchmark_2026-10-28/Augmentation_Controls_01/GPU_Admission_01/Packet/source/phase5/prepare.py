"""Resumable parallel preprocessing with per-clip receipts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from .common import config_file, resolve, digest, save_json, read_json, write_rows, rank, code_identity
from .importers import vlr_records, lrs2_records
from .prepare_worker import process, tokenizer, feasible


def cached_job(job):
    config, split, destination, original = job
    receipt = Path(destination) / 'clips' / (rank(original['clip_id']) + '.json')
    if receipt.exists():
        result = read_json(receipt)
        row = result.get('row')
        settings_match = row and row.get('core_frames', 125) == config.get('core_frames', 125)
        settings_match = settings_match and (row['frame_path'] is None or
            row.get('frame_cache_format', 'npy') == config.get('frame_cache_format', 'npy'))
        if settings_match and all(row.get(k) == v for k, v in original.items()):
            if 'full_eligible' not in row:
                tokens = tokenizer(str(resolve(config, 'tokenizer')), str(resolve(config, 'units')), config['language'])
                row['full_eligible'] = 1 <= row['frames'] <= 400 and feasible(tokens, row['text'], row['frames'])
                save_json(receipt, result)
            missing_full = config.get('cache_full_training', False) and row['full_eligible'] and row['frame_path'] is None
            if split == 'train' and not row['train_eligible'] and row['frame_path'] is None:
                if not missing_full and digest(row['video']) == row['video_sha256']:
                    return result
            if row['frame_path'] is not None:
                path = Path(destination) / row['frame_path']
                if path.exists() and digest(path) == row['frames_sha256'] and digest(row['video']) == row['video_sha256']:
                    return result
    result = process(job)
    save_json(receipt, result)
    return result


def prepare(config, split, limit=0, test_plan=None, workers=1):
    if limit < 0 or workers < 1:
        raise ValueError('Invalid limit or workers')
    if split == 'test':
        if not test_plan or limit:
            raise ValueError('Full test preparation requires a frozen test plan')
        from .selection import verify_plan
        verify_plan(test_plan, config)
    root = resolve(config, 'data_root')
    destination = resolve(config, 'prepared') / split
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / 'preparation.json'
    identity = {'config_sha256': config['config_sha256'], 'code_sha256': code_identity(), 'split': split, 'limit': limit}
    if lock.exists() and read_json(lock) != identity:
        raise ValueError('Preparation settings changed; choose a new cache path')
    save_json(lock, identity)
    importer = vlr_records(root, split) if config['dataset'] == 'vlr' else lrs2_records(root, split, resolve(config, f'{split}_list'))
    originals = list(islice(importer, limit)) if limit else list(importer)
    import json, hashlib
    source_hash = hashlib.sha256(json.dumps(originals, sort_keys=True).encode()).hexdigest()
    source_lock = destination / 'source.json'
    if source_lock.exists() and read_json(source_lock)['sha256'] != source_hash:
        raise ValueError('Source list or annotations changed; use a new prepared path')
    save_json(source_lock, {'sha256': source_hash, 'clips': len(originals)})
    if len({r['clip_id'] for r in originals}) != len(originals):
        raise ValueError('Duplicate clip ID')
    jobs = ((config, split, destination, row) for row in originals)
    rows, failures = [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(cached_job, jobs, chunksize=1) if pool else map(cached_job, jobs)
        for index, result in enumerate(results, 1):
            if 'row' in result:
                rows.append(result['row'])
            else:
                failures.append(result['failure'])
            if index % 100 == 0:
                print(f'{split}: {index}/{len(originals)}', flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    if not rows:
        raise ValueError('No successfully prepared clips')
    write_rows(destination / 'manifest.jsonl', rows)
    write_rows(destination / 'failures.jsonl', failures)
    receipt = {**identity, 'planned': len(originals), 'prepared': len(rows), 'failed': len(failures),
               'train_eligible': sum(r['train_eligible'] for r in rows),
               'full_eligible': sum(r['full_eligible'] for r in rows),
               'window_eligible': sum(r['window_eligible'] for r in rows),
               'invalid_timing': sum(not r['timing_valid'] for r in rows),
               'manifest_sha256': digest(destination / 'manifest.jsonl'),
               'failures_sha256': digest(destination / 'failures.jsonl')}
    save_json(destination / 'receipt.json', receipt)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--test-plan')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    prepare(config_file(args.config), args.split, args.limit, args.test_plan, args.workers)
