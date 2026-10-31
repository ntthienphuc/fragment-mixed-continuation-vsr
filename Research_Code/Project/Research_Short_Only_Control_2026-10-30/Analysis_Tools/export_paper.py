"""Admit completed short-only count exports to the paper, preserving old control data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

PHASE = Path(__file__).resolve().parent.parent
PAPER = PHASE.parent / 'PatternRecognition_CrossCorpus_Draft_2026-10-29'
ANALYSIS = PHASE / 'Analysis_01'
DESTINATION = PAPER / 'Data/Short_Only'
sys.path.insert(0, str(PAPER / 'Code'))
from short_only import reproduce


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prerequisites():
    if not (ANALYSIS / 'Summary.json').exists():
        return None
    summary = read(ANALYSIS / 'Summary.json')
    assert summary['status'] == 'both_locked_populations_recounted_and_compared'
    complete = read(PHASE / 'Assessment_Backup_Complete.json')
    assert complete['status'] == 'training_and_24_shards_backed_up' and len(complete['finished']) == 24
    assert complete['source_clips'] == 11307 and complete['destination_clips'] == 1167
    restored = read(PHASE / 'Analysis_Inputs_01/Complete.json')
    assert restored['status'] == 'all_24_backups_rehashed_and_restored'
    assert restored['assessment_receipt_sha256'] == sha(PHASE / 'Assessment_Backup_Complete.json')
    assert restored['training_receipt_sha256'] == sha(PHASE / 'Training_Backup_Verified.json')
    for name, digest in summary['output_sha256'].items():
        path = (ANALYSIS / name).resolve()
        assert path.is_relative_to(ANALYSIS.resolve()) and sha(path) == digest
    assert set(summary['code_sha256']) == {'analyze.py', 'collect_population.py', 'paired_statistics.py',
                                           'export_tables.py', 'restore_inputs.py'}
    for name, digest in summary['code_sha256'].items():
        assert sha(Path(__file__).parent / name) == digest, 'Analysis implementation changed since completion'
    return summary


def execution_comparability(complete):
    if complete.get('execution_host') != 'verified_split_hosts':
        return None
    folder = PHASE / 'Recovery_20261030_2040/Parallel_Assessment_01'
    protocol = read(folder / 'Comparability_Protocol.json')
    paid = read(folder / 'Paid_Backup_Complete.json')
    parity = read(folder / 'Extended_Parity_Local_Verified.json')
    assert paid['status'] == 'paid_subset_full_backup_and_cross_host_parity_verified'
    assert paid['protocol_sha256'] == sha(folder / 'Comparability_Protocol.json')
    assert parity['status'] == 'local_archives_rehashed_and_cross_host_parity_rechecked'
    assert not parity['overlap_shard_counted_in_final_population']
    checked = []
    unique = set()
    for comparison in parity['comparisons']:
        reference = Path(comparison['reference'])
        candidate = Path(comparison['candidate'])
        assert sha(reference) == comparison['reference_sha256']
        assert sha(candidate) == comparison['candidate_sha256']
        assert comparison['tokens_text_counts_pixels_trace_exact']
        assert comparison['score_atol'] == 1e-4 and comparison['max_score_error'] <= 1e-4
        with zipfile.ZipFile(reference) as archive:
            records = [json.loads(line) for line in archive.read('Run/Predictions.jsonl').splitlines()]
        assert len(records) == comparison['clips']
        unique.update((row['stage'], row['clip_id']) for row in records)
        checked.append({key: comparison[key] for key in ('task', 'clips', 'score_atol', 'max_score_error',
            'tokens_text_counts_pixels_trace_exact', 'reference_sha256', 'candidate_sha256')})
    assert sum(item['clips'] for item in checked) == 715
    assert len(complete['task_execution_hosts']) == 24
    for task in protocol['private_tasks']:
        assert complete['task_execution_hosts'][task] == 'WORKSTATION-TT'
    for task in protocol['paid_tasks']:
        assert complete['task_execution_hosts'][task] == 'Vast58265651'
    return {'status': 'completed_split_assessment_with_verified_cross_gpu_comparability',
        'comparison_records': 715, 'distinct_corpus_clip_pairs': len(unique),
        'checks': checked, 'overlap_shard_in_aggregation': False,
        'gpu_types': ['NVIDIA GeForce RTX 3060', 'NVIDIA GeForce RTX 4090'],
        'torch': '2.8.0+cu128', 'numpy': '1.26.4', 'precision': 'float32', 'tf32': False,
        'private_gpu_tasks': protocol['private_tasks'], 'paid_gpu_tasks': protocol['paid_tasks'],
        'same_checkpoint_code_pixels_tokenizer_and_decoder': True,
        'comparison_protocol_sha256': sha(folder / 'Comparability_Protocol.json'),
        'local_cross_host_verification_sha256': sha(folder / 'Extended_Parity_Local_Verified.json'),
        'scope': 'Archived actual GPU parity checks; portable package reproduces counts, not neural inference'}


def main(check=False):
    summary = prerequisites()
    if check:
        print(json.dumps({'complete_short_only_analysis_ready': summary is not None, 'data_written': False}))
        return
    assert summary is not None, 'Complete analysis is required; no population or paper data written'
    assert not DESTINATION.exists(), 'Preserve previous paper export'
    training = read(PHASE / 'Analysis_Inputs_01/Training_Verification.json')
    assert training['exposure']['presentations'] == 48000 and training['updates'] == 6000
    assert training['exposure']['full_fallbacks'] == 598
    exposures = read(PAPER / 'Data/Matched_Crops/Training_Exposure.json')
    assert set(exposures) == {'full', 'fixed_time', 'whole_word'}
    exposures['short_only'] = {'updates': 6000, 'exposure': training['exposure'],
        'full_slots': 598, 'final_endpoint_reached': True,
        'training_identity_sha256': training['identity_sha256'],
        'checkpoint_sha256': training['checkpoint_sha256']}
    DESTINATION.mkdir(exist_ok=False)
    sources = {name: ANALYSIS / name for name in summary['output_sha256']}
    sources.update({'Summary.json': ANALYSIS / 'Summary.json',
        'Comparison_Plan.json': PHASE / 'Comparison_Plan.json',
        'Training_Verification.json': PHASE / 'Analysis_Inputs_01/Training_Verification.json'})
    for relative, source in sources.items():
        target = DESTINATION / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    (DESTINATION / 'Training_Exposure.json').write_text(json.dumps(exposures, indent=2), encoding='utf-8')
    comparability = execution_comparability(read(PHASE / 'Assessment_Backup_Complete.json'))
    if comparability is not None:
        (DESTINATION / 'Execution_Comparability.json').write_text(
            json.dumps(comparability, indent=2), encoding='utf-8')
    tables, intervals = reproduce(DESTINATION)
    assert sum(len(rows) for _, rows in tables) == 8 and len(intervals) == 4
    proof = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete_short_only_exports_independently_recounted',
        'summary_sha256': sha(ANALYSIS / 'Summary.json'),
        'training_receipt_sha256': sha(PHASE / 'Training_Backup_Verified.json'),
        'assessment_receipt_sha256': sha(PHASE / 'Assessment_Backup_Complete.json'),
        'aggregate_rows': 8, 'paired_intervals': 4,
        'original_three_state_exports_unchanged': True,
        'source_and_destination_wer_pooled': False,
        'files': {path.relative_to(DESTINATION).as_posix(): sha(path)
                  for path in sorted(DESTINATION.rglob('*')) if path.is_file()}}
    target = PAPER / 'QA/Short_Only_Data_Verification.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps({'status': proof['status'], 'aggregate_rows': 8, 'paired_intervals': 4}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
