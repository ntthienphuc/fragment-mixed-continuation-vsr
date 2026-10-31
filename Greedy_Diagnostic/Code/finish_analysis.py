"""Wait for six locally backed-up populations, then verify and analyze both corpora."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    lock = {'analyze.py': sha(HERE / 'analyze.py'),
            'Analysis_Inputs_01/Manifest.json': sha(HERE / 'Analysis_Inputs_01/Manifest.json')}
    assert lock['Analysis_Inputs_01/Manifest.json'] == '0d1f983b752761cbc09232c873bf514333559eaae218189afb035a962fc16799'
    deadline = read(HERE / 'Admission.json')['deadline_unix'] + 900
    while not (HERE / 'Backup_Complete.json').exists():
        assert time.time() < deadline, 'Backup not complete by diagnostic collection deadline'
        assert not (HERE / 'Collector_Failure.txt').exists(), 'Inspect preserved collector failure'
        time.sleep(120)
    for relative, digest in lock.items():
        assert sha(HERE / relative) == digest
    backup = read(HERE / 'Backup_Complete.json')
    assert backup['status'] == 'six_complete_populations_locally_hash_verified'
    assert len(backup['files']) == 6
    pairs = {(row['stage'], row['model']) for row in backup['files']}
    assert pairs == {(stage, model) for stage in ('assessment', 'source')
                     for model in ('MATCHED_FC17', 'MATCHED_WW17', 'MATCHED_SO17')}
    for row in backup['files']:
        archive_path = Path(row['local'])
        assert archive_path.stat().st_size == row['bytes'] and sha(archive_path) == row['sha256']
        target = HERE / 'Collected' / row['stage'] / row['model']
        with zipfile.ZipFile(archive_path) as archive:
            expected = {'Start.json', 'Identity.json', 'Complete.json', 'Predictions.jsonl'}
            assert len(archive.namelist()) == 4 and set(archive.namelist()) == expected
            for name in expected:
                assert archive.read(name) == (target / name).read_bytes()
    output = HERE / 'Analysis_01'
    for stage in ('assessment', 'source'):
        proof_path = output / stage / 'Verification.json'
        if not proof_path.exists():
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-B', str(HERE / 'analyze.py'),
                '--inputs', str(HERE / 'Analysis_Inputs_01'), '--collected', str(HERE / 'Collected'),
                '--stage', stage, '--output', str(output / stage)],
                env=environment, capture_output=True, timeout=900)
            (HERE / (stage + '_analysis.stdout.log')).write_bytes(result.stdout)
            (HERE / (stage + '_analysis.stderr.log')).write_bytes(result.stderr)
            assert result.returncode == 0, result.stderr.decode(errors='replace')[-3000:]
        proof = read(proof_path)
        assert proof['status'] == 'complete_three_state_population_independently_recounted'
        assert proof['analysis_code_sha256'] == lock['analyze.py']
        assert proof['input_manifest_sha256'] == lock['Analysis_Inputs_01/Manifest.json']
        for name, digest in proof['files'].items():
            assert sha(output / stage / name) == digest
    aggregates = []
    comparisons = []
    contrasts = []
    for stage in ('source', 'assessment'):
        with (output / stage / 'Aggregate.csv').open(encoding='utf-8', newline='') as stream:
            aggregates.extend(csv.DictReader(stream))
        with (output / stage / 'Greedy_Minus_Common.csv').open(encoding='utf-8', newline='') as stream:
            comparisons.extend(csv.DictReader(stream))
        contrasts.extend(read(output / stage / 'Contrasts.json'))
    summary = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'six_backed_up_populations_independently_recounted_and_compared',
        'aggregate_rows': aggregates, 'contrasts': contrasts, 'greedy_minus_common': comparisons,
        'bootstrap_resamples': 10000, 'bootstrap_seed': 20261031,
        'new_blind_confirmation': False, 'original_primary_results_replaced': False,
        'blank_probability_limit': 'Saved sums bounds-checked, not recomputed from archived logits.',
        'backup_complete_sha256': sha(HERE / 'Backup_Complete.json'),
        'analysis_code_sha256': lock['analyze.py'],
        'input_manifest_sha256': lock['Analysis_Inputs_01/Manifest.json'],
        'stage_verification_sha256': {stage: sha(output / stage / 'Verification.json')
                                     for stage in ('source', 'assessment')}}
    with (output / 'Summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (HERE / 'Analysis_Failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
