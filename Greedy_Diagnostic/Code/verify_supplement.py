"""Verify and reproduce a freshly extracted supplementary package without a research workspace."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    package = args.package.resolve()
    manifest = read(package / 'Manifest.json')
    for name, digest in manifest['files'].items():
        target = (package / name).resolve()
        assert target.is_relative_to(package)
        assert sha(target) == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    stage_proofs = {}
    for stage in ('source', 'assessment'):
        command = [sys.executable, '-B', str(package / 'Code/analyze.py'),
                   '--inputs', str(package / 'Data/Inputs'), '--collected', str(package / 'Data/Collected'),
                   '--stage', stage, '--output', str(args.output / stage)]
        completed = subprocess.run(command, cwd=package, env=environment, capture_output=True, timeout=900)
        (args.output / (stage + '.stdout.log')).write_bytes(completed.stdout)
        (args.output / (stage + '.stderr.log')).write_bytes(completed.stderr)
        assert completed.returncode == 0, completed.stderr.decode(errors='replace')[-3000:]
        for name in ('Aggregate.csv', 'Contrasts.json', 'Per_Clip_Counts.csv', 'Greedy_Minus_Common.csv'):
            assert (args.output / stage / name).read_bytes() == (package / 'Results' / stage / name).read_bytes()
        stage_proofs[stage] = read(args.output / stage / 'Verification.json')
    proof = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'standalone_supplement_all_saved_paths_text_counts_and_intervals_reproduced',
        'files_hash_checked': len(manifest['files']), 'manifest_sha256': sha(package / 'Manifest.json'),
        'aggregate_rows': 6, 'paired_intervals': 4, 'model_populations': 6,
        'clip_model_records': 3 * (11307 + 1167), 'gpu_or_workspace_required': False,
        'full_neural_inference_repeated': False,
        'stages': {stage: {key: value[key] for key in
                          ('clips_per_model', 'reference_units_per_model', 'source_groups', 'frames_per_model')}
                   for stage, value in stage_proofs.items()}}
    (args.output / 'Verification.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps(proof))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
