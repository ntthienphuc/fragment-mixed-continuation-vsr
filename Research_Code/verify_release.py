"""Verify archived code identities and run the portable numerical reproduction."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE / 'Project'
PAPER_NAME = 'PatternRecognition_CrossCorpus_Draft_2026-10-29'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_sources():
    manifest = read(HERE / 'Source_Manifest.json')
    paths = [item['file'] for item in manifest['files']]
    assert len(paths) == len(set(paths))
    python_count = 0
    for item in manifest['files']:
        path = (PROJECT / item['file']).resolve()
        assert path.is_relative_to(PROJECT.resolve())
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=item['file'])
            python_count += 1
    base = PROJECT / 'Research_Phase_05_2026-10-13'
    spec = importlib.util.spec_from_file_location('release_original_common', base / 'phase5/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code_identity = module.code_identity()
    assert code_identity == '7e10ae375307c2e707c269d4c916c4940cedf87c1115ca553fb210a323d8240a'
    identities = read(HERE / 'Metadata/Training_Identities.json')
    mapping = {row['file']: row for row in read(HERE / 'Metadata/Matched_Source_Manifest.json')}
    for name, identity in identities.items():
        kind = 'short_only' if name == 'native_short_only' else name.split('_', 1)[0]
        for filename, expected in identity['code_sha256'].items():
            row = mapping['Drivers/' + kind + '/' + filename]
            assert sha(PROJECT / row['workspace_source']) == expected
    registry = read(PROJECT / 'Research_Server_Confirmation_2026-10-19/VC_Model_Registry.json')
    entry = registry['models']['VC_NATIVE']
    source = PROJECT / entry['source']
    values = {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert len(values) == entry['source_python_files'] and digest == entry['source_tree_sha256']
    for relative, expected in registry['support_sources_sha256'].items():
        assert sha(PROJECT / relative) == expected
    return {'files': len(paths), 'python_files_parsed': python_count,
            'original_training_code_identity': code_identity,
            'matched_executed_training_identities': len(identities),
            'native_reference_source_files': len(values)}


def command(command_line, cwd, output, label, timeout=300):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run(command_line, cwd=cwd, env=environment, capture_output=True, timeout=timeout)
    (output / (label + '.stdout.log')).write_bytes(result.stdout)
    (output / (label + '.stderr.log')).write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-4000:]
    return {'command': command_line, 'exit_code': result.returncode}


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = verify_sources()
    if args.test_python:
        result['unit_tests'] = command([str(args.test_python), '-B', '-m', 'unittest', 'discover',
            '-s', 'tests', '-v'], PROJECT / 'Research_Phase_05_2026-10-13', output, 'unit_tests')
    if args.data:
        result['numerical_command'] = command([sys.executable, '-B', str(PROJECT / PAPER_NAME / 'Code/reproduce.py'),
            '--data', str(args.data.resolve()), '--output', str(output / 'Recomputed')],
            HERE, output, 'numerical')
        proof = read(output / 'Recomputed/Verification.json')
        assert proof['aggregate_rows'] == 60 and proof['paired_intervals'] == 49
        result['aggregate_rows'] = proof['aggregate_rows']
        result['paired_intervals'] = proof['paired_intervals']
    result.update({'utc': datetime.now(timezone.utc).isoformat(),
                   'status': 'research_code_integrity_and_requested_checks_passed',
                   'full_neural_training_rerun': False,
                   'source_manifest_sha256': sha(HERE / 'Source_Manifest.json')})
    (output / 'Verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--test-python', type=Path)
    main(parser.parse_args())
