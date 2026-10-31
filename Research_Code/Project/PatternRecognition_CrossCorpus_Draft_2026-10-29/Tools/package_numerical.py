"""Build and actually run the portable Code/Data ZIPs from a separate extraction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PAPER = Path(__file__).resolve().parents[1]
CODE_FILES = ('LICENSE', 'README.md', 'recount.py', 'reproduce.py', 'requirements.txt', 'uncertainty.py',
              'short_only.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_checked(archive_path, folder, originals):
    folder.mkdir(exist_ok=False)
    expected = {path.relative_to(originals).as_posix(): path for path in originals.rglob('*')
                if path.is_file() and '__pycache__' not in path.parts}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and set(names) == set(expected)
        for name in names:
            target = (folder / name).resolve()
            assert target.is_relative_to(folder.resolve())
            data = archive.read(name)
            assert data == expected[name].read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def main(target, reference):
    target = target.resolve()
    reference = reference.resolve()
    assert not target.exists(), 'Keep earlier extraction runs'
    assert (reference / 'Recomputed_Results.json').exists()
    temporary = PAPER / 'Code.zip.pending'
    assert not temporary.exists()
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in CODE_FILES:
            archive.write(PAPER / 'Code' / name, name)
    temporary.replace(PAPER / 'Code.zip')
    target.mkdir(parents=True, exist_ok=False)
    extract_checked(PAPER / 'Code.zip', target / 'Code', PAPER / 'Code')
    extract_checked(PAPER / 'Data.zip', target / 'Data', PAPER / 'Data')
    command = [sys.executable, '-X', 'utf8', '-B', str(target / 'Code/reproduce.py'),
               '--data', str(target / 'Data'), '--output', str(target / 'Recomputed')]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run(command, cwd=target, env=environment, capture_output=True, timeout=180)
    (target / 'stdout.log').write_bytes(completed.stdout)
    (target / 'stderr.log').write_bytes(completed.stderr)
    assert completed.returncode == 0, completed.stderr.decode(errors='replace')
    assert (target / 'Recomputed/Recomputed_Results.json').read_bytes() == (
        reference / 'Recomputed_Results.json').read_bytes()
    proof = json.loads((target / 'Recomputed/Verification.json').read_text())
    result = {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'standalone_code_and_data_zip_reproduction_passed',
              'aggregate_rows': proof['aggregate_rows'], 'paired_intervals': proof['paired_intervals'],
              'command': command, 'exit_code': completed.returncode,
              'code_zip_sha256': sha(PAPER / 'Code.zip'), 'data_zip_sha256': sha(PAPER / 'Data.zip'),
              'equal_recomputed_results': True, 'research_workspace_imports_required': False,
              'gpu_or_network_used': False, 'full_neural_training_reproduced': False}
    (PAPER / 'QA/Numerical_Package_Verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.target, arguments.reference)
