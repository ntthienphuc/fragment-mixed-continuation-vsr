"""Archive, extract, numerically reproduce, and independently compile the supplement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main(args):
    build = read(args.built / 'Build_Verification.json')
    assert build['status'] == 'supplement_built_and_all_table_values_present'
    visual = read(args.built / 'Visual_Verification.json')
    assert visual['status'] == 'all_supplement_pages_visually_reviewed'
    assert visual['pdf_sha256'] == build['pdf_sha256'] == sha(args.built / 'Supplement.pdf')
    assert visual['pages_reviewed'] == list(range(1, build['pages'] + 1))
    args.output.mkdir(parents=True, exist_ok=False)
    release = args.output / 'Package'
    release.mkdir()
    for name in ('Code', 'Data', 'Project', 'Results', 'Source'):
        shutil.copytree(args.built / name, release / name)
    shutil.copy2(args.built / 'Supplement.pdf', release / 'Supplement.pdf')
    shutil.copy2(HERE / 'verify_supplement.py', release / 'Code/verify_supplement.py')
    shutil.copy2(HERE / 'supplement_README.md', release / 'README.md')
    save(release / 'Manifest.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'files': {path.relative_to(release).as_posix(): sha(path) for path in sorted(release.rglob('*'))
                  if path.is_file()}, 'summary_sha256': build['summary_sha256']})
    files = {path.relative_to(release).as_posix(): path for path in sorted(release.rglob('*')) if path.is_file()}
    archive_path = args.output / 'Greedy_Diagnostic.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
    extracted = args.output / 'Extracted'
    extracted.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files)
        assert set(archive.namelist()) == set(files)
        for name, path in files.items():
            payload = archive.read(name)
            assert payload == path.read_bytes()
            target = (extracted / name).resolve()
            assert target.is_relative_to(extracted.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-B', str(extracted / 'Code/verify_supplement.py'),
        '--package', str(extracted), '--output', str(args.output / 'Recomputed')],
        cwd=extracted, env=environment, capture_output=True, timeout=1200)
    (args.output / 'Reproduce.stdout.log').write_bytes(result.stdout)
    (args.output / 'Reproduce.stderr.log').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-3000:]
    numeric = read(args.output / 'Recomputed/Verification.json')
    assert numeric['aggregate_rows'] == 6 and numeric['paired_intervals'] == 4
    latex_build = args.output / 'Extracted_Tex_Build'
    latex_build.mkdir()
    for pass_number in (1, 2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
            '-output-directory=' + str(latex_build), 'main.tex'], cwd=extracted / 'Source',
            env=environment, capture_output=True, timeout=120)
        (latex_build / f'pass{pass_number}.stdout.log').write_bytes(result.stdout)
        assert result.returncode == 0, result.stdout.decode(errors='replace')[-3000:]
    import pymupdf

    with pymupdf.open(extracted / 'Supplement.pdf') as expected:
        with pymupdf.open(latex_build / 'main.pdf') as actual:
            assert len(expected) == len(actual) == build['pages']
            assert all(left.get_text() == right.get_text() for left, right in zip(expected, actual))
    proof = {**numeric, 'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'standalone_supplement_pdf_and_numerics_verified', 'pages': build['pages'],
        'archive_sha256': sha(archive_path), 'archive_bytes': archive_path.stat().st_size,
        'archive_members': len(files), 'pdf_sha256': build['pdf_sha256'],
        'manifest_sha256': sha(release / 'Manifest.json'), 'summary_sha256': build['summary_sha256'],
        'backup_complete_sha256': sha(HERE / 'Backup_Complete.json'),
        'visual_review_passed': True, 'fresh_extracted_numerical_reproduction_passed': True,
        'fresh_extracted_tex_compile_passed': True,
        'visual_receipt_sha256': sha(args.built / 'Visual_Verification.json'),
        'numerical_receipt_sha256': sha(args.output / 'Recomputed/Verification.json')}
    save(args.output / 'Verification.json', proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
