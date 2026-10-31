"""Extend a separate source snapshot with the actually executed short-only implementation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
PHASE = ROOT / 'Research_Short_Only_Control_2026-10-30'
RECOVERY = PHASE / 'Recovery_20261030_2040'
PREPARED = RECOVERY / 'Prepared_Training_Source_01'
ORIGINAL = PAPER / 'Matched_Training_Source'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    assert not PREPARED.exists(), 'Preserve earlier prepared source snapshots'
    backup = read(PHASE / 'Training_Backup_Verified.json')
    assert backup['status'] == 'all_training_files_locally_hash_verified'
    receipts = Path(backup['files']['receipts']['local'])
    assert sha(receipts) == backup['files']['receipts']['sha256']
    with zipfile.ZipFile(receipts) as archive:
        identity = json.loads(archive.read('Run/Identity.json'))
        verification = json.loads(archive.read('Training_Verification.json'))
        identity_hash = digest(json.dumps(identity, sort_keys=True).encode())
        assert identity_hash == backup['identity_sha256'] == verification['identity_sha256']
        assert identity['objective'] == 'matched_short_only' and identity['steps'] == 6000
        assert verification['updates'] == 6000 and verification['native_strict_load']
        assert verification['checkpoint_sha256'] == backup['files']['checkpoint']['sha256']
        driver = {}
        for name, expected in identity['code_sha256'].items():
            content = archive.read('Driver/' + name)
            assert digest(content) == expected == sha(PHASE / 'Driver' / name)
            driver[name] = content
    manifest = read(ORIGINAL / 'Source_Manifest.json')
    identities = read(ORIGINAL / 'Training_Identities.json')
    assert len(identities) == 5 and 'native_short_only' not in identities
    for entry in manifest:
        path = ORIGINAL / entry['file']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        assert path.read_bytes() == (ROOT / entry['workspace_source']).read_bytes()
    native = identities['native_full']
    for field in ('initializer_sha256', 'steps', 'accumulation', 'seed', 'lr', 'weight_decay',
                  'gradient_clip', 'precision', 'fused_adam', 'tf32'):
        assert native[field] == identity[field]
    summary = read(PHASE / 'Schedule/Summary.json')
    schedule_proof = read(PHASE / 'Schedule/Verification.json')
    assert sha(PHASE / 'Schedule/Summary.json') == schedule_proof['summary_sha256']
    assert schedule_proof['schedule_sha256'] == summary['schedule_sha256'] == identity['schedule']['schedule_sha256']
    assert schedule_proof['views_reconstructed'] == 48000
    assert schedule_proof['counts']['full_fallbacks'] == 598
    selected = []
    for name, content in driver.items():
        selected.append((PHASE / 'Driver' / name, 'Drivers/short_only/' + name,
                         identity['code_sha256'][name], 'executed_driver'))
    for name in ('prepare_schedule.py', 'Protocol.md'):
        key = str((PHASE / name).relative_to(ROOT))
        selected.append((PHASE / name, 'Preparation/Short_Only/' + name,
                         summary['dependencies'][key], 'verified_schedule_preparation'))
    selected.append((PHASE / 'verify_schedule.py', 'Preparation/Short_Only/verify_schedule.py',
                     schedule_proof['verification_code_sha256'], 'executed_schedule_verifier'))
    for source, relative, expected, evidence in selected:
        content = source.read_bytes()
        assert digest(content) == expected
        assert not re.search(rb'-----BEGIN .*PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}', content)
        manifest.append({'file': relative, 'sha256': expected, 'bytes': len(content),
            'workspace_source': source.relative_to(ROOT).as_posix(), 'evidence': evidence})
    state = {field: identity[field] for field in (
        'initializer_sha256', 'schedule', 'code_sha256', 'steps', 'accumulation', 'seed', 'lr',
        'weight_decay', 'gradient_clip', 'precision', 'fused_adam', 'tf32', 'native_graph',
        'packet_manifest_sha256', 'update_engine_sha256', 'objective', 'selection')}
    state.update({'run_identity_sha256': identity_hash, 'paper_model_id': 'MATCHED_SO17'})
    identities['native_short_only'] = state
    PREPARED.mkdir()
    output = PREPARED / 'Matched_Training_Source'
    shutil.copytree(ORIGINAL, output)
    for source, relative, expected, evidence in selected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == expected
    for name, value in [('Source_Manifest.json', manifest), ('Training_Identities.json', identities)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    training = {'model': 'MATCHED_SO17', 'status': verification['status'], 'updates': 6000,
        'identity_sha256': identity_hash, 'checkpoint_sha256': verification['checkpoint_sha256'],
        'exposure': verification['exposure'], 'native_strict_load': True,
        'batchnorm_buffers_unchanged': verification['batchnorm_buffers_unchanged'],
        'training_receipts_sha256': backup['files']['receipts']['sha256'],
        'recovered_checkpoint_locally_hash_verified': True,
        'full_neural_reproduction_claimed': False, 'assessment_results_included': False}
    (output / 'Short_Only_Training_Evidence.json').write_text(json.dumps(training, indent=2), encoding='utf-8')
    text = (output / 'README.md').read_text(encoding='utf-8')
    text = text.replace('whole-word follow-up and the paired token-normalized loss control.',
        'whole-word follow-up, its nominal whole-word short-only control, and the paired token-normalized loss control.')
    text = text.replace('`Training_Identities.json` identifies the three native and two normalized fitted\nstates.',
        '`Training_Identities.json` identifies four native and two normalized fitted\nstates.')
    text += '''

## Whole-word short-only control

`Drivers/short_only/` contains the five unchanged executed driver files, including
`objective.py`. `Preparation/Short_Only/` contains the exact schedule generator,
protocol and verifier associated with the checked schedule. The original odd-slot
word views are retained. Formerly-full even slots use the same word-span rule.
The arm completes 6,000 updates and 48,000 source presentations, with 47,402 short
views and 598 full-view fallbacks. It retains the native T0 loss and update engine.

`Short_Only_Training_Evidence.json` records the terminal identity, checkpoint hash
and actual exposure. This is training evidence, not an assessment result or a
claim of an independently repeated neural run. Population scores belong to the
separate numerical supplement only after its complete assessment verification.

As for the other drivers, original paths, schedule checks and finite admissions
are preserved. The short-only driver expects its verified `Schedule/` beside the
original driver folder; this inspection snapshot does not supply a new admission
or redistribute the frozen target-token schedule, videos or pretrained weights.
'''
    (output / 'README.md').write_text(text, encoding='utf-8')
    archive_path = PREPARED / 'Matched_Training_Source.zip'
    files = sorted(path for path in output.rglob('*') if path.is_file())
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(output).as_posix())
    extracted = PREPARED / 'Extracted'
    extracted.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and len(set(archive.namelist())) == len(files)
        for path in files:
            relative = path.relative_to(output)
            content = archive.read(relative.as_posix())
            assert content == path.read_bytes()
            target = extracted / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    result = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'six_state_training_source_prepared_and_extracted_bytes_verified',
        'source_files': len(manifest), 'archive_members': len(files), 'executed_identity_states': len(identities),
        'archive_sha256': sha(archive_path), 'archive_bytes': archive_path.stat().st_size,
        'original_five_state_source_preserved': True, 'canonical_paper_package_replaced': False,
        'full_neural_reproduction_claimed': False, 'assessment_results_added': False}
    (PREPARED / 'Verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
