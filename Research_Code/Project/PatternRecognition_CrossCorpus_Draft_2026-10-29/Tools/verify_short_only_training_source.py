"""Independently check all six identities and execute the extracted selector on every real schedule row."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
PHASE = ROOT / 'Research_Short_Only_Control_2026-10-30'
PREPARED = PHASE / 'Recovery_20261030_2040/Prepared_Training_Source_01'
EXTRACTED = PREPARED / 'Extracted'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    entries = read(EXTRACTED / 'Source_Manifest.json')
    identities = read(EXTRACTED / 'Training_Identities.json')
    expected_models = {'native_full', 'native_fixed_time', 'native_whole_word',
                       'normalized_full', 'normalized_fixed_time', 'native_short_only'}
    assert set(identities) == expected_models
    for entry in entries:
        path = EXTRACTED / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        assert path.read_bytes() == (ROOT / entry['workspace_source']).read_bytes()
    sys.path.insert(0, str(ROOT / 'Research_Finalization_2026-10-30'))
    from package_training_source import run_identity

    backup = read(PHASE / 'Training_Backup_Verified.json')
    receipt = Path(backup['files']['receipts']['local'])
    assert sha(receipt) == backup['files']['receipts']['sha256']
    with zipfile.ZipFile(receipt) as archive:
        short = json.loads(archive.read('Run/Identity.json'))
    for key in sorted(expected_models):
        kind, arm = key.split('_', 1)
        actual = short if key == 'native_short_only' else run_identity(kind, arm)
        digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        assert digest == identities[key]['run_identity_sha256']
        driver = 'short_only' if key == 'native_short_only' else kind
        for name, expected in actual['code_sha256'].items():
            assert sha(EXTRACTED / 'Drivers' / driver / name) == expected
        engine = EXTRACTED / 'Packet/support/Matched_Training_Runner_01/engine.py'
        assert actual['update_engine_sha256'] == sha(engine)
    assert identities['native_short_only']['run_identity_sha256'] == backup['identity_sha256']
    sys.path.insert(0, str(PHASE))
    import prepare_schedule

    sys.path.insert(0, str(EXTRACTED / 'Packet/source'))
    import phase5.text
    import phase5.common

    assert Path(phase5.text.__file__).is_relative_to(EXTRACTED)
    assert Path(phase5.common.__file__).is_relative_to(EXTRACTED)
    _, _, metadata, tokenizer, _, _, _, _ = prepare_schedule.setup()
    sys.path.insert(0, str(EXTRACTED / 'Packet/support/Native_Matched_Adapter_01'))
    import adapter

    assert Path(adapter.__file__).is_relative_to(EXTRACTED)
    # Python 3.9 uses the same byte reader implemented above, without changing exported source.
    adapter.sha = sha
    path = EXTRACTED / 'Drivers/short_only/objective.py'
    specification = importlib.util.spec_from_file_location('extracted_short_only_objective', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.SCHEDULE = PHASE / 'Schedule'
    objective = module.ShortOnlyObjective('vlr', 'short_only')
    assert objective.identity == short['schedule']
    model = SimpleNamespace(method='T0', tokenizer=tokenizer)
    maximum = max(row['frames'] for row in metadata.values())
    values = (np.arange(maximum) % 251).astype(np.uint8)[:, None, None]
    frames = np.broadcast_to(values, (maximum, 96, 96))
    fallbacks = 0
    for index, record in enumerate(objective.schedule):
        row = metadata[record['clip_id']]
        selected, text, numerator, denominator, fallback = objective.select(
            model, frames[:row['frames']], row, index, 17)
        assert np.array_equal(selected[:, 0, 0], frames[record['start']:record['stop'], 0, 0])
        assert numerator == denominator == 1 and fallback == record['full_fallback']
        fallbacks += int(fallback)
    assert len(objective.schedule) == 48000 and fallbacks == 598
    row = metadata[objective.schedule[0]['clip_id']]
    mutations = [(dict(row, clip_id='wrong'), 0, 17), (dict(row, split='test'), 0, 17),
                 (row, 0, 29), (row, -1, 17)]
    for changed, position, seed in mutations:
        try:
            objective.select(model, frames[:row['frames']], changed, position, seed)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid selector input accepted')
    assert sha(path) == short['code_sha256']['objective.py']
    result = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'six_executed_identities_and_extracted_short_only_selector_verified',
        'identity_states': 6, 'manifest_files': len(entries), 'actual_extracted_selector_cases': 48000,
        'negative_selector_cases': 4, 'full_fallbacks': fallbacks, 'synthetic_pixels_only': True,
        'real_archived_metadata_and_tokenizer': True, 'gpu_or_neural_training_used': False,
        'exported_code_edited_for_execution': False, 'schedule_path_relocated_in_test_interpreter': True,
        'python39_hash_reader_compatibility_in_test_only': True,
        'self_contained_neural_reproduction_claimed': False,
        'archive_sha256': sha(PREPARED / 'Matched_Training_Source.zip')}
    (PREPARED / 'Independent_Verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
