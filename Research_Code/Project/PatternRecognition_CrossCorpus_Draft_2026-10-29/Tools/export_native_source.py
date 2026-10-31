"""Export new native source/destination profiles only after the exact three-state matrix completes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
SOURCE = ROOT / 'Research_Matched_Continuation_Assessment_2026-10-30/Analysis_Native_Source_01'
DESTINATION = PAPER / 'Data/Matched_Crops'
OUTPUT = PAPER / 'Data/Matched_Source'
MODELS = ('MATCHED_FC17', 'MATCHED_FT17', 'MATCHED_WW17')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready():
    if not (SOURCE / 'Summary.json').exists():
        return False
    source = read(SOURCE / 'Summary.json')
    destination = read(DESTINATION / 'Summary.json')
    for summary, stage, clips, units in [(source, 'source', 11307, 225960),
                                        (destination, 'assessment', 1167, 30091)]:
        assert summary['status'] == 'complete_requested_matched_populations_recounted'
        assert summary['all_three_matched_models_complete'] and summary['stage'] == stage
        assert summary['readout'] == 'common' and set(summary['models']) == set(MODELS)
        assert all(row['clips'] == clips and row['units'] == units for row in summary['results'])
    assert source['assessment_plan_sha256'] == destination['assessment_plan_sha256']
    for model in MODELS:
        for field in ['binding_sha256', 'registry_sha256']:
            assert source['provenance'][model][field] == destination['provenance'][model][field]
    for name, digest in source['output_sha256'].items():
        assert sha(SOURCE / name) == digest
    return True


def main():
    if not ready():
        raise RuntimeError('The complete, verified new native source matrix is not available')
    source = read(SOURCE / 'Summary.json')
    destination = read(DESTINATION / 'Summary.json')
    source_rows = {row['model']: row for row in source['results']}
    target_rows = {row['model']: row for row in destination['results']}
    OUTPUT.mkdir(exist_ok=False)
    for name in ['Summary.json', 'Results.csv', 'Per_Clip_Counts.csv', 'Paired_Comparisons.csv']:
        shutil.copy2(SOURCE / name, OUTPUT / name)
        assert sha(SOURCE / name) == sha(OUTPUT / name)
    profiles = []
    for model in MODELS:
        before = source_rows[model]
        after = target_rows[model]
        assert before['decoder'] == after['decoder'] == 'beam_c0.1'
        profiles.append({'model': model, 'source_corpus': 'VLR', 'destination_corpus': 'ViCocktail',
                         'decoder': before['decoder'], 'source_wer_percent': before['wer_percent'],
                         'destination_wer_percent': after['wer_percent'],
                         'source_clips': before['clips'], 'destination_clips': after['clips'],
                         'source_reference_units': before['units'], 'destination_reference_units': after['units'],
                         'binding_sha256': source['provenance'][model]['binding_sha256']})
    with (OUTPUT / 'Source_Destination_Profile.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(),
               'status': 'new_native_source_destination_profiles_identity_matched',
               'source_summary_sha256': sha(SOURCE / 'Summary.json'),
               'destination_summary_sha256': sha(DESTINATION / 'Summary.json'),
               'models': list(MODELS), 'historical_source_values_reused': False,
               'source_and_destination_wer_pooled': False,
               'exports': {path.name: sha(path) for path in OUTPUT.iterdir() if path.is_file()}}
    (PAPER / 'QA/Native_Source_Profile_Verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps({'complete_source_ready': ready(), 'data_written': False}))
    else:
        main()
