"""Reproduce the manuscript's numerical tables from the accompanying derived CSV files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from recount import load_counts, verify_table
from uncertainty import loss_interaction, main_pairs, mechanism_pairs, source_or_crop_pairs

HERE = Path(__file__).resolve().parent


def main(data, output):
    data = data.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; previous reproduction is preserved')
    source_hashes = {path.relative_to(data).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in data.rglob('*') if path.is_file()}
    table_sets = []
    original = load_counts([data / 'Per_Clip_Errors.csv'], 'model_id', 'decoder_regime', 'reference_units')
    table_sets.append(('original_destination', verify_table(data / 'Results.csv', original,
                       'model_id', 'decoder_regime', 'reference_units')))
    source = load_counts([data / 'Source_Per_Clip_Errors.csv'], 'model', 'regime', 'units')
    table_sets.append(('original_source', verify_table(data / 'Source_Results.csv', source,
                       'model', 'regime', 'units')))
    crops = load_counts([data / 'Matched_Crops/Per_Clip_Counts.csv'], 'model', 'readout', 'units')
    table_sets.append(('matched_crop', verify_table(data / 'Matched_Crops/Results.csv', crops,
                       'model', 'readout', 'units')))
    normalized = load_counts([data / 'Loss_Reduction/Normalized_Per_Clip_Counts.csv'],
                             'model', 'readout', 'units')
    four_states = {key: state for key, state in crops.items() if key[0] != 'MATCHED_WW17'}
    four_states.update(normalized)
    table_sets.append(('loss_reduction', verify_table(data / 'Loss_Reduction/Results.csv', four_states,
                       'model', 'readout', 'units')))
    intervals = main_pairs(data, original)
    intervals += mechanism_pairs(data, original)
    intervals += source_or_crop_pairs(data / 'Source_Comparisons.csv', source, 'original_source')
    intervals += source_or_crop_pairs(data / 'Matched_Crops/Paired_Comparisons.csv', crops, 'matched_crop')
    intervals += loss_interaction(data, four_states)
    optional = data / 'Matched_Source'
    if optional.exists():
        new_source = load_counts([optional / 'Per_Clip_Counts.csv'], 'model', 'readout', 'units')
        table_sets.append(('matched_source', verify_table(optional / 'Results.csv', new_source,
                           'model', 'readout', 'units')))
        intervals += source_or_crop_pairs(optional / 'Paired_Comparisons.csv', new_source, 'matched_source')
    short_only = data / 'Short_Only'
    if short_only.exists():
        from short_only import reproduce as reproduce_short_only

        short_tables, short_intervals = reproduce_short_only(short_only)
        table_sets.extend(short_tables)
        intervals.extend(short_intervals)
    output.mkdir(parents=True, exist_ok=False)
    values = {'aggregate_tables': {name: rows for name, rows in table_sets}, 'paired_intervals': intervals}
    (output / 'Recomputed_Results.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed_derived_data_reproduction',
              'aggregate_rows': sum(len(rows) for name, rows in table_sets), 'paired_intervals': len(intervals),
              'bootstrap_resamples_per_interval': 10000, 'python': platform.python_version(),
              'numpy': np.__version__, 'input_sha256': source_hashes,
              'new_neural_inference': False, 'model_training_reproduced': False,
              'original_transcripts_realignments_recomputed': False,
              'scope': 'Recount exported edit counts and reproduce declared paired uncertainty; no new experiment.'}
    (output / 'Verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'aggregate_rows', 'paired_intervals', 'python', 'numpy']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE.parent / 'Data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.output)
