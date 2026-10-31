"""Portable independent recount of a COMPLETE three-state greedy population."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import unicodedata

import numpy as np

MODELS = ('MATCHED_FC17', 'MATCHED_WW17', 'MATCHED_SO17')
NUMERIC = ('errors', 'substitutions', 'deletions', 'insertions', 'units')
COUNTS = {'assessment': (1167, 30091, 20, 170078), 'source': (11307, 225960, 61, 1634458)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            assert line.endswith('\n') and line.strip(), 'Truncated or blank JSONL record'
            yield json.loads(line)


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())


def recount(reference, hypothesis):
    """Full scalar distance matrix plus independent traceback; diagonal/delete/insert tie order."""
    reference = normalized(reference).split()
    hypothesis = normalized(hypothesis).split()
    matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    matrix[0] = list(range(len(hypothesis) + 1))
    for row in range(1, len(reference) + 1):
        matrix[row][0] = row
        for column in range(1, len(hypothesis) + 1):
            matrix[row][column] = min(
                matrix[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1]),
                matrix[row - 1][column] + 1, matrix[row][column - 1] + 1)
    row, column = len(reference), len(hypothesis)
    substitutions = deletions = insertions = 0
    while row or column:
        mismatch = row and column and reference[row - 1] != hypothesis[column - 1]
        if row and column and matrix[row][column] == matrix[row - 1][column - 1] + mismatch:
            substitutions += int(mismatch)
            row -= 1
            column -= 1
        elif row and matrix[row][column] == matrix[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            assert column and matrix[row][column] == matrix[row][column - 1] + 1
            insertions += 1
            column -= 1
    assert len(hypothesis) == len(reference) - deletions + insertions
    assert matrix[-1][-1] == substitutions + deletions + insertions
    return {'errors': matrix[-1][-1], 'substitutions': substitutions, 'deletions': deletions,
            'insertions': insertions, 'units': len(reference), 'metric': 'wer'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_csv(path, values):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def population(folder, model, stage, cohort, vocabulary, expected, inputs):
    assert not (folder / 'Failure.json').exists()
    terminal = read(folder / 'Complete.json')
    assert terminal['status'] == 'complete_pending_independent_verification'
    assert terminal['mode'] == 'population' and terminal['model'] == model and terminal['stage'] == stage
    assert terminal['clips'] == len(cohort)
    for filename, key in [('Predictions.jsonl', 'predictions_sha256'), ('Start.json', 'start_sha256'),
                          ('Identity.json', 'identity_sha256')]:
        assert sha(folder / filename) == terminal[key], filename
    start = read(folder / 'Start.json')
    assert start['decoder'] == 'greedy_ctc_zero_blank_bias' and not start['references_used_for_inference']
    assert start['mode'] == 'population' and start['stage'] == stage and start['model'] == model
    for filename, key in [('Protocol.md', 'protocol_sha256'), ('Code_Lock.json', 'code_lock_sha256'),
                          ('Admission.json', 'admission_sha256')]:
        assert sha(inputs / filename) == start[key]
    assert datetime.fromisoformat(start['utc']) > datetime.fromisoformat(read(inputs / 'Admission.json')['utc'])
    identity = read(folder / 'Identity.json')
    for key, value in expected.items():
        assert identity[key] == value, key
    assert identity['precision'] == 'float32' and identity['tf32'] is False
    assert identity['deterministic_algorithms'] and identity['threads'] == 2
    assert not identity['reference_used_for_inference']
    verified = []
    prediction_rows = list(rows(folder / 'Predictions.jsonl'))
    assert len(prediction_rows) == len(cohort)
    for index, (prediction, original) in enumerate(zip(prediction_rows, cohort)):
        assert prediction['position'] == index == original['position']
        assert all(prediction[key] == value for key, value in original.items()), original['clip_id']
        assert prediction['model'] == model and prediction['stage'] == stage
        path = prediction['argmax_path']
        assert len(path) == original['frames'] and all(type(token) is int and 0 <= token < 2002 for token in path)
        tokens = [token for token, _ in groupby(path) if token not in (0, 2001)]
        assert tokens == prediction['tokens']
        text = normalized(''.join(vocabulary[token] for token in tokens).replace('\u2581', ' ').replace('<space>', ' '))
        assert text == prediction['text']
        counts = recount(original['reference'], text)
        assert counts == prediction['counts'], original['clip_id']
        encoding = prediction['encoding']
        assert encoding['model_id'] == model and encoding['precision'] == 'float32'
        assert encoding['frames'] == encoding['state_frames'] == len(path)
        assert encoding['pixel_sha256'] == original['pixel_sha256']
        assert encoding['color_space'] == 'gray' and encoding['transform'] == 'phase5_native_gray'
        blanks = sum(token == 0 for token in path)
        eos = sum(token == 2001 for token in path)
        assert blanks == prediction['blank_argmax_frames'] and eos == prediction['eos_argmax_frames']
        probability = prediction['blank_probability_sum']
        assert math.isfinite(probability) and 0 <= probability <= len(path)
        verified.append({'model': model, 'stage': stage, 'position': index,
            **{key: original[key] for key in ('clip_id', 'source_id', 'frames')},
            **{key: counts[key] for key in NUMERIC}, 'hypothesis_units': len(text.split()),
            'blank_argmax_frames': blanks, 'eos_argmax_frames': eos, 'blank_probability_sum': probability})
    assert terminal['frames'] == sum(row['frames'] for row in verified)
    return verified, {'complete_sha256': sha(folder / 'Complete.json'),
                      'predictions_sha256': terminal['predictions_sha256'], 'identity_sha256': terminal['identity_sha256']}


def paired(left, right, stage):
    groups = sorted({row['source_id'] for row in left})
    values = []
    for group in groups:
        matching = [(a, b) for a, b in zip(left, right) if a['source_id'] == group]
        assert all(a['clip_id'] == b['clip_id'] and a['units'] == b['units'] for a, b in matching)
        values.append([sum(a['errors'] - b['errors'] for a, b in matching), sum(a['units'] for a, b in matching)])
    values = np.asarray(values, dtype=np.int64)
    generator = np.random.default_rng(20261031)
    sampled = generator.integers(0, len(groups), size=(10000, len(groups)))
    numerator = values[sampled, 0].sum(axis=1)
    denominator = values[sampled, 1].sum(axis=1)
    distribution = 100.0 * numerator / denominator
    for index in range(16):
        scalar = 100.0 * sum(int(values[item, 0]) for item in sampled[index])
        scalar /= sum(int(values[item, 1]) for item in sampled[index])
        assert abs(scalar - distribution[index]) < 1e-12
    interval = np.percentile(distribution, [2.5, 97.5], method='linear').tolist()
    return {'stage': stage, 'contrast': left[0]['model'] + '_minus_' + right[0]['model'],
            'wer_difference_pp': 100.0 * int(values[:, 0].sum()) / int(values[:, 1].sum()),
            'ci95_low': interval[0], 'ci95_high': interval[1], 'source_groups': len(groups),
            'bootstrap_resamples': 10000, 'bootstrap_seed': 20261031, 'scalar_draws_checked': 16,
            'bootstrap_unit': 'archived_source_id', 'pointwise_unadjusted': True,
            'conditional_on_fitted_states': True, 'new_blind_confirmation': False}


def main(args):
    inputs = args.inputs.resolve()
    manifest = read(inputs / 'Manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(inputs / name) == digest, name
    cohort = list(rows(inputs / (args.stage + '_Cohort.jsonl')))
    expected_count, expected_units, expected_groups, expected_frames = COUNTS[args.stage]
    assert len(cohort) == len({row['clip_id'] for row in cohort}) == expected_count
    assert sum(len(normalized(row['reference']).split()) for row in cohort) == expected_units
    assert len({row['source_id'] for row in cohort}) == expected_groups
    assert sum(row['frames'] for row in cohort) == expected_frames
    vocabulary = {0: '<blank>', 2001: '<eos>'}
    for line in (inputs / 'vi_units.txt').read_text(encoding='utf-8').splitlines():
        piece, index = line.rsplit(' ', 1)
        assert int(index) not in vocabulary
        vocabulary[int(index)] = piece
    assert set(vocabulary) == set(range(2002))
    identities = read(inputs / 'Expected_Identities.json')
    outcomes = {}
    evidence = {}
    for model in MODELS:
        outcomes[model], evidence[model] = population(args.collected / args.stage / model, model, args.stage,
                                                     cohort, vocabulary, identities[model], inputs)
    aggregates = []
    for model, clips in outcomes.items():
        totals = {key: sum(row[key] for row in clips) for key in NUMERIC}
        assert totals['units'] == expected_units
        aggregates.append({'model': model, 'stage': args.stage, 'clips': len(clips), **totals,
            'wer': 100.0 * totals['errors'] / totals['units'],
            'hypothesis_units': sum(row['hypothesis_units'] for row in clips),
            'hypothesis_to_reference_ratio': sum(row['hypothesis_units'] for row in clips) / expected_units,
            'blank_argmax_fraction': sum(row['blank_argmax_frames'] for row in clips) / expected_frames,
            'mean_blank_probability': math.fsum(row['blank_probability_sum'] for row in clips) / expected_frames,
            'eos_argmax_frames': sum(row['eos_argmax_frames'] for row in clips)})
    with (inputs / (args.stage + '_Common.csv')).open(encoding='utf-8-sig', newline='') as stream:
        common_rows = list(csv.DictReader(stream))
    comparisons = []
    for aggregate in aggregates:
        model = aggregate['model']
        common = sorted((row for row in common_rows if row['model'] == model and row['readout'] == 'common'),
                        key=lambda row: int(row['position']))
        assert len(common) == len(cohort)
        for greedy, original in zip(outcomes[model], common):
            assert original['decoder'] == 'beam_c0.1'
            assert all(str(greedy[key]) == original[key] for key in
                       ('position', 'clip_id', 'source_id', 'frames', 'units'))
        totals = {key: sum(int(row[key]) for row in common) for key in NUMERIC}
        common_wer = 100.0 * totals['errors'] / totals['units']
        comparisons.append({'model': model, 'stage': args.stage, 'common_wer': common_wer,
            'greedy_wer': aggregate['wer'], 'greedy_minus_common_wer_pp': aggregate['wer'] - common_wer,
            **{'greedy_minus_common_' + key: aggregate[key] - totals[key] for key in NUMERIC}})
    contrasts = [paired(outcomes[model], outcomes['MATCHED_SO17'], args.stage) for model in MODELS[:2]]
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output / 'Aggregate.csv', aggregates)
    save_csv(args.output / 'Per_Clip_Counts.csv', [row for model in MODELS for row in outcomes[model]])
    save_csv(args.output / 'Greedy_Minus_Common.csv', comparisons)
    save(args.output / 'Contrasts.json', contrasts)
    save(args.output / 'Verification.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete_three_state_population_independently_recounted', 'stage': args.stage,
        'clips_per_model': expected_count, 'reference_units_per_model': expected_units,
        'source_groups': expected_groups, 'frames_per_model': expected_frames,
        'all_saved_paths_collapsed_independently': True, 'all_text_and_edit_counts_recomputed': True,
        'all_reference_and_pixel_identities_match': True, 'saved_logit_probability_sums_recomputed': False,
        'blank_probability_limit': 'Saved runtime sums checked for bounds; full logits were not archived.',
        'input_manifest_sha256': sha(inputs / 'Manifest.json'), 'analysis_code_sha256': sha(Path(__file__)),
        'numpy': np.__version__, 'evidence': evidence,
        'files': {path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file()}})
    print(json.dumps({'stage': args.stage, 'aggregates': aggregates, 'contrasts': contrasts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--collected', type=Path, required=True)
    parser.add_argument('--stage', choices=COUNTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
