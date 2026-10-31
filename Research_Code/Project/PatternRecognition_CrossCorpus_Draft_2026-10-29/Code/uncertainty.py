"""Reproduce the declared conditional bootstrap intervals from exported per-clip counts."""
from collections import defaultdict
import numpy as np
from recount import paired_values, read_csv


def paired_bootstrap(values, seed, method='indices', chunk=200):
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or np.any(values[:, 1] <= 0):
        raise ValueError('Invalid paired group matrix')
    rng = np.random.default_rng(seed)
    if method == 'multinomial':
        weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), 10000)
        counts = weights @ values
        samples = 100 * counts[:, 0] / counts[:, 1]
    elif method == 'indices':
        samples = []
        for offset in range(0, 10000, chunk):
            size = min(chunk, 10000 - offset)
            indices = rng.integers(0, len(values), size=(size, len(values)))
            counts = values[indices].sum(axis=1)
            samples.extend((100 * counts[:, 0] / counts[:, 1]).tolist())
    else:
        raise ValueError('Unknown resampling algorithm')
    low, high = np.quantile(samples, [.025, .975], method='linear')
    return {'estimate_pp': 100 * int(values[:, 0].sum()) / int(values[:, 1].sum()),
            'ci95_low_pp': float(low), 'ci95_high_pp': float(high), 'groups': len(values)}


def compare(actual, expected, estimate_field):
    for field, target in [('estimate_pp', estimate_field), ('ci95_low_pp', 'ci95_low_pp'),
                          ('ci95_high_pp', 'ci95_high_pp')]:
        if abs(actual[field] - float(expected[target])) > 1e-9:
            raise ValueError(f'Interval mismatch for {field}: {actual[field]} versus {expected[target]}')


def main_pairs(data, states):
    output = []
    for row in read_csv(data / 'Comparisons.csv'):
        language = row['language']
        suffix = '_29' if row['seed'] == '29' else ''
        prefix = 'VLR' if language == 'VI' else 'EN'
        left = states[(prefix + '_T0' + suffix + '_LAST', row['decoder_regime'])]
        right = states[(prefix + '_T1' + suffix + '_LAST', row['decoder_regime'])]
        values = paired_values(left, right, 'source' if language == 'VI' else 'clip')
        actual = paired_bootstrap(values, 20261027 if language == 'VI' else 20261028,
                                  'multinomial' if language == 'VI' else 'indices', chunk=256)
        compare(actual, row, 'gain_pp')
        output.append({'family': 'original_destination', 'language': language,
                       'seed': row['seed'], 'regime': row['decoder_regime'], **actual})
    return output


def source_or_crop_pairs(path, states, family):
    output = []
    for row in read_csv(path):
        regime = row.get('regime', row.get('readout'))
        left = states[(row['reference'], regime)]
        right = states[(row['comparison'], regime)]
        values = paired_values(left, right)
        seed = int(row.get('bootstrap_seed', row.get('seed')))
        actual = paired_bootstrap(values, seed)
        expected_groups = int(row.get('source_groups', row.get('groups')))
        if actual['groups'] != expected_groups:
            raise ValueError('Group count differs')
        compare(actual, row, 'reference_minus_comparison_pp')
        output.append({'family': family, 'reference': row['reference'], 'comparison': row['comparison'],
                       'regime': regime, **actual})
    return output


def mechanism_pairs(data, states):
    output = []
    for row in read_csv(data / 'Mechanism_Comparisons.csv'):
        language = row['language']
        if language not in ('VI', 'EN') or row['seed'] != '17':
            raise ValueError('Unexpected context-control population')
        expected_unit = 'video_source' if language == 'VI' else 'released_clip'
        if row['resampling_unit'] != expected_unit:
            raise ValueError('Context-control resampling unit differs')
        regime = row['decoder_regime']
        left = states[(row['baseline_model'], regime)]
        right = states[(row['variant_model'], regime)]
        values = paired_values(left, right, 'source' if language == 'VI' else 'clip')
        actual = paired_bootstrap(values, 20261027 if language == 'VI' else 20261028,
                                  'multinomial' if language == 'VI' else 'indices', chunk=256)
        compare(actual, row, 'gain_pp')
        output.append({'family': 'context_control', 'language': language, 'seed': row['seed'],
                       'reference': row['baseline_model'], 'comparison': row['variant_model'],
                       'regime': regime, 'role': row['contrast_role'], **actual})
    return output


def loss_interaction(data, states):
    order = ['MATCHED_FC17', 'MATCHED_FT17', 'NORMALIZED_FC17', 'NORMALIZED_FT17']
    population = set(states[(order[0], 'common')])
    if any(set(states[(model, 'common')]) != population for model in order):
        raise ValueError('Four-state population mismatch')
    grouped = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    for clip in sorted(population):
        rows = [states[(model, 'common')][clip] for model in order]
        if len({row['units'] for row in rows}) != 1 or len({row['source_id'] for row in rows}) != 1:
            raise ValueError('Four-state reference mismatch')
        grouped[rows[0]['source_id']][:4] += [row['errors'] for row in rows]
        grouped[rows[0]['source_id']][4] += rows[0]['units']
    values = np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.int64)
    weights = np.asarray([[1,-1,0,0], [0,0,1,-1], [-1,1,1,-1], [1,0,-1,0], [0,1,0,-1]])
    expected = read_csv(data / 'Loss_Reduction/Contrasts.csv')
    if len(expected) != 5 or len({row['bootstrap_seed'] for row in expected}) != 1:
        raise ValueError('Unexpected four-state contrast plan')
    rng = np.random.default_rng(int(expected[0]['bootstrap_seed']))
    draws = []
    for offset in range(0, 10000, 200):
        indices = rng.integers(0, len(values), size=(200, len(values)))
        counts = values[indices].sum(axis=1)
        draws.extend((100 * (counts[:, :4] @ weights.T) / counts[:, 4, None]).tolist())
    bounds = np.quantile(draws, [.025, .975], axis=0, method='linear')
    points = 100 * (weights @ values[:, :4].sum(axis=0)) / values[:, 4].sum()
    output = []
    for index, row in enumerate(expected):
        actual = {'estimate_pp': float(points[index]), 'ci95_low_pp': float(bounds[0, index]),
                  'ci95_high_pp': float(bounds[1, index]), 'groups': len(values)}
        compare(actual, row, 'estimate_pp')
        output.append({'family': 'loss_reduction', 'contrast': row['contrast'], **actual})
    return output
