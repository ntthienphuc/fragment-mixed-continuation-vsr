"""Paired source-cluster bootstrap on identical evaluation populations."""

import argparse
import numpy as np
from .common import read_rows, read_json, save_json
from pathlib import Path


def compare(left, right, replicates=10000, seed=17, comparisons=1):
    a = {r['clip_id']: r for r in left}
    b = {r['clip_id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Different or duplicate evaluation populations')
    groups = {}
    for key in sorted(a):
        for field in ['reference', 'source_id', 'view', 'context', 'decoder']:
            if a[key][field] != b[key][field]:
                raise ValueError(f'Paired comparison differs in {field}')
        source = a[key]['source_id']
        x, y = a[key]['scores']['word'], b[key]['scores']['word']
        if x['units'] != y['units']:
            raise ValueError('Reference denominator mismatch')
        groups.setdefault(source, np.zeros(3))
        groups[source] += [x['errors'], y['errors'], x['units']]
    values = np.array(list(groups.values()))
    if len(values) < 2 or values[:, 2].sum() == 0:
        raise ValueError('Need at least two source groups and nonempty references')
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))].sum(0)
        if sample[2] > 0:
            differences.append(100 * (sample[0] - sample[1]) / sample[2])
    total = values.sum(0)
    alpha = .05 / comparisons
    return {'clips': len(a), 'source_groups': len(values), 'replicates': replicates,
            'positive_means': 'right has lower WER', 'left_wer': 100 * total[0] / total[2],
            'right_wer': 100 * total[1] / total[2], 'gain_pp': 100 * (total[0] - total[1]) / total[2],
            'ci95': np.quantile(differences, [.025, .975]).tolist(),
            'familywise_ci': np.quantile(differences, [alpha / 2, 1 - alpha / 2]).tolist(),
            'family_size': comparisons, 'seed': seed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparisons', type=int, default=1)
    args = parser.parse_args()
    a = read_json(args.left / 'summary.json')
    b = read_json(args.right / 'summary.json')
    for key in ['dataset', 'manifest_sha256', 'scope', 'view', 'context', 'decoder', 'config_sha256', 'code_sha256']:
        if a[key] != b[key]:
            raise ValueError(f'Comparison differs in {key}')
    if a.get('training_seed') is not None and b.get('training_seed') is not None:
        if a['training_seed'] != b['training_seed']:
            raise ValueError('Compare matched training seeds')
    if args.comparisons < 1:
        raise ValueError('Family size must be positive')
    result = compare(read_rows(args.left / 'predictions.jsonl'),
                     read_rows(args.right / 'predictions.jsonl'), comparisons=args.comparisons)
    result['dataset'] = a['dataset']
    save_json(args.output, result)
    print(result)
