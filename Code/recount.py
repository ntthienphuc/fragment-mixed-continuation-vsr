"""Recount exported edit counts without access to a research workspace or GPU."""
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = ('errors', 'substitutions', 'deletions', 'insertions', 'units')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_counts(paths, model_field, regime_field, units_field):
    states = defaultdict(dict)
    for path in paths:
        for row in read_csv(path):
            key = (row[model_field], row[regime_field])
            clip = row['clip_id']
            if clip in states[key]:
                raise ValueError(f'Duplicate clip in {key}: {clip}')
            values = {name: int(row[units_field if name == 'units' else name]) for name in FIELDS}
            if min(values.values()) < 0 or values['units'] == 0:
                raise ValueError('Invalid edit count or empty reference')
            if values['errors'] != sum(values[name] for name in FIELDS[1:4]):
                raise ValueError('Edit decomposition differs')
            if values['substitutions'] + values['deletions'] > values['units']:
                raise ValueError('Substitutions and deletions exceed the reference')
            states[key][clip] = {**values, 'source_id': row.get('source_id', ''), 'decoder': row['decoder']}
    return dict(states)


def totals(state):
    values = {name: sum(row[name] for row in state.values()) for name in FIELDS}
    return {'clips': len(state), **values, 'wer_percent': 100 * values['errors'] / values['units']}


def verify_table(path, states, model_field, regime_field, units_field):
    output = []
    seen = set()
    for expected in read_csv(path):
        key = (expected[model_field], expected[regime_field])
        if key in seen:
            raise ValueError('Duplicate aggregate row')
        seen.add(key)
        actual = totals(states[key])
        for field in ('clips', *FIELDS):
            saved = expected[units_field if field == 'units' else field]
            if actual[field] != int(saved):
                raise ValueError(f'Aggregate mismatch: {key}, {field}')
        if abs(actual['wer_percent'] - float(expected['wer_percent'])) > 1e-9:
            raise ValueError(f'WER mismatch: {key}')
        if {row['decoder'] for row in states[key].values()} != {expected['decoder']}:
            raise ValueError(f'Decoder mismatch: {key}')
        output.append({'model': key[0], 'regime': key[1], **actual})
    if seen != set(states):
        raise ValueError('Unreported or missing aggregate states')
    return output


def paired_values(left, right, group_by='source'):
    if set(left) != set(right):
        raise ValueError('Unpaired populations')
    groups = defaultdict(lambda: [0, 0])
    for clip in sorted(left):
        first = left[clip]
        second = right[clip]
        if first['units'] != second['units'] or first['source_id'] != second['source_id']:
            raise ValueError('Paired reference metadata differ')
        group = clip if group_by == 'clip' else first['source_id']
        if not group:
            raise ValueError('Source-group identifier unavailable')
        groups[group][0] += first['errors'] - second['errors']
        groups[group][1] += first['units']
    return [groups[key] for key in sorted(groups)]
