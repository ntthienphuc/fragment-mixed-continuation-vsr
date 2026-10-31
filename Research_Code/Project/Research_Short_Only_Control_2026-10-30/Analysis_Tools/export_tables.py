"""Export complete paired populations to a portable, stage-specific count format."""

import csv
import json
from pathlib import Path

from paired_statistics import FIELDS, totals

DECODER = 'beam_c0.1'
IDENTITY_FIELDS = ('position', 'clip_id', 'source_id', 'frames')
COUNT_FIELDS = ('model', 'stage', 'readout', 'decoder', *IDENTITY_FIELDS, *FIELDS)
RESULT_FIELDS = ('model', 'stage', 'readout', 'decoder', 'clips', *FIELDS, 'wer_percent')
PAIR_FIELDS = ('role', 'reference', 'comparison', 'stage', 'readout', 'reference_decoder',
               'comparison_decoder', 'reference_minus_comparison_pp', 'ci95_low_pp', 'ci95_high_pp',
               'source_groups', 'bootstrap_unit', 'bootstrap_resamples', 'bootstrap_seed',
               'independent_scalar_checks', 'conditional_on_fitted_checkpoints', 'multiplicity_adjusted')
EXPECTED = {'source': (11307, 225960, 61), 'assessment': (1167, 30091, 20)}


def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(stage, models, cohort, comparisons):
    clips, units, groups = EXPECTED[stage]
    identities = {row['clip_id']: row for row in cohort}
    assert len(identities) == len(cohort) == clips
    assert sorted(row['position'] for row in cohort) == list(range(clips))
    assert len({row['source_id'] for row in cohort}) == groups
    assert sum(row['predictions'][DECODER]['counts']['units'] for row in cohort) == units
    records = []
    results = []
    for model, population in models.items():
        assert len(population) == clips
        assert {row['clip_id'] for row in population} == set(identities)
        total = totals(population, DECODER)
        assert total['units'] == units
        prefix = {'model': model, 'stage': stage, 'readout': 'common', 'decoder': DECODER}
        results.append({**prefix, 'clips': clips, **total})
        for row in sorted(population, key=lambda item: identities[item['clip_id']]['position']):
            identity = identities[row['clip_id']]
            counts = row['predictions'][DECODER]['counts']
            assert row['source_id'] == identity['source_id']
            assert counts['units'] == identity['predictions'][DECODER]['counts']['units']
            assert all(type(counts[field]) is int and counts[field] >= 0 for field in FIELDS)
            assert counts['errors'] == sum(counts[field] for field in FIELDS[1:4])
            assert counts['substitutions'] + counts['deletions'] <= counts['units']
            records.append({**prefix, **{name: identity[name] for name in IDENTITY_FIELDS},
                            **{name: counts[name] for name in FIELDS}})
    contrasts = []
    seen = set()
    for row in comparisons:
        pair = (row['reference'], row['comparison'])
        assert pair not in seen and all(model in models for model in pair)
        seen.add(pair)
        assert row['source_groups'] == groups and row['bootstrap_resamples'] == 10000
        assert row['bootstrap_unit'] == 'archived_source_id'
        assert row['conditional_on_fitted_checkpoints'] and not row['multiplicity_adjusted']
        assert row['ci95_low_pp'] <= row['ci95_high_pp']
        values = {**row, 'role': row.get('role', 'archived_control'), 'stage': stage,
                  'readout': 'common', 'reference_decoder': DECODER, 'comparison_decoder': DECODER}
        contrasts.append({field: values[field] for field in PAIR_FIELDS})
    assert contrasts
    return records, results, contrasts


def export(stage, models, cohort, comparisons, output):
    records, results, contrasts = prepare(stage, models, cohort, comparisons)
    output = Path(output)
    output.mkdir(exist_ok=False)
    write_csv(output / 'Per_Clip_Counts.csv', COUNT_FIELDS, records)
    write_csv(output / 'Results.csv', RESULT_FIELDS, results)
    write_csv(output / 'Paired_Comparisons.csv', PAIR_FIELDS, contrasts)
    return {'stage': stage, 'models': list(models), 'clips_per_model': EXPECTED[stage][0],
            'count_rows': len(records), 'aggregate_rows': len(results), 'paired_rows': len(contrasts)}
