"""Reproduce the complete short-only follow-up without a GPU or research checkout."""

import hashlib
import json

from recount import load_counts, read_csv, verify_table
from uncertainty import source_or_crop_pairs

MODELS = {'MATCHED_FC17', 'MATCHED_FT17', 'MATCHED_WW17', 'MATCHED_SO17'}
POPULATIONS = {'source': (11307, 225960, 61), 'assessment': (1167, 30091, 20)}
PAIRS = {'primary': ('MATCHED_WW17', 'MATCHED_SO17'),
         'reference': ('MATCHED_FC17', 'MATCHED_SO17')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(folder):
    summary = json.loads((folder / 'Summary.json').read_text(encoding='utf-8'))
    plan = json.loads((folder / 'Comparison_Plan.json').read_text(encoding='utf-8'))
    require(summary['status'] == 'both_locked_populations_recounted_and_compared', 'Incomplete short-only analysis')
    require(summary['comparison_plan_sha256'] == sha(folder / 'Comparison_Plan.json'), 'Comparison plan changed')
    require(plan['status'] == 'analysis_design_locked_before_short_only_training', 'Unregistered comparison')
    require(plan['readout'] == 'beam_c0.1' and plan['bootstrap_resamples'] == 10000, 'Analysis design changed')
    require(plan['bootstrap_unit'] == 'archived_source_id', 'Resampling unit changed')
    require(plan['conditional_on_fitted_checkpoints'] and not plan['multiplicity_adjusted'], 'Scope changed')
    require(plan['no_equivalence_margin_declared'], 'An equivalence margin was not preregistered')
    require(summary['conditional_on_fitted_checkpoints'] and not summary['multiplicity_adjusted'], 'Scope changed')
    require(not summary['new_blind_test'] and not summary['equivalence_margin_declared'], 'Unsupported claim')
    require(set(summary['stages']) == set(POPULATIONS), 'Missing population')
    require(summary['training_verification_sha256'] == sha(folder / 'Training_Verification.json'),
            'Training provenance changed')
    training = json.loads((folder / 'Training_Verification.json').read_text(encoding='utf-8'))
    require(training['updates'] == 6000 and training['exposure']['presentations'] == 48000,
            'Training endpoint differs')
    require(training['native_strict_load'] and training['batchnorm_buffers_unchanged'], 'Training unverified')
    for role, pair in PAIRS.items():
        require(tuple(plan[role + '_comparison']) == pair, 'Paired comparison changed')
    expected_files = {stage + '/' + filename for stage in POPULATIONS
                      for filename in ('Per_Clip_Counts.csv', 'Results.csv', 'Paired_Comparisons.csv')}
    require(set(summary['output_sha256']) == expected_files, 'Missing or unexpected exported evidence')
    for relative, digest in summary['output_sha256'].items():
        require(sha(folder / relative) == digest, 'Exported evidence changed: ' + relative)
    tables = []
    intervals = []
    for stage, (clips, units, groups) in POPULATIONS.items():
        directory = folder / stage
        saved = summary['stages'][stage]
        require(saved['clips'] == clips and set(saved['results']) == MODELS, 'Incomplete fitted-state matrix')
        require(saved['readout'] == 'common' and saved['decoder'] == 'beam_c0.1', 'Readout differs')
        states = load_counts([directory / 'Per_Clip_Counts.csv'], 'model', 'readout', 'units')
        require(set(states) == {(model, 'common') for model in MODELS}, 'Unexpected fitted states')
        cohorts = {frozenset(population) for population in states.values()}
        require(len(cohorts) == 1, 'Four states do not share the same population')
        actual = verify_table(directory / 'Results.csv', states, 'model', 'readout', 'units')
        for row in actual:
            require(row['clips'] == clips and row['units'] == units, 'Population size or denominator differs')
            require(all(abs(row[field] - value) < 1e-9
                        for field, value in saved['results'][row['model']].items()), 'Summary differs')
        pair_rows = read_csv(directory / 'Paired_Comparisons.csv')
        require(len(pair_rows) == 2 and {row['role'] for row in pair_rows} == set(PAIRS), 'Unexpected contrasts')
        saved_pairs = {row['role']: row for row in saved['comparisons']}
        require(set(saved_pairs) == set(PAIRS) and len(saved['comparisons']) == 2, 'Summary contrasts differ')
        for row in pair_rows:
            pair = (row['reference'], row['comparison'])
            require(pair == PAIRS[row['role']] and row['stage'] == stage, 'Wrong contrast or corpus')
            require(row['readout'] == 'common', 'Readout differs')
            require(row['reference_decoder'] == row['comparison_decoder'] == 'beam_c0.1', 'Decoder differs')
            require(int(row['source_groups']) == groups and int(row['bootstrap_resamples']) == 10000,
                    'Group or resample count differs')
            require(row['bootstrap_unit'] == 'archived_source_id', 'Bootstrap unit differs')
            require(row['conditional_on_fitted_checkpoints'] == 'True'
                    and row['multiplicity_adjusted'] == 'False', 'Inference scope differs')
            label = '|'.join([stage, 'common', *pair])
            seed = int.from_bytes(hashlib.sha256(('20261030|' + label).encode()).digest()[:8], 'big')
            require(int(row['bootstrap_seed']) == seed, 'Bootstrap seed differs from locked rule')
            previous = saved_pairs[row['role']]
            require((previous['reference'], previous['comparison']) == pair, 'Summary contrast differs')
            for field in ('reference_minus_comparison_pp', 'ci95_low_pp', 'ci95_high_pp'):
                require(abs(float(row[field]) - previous[field]) < 1e-9, 'Summary interval differs')
        family = 'short_only_' + stage
        tables.append((family, actual))
        intervals.extend(source_or_crop_pairs(directory / 'Paired_Comparisons.csv', states, family))
    return tables, intervals
