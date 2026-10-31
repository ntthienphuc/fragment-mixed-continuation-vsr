"""Export the completed loss-reduction control and its four-state paired analysis."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PHASE = ROOT / 'Research_Normalized_Continuation_Assessment_2026-10-30'
SOURCE = PHASE / 'Analysis_Interaction_01'
TARGET = HERE / 'Data/Loss_Reduction'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((SOURCE / 'Summary.json').read_text())
    assert summary['status'] == 'complete_four_state_paired_interaction_recounted'
    assert summary['clips'] == 1167 and summary['reference_units'] == 30091
    runner = json.loads((PHASE / 'Analysis_Preparation_01/Runner_Complete.json').read_text())
    assert runner['status'] == 'both_recount_commands_passed'
    TARGET.mkdir(exist_ok=True)
    for name in ('Summary.json', 'Contrasts.csv'):
        shutil.copy2(SOURCE / name, TARGET / name)
        assert sha(SOURCE / name) == sha(TARGET / name)
    results = summary['results']
    for row in results:
        assert row['errors'] == row['substitutions'] + row['deletions'] + row['insertions']
        assert abs(row['wer_percent'] - 100 * row['errors'] / row['units']) < 1e-10
    with (TARGET / 'Results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    original = ROOT / 'Research_Matched_Continuation_Assessment_2026-10-30/Analysis_Native_ThreeArm_01'
    for label, source in [('Native', original), ('Normalized', PHASE / 'Analysis_Destination_01')]:
        shutil.copy2(source / 'Per_Clip_Counts.csv', TARGET / (label + '_Per_Clip_Counts.csv'))
    shutil.copy2(PHASE / 'Analysis_Preparation_01/Analysis_Plan.json', TARGET / 'Analysis_Plan.json')
    identities = json.loads((HERE / 'Data/Matched_Crops/Model_Identities.json').read_text())
    identities.pop('MATCHED_WW17')
    native_exposure = json.loads((HERE / 'Data/Matched_Crops/Training_Exposure.json').read_text())
    normalized_exposure = {}
    training = ROOT / 'Research_Vast_Source_2026-10-29/Normalized_Production_01/Production_01'
    for arm, model in [('full', 'NORMALIZED_FC17'), ('fixed_time', 'NORMALIZED_FT17')]:
        proof = json.loads((training / 'Training_Verification' / (arm + '.json')).read_text())
        assert proof['status'] == 'normalized_training_records_checkpoint_and_strict_load_verified'
        record = proof['records']['training']
        assert record['updates'] == 6000 and record['exposure']['presentations'] == 48000
        assert record['exposure'] == native_exposure[arm]['exposure']
        assert record['full_slots'] == native_exposure[arm]['full_slots']
        normalized_exposure[arm] = record
        identities[model] = {key: proof[key] for key in ('identity_sha256', 'checkpoint_sha256')}
        binding_path = PHASE / 'Analysis_Preparation_01' / arm / 'Binding'
        binding = json.loads((binding_path / 'Binding.json').read_text())
        identities[model]['binding_sha256'] = sha(binding_path / 'Binding.json')
        identities[model]['inference_checkpoint_sha256'] = binding['checkpoint_sha256']
    (TARGET / 'Model_Identities.json').write_text(json.dumps(identities, indent=2))
    (TARGET / 'Training_Exposure.json').write_text(json.dumps(normalized_exposure, indent=2))
    indexed = {row['model']: row for row in results}
    lines = [r'\begin{table}[tbp]', r'\centering\small',
             r'\caption{Loss-reduction control on the complete ViCocktail assessment at seed 17.',
             r'All four fitted states use the common decoder. Gain is full-only minus fixed-time WER.}',
             r'\label{tab:reduction}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Loss reduction & Full-only WER & Fixed-time WER & Gain (pp)\\', r'\midrule']
    for label, prefix in [('Native', 'MATCHED'), ('Token-normalized', 'NORMALIZED')]:
        full = indexed[prefix + '_FC17']['wer_percent']
        mixed = indexed[prefix + '_FT17']['wer_percent']
        lines.append(f'{label} & {full:.3f} & {mixed:.3f} & {full-mixed:.3f}' + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (HERE / 'Source/tables/loss_reduction.tex').write_text('\n'.join(lines) + '\n')
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_loss_control_export_verified',
               'source_summary_sha256': sha(SOURCE / 'Summary.json'), 'models': 4, 'clips_each': 1167,
               'manuscript_results_from_complete_cohort_only': True,
               'export_sha256': {p.name: sha(p) for p in TARGET.iterdir() if p.is_file()}}
    (HERE / 'QA/Loss_Control_Verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'status': receipt['status'], 'models': 4, 'clips_each': 1167}))


if __name__ == '__main__':
    main()
