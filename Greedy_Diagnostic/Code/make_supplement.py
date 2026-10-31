"""Build a separate anonymous supplement only from both verified full populations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
LABELS = {'MATCHED_FC17': 'FC', 'MATCHED_WW17': 'WW', 'MATCHED_SO17': 'SO'}
STAGES = {'source': 'VLR', 'assessment': 'ViCocktail'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def table(header, rows, columns, caption):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\small', '\\caption{' + caption + '}',
        '\\begin{tabular}{' + columns + '}', r'\toprule', header + r'\\', r'\midrule',
        *[row + r'\\' for row in rows], r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def result_tex(summary):
    aggregates = summary['aggregate_rows']
    comparison = {(row['stage'], row['model']): row for row in summary['greedy_minus_common']}
    wer_rows = []
    edit_rows = []
    blank_rows = []
    displayed = []
    for row in aggregates:
        stage, model = row['stage'], row['model']
        label = STAGES[stage] + ' & ' + LABELS[model]
        common = comparison[(stage, model)]
        values = [f'{float(common["common_wer"]):.3f}', f'{float(row["wer"]):.3f}',
                  f'{float(common["greedy_minus_common_wer_pp"]):+.3f}']
        displayed.extend(values)
        wer_rows.append(label + ' & ' + ' & '.join(values))
        edits = [str(row[key]) for key in ('substitutions', 'deletions', 'insertions', 'hypothesis_units')]
        displayed.extend(edits)
        edit_rows.append(label + ' & ' + ' & '.join(edits))
        blanks = [f'{100 * float(row[key]):.3f}' for key in
                  ('hypothesis_to_reference_ratio', 'blank_argmax_fraction', 'mean_blank_probability')]
        displayed.extend(blanks)
        blank_rows.append(label + ' & ' + ' & '.join(blanks))
    contrast_rows = []
    for contrast in summary['contrasts']:
        left, right = contrast['contrast'].split('_minus_')
        values = [f'{contrast["wer_difference_pp"]:+.3f}', f'{contrast["ci95_low"]:+.3f}',
                  f'{contrast["ci95_high"]:+.3f}']
        displayed.extend(values)
        contrast_rows.append(STAGES[contrast['stage']] + ' & ' + LABELS[left] + '--' + LABELS[right]
                             + ' & ' + values[0] + ' & [' + values[1] + ', ' + values[2] + ']')
    text = table('Corpus & State & Common WER & Greedy WER & $\Delta$ (pp)', wer_rows, 'llrrr',
                 'Complete-population WER (percent). $\Delta$ is greedy minus common; smaller WER is better.')
    text += table('Corpus & State & S & D & I & Output units', edit_rows, 'llrrrr',
                  'Greedy substitutions, deletions, insertions and total predicted whitespace units.')
    text += table('Corpus & State & Output/reference & Blank argmax & Mean $p(\mathrm{blank})$',
                  blank_rows, 'llrrr', 'Readout descriptions (percent). Output/reference uses transcript units; '
                  'blank statistics are weighted by all encoded frames, not averaged equally over clips.')
    text += table('Corpus & Contrast & Difference (pp) & 95\% interval', contrast_rows, 'llrr',
                  'Paired archived-source-group contrasts under the fixed greedy readout. Positive differences favor SO.')
    text += r'\clearpage' + '\n'
    for stage, corpus in STAGES.items():
        values = {row['model']: row for row in aggregates if row['stage'] == stage}
        short = values['MATCHED_SO17']
        differences = {model: float(short['wer']) - float(values[model]['wer'])
                       for model in ('MATCHED_FC17', 'MATCHED_WW17')}
        deletion_differences = {model: int(short['deletions']) - int(values[model]['deletions'])
                               for model in ('MATCHED_FC17', 'MATCHED_WW17')}
        text += (f'On {corpus}, SO minus FC is {differences["MATCHED_FC17"]:+.3f} WER points and '
                 f'SO minus WW is {differences["MATCHED_WW17"]:+.3f} points. The corresponding '
                 f'deletion-count differences are {deletion_differences["MATCHED_FC17"]:+d} and '
                 f'{deletion_differences["MATCHED_WW17"]:+d}, respectively.\n\n')
    return text, displayed


def main(output):
    summary = read(HERE / 'Analysis_01/Summary.json')
    assert summary['status'] == 'six_backed_up_populations_independently_recounted_and_compared'
    assert len(summary['aggregate_rows']) == 6 and len(summary['contrasts']) == 4
    for stage, digest in summary['stage_verification_sha256'].items():
        assert sha(HERE / 'Analysis_01' / stage / 'Verification.json') == digest
    assert sha(HERE / 'analyze.py') == summary['analysis_code_sha256']
    assert sha(HERE / 'Analysis_Inputs_01/Manifest.json') == summary['input_manifest_sha256']
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(HERE / 'Analysis_Inputs_01', output / 'Data/Inputs')
    shutil.copytree(HERE / 'Collected', output / 'Data/Collected')
    shutil.copytree(HERE / 'Analysis_01', output / 'Results')
    (output / 'Code').mkdir()
    for name in ('analyze.py', 'prepare_analysis_inputs.py', 'finish_analysis.py',
                 'make_supplement.py', 'package_supplement.py', 'supplement_template.tex',
                 'supplement_README.md'):
        shutil.copy2(HERE / name, output / 'Code' / name)
    inference = output / 'Project' / HERE.name
    inference.mkdir(parents=True)
    shutil.copytree(HERE / 'Bindings', inference / 'Bindings')
    for name in ('run.py', 'Protocol.md', 'Code_Lock.json', 'Admission.json', 'execute_population.py',
                 'Assets_Complete.json', 'Probe_Verification.json', 'Backup_Complete.json'):
        shutil.copy2(HERE / name, inference / name)
    source = output / 'Source'
    source.mkdir()
    shutil.copy2(HERE / 'supplement_template.tex', source / 'main.tex')
    tables, displayed = result_tex(summary)
    (source / 'results.tex').write_text(tables, encoding='utf-8')
    build = output / 'Build'
    build.mkdir()
    for pass_number in (1, 2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
            '-output-directory=' + str(build), 'main.tex'], cwd=source, capture_output=True, timeout=120)
        (build / f'pass{pass_number}.stdout.log').write_bytes(result.stdout)
        (build / f'pass{pass_number}.stderr.log').write_bytes(result.stderr)
        assert result.returncode == 0, result.stdout.decode(errors='replace')[-3000:]
    log = (build / 'main.log').read_text(encoding='utf-8', errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    shutil.copy2(build / 'main.pdf', output / 'Supplement.pdf')
    import pymupdf

    render = output / 'Render'
    render.mkdir()
    with pymupdf.open(output / 'Supplement.pdf') as document:
        text = '\n'.join(page.get_text() for page in document)
        assert 'Anonymous authors' in text and document.metadata['author'] == 'Anonymous authors'
        for value in displayed:
            assert value in text or value.replace('-', '\u2212') in text, value
        for index, page in enumerate(document):
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(str(render / f'page-{index + 1:02}.png'))
        pages = len(document)
    save(output / 'Build_Verification.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'supplement_built_and_all_table_values_present', 'pages': pages,
        'pdf_sha256': sha(output / 'Supplement.pdf'), 'summary_sha256': sha(HERE / 'Analysis_01/Summary.json'),
        'displayed_numeric_values_checked': len(displayed), 'anonymous': True,
        'visual_review_pending': True, 'portable_extracted_reproduction_pending': True})
    print(json.dumps({'status': 'built', 'pages': pages, 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
