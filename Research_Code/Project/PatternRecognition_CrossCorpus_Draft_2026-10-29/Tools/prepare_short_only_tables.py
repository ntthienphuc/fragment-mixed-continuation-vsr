"""Render only complete independently recounted results into two staged LaTeX tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / 'Code'))
from short_only import reproduce
from verify_short_only_manuscript import verify_tables

MODEL_ROWS = [('Full-only', 'MATCHED_FC17', 'full'),
              ('Fixed-time mixed', 'MATCHED_FT17', 'fixed_time'),
              ('Whole-word mixed', 'MATCHED_WW17', 'whole_word'),
              ('Whole-word short-only', 'MATCHED_SO17', 'short_only')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(tables, intervals, exposure):
    counts = {name.removeprefix('short_only_'): {row['model']: row for row in rows}
              for name, rows in tables}
    rows = []
    for label, model, arm in MODEL_ROWS:
        values = [counts[stage][model]['wer_percent'] for stage in ('source', 'assessment')]
        values += [exposure[arm]['exposure'][field] / 1e6 for field in ('input_frames', 'target_tokens')]
        rows.append(label + ' & ' + ' & '.join(f'{value:.3f}' for value in values) + r'\\')
    crop = r'''\begin{table}[tbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Source and destination performance of the matched continuation recipes
at seed 17. The same fitted states use a common decoder on 11,307 VLR and
1,167 ViCocktail clips. Training frame and native-token counts are in millions.}
\label{tab:crop}
\begin{tabular}{lrrrr}
\toprule
Training views & VLR WER (\%) & ViCocktail WER (\%) & Frames & Tokens\\
\midrule
ROWS
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Each arm uses 48,000 source presentations and 6,000 updates. Both mixed arms
share 317 full-clip fallbacks; short-only uses 598 fallbacks (1.25\%).
These fitted states are distinct from the original FC/FM comparison.
Frame and token exposures are not equal.
\end{minipage}
\end{table}
'''.replace('ROWS', '\n'.join(rows))
    pairs = {(row['family'].removeprefix('short_only_'), row['reference']): row for row in intervals}
    rows = []
    for stage, label in [('source', 'VLR'), ('assessment', 'ViCocktail')]:
        for reference, contrast in [('MATCHED_WW17', 'Mixed--short-only'),
                                     ('MATCHED_FC17', 'Full--short-only')]:
            row = pairs[(stage, reference)]
            rows.append(f"{label} & {contrast} & {row['estimate_pp']:.3f} & " +
                        f"[{row['ci95_low_pp']:.3f}, {row['ci95_high_pp']:.3f}]" + r'\\')
    comparisons = r'''\begin{table}[tbp]
\centering\small
\caption{Predeclared comparisons with whole-word short-only continuation.
Differences are reference minus short-only WER in percentage points;
positive values favor short-only. Mixed denotes whole-word mixing.}
\label{tab:shortonly}
\begin{tabular}{llrr}
\toprule
Population & Contrast & Difference & 95\% interval\\
\midrule
ROWS
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Pointwise intervals use 10,000 paired bootstrap resamples of 61 archived
VLR source groups or 20 ViCocktail recording groups, conditional on the fitted
states. Whole-word mixed versus short-only is the primary contrast;
full-only versus short-only is the reference contrast.
\end{minipage}
\end{table}
'''.replace('ROWS', '\n'.join(rows))
    return crop, comparisons


def main(output, check):
    data = PAPER / 'Data/Short_Only'
    proof_path = PAPER / 'QA/Short_Only_Data_Verification.json'
    ready = (data / 'Summary.json').exists() and proof_path.exists()
    if check:
        print(json.dumps({'ready': ready, 'files_written': False}))
        return
    assert ready, 'Complete real short-only exports are required before creating tables'
    proof = json.loads(proof_path.read_text())
    summary = json.loads((data / 'Summary.json').read_text())
    assert proof['status'] == 'complete_short_only_exports_independently_recounted'
    assert not summary.get('synthetic_test_fixture_not_scientific_evidence', False)
    for relative, digest in proof['files'].items():
        assert sha(data / relative) == digest
    assert output is not None and not output.exists(), 'Choose a fresh staging directory'
    tables, intervals = reproduce(data)
    exposure = json.loads((data / 'Training_Exposure.json').read_text())
    crop, comparisons = render(tables, intervals, exposure)
    (output / 'tables').mkdir(parents=True, exist_ok=False)
    (output / 'tables/crop_comparison.tex').write_text(crop, encoding='utf-8')
    (output / 'tables/short_only_comparisons.tex').write_text(comparisons, encoding='utf-8')
    verify_tables(output, tables, intervals, exposure)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'actual_four_state_tables_and_locked_intervals_rendered_and_rechecked',
        'models': 4, 'wer_cells': 8, 'exposure_cells': 8, 'paired_intervals': 4,
        'summary_sha256': sha(data / 'Summary.json'), 'data_receipt_sha256': sha(proof_path),
        'files': {path.relative_to(output).as_posix(): sha(path)
                  for path in sorted((output / 'tables').glob('*.tex'))}}
    (output / 'Receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    main(arguments.output, arguments.check)
