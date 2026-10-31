"""Prepare the final study design; draw result profiles only from the complete verified export."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAPER = Path(__file__).resolve().parents[1]
PHASE = PAPER.parent / 'Research_Short_Only_Control_2026-10-30'
COLORS = ['#246a9a', '#19786e', '#b45d33', '#a33245']
MODELS = ['MATCHED_FC17', 'MATCHED_FT17', 'MATCHED_WW17', 'MATCHED_SO17']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def design(output):
    proof = json.loads((PHASE / 'Recovery_20261030_2040/Training_Verification.json').read_text())
    assert proof['updates'] == 6000 and proof['exposure']['presentations'] == 48000
    assert proof['exposure']['full_fallbacks'] == 598
    exposure = json.loads((PAPER / 'Data/Matched_Crops/Training_Exposure.json').read_text())
    assert exposure['full']['full_slots'] == 48000
    assert exposure['fixed_time']['full_slots'] == exposure['whole_word']['full_slots'] == 24317
    assert all(item['updates'] == 6000 and item['exposure']['presentations'] == 48000
               for item in exposure.values())
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    axis.set(xlim=(0, 7.2), ylim=(0, 5.6))
    axis.axis('off')

    def box(left, bottom, title, body, color):
        width, height = 3.32, 0.93
        axis.add_patch(FancyBboxPatch((left, bottom), width, height,
            boxstyle='round,pad=0.02,rounding_size=0.05', linewidth=.8,
            edgecolor=color, facecolor='#f8fafc'))
        axis.text(left + .11, bottom + .76, title, fontsize=9.7, fontweight='bold', color=color, va='top')
        axis.text(left + .11, bottom + .49, body, fontsize=8.9, color='#253247', va='top', linespacing=1.35)

    axis.text(.15, 5.36, '(a) Original comparison', fontsize=11.2, fontweight='bold', color='#193b5b')
    axis.text(3.75, 5.36, '(b) Matched supervision', fontsize=11.2, fontweight='bold', color='#193b5b')
    axis.text(.15, 5.10, 'FC / FM and same-target context controls', fontsize=8.5, color='#526073')
    axis.text(3.75, 5.10, 'Same source order, seed and 6,000 updates', fontsize=8.5, color='#526073')
    rows = [3.98, 2.90, 1.82, .74]
    left = [
        ('FC | Full clips', 'Complete input\nComplete transcript', '#246a9a'),
        ('FM | Full + fixed-time views', 'Alternate full clips and 2 s crops\nCrop labels include overlapping words', '#19786e'),
        ('FM-ST | Full-view self-teaching', 'FM inputs + detached full-view targets\nTests predictive context', '#77508c'),
        ('CP | Context-preserving views', 'Restore neighboring input frames\nRetain the same crop transcript', '#a26322')]
    right = [
        ('Full-only', '48,000 complete inputs\nComplete transcript at every presentation', COLORS[0]),
        ('Fixed-time mixed', '24,317 full + 23,683 short views\n2 s crops with overlap-based labels', COLORS[1]),
        ('Whole-word mixed', '24,317 full + 23,683 short views\n2.00–3.28 s word-aligned spans', COLORS[2]),
        ('Whole-word short-only', '47,402 short views + 598 full fallbacks\nSame word-span rule; no source clip dropped', COLORS[3])]
    for bottom, first, second in zip(rows, left, right):
        box(.16, bottom, *first)
        box(3.76, bottom, *second)
    axis.plot([.15, 7.08], [.48, .48], color='#b8c4d0', linewidth=.8)
    axis.text(3.6, .22, 'All states: complete video → unchanged recognizer → complete transcript',
              fontsize=9.4, fontweight='bold', color='#193b5b', ha='center', va='center')
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for suffix in ('pdf', 'png'):
        figure.savefig(output / ('training_design.' + suffix), bbox_inches='tight', pad_inches=.04, dpi=180)
    plt.close(figure)


def profiles(output):
    sys.path.insert(0, str(PAPER / 'Code'))
    from short_only import reproduce

    data = PAPER / 'Data/Short_Only'
    tables, _ = reproduce(data)
    matched = {name.removeprefix('short_only_'): {row['model']: row['wer_percent'] for row in rows}
               for name, rows in tables}
    old = {row['model']: row for row in read_csv(PAPER / 'Data/Source_Destination_Profile.csv')
           if row['regime'] == 'common'}
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 7.3))
    measurements = []
    for row in range(3):
        for column in range(2):
            if row == 0:
                stage = 'source' if column == 0 else 'assessment'
                values = [matched[stage][model] for model in MODELS]
                labels = ['Full', 'Fixed-time\nmixed', 'Whole-word\nmixed', 'Whole-word\nshort-only']
                title = 'Matched: ' + ('VLR source' if column == 0 else 'ViCocktail destination')
                models = MODELS
                colors = COLORS
            else:
                prefix = 'VLR' if row == 1 else 'EN'
                models = [prefix + suffix for suffix in ('_B0', '_T0_LAST', '_T1_LAST')]
                metric = 'source_wer' if column == 0 else 'destination_wer'
                values = [float(old[model][metric]) for model in models]
                labels = ['Initializer', 'FC', 'FM']
                names = ['VLR source', 'ViCocktail destination'] if row == 1 else [
                    'LRS2 source', 'WildVSR destination']
                title = 'Original: ' + names[column]
                colors = ['#637587', '#246a9a', '#19786e']
            axis = axes[row, column]
            axis.scatter(range(len(values)), values, c=colors, s=37, zorder=3)
            for index, value in enumerate(values):
                axis.annotate(f'{value:.3f}', (index, value), xytext=(0, 8),
                              textcoords='offset points', ha='center', fontsize=8.3)
                measurements.append({'panel': row * 2 + column, 'title': title,
                                     'model': models[index], 'wer_percent': value})
            span = max(values) - min(values)
            margin = max(span * .27, .3)
            axis.set_ylim(min(values) - margin, max(values) + 1.5 * margin)
            axis.set_xlim(-.4, len(values) - .6)
            axis.set_xticks(range(len(values)), labels, fontsize=8.2)
            axis.set_title(f'({chr(97 + row * 2 + column)}) ' + title, fontsize=9.2)
            axis.set_ylabel('WER (%)')
            axis.grid(axis='y', alpha=.2)
            axis.spines[['right', 'top']].set_visible(False)
    figure.tight_layout(pad=1.25, h_pad=2.0)
    for suffix in ('pdf', 'png'):
        figure.savefig(output / ('source_destination_profile.' + suffix), bbox_inches='tight', dpi=180)
    plt.close(figure)
    return measurements


def main(output, only_design):
    assert not output.exists(), 'Preserve previous figure output'
    if not only_design:
        evidence = json.loads((PAPER / 'QA/Short_Only_Data_Verification.json').read_text())
        assert evidence['status'] == 'complete_short_only_exports_independently_recounted'
        for relative, digest in evidence['files'].items():
            assert sha(PAPER / 'Data/Short_Only' / relative) == digest
    output.mkdir(parents=True, exist_ok=False)
    design(output)
    values = [] if only_design else profiles(output)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'training_design_prepared' if only_design else 'complete_study_figures_rendered',
        'design_only_no_population_result': only_design, 'figure_values': values,
        'visual_inspection_required': True,
        'files': {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / 'Receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-only', action='store_true')
    args = parser.parse_args()
    main(args.output, args.design_only)
