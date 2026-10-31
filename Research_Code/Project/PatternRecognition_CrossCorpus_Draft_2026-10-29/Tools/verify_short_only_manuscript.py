"""Independently check the final four-state table, new paired intervals and anonymous PDF."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pymupdf

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / 'Code'))
from short_only import reproduce

MODELS = {'Full-only': ('MATCHED_FC17', 'full'),
          'Fixed-time mixed': ('MATCHED_FT17', 'fixed_time'),
          'Whole-word mixed': ('MATCHED_WW17', 'whole_word'),
          'Whole-word short-only': ('MATCHED_SO17', 'short_only')}
CONTRASTS = {'Mixed--short-only': 'MATCHED_WW17', 'Full--short-only': 'MATCHED_FC17'}
STAGES = {'VLR': 'source', 'ViCocktail': 'assessment'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(path):
    return [[part.strip() for part in line.removesuffix(r'\\').split('&')]
            for line in path.read_text(encoding='utf-8').splitlines() if '&' in line]


def verify_tables(source, tables, intervals, exposure):
    assert len(intervals) == 4 and sum(len(rows) for _, rows in tables) == 8
    counts = {name.removeprefix('short_only_'): {row['model']: row for row in rows}
              for name, rows in tables}
    actual = [row for row in cells(source / 'tables/crop_comparison.tex') if row[0] in MODELS]
    assert len(actual) == 4 and {row[0] for row in actual} == set(MODELS)
    for row in actual:
        model, arm = MODELS[row[0]]
        expected = [f"{counts[stage][model]['wer_percent']:.3f}" for stage in ('source', 'assessment')]
        values = exposure[arm]['exposure']
        expected += [f"{values['input_frames'] / 1e6:.3f}", f"{values['target_tokens'] / 1e6:.3f}"]
        assert row[1:] == expected, (model, row[1:], expected)
    saved_pairs = {(row['family'].removeprefix('short_only_'), row['reference']): row for row in intervals}
    assert len(saved_pairs) == 4
    rows = [row for row in cells(source / 'tables/short_only_comparisons.tex') if row[0] in STAGES]
    assert len(rows) == 4
    seen = set()
    for row in rows:
        assert len(row) == 4 and row[1] in CONTRASTS
        key = (STAGES[row[0]], CONTRASTS[row[1]])
        assert key not in seen
        seen.add(key)
        expected = saved_pairs[key]
        assert row[2] == f"{expected['estimate_pp']:.3f}"
        assert row[3] == f"[{expected['ci95_low_pp']:.3f}, {expected['ci95_high_pp']:.3f}]"
    assert seen == {(stage, model) for stage in STAGES.values() for model in CONTRASTS.values()}


def verify():
    data = PAPER / 'Data/Short_Only'
    evidence = read(PAPER / 'QA/Short_Only_Data_Verification.json')
    assert evidence['status'] == 'complete_short_only_exports_independently_recounted'
    for relative, digest in evidence['files'].items():
        assert sha(data / relative) == digest
    summary = read(data / 'Summary.json')
    assert not summary.get('synthetic_test_fixture_not_scientific_evidence', False)
    tables, intervals = reproduce(data)
    exposure = read(data / 'Training_Exposure.json')
    source = PAPER / 'Source'
    verify_tables(source, tables, intervals, exposure)
    main = (source / 'main.tex').read_text(encoding='utf-8')
    assert r'\author{Anonymous authors}' in main and r'\affiliation' not in main
    method = (source / 'sections/method.tex').read_text(encoding='utf-8')
    assert all(value in method for value in ('47,402', '598', '2,719,321', '422,153', '470,153'))
    experiments = (source / 'sections/experiments.tex').read_text(encoding='utf-8')
    assert all(value in experiments for value in ('short-only', '61', '20', '10,000', 'equivalence margin'))
    results = (source / 'sections/results.tex').read_text(encoding='utf-8')
    assert r'\input{tables/short_only_comparisons}' in results
    with pymupdf.open(PAPER / 'Manuscript.pdf') as document:
        assert 'Anonymous authors' in document[0].get_text()
        authors = [value.strip() for value in document.metadata.get('author', '').split(';')]
        assert all(value in ('', 'Anonymous authors') for value in authors), authors
    return {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'short_only_tables_intervals_exposure_and_anonymity_verified',
        'matched_models': 4, 'aggregate_wer_cells': 8, 'training_exposure_cells': 8,
        'paired_intervals': 4, 'anonymous_review': True, 'new_blind_test': False,
        'pdf_sha256': sha(PAPER / 'Manuscript.pdf'), 'summary_sha256': sha(data / 'Summary.json'),
        'source_sha256': {path.relative_to(source).as_posix(): sha(path)
                          for path in sorted(source.rglob('*.tex'))},
        'statistical_recount_independent_of_latex_table_generation': True,
        'scientific_narrative_and_visual_review_required_separately': True}


def main():
    result = verify()
    (PAPER / 'QA/Short_Only_Manuscript_Verification.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
