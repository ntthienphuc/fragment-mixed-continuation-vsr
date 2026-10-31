"""Exercise manuscript numeric checks on synthetic fixtures and the closed delivery gate."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

from check_paper_integration import PaperChecks, PAPER
from common import read, rows, sha

sys.path.insert(0, str(PAPER / 'Tools'))
from verify_short_only_manuscript import verify_tables
from short_only import reproduce
import short_only_delivery


class FinalPaperChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PaperChecks.setUpClass()
        cls.fixture = PaperChecks
        cls.source = PaperChecks.root / 'Source'
        (cls.source / 'tables').mkdir(parents=True)
        cls.tables, cls.intervals = reproduce(PaperChecks.folder)
        cls.exposure = read(PAPER / 'Data/Matched_Crops/Training_Exposure.json')
        proof = read(PaperChecks.folder / 'Training_Verification.json')
        cls.exposure['short_only'] = {'exposure': proof['exposure']}
        # Existing control rows remain literal; SO is deliberately an FC alias only for this test.
        cls.crop = '\n'.join([
            r'Full-only & 38.029 & 67.369 & 7.307 & 1.145\\',
            r'Fixed-time mixed & 38.580 & 63.846 & 4.888 & 0.798\\',
            r'Whole-word mixed & 38.606 & 63.388 & 5.013 & 0.784\\',
            r'Whole-word short-only & 38.029 & 67.369 & 2.719 & 0.422\\'])
        comparisons = []
        import csv

        for stage, label in [('source', 'VLR'), ('assessment', 'ViCocktail')]:
            path = PaperChecks.folder / stage / 'Paired_Comparisons.csv'
            with path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    contrast = 'Mixed--short-only' if row['role'] == 'primary' else 'Full--short-only'
                    point = float(row['reference_minus_comparison_pp'])
                    lower = float(row['ci95_low_pp'])
                    upper = float(row['ci95_high_pp'])
                    comparisons.append(f'{label} & {contrast} & {point:.3f} & [{lower:.3f}, {upper:.3f}]' + r'\\')
        cls.pairs = '\n'.join(comparisons)

    @classmethod
    def tearDownClass(cls):
        PaperChecks.tearDownClass()

    def setUp(self):
        (self.source / 'tables/crop_comparison.tex').write_text(self.crop, encoding='utf-8')
        (self.source / 'tables/short_only_comparisons.tex').write_text(self.pairs, encoding='utf-8')

    def verify(self):
        verify_tables(self.source, self.tables, self.intervals, self.exposure)

    def test_complete_fixture_matches_independent_recount(self):
        self.verify()

    def test_changed_short_only_source_score_rejected(self):
        changed = self.crop.replace('Whole-word short-only & 38.029', 'Whole-word short-only & 38.030')
        (self.source / 'tables/crop_comparison.tex').write_text(changed, encoding='utf-8')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_swapped_corpora_rejected(self):
        changed = self.crop.replace('Full-only & 38.029 & 67.369', 'Full-only & 67.369 & 38.029')
        (self.source / 'tables/crop_comparison.tex').write_text(changed, encoding='utf-8')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_training_exposure_rejected(self):
        changed = self.crop.replace('2.719 & 0.422', '2.719 & 0.423')
        (self.source / 'tables/crop_comparison.tex').write_text(changed, encoding='utf-8')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_missing_comparison_rejected(self):
        changed = '\n'.join(self.pairs.splitlines()[:-1])
        (self.source / 'tables/short_only_comparisons.tex').write_text(changed, encoding='utf-8')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_interval_rejected(self):
        changed = self.pairs.replace('[0.000, 0.000]', '[0.000, 0.001]', 1)
        self.assertNotEqual(changed, self.pairs)
        (self.source / 'tables/short_only_comparisons.tex').write_text(changed, encoding='utf-8')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_no_final_delivery_without_required_evidence(self):
        from unittest.mock import patch

        output = self.fixture.root / 'Forbidden_Delivery'
        missing = self.fixture.root / 'Absent_Full_Assessment.json'
        with patch.object(short_only_delivery, 'REQUIRED', [missing]):
            with self.assertRaises(FileNotFoundError):
                short_only_delivery.main(False, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FinalPaperChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': result.wasSuccessful(),
        'tests': result.testsRun, 'new_short_only_result_used': False,
        'synthetic_so_alias_fixture_only': True, 'final_delivery_performed': False,
        'positive_real_data_delivery_validation_pending': True,
        'scope': 'Independent table recount comparison, interval/exposure errors, incomplete-delivery rejection',
        'code_sha256': {name: sha(PAPER / 'Tools' / name) for name in (
            'verify_short_only_manuscript.py', 'short_only_delivery.py')}}
    destination = PAPER.parent / 'Research_Short_Only_Control_2026-10-30/Recovery_20261030_2040'
    (destination / 'Final_Paper_Gate_Tests.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
