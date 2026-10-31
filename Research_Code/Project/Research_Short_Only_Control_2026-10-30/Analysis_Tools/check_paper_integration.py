"""Test the portable new branch using explicitly synthetic SO aliases of archived FC data."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from analyze import CONTROL, DECODER, load_old
from common import read, rows, sha
from export_tables import export
from paired_statistics import paired, totals
import export_paper

PAPER = CONTROL.parent / 'PatternRecognition_CrossCorpus_Draft_2026-10-29'
sys.path.insert(0, str(PAPER / 'Code'))
from short_only import reproduce


class PaperChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='lip_synthetic_so_fixture_')
        cls.root = Path(cls.temporary.name)
        cls.folder = cls.root / 'Short_Only'
        cls.folder.mkdir()
        design = read(CONTROL / 'Comparison_Plan.json')
        plan_path = cls.folder / 'Comparison_Plan.json'
        shutil.copy2(CONTROL / 'Comparison_Plan.json', plan_path)
        proof_path = cls.folder / 'Training_Verification.json'
        shutil.copy2(CONTROL / 'Recovery_20261030_2040/Training_Verification.json', proof_path)
        stages = {}
        for stage in ('source', 'assessment'):
            lock = design['old_evidence'][stage]
            archived = read(lock['summary'])
            cohort = []
            for shard in archived['provenance']['MATCHED_FC17']['shards']:
                path = Path(shard['folder']) / 'Predictions.jsonl'
                assert sha(path) == shard['files'][path.name]
                cohort.extend(rows(path))
            models, _ = load_old(stage, lock, cohort)
            models['MATCHED_SO17'] = deepcopy(models['MATCHED_FC17'])
            comparisons = []
            for role in ('primary', 'reference'):
                left, right = design[role + '_comparison']
                comparison = paired(models[left], models[right], DECODER, DECODER,
                                    '|'.join([stage, 'common', left, right]))
                comparisons.append({'role': role, 'reference': left, 'comparison': right, **comparison})
            export(stage, models, cohort, comparisons, cls.folder / stage)
            stages[stage] = {'clips': len(cohort), 'readout': 'common', 'decoder': DECODER,
                'results': {model: totals(population, DECODER) for model, population in models.items()},
                'comparisons': comparisons}
        summary = {'status': 'both_locked_populations_recounted_and_compared', 'stages': stages,
            'comparison_plan_sha256': sha(plan_path), 'training_verification_sha256': sha(proof_path),
            'conditional_on_fitted_checkpoints': True, 'multiplicity_adjusted': False,
            'new_blind_test': False, 'equivalence_margin_declared': False,
            'synthetic_test_fixture_not_scientific_evidence': True,
            'output_sha256': {path.relative_to(cls.folder).as_posix(): sha(path)
                              for path in cls.folder.rglob('*.csv')}}
        cls.summary_path = cls.folder / 'Summary.json'
        cls.summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        cls.original_summary = cls.summary_path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def tearDown(self):
        self.summary_path.write_bytes(self.original_summary)

    def test_full_new_branch_recounts_fixture_and_all_four_intervals(self):
        tables, intervals = reproduce(self.folder)
        self.assertEqual([name for name, _ in tables], ['short_only_source', 'short_only_assessment'])
        self.assertEqual(sum(len(rows) for _, rows in tables), 8)
        self.assertEqual(len(intervals), 4)
        for _, population in tables:
            by_model = {row['model']: row for row in population}
            self.assertEqual(by_model['MATCHED_SO17']['errors'], by_model['MATCHED_FC17']['errors'])
        references = [row for row in intervals if row['reference'] == 'MATCHED_FC17']
        self.assertTrue(all(row['estimate_pp'] == row['ci95_low_pp'] == row['ci95_high_pp'] == 0
                            for row in references))

    def test_changed_summary_interval_rejected(self):
        summary = read(self.summary_path)
        summary['stages']['source']['comparisons'][0]['ci95_high_pp'] += 0.001
        self.summary_path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Summary interval differs'):
            reproduce(self.folder)

    def test_missing_population_rejected(self):
        summary = read(self.summary_path)
        del summary['stages']['source']
        self.summary_path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Missing population'):
            reproduce(self.folder)

    def test_changed_csv_rejected_by_summary_hash(self):
        target = self.folder / 'source/Results.csv'
        original = target.read_bytes()
        try:
            target.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Exported evidence changed'):
                reproduce(self.folder)
        finally:
            target.write_bytes(original)

    def test_unsupported_equivalence_claim_rejected(self):
        summary = read(self.summary_path)
        summary['equivalence_margin_declared'] = True
        self.summary_path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unsupported claim'):
            reproduce(self.folder)

    def test_export_gate_rejects_missing_analysis(self):
        missing = self.root / 'missing_analysis'
        with patch.object(export_paper, 'ANALYSIS', missing):
            self.assertIsNone(export_paper.prerequisites())
        self.assertFalse(missing.exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PaperChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': result.wasSuccessful(),
        'tests': result.testsRun, 'new_short_only_results_used': False,
        'short_only_fixture_is_synthetic': True,
        'fixture_description': 'Previously verified FC counts duplicated under SO test alias; no neural result',
        'scope': 'portable four-state/two-population branch, intervals, integrity and closed real-result gate',
        'code_sha256': {path.name: sha(path) for path in (PAPER / 'Code').glob('*.py')}}
    destination = CONTROL / 'Recovery_20261030_2040/Paper_Integration_Tests.json'
    destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
