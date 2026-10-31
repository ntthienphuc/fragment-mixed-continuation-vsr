"""Round-trip prior completed populations through exports and independent portable code."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

from analyze import CONTROL, load_old
from common import read, rows, sha
from export_tables import export, prepare

PAPER = CONTROL.parent / 'PatternRecognition_CrossCorpus_Draft_2026-10-29'
sys.path.insert(0, str(PAPER / 'Code'))
from recount import load_counts, verify_table
from uncertainty import source_or_crop_pairs


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = read(CONTROL / 'Comparison_Plan.json')
        cls.fixture = {}
        for stage in ('source', 'assessment'):
            lock = cls.design['old_evidence'][stage]
            summary = read(lock['summary'])
            current = []
            for shard in summary['provenance']['MATCHED_FC17']['shards']:
                path = Path(shard['folder']) / 'Predictions.jsonl'
                assert sha(path) == shard['files'][path.name]
                current.extend(rows(path))
            models, archived = load_old(stage, lock, current)
            cls.fixture[stage] = (models, current, archived['comparisons'])

    def test_two_real_populations_roundtrip_independently(self):
        for stage, (models, cohort, comparisons) in self.fixture.items():
            with tempfile.TemporaryDirectory(prefix='lip_prior_export_') as directory:
                folder = Path(directory) / stage
                receipt = export(stage, models, cohort, comparisons, folder)
                counts = load_counts([folder / 'Per_Clip_Counts.csv'], 'model', 'readout', 'units')
                actual = verify_table(folder / 'Results.csv', counts, 'model', 'readout', 'units')
                intervals = source_or_crop_pairs(folder / 'Paired_Comparisons.csv', counts, stage)
                self.assertEqual(len(actual), 3)
                self.assertEqual(len(intervals), 3)
                self.assertEqual(receipt['count_rows'], 3 * len(cohort))
                self.assertTrue(all(row['clips'] == len(cohort) for row in actual))

    def test_missing_cohort_record_rejected(self):
        models, cohort, comparisons = self.fixture['assessment']
        with self.assertRaises(AssertionError):
            prepare('assessment', models, cohort[:-1], comparisons)

    def test_wrong_denominator_rejected(self):
        models, cohort, comparisons = self.fixture['assessment']
        changed = deepcopy(models)
        changed['MATCHED_FC17'][0]['predictions']['beam_c0.1']['counts']['units'] += 1
        with self.assertRaises(AssertionError):
            prepare('assessment', changed, cohort, comparisons)

    def test_wrong_group_identifier_rejected(self):
        models, cohort, comparisons = self.fixture['assessment']
        changed = deepcopy(models)
        changed['MATCHED_FC17'][0]['source_id'] = 'wrong-recording'
        with self.assertRaises(AssertionError):
            prepare('assessment', changed, cohort, comparisons)

    def test_wrong_interval_protocol_rejected(self):
        models, cohort, comparisons = self.fixture['assessment']
        for field, value in [('bootstrap_resamples', 9999), ('source_groups', 19),
                             ('multiplicity_adjusted', True)]:
            changed = deepcopy(comparisons)
            changed[0][field] = value
            with self.assertRaises(AssertionError):
                prepare('assessment', models, cohort, changed)

    def test_duplicate_contrast_rejected(self):
        models, cohort, comparisons = self.fixture['assessment']
        with self.assertRaises(AssertionError):
            prepare('assessment', models, cohort, comparisons + comparisons[:1])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExportChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'tests': outcome.testsRun,
        'passed': outcome.wasSuccessful(), 'new_short_only_result': False,
        'prior_completed_count_rows_checked': 3 * (11307 + 1167),
        'prior_paired_intervals_independently_reproduced': 6,
        'scope': 'archived controls exported and recounted using portable paper code',
        'code_sha256': {path.name: sha(path) for path in Path(__file__).parent.glob('*.py')}}
    destination = CONTROL / 'Recovery_20261030_2040/Export_Verification.json'
    destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
