"""Whole-corpus baselines must not silently share the long-window population."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from phase5.common import write_rows, save_json, digest, code_identity
from phase5.data import PreparedData
from phase5.training import train


class CohortTests(unittest.TestCase):
    def test_short_utterance_is_full_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'clip_id': 'short', 'split': 'train', 'train_eligible': False, 'full_eligible': True},
                    {'clip_id': 'long', 'split': 'train', 'train_eligible': True, 'full_eligible': True},
                    {'clip_id': 'bad_ctc', 'split': 'train', 'train_eligible': False, 'full_eligible': False}]
            write_rows(root/'manifest.jsonl', rows)
            write_rows(root/'failures.jsonl', [])
            save_json(root/'receipt.json', {'code_sha256': code_identity(),
                'manifest_sha256': digest(root/'manifest.jsonl'),
                'failures_sha256': digest(root/'failures.jsonl')})
            window = PreparedData(root, training=True)
            full = PreparedData(root, training=True, cohort='full')
            self.assertEqual([r['clip_id'] for r in window.rows], ['long'])
            self.assertEqual([r['clip_id'] for r in full.rows], ['short', 'long'])
            self.assertEqual(len(window.all_rows), 3)

    def test_full_cohort_cannot_feed_window_method(self):
        with self.assertRaises(ValueError):
            train(SimpleNamespace(cohort='full', method='T1', selection_view='full'))
        with self.assertRaises(ValueError):
            train(SimpleNamespace(cohort='full', method='T0', selection_view='window'))
