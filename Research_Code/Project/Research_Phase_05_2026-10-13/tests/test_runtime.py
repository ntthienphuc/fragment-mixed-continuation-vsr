import random
import unittest
import numpy as np
import torch
from phase5.runtime import rng_state, restore_rng, samples
from phase5.common import ROOT, read_json
from phase5.model import METHODS
import csv


class Runtime(unittest.TestCase):
    def test_rng_roundtrip(self):
        state = rng_state()
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        restore_rng(state)
        actual = (random.random(), np.random.rand(4), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(np.array_equal(expected[1], actual[1]))
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_prefetch_preserves_order(self):
        class Data:
            def sample(self, number, seed):
                return number, seed
        expected = list(samples(Data(), 3, 23, 17, 0))
        self.assertEqual(expected, list(samples(Data(), 3, 23, 17, 3)))

    def test_method_mapping_and_plan(self):
        with (ROOT / 'Method_Mapping.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r['method_id'] for r in rows if r['method_id'] != 'F0'], list(METHODS))
        self.assertEqual(len({r['method_id'] for r in rows}), len(rows))
        baseline = next(r for r in rows if r['method_id'] == 'F0')
        self.assertEqual((baseline['architecture_id'], baseline['train_cohort'], baseline['selection_view']),
                         ('T0', 'full', 'full'))
        self.assertTrue(all(r['architecture_id'] in METHODS for r in rows))
        plan = read_json(ROOT / 'plans/pilot.json')
        self.assertEqual(len(plan['jobs']), 14)
        for job in plan['jobs']:
            args = job['args']
            self.assertEqual(args[args.index('--accumulation') + 1], '8')
            self.assertEqual(args[args.index('--scope') + 1], 'all')
            self.assertEqual(args[args.index('--steps') + 1], '200')
