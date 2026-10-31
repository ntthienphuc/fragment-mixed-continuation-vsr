"""Exercise joins, closed gates and bootstrap on prior evidence, never as new SO results."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from analyze import CONTROL, DECODER, load_old, runtime_comparison
from collect_population import population_metadata, verify_records
from common import read, rows, sha
from paired_statistics import paired
import restore_inputs


class AnalysisChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = read(CONTROL / "Comparison_Plan.json")
        cls.fixture = {}
        for stage in ("source", "assessment"):
            lock = cls.design["old_evidence"][stage]
            summary = read(lock["summary"])
            provenance = summary["provenance"]["MATCHED_FC17"]
            current = []
            for shard in provenance["shards"]:
                file = Path(shard["folder"]) / "Predictions.jsonl"
                assert sha(file) == shard["files"][file.name]
                current.extend(rows(file))
            cls.fixture[stage] = (current, summary, provenance)

    def test_full_populations_match_locked_metadata(self):
        for stage, (current, _, _) in self.fixture.items():
            dataset, spec = population_metadata(stage)
            self.assertEqual(len(current), spec["clips"])
            checks = verify_records(dataset, [item["position"] for item in current], current,
                                    [DECODER], "MATCHED_FC17", "population")
            self.assertEqual(checks, spec["clips"])

    def test_existing_totals_and_group_intervals_reproduce(self):
        for stage, (current, summary, _) in self.fixture.items():
            models, _ = load_old(stage, self.design["old_evidence"][stage], current)
            for expected in summary["comparisons"]:
                left = expected["reference"]
                right = expected["comparison"]
                label = "|".join([stage, "common", left, right])
                observed = paired(models[left], models[right], DECODER, DECODER, label)
                for key, value in observed.items():
                    self.assertEqual(value, expected[key])

    def test_existing_common_runtimes_compare(self):
        for _, summary, provenance in self.fixture.values():
            self.assertEqual(len(runtime_comparison(summary, provenance)), 3)

    def test_wrong_source_and_frame_identity_rejected(self):
        current, _, _ = self.fixture["assessment"]
        for field, replacement in (("source_id", "not-the-archived-source"), ("frames", -1)):
            changed = deepcopy(current)
            changed[0][field] = replacement
            with self.assertRaises(AssertionError):
                load_old("assessment", self.design["old_evidence"]["assessment"], changed)

    def test_wrong_denominator_rejected(self):
        current, _, _ = self.fixture["assessment"]
        changed = deepcopy(current)
        changed[0]["predictions"][DECODER]["counts"]["units"] += 1
        with self.assertRaises(AssertionError):
            load_old("assessment", self.design["old_evidence"]["assessment"], changed)

    def test_incomplete_matrix_cannot_restore(self):
        incomplete = {"status": "training_and_24_shards_backed_up", "source_clips": 11307,
                      "destination_clips": 1167, "finished": []}
        with patch.object(restore_inputs, "read", return_value=incomplete):
            with self.assertRaises(AssertionError):
                restore_inputs.restore()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = {
        "utc": datetime.now(timezone.utc).isoformat(), "tests": outcome.testsRun,
        "passed": outcome.wasSuccessful(), "new_short_only_result": False,
        "scope": "existing corpus identity, count and interval replay; rejected corrupt joins and partial matrix",
        "code_sha256": {path.name: sha(path) for path in Path(__file__).parent.glob("*.py")},
    }
    (CONTROL / "Analysis_Plumbing_Verification.json").write_text(json.dumps(proof, indent=2))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
