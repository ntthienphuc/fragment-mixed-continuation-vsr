"""Lock one decoder per model using complete development predictions only."""

import argparse
import sys

from common import HERE, ROOT, lines, read, save, sha, utc

sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
from verify_probe import counts


def choose(cohort, predictions, candidates):
    """Use aggregate edit errors, not mean per-clip WER or rounded percentages."""
    assert cohort and candidates and len(set(candidates)) == len(candidates)
    assert all(row["split"] == "development" for row in cohort)
    references = {row["clip_id"]: row for row in cohort}
    assert len(references) == len(cohort)
    assert len(predictions) == len(cohort)
    assert len({row["clip_id"] for row in predictions}) == len(predictions)
    assert set(references) == {row["clip_id"] for row in predictions}
    totals = {candidate: 0 for candidate in candidates}
    units = sum(len(row["reference"].split()) for row in cohort)
    assert units > 0
    checks = 0
    for row in predictions:
        reference = references[row["clip_id"]]
        assert row["split"] == "development"
        assert row["source_id"] == reference["source_id"]
        assert set(row["predictions"]) == set(candidates)
        for candidate in candidates:
            result = row["predictions"][candidate]
            actual = counts(reference["reference"], result["text"])
            assert actual == result["counts"], (row["clip_id"], candidate)
            totals[candidate] += actual["errors"]
            checks += 1
    selected = min(candidates, key=lambda candidate: totals[candidate])
    return {"decoder": selected, "reference_units": units, "clips": len(cohort),
            "errors": totals, "wer": {key: 100.0 * value / units for key, value in totals.items()},
            "independent_prediction_counts_checked": checks, "ties_use_frozen_candidate_order": True}


def main(manifest_path):
    plan = read(HERE / "Plan.json")
    for name, expected in plan["frozen_sha256"].items():
        assert sha(ROOT / name) == expected, name
    manifest = read(manifest_path)
    assert manifest["status"] == "complete_verified_development_bundle"
    assert manifest["plan_sha256"] == sha(HERE / "Plan.json")
    models = [*plan["matched_models"], plan["practical_reference"]]
    assert set(manifest["models"]) == set(models)
    cohort = lines(HERE / "Development.jsonl")
    assert len(cohort) == 768
    expected_entries = read(HERE / "model_registry.json")["models"]
    expected_entries.update(read(HERE / "Native_Reference_Registry.json")["models"])
    choices = {}
    manifest_key = manifest_path.resolve().relative_to(ROOT.resolve()).as_posix()
    bound = {manifest_key: sha(manifest_path)}
    for model in models:
        item = manifest["models"][model]
        path = ROOT / item["predictions_path"]
        proof_path = ROOT / item["verification_path"]
        assert sha(path) == item["predictions_sha256"]
        assert sha(proof_path) == item["verification_sha256"]
        proof = read(proof_path)
        assert proof["status"] == "passed" and proof["model"] == model
        assert proof["split"] == "development" and proof["clips"] == 768
        assert proof["plan_sha256"] == sha(HERE / "Plan.json")
        assert proof["predictions_sha256"] == sha(path)
        assert proof["checkpoint_sha256"] == expected_entries[model]["checkpoint_sha256"]
        assert proof["candidate_order"] == plan["candidate_order"]
        predictions = lines(path)
        assert all(row["model"] == model for row in predictions)
        choices[model] = choose(cohort, predictions, plan["candidate_order"])
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
        bound[proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
    result = {"status": "all_development_decoders_locked_before_test", "utc": utc(),
              "models": choices, "plan_sha256": sha(HERE / "Plan.json"),
              "selection_code_sha256": sha(__file__), "input_sha256": bound,
              "assessment_manifest_sha256": sha(HERE / "Assessment.jsonl"),
              "test_predictions_read": False, "checkpoint_selection_on_test": False}
    save(HERE / "Decoder_Selection.json", result)
    print({model: value["decoder"] for model, value in choices.items()}, flush=True)


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    main(parser.parse_args().manifest)
