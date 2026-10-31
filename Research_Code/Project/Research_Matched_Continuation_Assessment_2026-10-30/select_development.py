"""Lock a new model's decoder from complete verified development shards only."""

import argparse
from pathlib import Path

from common import HERE, ROOT, load_module, read, rows, save, sha, utc


def choose(cohort, predictions, candidates):
    assert cohort and candidates and len(set(candidates)) == len(candidates)
    assert all(row["stage"] == "development" for row in cohort)
    references = {row["clip_id"]: row for row in cohort}
    assert len(references) == len(cohort) == len(predictions)
    assert len({row["clip_id"] for row in predictions}) == len(predictions)
    assert set(references) == {row["clip_id"] for row in predictions}
    independent = load_module("matched_independent_counts", ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py")
    totals = {candidate: 0 for candidate in candidates}
    units = sum(len(row["reference"].split()) for row in cohort)
    assert units > 0
    for row in predictions:
        reference = references[row["clip_id"]]
        assert row["stage"] == "development" and row["source_id"] == reference["source_id"]
        assert list(row["predictions"]) == candidates
        for candidate in candidates:
            result = row["predictions"][candidate]
            actual = independent.counts(reference["reference"], result["text"])
            assert actual == result["counts"], (row["clip_id"], candidate)
            totals[candidate] += actual["errors"]
    selected = min(candidates, key=lambda candidate: totals[candidate])
    return {"decoder": selected, "reference_units": units, "clips": len(cohort), "errors": totals,
            "wer": {key: 100.0 * value / units for key, value in totals.items()},
            "independent_prediction_counts_checked": len(cohort) * len(candidates),
            "ties_use_frozen_candidate_order": True}


def main(arguments):
    plan = read(HERE / "Plan.json")
    assert arguments.model in plan["models"]
    registry_sha = sha(arguments.binding / "model_registry.json")
    binding = read(arguments.binding / "Binding.json")
    assert binding["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    assert binding["registry_sha256"] == registry_sha
    spec = plan["destination"]["development"]
    source = arguments.destination_root / spec["manifest"]
    assert sha(source) == spec["sha256"]
    cohort = [{**row, "stage": row["split"]} for row in rows(source)]
    assert len(cohort) == spec["clips"]
    choices = plan["secondary_candidate_order"]
    predictions = []
    bound = {}
    for index in range(8):
        task = f"development_{index:02}"
        folder = arguments.shards / task
        proof = read(folder / "Verification.json")
        assert proof["status"] == "all_records_recounted_and_fresh_replay_passed"
        assert proof["mode"] == "population" and proof["stage"] == "development"
        assert proof["task"] == task and proof["readout"] == "grid" and proof["model"] == arguments.model
        assert proof["registry_sha256"] == registry_sha and proof["candidates"] == choices
        assert proof["assessment_plan_sha256"] == sha(HERE / "Plan.json")
        assert proof["code_lock_sha256"] == sha(HERE / "Code_Lock.json")
        for name, key in (("Predictions.jsonl", "predictions_sha256"), ("Identity.json", "identity_sha256")):
            assert sha(folder / name) == proof[key]
        chunk = rows(folder / "Predictions.jsonl")
        assert len(chunk) == proof["clips"]
        assert all(row["model"] == arguments.model and row["mode"] == "population" for row in chunk)
        predictions.extend(chunk)
        bound[task] = {"predictions_sha256": sha(folder / "Predictions.jsonl"),
                       "verification_sha256": sha(folder / "Verification.json"), "clips": len(chunk)}
    assert sorted(row["position"] for row in predictions) == list(range(spec["clips"]))
    result = choose(cohort, predictions, choices)
    assert result["reference_units"] == spec["reference_units"]
    save(arguments.output, {
        "utc": utc(), "status": "complete_development_selection", "model": arguments.model, **result,
        "registry_sha256": registry_sha, "assessment_plan_sha256": sha(HERE / "Plan.json"),
        "code_lock_sha256": sha(HERE / "Code_Lock.json"), "selection_code_sha256": sha(__file__),
        "development_manifest_sha256": sha(source), "bound_shards": bound, "assessment_predictions_read": False,
        "primary_common_decoder_unchanged": plan["primary_decoder"],
    })
    print({"model": arguments.model, "decoder": result["decoder"], "clips": result["clips"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
