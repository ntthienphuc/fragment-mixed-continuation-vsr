"""Choose one source-development decoder per model before destination recognition."""

import argparse
from pathlib import Path
import sys

from english_common import HERE, ROOT, lines, read, require, save, sha, utc

sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
from verify_probe import counts


def choose(population, predictions, candidates):
    require(bool(population) and len(population) == len(predictions), "Incomplete development population")
    require(all(row["split"] == "development" for row in population), "Only source development may select")
    require(len(candidates) == len(set(candidates)) and bool(candidates), "Bad candidate order")
    by_id = {row["clip_id"]: row for row in population}
    require(len(by_id) == len(population), "Duplicate source clips")
    require(len({row["clip_id"] for row in predictions}) == len(predictions), "Duplicated predictions")
    require(set(by_id) == {row["clip_id"] for row in predictions}, "Wrong source population")
    totals = {candidate: 0 for candidate in candidates}
    checked = 0
    for predicted in predictions:
        original = by_id[predicted["clip_id"]]
        require(predicted["split"] == "development", "Prediction stage is not development")
        require(predicted["source_id"] == original["source_id"], "Source group differs")
        require(set(predicted["predictions"]) == set(candidates), "Missing decoder candidate")
        for candidate in candidates:
            result = predicted["predictions"][candidate]
            actual = counts(original["reference"], result["text"])
            require(result["counts"] == actual, "Selection recount differs")
            totals[candidate] += actual["errors"]
            checked += 1
    words = sum(len(row["reference"].split()) for row in population)
    require(words > 0, "No source reference words")
    return {"decoder": min(candidates, key=lambda candidate: totals[candidate]), "clips": len(population),
            "reference_units": words, "errors": totals, "wer": {key: 100 * value / words for key, value in totals.items()},
            "independent_counts_checked": checked, "ties_use_frozen_candidate_order": True}


def main(manifest_path):
    from collect_results import collect

    plan = read(HERE / "Plan.json")
    for name, expected in plan["frozen_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen plan input changed")
    records, inputs = collect("development", manifest_path, plan)
    population = lines(HERE / "Development.jsonl")
    selections = {model: choose(population, records[model], plan["candidate_order"]) for model in plan["models"]}
    save(HERE / "Decoder_Selection.json", {
        "utc": utc(), "status": "all_source_development_decoders_locked_before_test", "models": selections,
        "plan_sha256": sha(HERE / "Plan.json"), "selection_code_sha256": sha(__file__), "input_sha256": inputs,
        "assessment_manifest_sha256": sha(HERE / "Assessment.jsonl"), "test_predictions_read": False,
        "checkpoint_selection_on_test": False, "destination_calibration_labels_used": False})
    print({model: value["decoder"] for model, value in selections.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    main(parser.parse_args().manifest.resolve())
