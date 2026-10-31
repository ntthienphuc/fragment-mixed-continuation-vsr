"""Select one decoder using the complete LRS2 development population only."""

import argparse
from pathlib import Path

from collect_full_results import collect
from execution_common import checked_execution, require
from experiment_core import cohort, independent_counts, validate_choices
from mechanism_common import HERE, save, sha, utc


def choose(population, predictions, order):
    require(bool(population) and len(population) == len(predictions), "Incomplete development population")
    require(len(set(order)) == len(order) and bool(order), "Invalid candidate order")
    require(all(row["split"] == "development" for row in population), "Assessment cannot select a decoder")
    by_id = {row["clip_id"]: row for row in population}
    require(len(by_id) == len(population), "Duplicated population")
    require(len({row["clip_id"] for row in predictions}) == len(predictions), "Duplicated predictions")
    require(set(by_id) == {row["clip_id"] for row in predictions}, "Wrong development population")
    errors = dict.fromkeys(order, 0)
    checked = 0
    for predicted in predictions:
        row = by_id[predicted["clip_id"]]
        require(predicted["split"] == "development" and predicted["source_id"] == row["source_id"],
                "Prediction split or source changed")
        require(set(predicted["predictions"]) == set(order), "Incomplete decoder grid")
        for candidate, result in predicted["predictions"].items():
            counts = independent_counts(row["reference"], result["text"])
            require(counts == result["counts"], "Independent selection recount differs")
            errors[candidate] += counts["errors"]
            checked += 1
    words = sum(len(row["reference"].split()) for row in population)
    require(words > 0, "Empty development denominator")
    return {"decoder": min(order, key=lambda name: errors[name]), "clips": len(population),
            "reference_units": words, "independent_counts_checked": checked, "errors": errors,
            "wer": {name: 100 * value / words for name, value in errors.items()},
            "ties_use_frozen_candidate_order": True}


def main(manifest):
    plan, lock = checked_execution()
    require(not (HERE / "Decoder_Selection.json").exists(), "Preserve the earlier selection")
    records, inputs = collect("development", manifest, plan)
    result = choose(cohort(plan, "en", "development"), records, plan["candidate_order"])
    selection = {"utc": utc(), "status": "all_mechanism_development_decoders_locked_before_assessment",
                 "models": {"EN_F0_LAST": result}, "plan_sha256": sha(HERE / "Plan.json"),
                 "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
                 "selection_code_sha256": sha(__file__), "input_sha256": inputs,
                 "assessment_manifest_sha256": plan["cohorts"]["en"]["assessment"]["sha256"],
                 "test_predictions_read": False, "checkpoint_selection_on_test": False,
                 "destination_calibration_labels_used": False}
    validate_choices(plan, selection)
    save(HERE / "Decoder_Selection.json", selection)
    print({"model": "EN_F0_LAST", "decoder": result["decoder"], "development_clips": result["clips"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    main(parser.parse_args().manifest.resolve())
