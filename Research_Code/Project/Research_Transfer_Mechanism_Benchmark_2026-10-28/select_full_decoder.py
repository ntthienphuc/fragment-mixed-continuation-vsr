"""Select decoders on the declared development population for each language."""

import argparse
from pathlib import Path

from collect_full_results import collect
from execution_common import checked_execution, require
from experiment_core import cohort, independent_counts, validate_choices
from mechanism_common import HERE, read, save, sha, utc


def calibration_provenance(plan):
    expected = {"vi": "target_corpus_development", "en": "source_corpus_development"}
    regimes = plan["calibration_regime"]
    require(regimes == expected, "Calibration regimes differ from this frozen comparison")
    by_language = {language: value == "target_corpus_development" for language, value in regimes.items()}
    return {"calibration_regime": dict(regimes),
            "destination_calibration_labels_used_by_language": by_language,
            "destination_calibration_labels_used": any(by_language.values())}


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
    registry = read(HERE / "model_registry.json")["models"]
    result = {model: choose(cohort(plan, registry[model]["language"], "development"),
                            records[model], plan["candidate_order"]) for model in plan["models"]}
    selection = {"utc": utc(), "status": "all_mechanism_development_decoders_locked_before_assessment",
                 "models": result, "plan_sha256": sha(HERE / "Plan.json"),
                 "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
                 "selection_code_sha256": sha(__file__), "input_sha256": inputs,
                 "assessment_manifest_sha256": {language: plan["cohorts"][language]["assessment"]["sha256"] for language in ("vi", "en")},
                 "test_predictions_read": False, "checkpoint_selection_on_test": False,
                 **calibration_provenance(plan)}
    validate_choices(plan, selection)
    save(HERE / "Decoder_Selection.json", selection)
    print({model: {"decoder": value["decoder"], "clips": value["clips"]} for model, value in result.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    main(parser.parse_args().manifest.resolve())
