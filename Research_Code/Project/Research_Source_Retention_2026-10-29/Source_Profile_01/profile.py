"""Partition completed English predictions at fixed recipe-related duration boundaries."""

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "Analysis_01"
sys.path.insert(0, str(ANALYSIS))
from collect import digest, metadata, model, normalized, plans, require

DECODER = "beam_c0.1"
NAMES = ("EN_B0", "EN_T0_LAST", "EN_T1_LAST")
FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def band(frames):
    if frames < 50:
        return "below_2_seconds"
    if frames < 82:
        return "2_to_below_3.28_seconds"
    return "at_least_3.28_seconds"


def totals(records, positions):
    result = {field: sum(records[index]["predictions"][DECODER]["counts"][field]
                        for index in positions) for field in FIELDS}
    predicted = sum(len(normalized(records[index]["predictions"][DECODER]["text"]).split())
                    for index in positions)
    require(predicted == result["units"] - result["deletions"] + result["insertions"],
            "Output word counts do not reconcile with the edit decomposition")
    return {**result, "wer_percent": 100 * result["errors"] / result["units"],
            "predicted_units": predicted, "prediction_reference_ratio": predicted / result["units"]}


def main():
    parent, assessment = plans()
    rows, pixels = metadata("lrs2")
    completed = {}
    provenance = []
    for name in NAMES:
        state, records, sources = model("lrs2", name, parent, assessment, rows, pixels)
        require(state["status"] == "complete_verified_population", "An English state is incomplete")
        completed[name] = records
        provenance.extend(sources)
    previous_path = ANALYSIS / "Main_Contrast_01/Summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    groups = {name: [index for index, row in enumerate(rows) if band(row["frames"]) == name]
              for name in ("below_2_seconds", "2_to_below_3.28_seconds", "at_least_3.28_seconds")}
    require(sorted(index for values in groups.values() for index in values) == list(range(len(rows))),
            "Strata must partition the complete cohort")
    summaries = []
    for name, positions in groups.items():
        results = {state: totals(records, positions) for state, records in completed.items()}
        differences = []
        for index in positions:
            left = completed["EN_T0_LAST"][index]["predictions"][DECODER]["counts"]["errors"]
            right = completed["EN_T1_LAST"][index]["predictions"][DECODER]["counts"]["errors"]
            differences.append(left - right)
        summaries.append({"stratum": name, "clips": len(positions),
                          "source_groups": len({rows[index]["source_id"] for index in positions}),
                          "median_seconds": statistics.median(rows[index]["frames"] / 25 for index in positions),
                          "results": results, "fc_minus_fm_errors": sum(differences),
                          "fm_improved_clips": sum(value > 0 for value in differences),
                          "fm_worsened_clips": sum(value < 0 for value in differences),
                          "fm_tied_clips": sum(value == 0 for value in differences)})
    for name in NAMES:
        baseline = next(row for row in previous["results"] if row["model"] == name and row["regime"] == "common")
        for field in FIELDS:
            require(sum(group["results"][name][field] for group in summaries) == baseline[field],
                    "Strata do not reconcile with the independently completed cohort")
    require(sum(group["fc_minus_fm_errors"] for group in summaries) == -97, "FC/FM difference changed")
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cohort_profile_reconciled",
               "retrospective": True, "decoder": DECODER, "clips": len(rows),
               "median_seconds": statistics.median(row["frames"] / 25 for row in rows),
               "under_native_training_minimum": sum(row["frames"] < 82 for row in rows),
               "under_native_training_minimum_percent": 100 * sum(row["frames"] < 82 for row in rows) / len(rows),
               "strata": summaries, "inputs": provenance, "parent_summary_sha256": digest(previous_path.read_bytes()),
               "code_sha256": digest(Path(__file__).read_bytes()),
               "protocol_sha256": digest((HERE / "Protocol.md").read_bytes()),
               "blank_posteriors_measured": False, "recognition_rerun": False,
               "manuscript_changed": False, "new_hypothesis_confirmed": False}
    output = HERE / "Results_01"
    output.mkdir(exist_ok=False)
    with (output / "Summary.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in ("status", "clips", "median_seconds",
                     "under_native_training_minimum_percent", "strata")}))


if __name__ == "__main__":
    main()
