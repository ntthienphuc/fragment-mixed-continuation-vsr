"""Compare selection with complete archived development evidence, not new inference."""

from copy import deepcopy

from common import HERE, ROOT, read, rows, save, sha, utc
from select_development import choose


def main():
    original = ROOT / "Research_Transfer_Full_Benchmark_2026-10-27"
    manifest = read(original / "Development_Bundle_01/Manifest.json")
    expected = read(original / "Decoder_Selection.json")
    plan = read(HERE / "Plan.json")
    assert sha(original / "Development.jsonl") == plan["destination"]["development"]["sha256"]
    cohort = [{**row, "stage": row["split"]} for row in rows(original / "Development.jsonl")]
    candidates = plan["secondary_candidate_order"]
    checked = {}
    for model in ("VLR_B0", "VLR_T1_LAST"):
        item = manifest["models"][model]
        path = ROOT / item["predictions_path"]
        assert sha(path) == item["predictions_sha256"]
        predictions = [{**row, "stage": row["split"]} for row in rows(path)]
        actual = choose(cohort, predictions, candidates)
        assert actual == expected["models"][model]
        checked[model] = {"decoder": actual["decoder"], "count_checks": actual["independent_prediction_counts_checked"],
                          "predictions_sha256": sha(path)}
    bad_cases = {}
    bad_cases["missing_clip"] = predictions[:-1]
    bad_cases["duplicate_clip"] = predictions[:-1] + [predictions[0]]
    for label, key, value in (("assessment_row", "stage", "assessment"), ("source_mismatch", "source_id", "wrong")):
        bad = deepcopy(predictions)
        bad[0][key] = value
        bad_cases[label] = bad
    bad = deepcopy(predictions)
    bad[0]["predictions"][candidates[0]]["counts"]["errors"] += 1
    bad_cases["wrong_count"] = bad
    bad = deepcopy(predictions)
    del bad[0]["predictions"][candidates[-1]]
    bad_cases["missing_candidate"] = bad
    rejected = []
    for label, values in bad_cases.items():
        try:
            choose(cohort, values, candidates)
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError("Malformed development evidence admitted: " + label)
    tie = deepcopy(predictions)
    first = candidates[0]
    for row in tie:
        row["predictions"] = {candidate: deepcopy(row["predictions"][first]) for candidate in candidates}
    assert choose(cohort, tie, candidates)["decoder"] == first
    save(HERE / "Selection_Regression.json", {
        "utc": utc(), "status": "complete_historical_selection_reproduced_and_invalid_inputs_rejected",
        "checked": checked, "malformed_cases_rejected": rejected, "tie_order_verified": True,
        "selection_code_sha256": sha(HERE / "select_development.py"), "new_checkpoint_assessed": False,
        "historical_selection_sha256": sha(original / "Decoder_Selection.json"),
    })
    print({"checked": checked, "rejected": rejected})


if __name__ == "__main__":
    main()
