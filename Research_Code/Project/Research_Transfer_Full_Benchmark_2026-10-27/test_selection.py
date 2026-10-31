"""Verify selection edge cases and independently reconstruct a complete real cohort."""

import copy

from common import HERE, PARENT, ROOT, lines, read, save, sha, utc
from select_decoder import choose, counts


def record(row, alternatives):
    return {"clip_id": row["clip_id"], "source_id": row["source_id"], "split": "development",
            "predictions": {name: {"text": text, "counts": counts(row["reference"], text)}
                            for name, text in alternatives.items()}}


def rejects(function):
    try:
        function()
    except AssertionError:
        return True
    raise AssertionError("Invalid selection input was accepted")


def main():
    short = {"clip_id": "short", "source_id": "s1", "reference": "a", "split": "development"}
    long = {"clip_id": "long", "source_id": "s2", "reference": "b c d e f g h i j k",
            "split": "development"}
    cohort = [short, long]
    rows = [record(short, {"first": "", "second": "a"}),
            record(long, {"first": long["reference"], "second": "b c d e f g h i"})]
    result = choose(cohort, rows, ["first", "second"])
    assert result["decoder"] == "first" and result["errors"] == {"first": 1, "second": 2}
    tie = [record(row, {"first": row["reference"], "second": row["reference"]}) for row in cohort]
    assert choose(cohort, tie, ["second", "first"])["decoder"] == "second"
    rejected = 0
    rejected += rejects(lambda: choose(cohort, rows[:-1], ["first", "second"]))
    rejected += rejects(lambda: choose(cohort, [rows[0], rows[0]], ["first", "second"]))
    test_cohort = [{**row, "split": "assessment"} for row in cohort]
    rejected += rejects(lambda: choose(test_cohort, rows, ["first", "second"]))
    contaminated = copy.deepcopy(rows)
    contaminated[0]["split"] = "assessment"
    rejected += rejects(lambda: choose(cohort, contaminated, ["first", "second"]))
    wrong_count = copy.deepcopy(rows)
    wrong_count[0]["predictions"]["first"]["counts"]["errors"] += 1
    rejected += rejects(lambda: choose(cohort, wrong_count, ["first", "second"]))
    extra_candidate = copy.deepcopy(rows)
    extra_candidate[0]["predictions"]["unregistered"] = extra_candidate[0]["predictions"]["first"]
    rejected += rejects(lambda: choose(cohort, extra_candidate, ["first", "second"]))
    frozen = read(PARENT / "Beam_Execution/Execution_Lock.json")
    records = {}
    for task, description in frozen["tasks"].items():
        if description["model"] != "VLR_T0_LAST":
            continue
        matches = list((PARENT / "Beam_Execution/Runs").glob("*/" + task + "/Verification.json"))
        assert len(matches) == 1
        proof = read(matches[0])
        path = matches[0].parent / "Predictions.jsonl"
        assert proof["status"] == "passed" and proof["predictions_sha256"] == sha(path)
        for row in lines(path):
            assert row["clip_id"] not in records
            records[row["clip_id"]] = {**row, "split": "development"}
    greedy_path = PARENT / "Greedy_01/VLR_T0_LAST/Predictions.jsonl"
    greedy_proof = read(greedy_path.parent / "Verification.json")
    assert greedy_proof["predictions_sha256"] == sha(greedy_path)
    for row in lines(greedy_path):
        records[row["clip_id"]]["predictions"].update(row["predictions"])
    full_cohort = lines(ROOT / "Research_Interval_Transfer_2026-10-27/Cohort.jsonl")
    order = read(HERE / "Plan.json")["candidate_order"]
    actual = choose(full_cohort, list(records.values()), order)
    assert actual["independent_prediction_counts_checked"] == 531 * 11
    independent_totals = {candidate: sum(row["predictions"][candidate]["counts"]["errors"]
                                        for row in records.values()) for candidate in order}
    assert actual["errors"] == independent_totals
    expected = sorted(enumerate(order), key=lambda pair: (independent_totals[pair[1]], pair[0]))[0][1]
    assert actual["decoder"] == expected
    save(HERE / "Selection_Verification.json", {
        "utc": utc(), "status": "passed", "invalid_inputs_rejected": rejected,
        "pooled_vs_mean_wer_counterexample_passed": True, "frozen_tie_order_passed": True,
        "historical_prediction_counts_checked": 531 * 11,
        "current_full_benchmark_decoder_locked": False, "test_outcomes_read": False,
        "code_sha256": {name: sha(HERE / name) for name in ("select_decoder.py", "test_selection.py")}})
    print({"status": "passed", "invalid_rejected": rejected, "real_prediction_checks": 531 * 11})


if __name__ == "__main__":
    main()
