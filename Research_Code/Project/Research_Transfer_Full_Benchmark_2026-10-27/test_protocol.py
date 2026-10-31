"""Check seed retention, full-sentence paired inference, and mismatched-record rejection."""

import copy
import sys

import numpy as np

from admission import validate_parent
from analyze_assessment import FIELDS, describe_source_counts
from common import HERE, ROOT, lines, read, save, sha, utc
from reuse_development import prior_predictions
from verify_task import verify_records


def rejects(function):
    try:
        function()
    except (ValueError, KeyError):
        return 1
    raise AssertionError("Incorrect input was accepted")


def main():
    receipt = {"status": "seed29_full_matrix_analysis_verified", "verified_tasks": 16,
               "independent_counts_checked": 11682, "summary_sha256": "summary"}
    summary = {"status": "complete_verified_seed29_final_pair_decoder_matrix"}
    review = {"status": "complete_seed29_result_reviewed", "summary_sha256": "summary",
              "decision": "proceed_complete_benchmark", "both_training_seeds_retained": True,
              "scientific_reason": "Synthetic validation fixture: retain both positive and negative seeds."}
    admission = {"parent_summary_sha256": "summary", "parent_receipt_sha256": "receipt",
                 "parent_review_sha256": "review"}
    args = (receipt, summary, review, "receipt", "summary", "review")
    validate_parent(admission, *args)
    rejected = 0
    for name, value in (("status", "running"), ("verified_tasks", 15), ("independent_counts_checked", 11000)):
        invalid = {**receipt, name: value}
        rejected += rejects(lambda invalid=invalid: validate_parent(
            admission, invalid, summary, review, "receipt", "summary", "review"))
    negative_dropped = {**review, "both_training_seeds_retained": False}
    rejected += rejects(lambda: validate_parent(
        admission, receipt, summary, negative_dropped, "receipt", "summary", "review"))
    models = ["VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST", "VLR_T0_29_LAST", "VLR_T1_29_LAST", "VC_NATIVE"]
    values = np.zeros((6, 2, 5), dtype=np.int64)
    values[:, :, 4] = [10, 1000]
    values[:, :, 0] = [4, 100]
    values[:, :, 1] = values[:, :, 0]
    values[2, :, 0] = [3, 90]
    values[2, :, 1] = [3, 90]
    values[4, :, 0] = [5, 110]
    values[4, :, 1] = [5, 110]
    weights = np.array([[1, 1], [2, 0], [0, 2]] * 40, dtype=np.int64)
    result = describe_source_counts(values, models, ["small", "large"], weights)
    positive = result["pairs"]["VLR_T0_LAST_minus_VLR_T1_LAST"]
    negative = result["pairs"]["VLR_T0_29_LAST_minus_VLR_T1_29_LAST"]
    assert abs(positive["gain_pp"] - 100 * 11 / 1010) < 1e-12
    assert abs(negative["gain_pp"] + positive["gain_pp"]) < 1e-12
    assert abs(positive["unweighted_mean_source_gain_pp"] - 5.5) < 1e-12
    assert abs(result["fixed_two_seed_mean"]["gain_pp"]) < 1e-12
    assert np.allclose(result["fixed_two_seed_mean"]["source_ci95_pp"], [0, 0], atol=1e-12)
    invalid_counts = values.copy()
    invalid_counts[2, 0, 4] += 1
    rejected += rejects(lambda: describe_source_counts(invalid_counts, models, ["small", "large"], weights))
    invalid_counts = values.copy()
    invalid_counts[2, 0, 0] += 1
    rejected += rejects(lambda: describe_source_counts(invalid_counts, models, ["small", "large"], weights))
    plan = read(HERE / "Plan.json")
    old_rows = lines(ROOT / "Research_Interval_Transfer_2026-10-27/Cohort.jsonl")
    cache_path = plan["development_reuse"]["VLR_B0"]["cache_folder"]
    from pathlib import Path
    cached = {row["clip_id"]: row for row in lines(Path(cache_path) / "Index.jsonl")}
    pixels = {row["clip_id"]: row for row in lines(HERE / "Pixels_01/Index.jsonl")}
    records, _ = prior_predictions("VLR_B0")
    actual_rows = []
    for index, row in enumerate(old_rows):
        record = records[row["clip_id"]]
        record.update({"index": index, "frames": row["frames"], "media_sha256": row["input"]["sha256"],
                       "encoding": cached[row["clip_id"]]["encoding"]})
        actual_rows.append(record)
    checked = verify_records(list(enumerate(old_rows)), actual_rows, "VLR_B0", "development",
                             plan["candidate_order"], pixels)
    assert checked == 5841
    for key, wrong in (("split", "assessment"), ("model", "VC_NATIVE"), ("media_sha256", "wrong")):
        corrupt = copy.deepcopy(actual_rows[:1])
        corrupt[0][key] = wrong
        rejected += rejects(lambda corrupt=corrupt: verify_records(
            [(0, old_rows[0])], corrupt, "VLR_B0", "development", plan["candidate_order"], pixels))
    corrupt = copy.deepcopy(actual_rows[:1])
    corrupt[0]["predictions"]["beam_c0.1"]["cap"] += 1
    rejected += rejects(lambda: verify_records(
        [(0, old_rows[0])], corrupt, "VLR_B0", "development", plan["candidate_order"], pixels))
    save(HERE / "Protocol_Verification.json", {
        "status": "passed", "utc": utc(), "invalid_inputs_rejected": rejected,
        "negative_seed_retention_checked": True, "same_draw_two_seed_mean_checked": True,
        "pooled_and_source_macro_distinguished": True, "real_record_count_checks": checked,
        "independent_scalar_bootstrap_checks": result["independent_scalar_bootstrap_checks"],
        "synthetic_results_not_research_results": True, "new_inference_run": False,
        "code_sha256": {name: sha(HERE / name) for name in (
            "admission.py", "verify_task.py", "analyze_assessment.py", "test_protocol.py")},
    })
    print({"status": "passed", "rejected": rejected, "real_recounts": checked}, flush=True)


if __name__ == "__main__":
    main()
