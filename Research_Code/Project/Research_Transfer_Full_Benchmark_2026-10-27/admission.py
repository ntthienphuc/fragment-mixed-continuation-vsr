"""Reject unreviewed, unbounded, or test-tuned benchmark execution before model loading."""

import math
import time

from common import HERE, ROOT, SEED29, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_plan():
    plan = read(HERE / "Plan.json")
    for name, expected in plan["frozen_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen input changed: " + name)
    return plan


def validate_budget(admission, now):
    for field in ("admitted_unix", "deadline_unix", "maximum_execution_seconds"):
        require(isinstance(admission[field], (int, float)) and math.isfinite(admission[field]), field)
    seconds = admission["maximum_execution_seconds"]
    require(0 < seconds <= 86400, "Execution must have a finite maximum of one day")
    require(admission["deadline_unix"] == admission["admitted_unix"] + seconds, "Deadline changed")
    require(admission["admitted_unix"] <= now < admission["deadline_unix"], "Outside admission window")
    require(admission["execution_location"] in {"local", "server"}, "Unknown execution location")
    if admission["execution_location"] == "server":
        fields = ("usd_per_hour", "maximum_cost_usd", "observed_balance_usd", "reserved_balance_usd")
        require(all(math.isfinite(admission[field]) for field in fields), "Nonfinite budget")
        require(0 < admission["usd_per_hour"] < 1, "Pod price must remain below one dollar/hour")
        require(admission["reserved_balance_usd"] >= 1, "Reserve is required")
        cost = admission["usd_per_hour"] * seconds / 3600
        require(cost <= admission["maximum_cost_usd"], "Lease exceeds allocated cost")
        available = admission["observed_balance_usd"] - admission["reserved_balance_usd"]
        require(admission["maximum_cost_usd"] <= available, "Allocation exceeds available funds")
        require(admission["provider_guard_verified"] is True, "Provider stop guard was not verified")
        require(admission["backup_restore_verified"] is True, "Backup readback is required")


def validate_parent(admission, receipt, summary, review, receipt_hash, summary_hash, review_hash):
    require(receipt["status"] == "seed29_full_matrix_analysis_verified", "Seed 29 is incomplete")
    require(receipt["verified_tasks"] == 16 and receipt["independent_counts_checked"] == 11682,
            "The entire second-seed matrix is required")
    require(receipt["summary_sha256"] == summary_hash == admission["parent_summary_sha256"],
            "Parent summary identity differs")
    require(receipt_hash == admission["parent_receipt_sha256"], "Parent receipt identity differs")
    require(summary["status"] == "complete_verified_seed29_final_pair_decoder_matrix", "Incomplete matrix")
    require(review_hash == admission["parent_review_sha256"], "Scientific review identity differs")
    require(review["status"] == "complete_seed29_result_reviewed", "Scientific review is pending")
    require(review["summary_sha256"] == summary_hash, "Review concerns another result")
    require(review["decision"] == "proceed_complete_benchmark", "Expansion was not selected")
    require(review["both_training_seeds_retained"] is True, "Both training seeds must be retained")
    require(len(review["scientific_reason"].strip()) >= 40, "Missing scientific interpretation")


def validate_selection(selection, model, plan, plan_hash, assessment_hash):
    require(selection["status"] == "all_development_decoders_locked_before_test", "Selection incomplete")
    require(selection["plan_sha256"] == plan_hash, "Selection concerns another plan")
    require(selection["assessment_manifest_sha256"] == assessment_hash, "Test population changed")
    require(selection["test_predictions_read"] is False, "Test-tuned selection is forbidden")
    models = [*plan["matched_models"], plan["practical_reference"]]
    require(set(selection["models"]) == set(models) and model in models, "Incomplete model selection")
    for value in selection["models"].values():
        require(value["clips"] == 768 and value["reference_units"] == 15919, "Incomplete development")
        require(value["independent_prediction_counts_checked"] == 8448, "Incomplete candidate grid")
        require(value["decoder"] in plan["candidate_order"], "Unregistered selected decoder")
    return selection["models"][model]["decoder"]


def checked(stage, model):
    require(stage in {"preflight", "development", "assessment"}, "Unknown stage")
    plan = verify_plan()
    require(model in [*plan["matched_models"], plan["practical_reference"]], "Unknown model")
    path = HERE / "Admission.json"
    if not path.exists():
        raise RuntimeError("Full benchmark not admitted: complete and review seed 29 first")
    admission = read(path)
    require(admission["status"] == "admitted_after_seed29_review", "Admission not active")
    require(admission["plan_sha256"] == sha(HERE / "Plan.json"), "Admission plan changed")
    require(stage in admission["stages"], "This stage was not admitted")
    validate_budget(admission, time.time())
    receipt_path = SEED29 / "Completion_Receipt.json"
    summary_path = SEED29 / "Analysis_01/Summary.json"
    review_path = ROOT / admission["parent_review_path"]
    summary = read(summary_path)
    validate_parent(admission, read(receipt_path), summary, read(review_path),
                    sha(receipt_path), sha(summary_path), sha(review_path))
    for name, expected in summary["input_sha256"].items():
        require(sha(ROOT / name) == expected, "Parent prediction changed: " + name)
    lock_path = HERE / "Execution_Lock.json"
    require(sha(lock_path) == admission["execution_lock_sha256"], "Execution lock changed")
    lock = read(lock_path)
    require(lock["plan_sha256"] == sha(HERE / "Plan.json"), "Execution uses another plan")
    for name, expected in lock["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Execution code changed: " + name)
    if stage != "preflight":
        for family in ("standard", "vicocktail"):
            item = admission["runtime_preflights"][family]
            proof_path = ROOT / item["path"]
            require(sha(proof_path) == item["sha256"], "Runtime proof changed")
            proof = read(proof_path)
            require(proof["status"] == "passed" and proof["family"] == family, "Runtime not verified")
            require(proof["execution_lock_sha256"] == sha(lock_path), "Runtime proof uses another code lock")
    selection = None
    if stage == "assessment":
        selection = read(HERE / "Decoder_Selection.json")
        validate_selection(selection, model, plan, sha(HERE / "Plan.json"), sha(HERE / "Assessment.jsonl"))
        require(selection["selection_code_sha256"] == sha(HERE / "select_decoder.py"), "Selector changed")
        for name, expected in selection["input_sha256"].items():
            require(sha(ROOT / name) == expected, "Selection evidence changed: " + name)
    return plan, admission, selection
