"""Fail before model loading if code, input, selection or resource bounds differ."""

import math
import time

from english_common import HERE, ROOT, read, require, sha


def validate_budget(value, now):
    keys = ("admitted_unix", "deadline_unix", "maximum_seconds")
    require(all(type(value[key]) in {int, float} and math.isfinite(value[key]) for key in keys), "Invalid clock")
    require(0 < value["maximum_seconds"] <= 86400, "Unbounded execution")
    require(value["deadline_unix"] == value["admitted_unix"] + value["maximum_seconds"], "Deadline changed")
    require(value["admitted_unix"] <= now < value["deadline_unix"], "Outside resource window")
    require(value["execution_location"] in {"local", "server"}, "Unknown execution location")
    if value["execution_location"] == "local":
        require(value["maximum_seconds"] <= 7200 and value["paid_allocation"] is False, "Local bound changed")
    else:
        money_fields = ("usd_per_hour", "maximum_cost_usd", "observed_balance_usd", "reserved_balance_usd")
        require(all(type(value[key]) in {int, float} and math.isfinite(value[key]) for key in money_fields),
                "Invalid monetary bound")
        require(0 < value["usd_per_hour"] < 1 and math.isfinite(value["usd_per_hour"]), "Wrong server price")
        require(value["reserved_balance_usd"] >= 1, "No backup reserve")
        cost = value["usd_per_hour"] * value["maximum_seconds"] / 3600
        require(cost <= value["maximum_cost_usd"], "Lease exceeds allocation")
        require(value["maximum_cost_usd"] <= value["observed_balance_usd"] - value["reserved_balance_usd"],
                "Budget exceeds available funds")
        require(value["provider_guard_verified"] is True and value["backup_restore_verified"] is True,
                "Missing server guard")


def validate_selection(selection, plan):
    require(selection["status"] == "all_source_development_decoders_locked_before_test", "Incomplete selection")
    require(set(selection["models"]) == set(plan["models"]), "Incomplete model grid")
    require(selection["test_predictions_read"] is False, "Test-tuned selection")
    for selected in selection["models"].values():
        require(selected["clips"] == plan["development_clips"], "Partial development selection")
        require(selected["reference_units"] == plan["development_reference_words"], "Development denominator changed")
        require(selected["independent_counts_checked"] == 11902, "Incomplete decoder grid")
        require(selected["decoder"] in plan["candidate_order"], "Unknown selected decoder")


def checked(stage, model, admission_path, task_name=None):
    require(stage in {"preflight", "development", "assessment"}, "Unknown stage")
    plan = read(HERE / "Plan.json")
    require(model in plan["models"], "Unknown model")
    for name, expected in plan["frozen_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen input changed: " + name)
    execution_path = HERE / "Execution_Lock.json"
    execution = read(execution_path)
    require(execution["plan_sha256"] == sha(HERE / "Plan.json"), "Execution plan changed")
    for name, expected in execution["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Execution code changed: " + name)
    admission = read(admission_path)
    require(admission["status"] == "bounded_english_execution_admitted", "Not admitted")
    require(admission["plan_sha256"] == sha(HERE / "Plan.json"), "Admission plan changed")
    require(admission["execution_lock_sha256"] == sha(execution_path), "Admission code changed")
    require(stage in admission["stages"] and model in admission["models"], "Stage or model not admitted")
    validate_budget(admission, time.time())
    if task_name is not None:
        require(task_name in admission["tasks"] and task_name in execution["tasks"], "Task not admitted")
    locations_path = ROOT / admission["locations_path"]
    require(sha(locations_path) == admission["locations_sha256"], "Locations changed after admission")
    if stage != "preflight":
        require(set(admission["runtime_preflights"]) == set(plan["models"]), "Incomplete runtime preflights")
        for identity, record in admission["runtime_preflights"].items():
            proof_path = ROOT / record["path"]
            require(sha(proof_path) == record["sha256"], "Runtime proof changed")
            proof = read(proof_path)
            require(proof["status"] == "native_source_cache_preflight_passed", "Runtime preflight incomplete")
            require(proof["model"] == identity and proof["execution_lock_sha256"] == sha(execution_path),
                    "Runtime identity changed")
            require(proof["prediction_comparisons"] == 33 and proof["test_clips_used"] is False,
                    "Runtime comparison incomplete")
    selection = None
    if stage == "assessment":
        selection = read(HERE / "Decoder_Selection.json")
        validate_selection(selection, plan)
        require(selection["plan_sha256"] == sha(HERE / "Plan.json"), "Selection uses another plan")
        require(selection["assessment_manifest_sha256"] == sha(HERE / "Assessment.jsonl"), "Test data changed")
        require(selection["selection_code_sha256"] == sha(HERE / "select_decoder.py"), "Selector changed")
        for name, expected in selection["input_sha256"].items():
            require(sha(ROOT / name) == expected, "Selection evidence changed: " + name)
    return plan, admission, selection, read(locations_path)
