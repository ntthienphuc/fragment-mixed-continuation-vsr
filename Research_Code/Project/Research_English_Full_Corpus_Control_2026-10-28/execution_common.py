"""Bound execution to the frozen F0 population, decoder policy, device and budget."""

import math
from pathlib import Path
import time

from experiment_core import cohort, task_rows, validate_choices
from mechanism_common import EN, HERE, ROOT, read, sha, validate_inputs


VALIDATION_TASK = "source_integration_EN_F0_LAST"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_budget(value, now):
    fields = ("admitted_unix", "deadline_unix", "maximum_seconds")
    require(all(type(value[key]) in {int, float} and math.isfinite(value[key]) for key in fields), "Invalid clock")
    require(0 < value["maximum_seconds"] <= 21600, "Unbounded admission")
    require(value["deadline_unix"] == value["admitted_unix"] + value["maximum_seconds"], "Deadline changed")
    require(value["admitted_unix"] <= now < value["deadline_unix"], "Admission expired or not started")
    require(value["execution_location"] in {"local", "server"}, "Unknown execution location")
    require(value["device"] in {"cpu", "cuda"} and value["threads"] == 2, "Device or thread policy changed")
    if value["execution_location"] == "local":
        require(value["paid_allocation"] is False and value["maximum_seconds"] <= 14400, "Local budget changed")
    else:
        require(value["paid_allocation"] is True, "Server allocation must declare its paid budget")
        fields = ("usd_per_hour", "maximum_cost_usd", "observed_balance_usd", "reserved_balance_usd")
        require(all(type(value[key]) in {int, float} and math.isfinite(value[key]) for key in fields), "Invalid cost")
        require(0 < value["usd_per_hour"] < 1 and value["reserved_balance_usd"] >= 1, "Rate or reserve changed")
        cost = value["maximum_seconds"] * value["usd_per_hour"] / 3600
        require(cost <= value["maximum_cost_usd"], "Allocation exceeds cost limit")
        require(value["maximum_cost_usd"] <= value["observed_balance_usd"] - value["reserved_balance_usd"],
                "Allocation exceeds balance after reserve")
        require(value["provider_guard_verified"] is True and value["backup_restore_verified"] is True,
                "Missing verified guard or backup")


def selected_rows(plan, task_name):
    if task_name != VALIDATION_TASK:
        return task_rows(plan, task_name)
    rows = cohort(plan, "en", "development")
    positions = plan["source_probe_positions"]["en"]
    task = {"stage": "development", "model": "EN_F0_LAST", "language": "en", "positions": positions}
    return task, [(position, rows[position]) for position in positions]


def check_scope(value, task_name, task, device):
    require(value["status"] == "bounded_F0_execution_admitted", "Not admitted")
    require(value["device"] == device, "Unadmitted device")
    require(value["models"] == ["EN_F0_LAST"] and task["model"] == "EN_F0_LAST", "Model changed")
    require(task_name in value["tasks"] and task["stage"] in value["stages"], "Task or stage not admitted")
    require(value["purpose"] in {"source_integration", "full_evaluation"}, "Unknown purpose")
    if value["purpose"] == "source_integration":
        require(value["tasks"] == [VALIDATION_TASK] and value["stages"] == ["development"],
                "Integration admission cannot run benchmark data")
        require(value["device"] == "cpu" and value["execution_location"] == "local", "Integration device changed")
    else:
        require(task_name != VALIDATION_TASK, "Integration output cannot enter the benchmark")


def checked_execution():
    plan = validate_inputs()
    path = HERE / "Execution_Lock.json"
    lock = read(path)
    require(lock["status"] == "F0_execution_frozen", "Execution not frozen")
    require(lock["plan_sha256"] == sha(HERE / "Plan.json"), "Plan changed")
    require(lock["tasks"] == plan["tasks"], "Complete population changed")
    for name, digest in lock["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Execution source changed: " + name)
    for name, digest in lock["prerequisite_sha256"].items():
        require(sha(ROOT / name) == digest, "Prerequisite changed: " + name)
    return plan, lock


def checked_selection(plan):
    path = HERE / "Decoder_Selection.json"
    selection = read(path)
    validate_choices(plan, selection)
    require(selection["plan_sha256"] == sha(HERE / "Plan.json"), "Selection plan changed")
    require(selection["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Selection code changed")
    require(selection["selection_code_sha256"] == sha(HERE / "select_full_decoder.py"), "Selector changed")
    require(selection["assessment_manifest_sha256"] == plan["cohorts"]["en"]["assessment"]["sha256"],
            "Assessment manifest changed")
    for name, digest in selection["input_sha256"].items():
        require(sha(ROOT / name) == digest, "Development selection evidence changed")
    return selection


def checked(admission_path, task_name, device):
    plan, lock = checked_execution()
    task, rows = selected_rows(plan, task_name)
    admission = read(admission_path)
    check_scope(admission, task_name, task, device)
    validate_budget(admission, time.time())
    require(admission["plan_sha256"] == sha(HERE / "Plan.json"), "Admission plan changed")
    require(admission["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Admission code changed")
    location = ROOT / admission["locations_path"]
    require(sha(location) == admission["locations_sha256"], "Input locations changed")
    proof_path = ROOT / admission["runtime_preflight"]["path"]
    require(sha(proof_path) == admission["runtime_preflight"]["sha256"], "Runtime proof changed")
    proof = read(proof_path)
    require(proof["status"] == "three_source_inputs_full_grid_verified", "Runtime proof incomplete")
    require(proof["model"] == task["model"] and proof["device"] == device, "Wrong runtime model or device")
    require(proof["plan_sha256"] == sha(HERE / "Plan.json") and proof["assessment_used"] is False,
            "Runtime proof has wrong scope")
    require(proof["independent_count_checks"] == len(proof["comparisons"]) == 33, "Incomplete native grid")
    for name, digest in proof["scripts"].items():
        require(sha(HERE / name) == digest, "Runtime implementation changed")
    selection = None
    if task["stage"] == "assessment":
        selection = checked_selection(plan)
        require(admission["selection_sha256"] == sha(HERE / "Decoder_Selection.json"), "Selection not admitted")
        parent_path = EN / "Completion_Receipt.json"
        require(admission["parent_completion_sha256"] == sha(parent_path), "Parent benchmark not complete")
        require(read(parent_path)["status"] == "english_full_benchmark_verified", "Parent benchmark unverified")
    return plan, task, rows, admission, selection, read(location)
