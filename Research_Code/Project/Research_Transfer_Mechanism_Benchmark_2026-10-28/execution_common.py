"""Bound the four prespecified control states to complete cohorts and source-only selection."""

import math
from pathlib import Path
import time

from experiment_core import cohort, task_rows, validate_choices
from mechanism_common import EN, HERE, ROOT, VI, read, sha, validate_inputs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_device_matches(actual, requested):
    if requested == "cuda":
        return actual in {"cuda", "cuda:0"}
    return requested == "cpu" and actual == "cpu"


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
        return
    require(value["paid_allocation"] is True, "Missing paid budget")
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
    if not task_name.startswith("source_integration_"):
        return task_rows(plan, task_name)
    model = task_name.removeprefix("source_integration_")
    require(model in plan["models"], "Unknown integration model")
    language = read(HERE / "model_registry.json")["models"][model]["language"]
    rows = cohort(plan, language, "development")
    positions = plan["source_probe_positions"][language]
    task = {"stage": "development", "model": model, "language": language, "positions": positions}
    return task, [(position, rows[position]) for position in positions]


def check_scope(value, task_name, task, device, plan):
    require(value["status"] == "bounded_mechanism_execution_admitted", "Not admitted")
    require(value["device"] == device, "Unadmitted device")
    require(value["models"] == plan["models"] and task["model"] in plan["models"], "Models changed")
    require(value["tasks"] and len(set(value["tasks"])) == len(value["tasks"]), "Duplicated admitted task")
    require(value["stages"] in (["development"], ["assessment"]), "Stages must be admitted separately")
    require(task_name in value["tasks"] and task["stage"] in value["stages"], "Task not admitted")
    require(value["purpose"] in {"source_integration", "full_evaluation"}, "Unknown purpose")
    if value["purpose"] == "source_integration":
        names = {"source_integration_" + model for model in plan["models"]}
        require(set(value["tasks"]).issubset(names) and value["stages"] == ["development"], "Wrong integration")
    else:
        require(set(value["tasks"]).issubset(plan["tasks"]), "Integration cannot enter full evaluation")
        require(all(plan["tasks"][name]["stage"] in value["stages"] for name in value["tasks"]), "Stage changed")


def checked_execution():
    plan = validate_inputs()
    lock = read(HERE / "Execution_Lock.json")
    require(lock["status"] == "mechanism_execution_frozen", "Execution not frozen")
    require(lock["plan_sha256"] == sha(HERE / "Plan.json") and lock["tasks"] == plan["tasks"], "Plan changed")
    for key in ("code_sha256", "prerequisite_sha256"):
        for name, digest in lock[key].items():
            require(sha(ROOT / name) == digest, "Locked input changed: " + name)
    return plan, lock


def checked_selection(plan):
    selection = read(HERE / "Decoder_Selection.json")
    validate_choices(plan, selection)
    require(selection["plan_sha256"] == sha(HERE / "Plan.json"), "Selection plan changed")
    require(selection["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Selection code changed")
    require(selection["selection_code_sha256"] == sha(HERE / "select_full_decoder.py"), "Selector changed")
    for language in ("vi", "en"):
        require(selection["assessment_manifest_sha256"][language] == plan["cohorts"][language]["assessment"]["sha256"],
                "Assessment manifest changed")
    for name, digest in selection["input_sha256"].items():
        require(sha(ROOT / name) == digest, "Selection evidence changed")
    return selection


def checked(admission_path, task_name, device):
    plan, lock = checked_execution()
    task, rows = selected_rows(plan, task_name)
    admission = read(admission_path)
    check_scope(admission, task_name, task, device, plan)
    validate_budget(admission, time.time())
    require(admission["plan_sha256"] == sha(HERE / "Plan.json"), "Admission plan changed")
    require(admission["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Admission code changed")
    location = ROOT / admission["locations_path"]
    require(sha(location) == admission["locations_sha256"], "Locations changed")
    entry = admission["runtime_preflights"][task["model"]]
    proof_path = ROOT / entry["path"]
    require(sha(proof_path) == entry["sha256"], "Runtime proof changed")
    proof = read(proof_path)
    require(proof["status"] == "three_source_inputs_full_grid_verified", "Runtime proof incomplete")
    require(proof["model"] == task["model"] and runtime_device_matches(proof["device"], device),
            "Wrong runtime model or device")
    require(proof["plan_sha256"] == sha(HERE / "Plan.json") and proof["assessment_used"] is False,
            "Wrong runtime proof scope")
    registry = read(HERE / "model_registry.json")["models"]
    require(proof["checkpoint_sha256"] == registry[task["model"]]["checkpoint_sha256"], "Wrong checkpoint")
    require(proof["independent_count_checks"] == len(proof["comparisons"]) == 33, "Incomplete native grid")
    for name, digest in proof["scripts"].items():
        require(sha(HERE / name) == digest, "Runtime implementation changed")
    selection = None
    if task["stage"] == "assessment":
        selection = checked_selection(plan)
        require(admission["selection_sha256"] == sha(HERE / "Decoder_Selection.json"), "Selection not admitted")
        for parent, expected in ((VI, "complete_full_benchmark_verified"),
                                 (EN, "english_full_benchmark_verified")):
            receipt = parent / "Completion_Receipt.json"
            require(admission["parent_completion_sha256"][parent.name] == sha(receipt), "Parent is incomplete")
            require(read(receipt)["status"] == expected, "Unverified parent")
    return plan, task, rows, admission, selection, read(location)
