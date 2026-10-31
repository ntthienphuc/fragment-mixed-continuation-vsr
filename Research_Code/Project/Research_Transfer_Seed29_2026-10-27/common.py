"""Exclusive evidence outputs and explicit admission for seed-29 inference."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
PRIOR = ROOT / "Research_Transfer_Initialization_2026-10-27"
COHORT = ROOT / "Research_Interval_Transfer_2026-10-27/Cohort.jsonl"
STORE = Path(os.environ.get("LIP_SEED29_STORE", str(
    Path.home() / "Lip_Research_Backups/Transfer_Seed29_20261027")))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_admission(admission, receipt, parent_result, plan_hash, receipt_hash, summary_hash):
    assert admission["status"] == "admitted_after_parent_review"
    assert admission["plan_sha256"] == plan_hash
    assert admission["parent_receipt_sha256"] == receipt_hash
    assert admission["parent_summary_sha256"] == summary_hash == receipt["summary_sha256"]
    assert receipt["status"] == "complete_matrix_analysis_verified"
    assert receipt["verified_new_tasks"] == 24 and receipt["independent_counts_checked"] == 29205
    assert parent_result["status"] == "complete_verified_five_model_decoder_matrix"
    assert parent_result["independent_counts_checked"] == 29205
    assert admission["complete_decoder_grid_required"] is True
    assert admission["models"] == ["VLR_T0_29_LAST", "VLR_T1_29_LAST"]
    assert isinstance(admission["scientific_decision"], str) and len(admission["scientific_decision"].strip()) >= 40
    assert admission["maximum_execution_seconds"] > 0
    assert math.isfinite(admission["maximum_execution_seconds"])
    assert 0 < admission["maximum_execution_seconds"] <= 86400
    assert math.isfinite(admission["admitted_unix"]) and math.isfinite(admission["deadline_unix"])
    assert admission["deadline_unix"] == admission["admitted_unix"] + admission["maximum_execution_seconds"]
    assert admission["execution_location"] in ("local", "server")
    if admission["execution_location"] == "server":
        assert 0 < admission["usd_per_hour"] < 1
        assert admission["maximum_cost_usd"] > 0 and admission["reserved_balance_usd"] >= 1
        assert admission["maximum_cost_usd"] >= (
            admission["usd_per_hour"] * admission["maximum_execution_seconds"] / 3600)
        assert admission["maximum_cost_usd"] <= (
            admission["observed_balance_usd"] - admission["reserved_balance_usd"])


def checked_plan():
    value = read(HERE / "Plan.json")
    assert value["status"] == "prepared_before_seed29_destination_inference"
    assert sha(COHORT) == value["cohort_sha256"]
    for name, expected in value["frozen_sha256"].items():
        assert sha(ROOT / name) == expected, name
    parent_plan = read(PRIOR / "Plan.json")
    assert sha(PRIOR / "Plan.json") == value["parent_initialization_plan_sha256"]
    for name, expected in parent_plan["frozen_sha256"].items():
        assert sha(ROOT / name) == expected, name
    path = HERE / "Admission.json"
    if not path.exists():
        raise RuntimeError("Seed-29 inference is not admitted: complete and review the parent matrix first")
    admission = read(path)
    receipt_path = PRIOR / "Completion_Receipt.json"
    summary_path = PRIOR / "Analysis_01/Summary.json"
    receipt = read(receipt_path)
    result = read(summary_path)
    validate_admission(admission, receipt, result, sha(HERE / "Plan.json"),
                       sha(receipt_path), sha(summary_path))
    assert admission["admitted_unix"] <= time.time() < admission["deadline_unix"]
    for name, expected in result["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    return {**value, "admission_deadline_unix": admission["deadline_unix"]}
