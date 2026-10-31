"""Phase-local adapter for the unchanged native beam worker and its evidence checks."""

import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
SOURCE = ROOT / "Research_Interval_Transfer_2026-10-27"
VENDOR = ROOT / "Research_Phase_05_2026-10-13/vendor/auto_avsr"

specification = importlib.util.spec_from_file_location("seed29_input_contract", PHASE / "common.py")
contract = importlib.util.module_from_spec(specification)
specification.loader.exec_module(contract)
STORE = contract.STORE
read = contract.read
lines = contract.lines
sha = contract.sha
save = contract.save
utc = contract.utc


def configure():
    import torch
    torch.manual_seed(17)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def plan():
    admission = contract.checked_plan()
    value = read(HERE / "Plan.json")
    assert value["seed29_input_plan_sha256"] == sha(PHASE / "Plan.json")
    assert value["cohort_sha256"] == sha(SOURCE / "Cohort.jsonl")
    assert value["protocol_sha256"] == sha(PHASE / "Execution_Design.md")
    assert list(value["models"]) == admission["models"]
    assert value["candidate_tie_order"] == admission["candidate_order"]
    return {**value, "admission_deadline_unix": admission["admission_deadline_unix"]}
