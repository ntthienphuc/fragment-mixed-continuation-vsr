"""Reproduce the adopted, complete mechanism table while retaining both Vietnamese seeds."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "Data"
# Completed-study admission is independent of manuscript directory names.
ORIGIN = PROJECT / "Research_Transfer_Mechanism_Benchmark_2026-10-28"
STAGING = ORIGIN / "Paper_Evidence_01/Staged_01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def active():
    return (DATA / "Mechanism_Adoption.json").exists()


def gate():
    admission = read(DATA / "Mechanism_Adoption.json")
    assert admission["status"] == "completed_matrix_selected_for_paper"
    assert admission["staging_verification_sha256"] == sha(STAGING / "Verification.json")
    proof = read(STAGING / "Verification.json")
    assert proof["combined_rows"] == 58628 and proof["aggregate_rows"] == 30
    for name, digest in proof["output_sha256"].items():
        assert sha(STAGING / name) == digest, name
    sys.path.insert(0, str(ORIGIN / "Paper_Evidence_01"))
    import stage_completed
    finalized = stage_completed.check_finalized()
    assert finalized is not None
    final = finalized[0]
    assert final["sha256"] == proof["matrix_backup_sha256"] == admission["matrix_backup_sha256"]
    return stage_completed.ANALYSIS

