"""Frozen inputs and portable paths for the decoder-controlled transfer stage."""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "Research_Interval_Transfer_2026-10-27"
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
VENDOR = ROOT / "Research_Phase_05_2026-10-13/vendor/auto_avsr"
STORE = Path(os.environ.get("LIP_TRANSFER_STORE", str(
    Path.home() / "Lip_Research_Backups/Transfer_Initialization_20261027")))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_cache():
    override = os.environ.get("LIP_TRANSFER_CACHE")
    return Path(override) if override else Path(read(SOURCE / "Cache_Plan.json")["destination"])


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
    value = read(HERE / "Plan.json")
    assert sha(SOURCE / "Cohort.jsonl") == value["cohort_sha256"]
    assert sha(HERE / "Protocol.md") == value["protocol_sha256"]
    return value
