"""Local-only English runtime proof; no test-set selection or paid allocation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
INPUTS = PHASE / "Transfer_Inputs_01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_lock():
    lock = read(HERE / "Lock.json")
    assert lock["status"] == "local_runtime_preflight_only"
    assert lock["test_inference_authorized"] is False
    for name, expected in lock["sha256"].items():
        assert sha(ROOT / name) == expected, name
    return lock
