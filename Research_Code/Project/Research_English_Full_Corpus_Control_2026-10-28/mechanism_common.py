"""Inputs for a separate T2/T4 extension; never mutate the active benchmarks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
VI = ROOT / "Research_Transfer_Full_Benchmark_2026-10-27"
EN = ROOT / "Research_English_Transfer_Benchmark_2026-10-28"
DIAGNOSTIC = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    result = []
    with Path(path).open("rb") as stream:
        for value in stream:
            assert value.endswith(b"\n"), "Incomplete JSONL record"
            if value.strip():
                result.append(json.loads(value))
    return result


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def logical(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def validate_inputs():
    value = read(HERE / "Plan.json")
    for name, digest in value["input_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return value
