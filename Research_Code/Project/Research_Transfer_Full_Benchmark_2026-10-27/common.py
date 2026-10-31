"""Phase-local immutable evidence utilities for the full ViCocktail benchmark."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
PARENT = ROOT / "Research_Transfer_Initialization_2026-10-27"
SEED29 = ROOT / "Research_Transfer_Seed29_2026-10-27"
READINESS = ROOT / "Research_Transfer_Confirmation_Readiness_2026-10-27"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()
