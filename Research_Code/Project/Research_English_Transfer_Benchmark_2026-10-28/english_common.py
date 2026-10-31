"""Immutable phase-local utilities for complete English transfer evaluation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
READY = ROOT / "Research_English_Transfer_Readiness_2026-10-27"
ANCHOR = ROOT / "Research_Transfer_Decoder_Anchor_2026-10-28"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    values = []
    with Path(path).open("rb") as stream:
        for number, payload in enumerate(stream, 1):
            require(payload.endswith(b"\n"), f"Incomplete record {number}: {path}")
            if payload.strip():
                values.append(json.loads(payload))
    return values


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def root_path(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def exact_copy(source, target):
    target.parent.mkdir(exist_ok=True, parents=True)
    payload = source.read_bytes()
    if target.exists():
        require(target.read_bytes() == payload, "Existing immutable copy differs")
    else:
        with target.open("xb") as stream:
            stream.write(payload)
