"""Small helpers for read-only scheduling reconstruction."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE5 = ROOT / "Research_Phase_05_2026-10-13"
ARCHIVE = ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip"
MEMBER = "Phase05/prepared/vlr/train/manifest.jsonl"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_training():
    eligible = []
    digest = hashlib.sha256()
    prepared = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        receipt = json.loads(archive.read("Phase05/prepared/vlr/train/receipt.json"))
        with archive.open(MEMBER) as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                prepared += 1
                assert row["split"] == "train"
                if row["train_eligible"]:
                    selected = {key: row[key] for key in ("clip_id", "source_id", "frames", "text",
                                                         "train_starts", "words", "core_frames")}
                    eligible.append(selected)
    assert digest.hexdigest() == receipt["manifest_sha256"]
    assert prepared == receipt["prepared"] == 92632
    assert len(eligible) == receipt["train_eligible"] == 90519
    assert len({row["clip_id"] for row in eligible}) == len(eligible)
    return eligible, receipt
