"""Use original VLR training pixels and source validation; never read target assessment."""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import sys
import zipfile
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "data/official_raw/vlr/extracted/cloud_dataset_signer_split/data/train"
META = ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip"
VALIDATION = ROOT / "Server_Rescue_2026-10-15/Data/VLR/Prepared_Validation.zip"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class TrainingData:
    def __init__(self, objective, spec):
        if objective.dataset != "vlr":
            raise ValueError("English training media have not been admitted")
        self.objective = objective
        with zipfile.ZipFile(META) as archive:
            raw = archive.read("Phase05/prepared/vlr/train/manifest.jsonl")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != spec["identity"]["train_manifest_sha256"]:
            raise ValueError("Original training manifest changed")
        self.all_rows = rows(raw)
        self.rows = [row for row in self.all_rows if row["train_eligible"]]
        self.by_id = {row["clip_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows) or any(row["split"] != "train" for row in self.all_rows):
            raise ValueError("Training identity or split is invalid")
        for number, record in enumerate(objective.schedule):
            epoch, position = divmod(number, len(self.rows))
            expected = self.rows[self.order(epoch)[position]]["clip_id"]
            if expected != record["clip_id"]:
                raise ValueError("Explicit schedule differs from native source sampler")

    @lru_cache(maxsize=2)
    def order(self, epoch):
        indices = list(range(len(self.rows)))
        random.Random(17 + epoch).shuffle(indices)
        return indices

    @lru_cache(maxsize=8)
    def frames(self, clip_id):
        from phase5.media import read_video

        row = self.by_id[clip_id]
        path = (RAW / (clip_id + ".mp4")).resolve()
        if not path.is_relative_to(RAW.resolve()) or sha(path) != row["video_sha256"]:
            raise ValueError("Training media path or bytes changed")
        pixels = read_video(path)
        if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96):
            raise ValueError("Training pixel shape or dtype changed")
        if row.get("pixels_sha256"):
            if hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixels_sha256"]:
                raise ValueError("Training pixels changed")
        else:
            sys.path.insert(0, str(HERE.parent / "Readiness_02"))
            from verify_vlr_media import archived_cache_bytes

            if hashlib.sha256(archived_cache_bytes(pixels)).hexdigest() != row["frames_sha256"]:
                raise ValueError("Training pixels differ from archived native cache")
        return pixels

    def sample(self, number, seed):
        if seed != 17 or not 0 <= number < 48000:
            raise ValueError("Sample outside the locked seed/order")
        clip_id = self.objective.schedule[number]["clip_id"]
        return self.by_id[clip_id], self.frames(clip_id)


class ValidationData:
    def __init__(self, spec):
        prefix = "prepared/vlr/val/"
        self.prefix = prefix
        with zipfile.ZipFile(VALIDATION) as archive:
            raw = archive.read(prefix + "manifest.jsonl")
            failure_raw = archive.read(prefix + "failures.jsonl")
            self.receipt = json.loads(archive.read(prefix + "receipt.json"))
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != spec["identity"]["val_manifest_sha256"]:
            raise ValueError("Original source-validation manifest changed")
        if self.receipt["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("Validation receipt disagrees with manifest")
        for field in ("config_sha256", "code_sha256"):
            if self.receipt[field] != spec["identity"][field]:
                raise ValueError("Validation preparation identity changed")
        if hashlib.sha256(failure_raw).hexdigest() != self.receipt["failures_sha256"]:
            raise ValueError("Validation failure coverage changed")
        self.rows = rows(raw)
        self.all_rows = self.rows
        self.failures = rows(failure_raw)
        self.by_id = {row["clip_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows) or any(row["split"] != "val" for row in self.rows):
            raise ValueError("Only unique source-validation rows are accepted")

    def selection(self, limit=0):
        from phase5.common import rank

        selected = sorted(self.rows, key=lambda row: rank(row["clip_id"]))
        return selected[:limit] if limit else selected

    @lru_cache(maxsize=8)
    def frames(self, clip_id):
        row = self.by_id[clip_id]
        relative = PurePosixPath(row["frame_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Validation member escapes the split")
        with zipfile.ZipFile(VALIDATION) as archive:
            content = archive.read(self.prefix + relative.as_posix())
        if hashlib.sha256(content).hexdigest() != row["frames_sha256"]:
            raise ValueError("Source-validation cache bytes changed")
        with np.load(io.BytesIO(content), allow_pickle=False) as frames:
            if frames.files != ["frames"]:
                raise ValueError("Unexpected validation cache fields")
            pixels = frames["frames"]
        if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96):
            raise ValueError("Source-validation pixels have wrong shape or dtype")
        return pixels
