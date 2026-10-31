"""Read hash-bound staged source inputs without treating partial media as a full dataset."""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def rows(content):
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def relative_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("Media member is outside the declared split")
    return path


def archived_cache_bytes(pixels):
    from phase5.frame_cache import encode_frames

    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(encode_frames(pixels))) as source:
        info = source.getinfo("frames.npy")
        content = source.read("frames.npy")
        info.create_system = 3
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
            target.writestr(info, content, compresslevel=1)
    return output.getvalue()


class InputPaths:
    def __init__(self, runtime_file, expected_sha256):
        path = Path(runtime_file).resolve()
        if sha(path) != expected_sha256:
            raise ValueError("Runtime input receipt changed")
        self.receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        receipt = self.receipt
        if receipt["status"] != "scheduled_training_and_source_validation_inputs_verified":
            raise ValueError("Input staging did not finish")
        if receipt["training_clips_available"] != 48000 or receipt["validation_cached_clips"] != 512:
            raise ValueError("Input population changed")
        if receipt["validation_full_manifest_rows"] != 11578:
            raise ValueError("Source-validation manifest population changed")
        if receipt["validation_selection"] != "first_512_sha256_clip_id":
            raise ValueError("Validation selection changed")
        self.scope = path.parent
        expected = {"train_manifest": "Training/train_manifest.jsonl", "train_raw_root": "Training/raw",
                    "validation_archive": "Source_Validation_512.zip"}
        for key, relative in expected.items():
            value = Path(receipt[key]).resolve()
            if value != (self.scope / relative).resolve() or not value.is_relative_to(self.scope):
                raise ValueError("Input mapping escaped the verified staging scope")
            setattr(self, key, value)
        if sha(self.train_manifest) != receipt["train_manifest_sha256"]:
            raise ValueError("Training manifest changed")
        if sha(self.validation_archive) != receipt["validation_archive_sha256"]:
            raise ValueError("Source-validation archive changed")
        self.identity = {"runtime_sha256": expected_sha256, "provider_sha256": sha(Path(__file__)),
                         "train_manifest_sha256": receipt["train_manifest_sha256"],
                         "val_manifest_sha256": receipt["val_manifest_sha256"],
                         "validation_archive_sha256": receipt["validation_archive_sha256"],
                         "cached_validation_limit": 512}


class TrainingData:
    def __init__(self, objective, spec, paths):
        if objective.dataset != "vlr":
            raise ValueError("Only the verified VLR training package is admitted")
        self.objective = objective
        self.paths = paths
        self.manifest_sha256 = paths.receipt["train_manifest_sha256"]
        if self.manifest_sha256 != spec["identity"]["train_manifest_sha256"]:
            raise ValueError("Training metadata differs from the original native identity")
        self.all_rows = rows(paths.train_manifest.read_bytes())
        self.rows = [row for row in self.all_rows if row["train_eligible"]]
        self.by_id = {row["clip_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows) or any(row["split"] != "train" for row in self.all_rows):
            raise ValueError("Training identities or splits are invalid")
        if len(objective.schedule) != 48000:
            raise ValueError("Locked presentation count changed")
        self.scheduled = {row["clip_id"] for row in objective.schedule}
        if len(self.scheduled) != 48000:
            raise ValueError("Expected the 48,000 distinct VLR source clips")
        for number, record in enumerate(objective.schedule):
            epoch, position = divmod(number, len(self.rows))
            expected = self.rows[self.order(epoch)[position]]["clip_id"]
            if expected != record["clip_id"]:
                raise ValueError("Explicit order differs from the native source sampler")
            if not self.locate(expected).is_file():
                raise ValueError("A scheduled training video is absent")

    @lru_cache(maxsize=2)
    def order(self, epoch):
        indices = list(range(len(self.rows)))
        random.Random(17 + epoch).shuffle(indices)
        return indices

    def locate(self, clip_id):
        relative = relative_name(clip_id)
        if clip_id not in self.scheduled:
            raise ValueError("Training media requested outside the staged schedule")
        path = (self.paths.train_raw_root / (relative.as_posix() + ".mp4")).resolve()
        if not path.is_relative_to(self.paths.train_raw_root):
            raise ValueError("Training path escaped its media root")
        return path

    @lru_cache(maxsize=8)
    def frames(self, clip_id):
        from phase5.media import read_video

        path = self.locate(clip_id)
        row = self.by_id[clip_id]
        if sha(path) != row["video_sha256"]:
            raise ValueError("Training video bytes changed")
        pixels = read_video(path)
        if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96):
            raise ValueError("Training pixels have wrong shape or dtype")
        if row.get("pixels_sha256"):
            valid = digest(pixels.tobytes()) == row["pixels_sha256"]
        else:
            valid = digest(archived_cache_bytes(pixels)) == row["frames_sha256"]
        if not valid:
            raise ValueError("Training pixels differ from the archived native cache")
        return pixels

    def sample(self, number, seed):
        if type(number) is not int or not 0 <= number < 48000 or seed != 17:
            raise ValueError("Sample is outside the locked seed/order")
        clip_id = self.objective.schedule[number]["clip_id"]
        return self.by_id[clip_id], self.frames(clip_id)


class ValidationData:
    def __init__(self, spec, paths):
        self.paths = paths
        self.prefix = "prepared/vlr/val/"
        with zipfile.ZipFile(paths.validation_archive) as archive:
            raw = archive.read(self.prefix + "manifest.jsonl")
            failures = archive.read(self.prefix + "failures.jsonl")
            self.receipt = json.loads(archive.read(self.prefix + "receipt.json"))
            names = archive.namelist()
        self.manifest_sha256 = digest(raw)
        if self.manifest_sha256 != spec["identity"]["val_manifest_sha256"]:
            raise ValueError("Source-validation manifest differs from original metadata")
        if self.receipt["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("Source-validation receipt disagrees with manifest")
        for field in ("config_sha256", "code_sha256"):
            if self.receipt[field] != spec["identity"][field]:
                raise ValueError("Validation preparation identity changed")
        if digest(failures) != self.receipt["failures_sha256"]:
            raise ValueError("Validation failure coverage changed")
        self.rows = rows(raw)
        self.all_rows = self.rows
        self.failures = rows(failures)
        self.by_id = {row["clip_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows) or len(self.rows) != 11578:
            raise ValueError("Validation identities or full metadata count changed")
        if any(row["split"] != "val" for row in self.rows):
            raise ValueError("Only source-validation rows are accepted")
        self.available = sorted(self.rows, key=lambda row: digest(row["clip_id"].encode()))[:512]
        self.cached = {row["clip_id"] for row in self.available}
        expected = {self.prefix + relative_name(row["frame_path"]).as_posix() for row in self.available}
        expected.update(self.prefix + name for name in ("manifest.jsonl", "receipt.json", "failures.jsonl"))
        if len(names) != 515 or set(names) != expected:
            raise ValueError("Validation archive does not contain exactly the fixed 512-cache selection")

    def selection(self, limit=0):
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("Only an explicit subset of at most 512 cached validation clips is available")
        return self.available[:limit]

    @lru_cache(maxsize=8)
    def frames(self, clip_id):
        if clip_id not in self.cached:
            raise ValueError("Validation media requested outside the explicitly cached subset")
        row = self.by_id[clip_id]
        relative = relative_name(row["frame_path"])
        with zipfile.ZipFile(self.paths.validation_archive) as archive:
            content = archive.read(self.prefix + relative.as_posix())
        if digest(content) != row["frames_sha256"]:
            raise ValueError("Validation frame-cache bytes changed")
        with np.load(io.BytesIO(content), allow_pickle=False) as cache:
            if cache.files != ["frames"]:
                raise ValueError("Unexpected validation cache fields")
            pixels = cache["frames"]
        if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96):
            raise ValueError("Validation pixels have wrong shape or dtype")
        return pixels
