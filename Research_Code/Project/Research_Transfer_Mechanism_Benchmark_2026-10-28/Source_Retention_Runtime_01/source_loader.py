"""Read the complete locked source-test cohorts without extracting duplicate data."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
READY = HERE.parent / "Source_Retention_Readiness_01"
PLAN_SHA256 = "57043031647da4c158d8b83b5680d4e2031dfbd8ec1d85af4d21c7521851e084"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_frames(payload, row, pixel):
    require(digest(payload) == row["frames_sha256"], "Frame-cache digest changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
        require(cache.files == ["frames"], "Unexpected frame-cache members")
        frames = cache["frames"]
    require(frames.dtype == np.uint8, "Unexpected pixel type")
    require(frames.shape == (row["frames"], 96, 96), "Unexpected complete-clip shape")
    require(row["clip_id"] == pixel["clip_id"] and len(frames) == pixel["frames"], "Pixel identity mismatch")
    require(digest(frames.tobytes()) == pixel["pixel_sha256"], "Decoded pixels changed")
    return frames


class SourceTest:
    """Positions retain the archived manifest order; no eligibility filter is applied."""

    def __init__(self, archive, rows, pixels, manifest, identity):
        self.archive = archive
        self.rows = tuple(rows)
        self.pixels = pixels
        self.manifest = manifest
        self.identity = dict(identity)

    def __len__(self):
        return len(self.rows)

    def load(self, position):
        if type(position) is not int or not 0 <= position < len(self):
            raise IndexError("Source-test position is outside the complete population")
        row = self.rows[position]
        name = "prepared/" + row["frame_path"]
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
                "Unsafe archive member")
        require(name in self.manifest, "Frame member is absent from archive manifest")
        payload = self.archive.read(name)
        expected = self.manifest[name]
        require(len(payload) == expected["bytes"] and digest(payload) == expected["sha256"],
                "Frame payload differs from archive manifest")
        frames = decode_frames(payload, row, self.pixels[row["clip_id"]])
        return dict(row), frames


@contextmanager
def open_source(language):
    require(language in {"vlr", "lrs2"}, "Unsupported source corpus")
    plan_bytes = (READY / "Prepared_Plan.json").read_bytes()
    require(digest(plan_bytes) == PLAN_SHA256, "Source preparation plan changed")
    plan = json.loads(plan_bytes)
    expected = plan["corpora"][language]
    pixel_bytes = (READY / (language + "_Pixel_Manifest.jsonl")).read_bytes()
    require(digest(pixel_bytes) == expected["pixel_manifest_sha256"], "Pixel manifest changed")
    pixel_rows = [json.loads(line) for line in pixel_bytes.splitlines()]
    pixels = {row["clip_id"]: row for row in pixel_rows}
    require(len(pixels) == len(pixel_rows) == expected["clips"], "Pixel cohort mismatch")
    path = ROOT / expected["archive"].replace("\\", "/")
    require(path.stat().st_size == expected["archive_bytes"], "Archive size changed")
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == expected["archive_sha256"],
                "Archive digest changed")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            payload = archive.read("Manifest.json")
            require(digest(payload) == expected["archive_manifest_sha256"], "Archive manifest changed")
            members = json.loads(payload)["files"]
            require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP member")
            require(set(archive.namelist()) == set(members) | {"Manifest.json"}, "ZIP coverage changed")
            metadata = archive.read("prepared/manifest.jsonl")
            require(digest(metadata) == expected["prepared_manifest_sha256"], "Prepared metadata changed")
            rows = [json.loads(line) for line in metadata.splitlines() if line.strip()]
            require(len(rows) == expected["clips"], "Complete cohort size changed")
            require({row["clip_id"] for row in rows} == set(pixels), "Clip population changed")
            require(all(row["split"] == "test" and row["frames"] > 0 and row["text"].strip()
                        for row in rows), "Invalid source-test metadata")
            require(sum(len(row["text"].split()) for row in rows) == expected["reference_whitespace_units"],
                    "Reference denominator changed")
            require(sum(row["frames"] for row in rows) == expected["total_frames"], "Frame coverage changed")
            yield SourceTest(archive, rows, pixels, members, expected)
