"""Hash all scheduled VLR source videos and replay a fixed set of archived pixel digests."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MEDIA = ROOT / "data/official_raw/vlr/extracted/cloud_dataset_signer_split/data/train"
REPLAY = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"
sys.path.insert(0, str(ROOT / "Research_Phase_05_2026-10-13"))
from phase5.frame_cache import encode_frames
from phase5.media import read_video


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archived_cache_bytes(frames):
    encoded = encode_frames(frames)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(encoded)) as source:
        info = source.getinfo("frames.npy")
        payload = source.read("frames.npy")
        info.create_system = 3
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
            target.writestr(info, payload, compresslevel=1)
    restored = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(restored)) as target:
        assert target.read("frames.npy") == payload
    return restored


def locate(identity):
    relative = PurePosixPath(identity)
    assert not relative.is_absolute() and ".." not in relative.parts and ":" not in identity
    path = (MEDIA / (identity + ".mp4")).resolve()
    path.relative_to(MEDIA.resolve())
    return path


def raw_check(row):
    path = locate(row["clip_id"])
    result = {"clip_id": row["clip_id"], "expected_sha256": row["video_sha256"]}
    if not path.is_file():
        return {**result, "status": "missing"}
    digest = sha(path)
    return {**result, "sha256": digest, "bytes": path.stat().st_size,
            "status": "matched" if digest == row["video_sha256"] else "mismatch"}


def main():
    assert not (HERE / "Media_Coverage.json").exists()
    parent = read(REPLAY / "Summary.json")
    parent = next(item for item in parent["datasets"] if item["dataset"] == "vlr")
    trace_path = REPLAY / "vlr_trace.jsonl.gz"
    assert sha(trace_path) == parent["trace_sha256"]
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    wanted = {row["clip_id"] for row in trace}
    assert len(trace) == len(wanted) == 48000
    rows = {}
    checksum = hashlib.sha256()
    fields = ("clip_id", "frames", "video_sha256", "pixels_sha256", "frames_sha256")
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        with archive.open("Phase05/prepared/vlr/train/manifest.jsonl") as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                assert row["split"] == "train"
                if row["clip_id"] in wanted:
                    rows[row["clip_id"]] = {field: row.get(field) for field in fields}
    assert checksum.hexdigest() == parent["manifest_sha256"] and set(rows) == wanted
    scheduled_path = HERE / "vlr_short_slots.jsonl.gz"
    verification = read(HERE / "Verification.json")
    assert verification["status"] == "complete_training_schedule_verified"
    assert sha(scheduled_path) == verification["datasets"][0]["schedule_sha256"]
    with gzip.open(scheduled_path, "rt", encoding="utf-8") as stream:
        scheduled = [json.loads(line) for line in stream]
    probes = set(sorted(wanted, key=lambda name: hashlib.sha256(name.encode()).hexdigest())[:32])
    probes.update(record["clip_id"] for record in scheduled
                  if record["view"] == "shared_full_fallback" or record["stop"] - record["start"] == 1)
    probes.update(row["clip_id"] for row in sorted(rows.values(), key=lambda item: item["frames"])[:2])
    probes.update(row["clip_id"] for row in sorted(rows.values(), key=lambda item: -item["frames"])[:2])
    probes.update(sorted(name for name, row in rows.items() if row["pixels_sha256"] is None)[:2])
    plan = {"utc": datetime.now(timezone.utc).isoformat(), "raw_video_count": len(rows),
            "pixel_probes": sorted(probes),
            "selection": "32 hash-ranked plus frame extremes, fallbacks, one-frame views and two absent pixel digests",
            "source_manifest_sha256": checksum.hexdigest(), "trace_sha256": sha(trace_path),
            "schedule_sha256": sha(scheduled_path), "script_sha256": sha(Path(__file__).resolve()),
            "direct_pixel_digest_available": sum(bool(row["pixels_sha256"]) for row in rows.values()),
            "cache_container_create_system": 3}
    with (HERE / "Media_Plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    began = time.monotonic()
    counts = Counter()
    total_bytes = 0
    destination = HERE / "VLR_Raw_Hashes.jsonl.gz"
    with gzip.open(destination, "xt", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=2) as pool:
        for index, result in enumerate(pool.map(raw_check, rows.values()), 1):
            counts[result["status"]] += 1
            total_bytes += result.get("bytes", 0)
            output.write(json.dumps(result) + "\n")
            if index % 4000 == 0:
                seconds = time.monotonic() - began
                print({"utc": datetime.now(timezone.utc).isoformat(), "checked": index, "counts": dict(counts),
                       "seconds": seconds, "eta_seconds": seconds / index * (len(rows) - index)}, flush=True)
    pixels = []
    if counts["matched"] == 48000:
        for identity in sorted(probes):
            row = rows[identity]
            frames = read_video(locate(identity))
            digest = hashlib.sha256(frames.tobytes()).hexdigest()
            encoded_digest = hashlib.sha256(archived_cache_bytes(frames)).hexdigest()
            pixels.append({"clip_id": identity, "shape": list(frames.shape),
                           "shape_matches": frames.shape == (row["frames"], 96, 96),
                           "pixels_sha256": digest, "expected_pixels_sha256": row["pixels_sha256"],
                           "pixels_match": digest == row["pixels_sha256"] if row["pixels_sha256"] else None,
                           "encoded_cache_sha256": encoded_digest,
                           "archived_cache_sha256": row["frames_sha256"],
                           "encoded_bytes_match": encoded_digest == row["frames_sha256"]})
    passed = counts["matched"] == 48000 and len(pixels) == len(probes)
    passed = passed and all(item["shape_matches"] and item["pixels_match"] is not False
                           and item["encoded_bytes_match"] for item in pixels)
    proof = {"utc": datetime.now(timezone.utc).isoformat(),
             "status": "raw_inventory_and_fixed_pixel_probes_passed" if passed else "media_checks_incomplete_or_failed",
             "counts": dict(counts), "total_raw_bytes": total_bytes, "seconds": time.monotonic() - began,
             "pixels": pixels, "pixel_probe_count": len(pixels), "all_videos_decoded": False,
             "raw_hashes_sha256": sha(destination), "media_plan_sha256": sha(HERE / "Media_Plan.json"),
             "bulk_cache_rebuilt": False, "training_admitted": False}
    with (HERE / "Media_Coverage.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps({key: value for key, value in proof.items() if key != "pixels"}), flush=True)
    assert passed


if __name__ == "__main__":
    main()
