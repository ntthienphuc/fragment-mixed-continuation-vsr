"""Freeze all source clips and numerical code before writing encoder caches."""

import json
from pathlib import Path
import shutil

from encoding_common import HERE, INPUTS, ROOT, RUNTIME_CHECK, read, save, sha, utc


def main():
    previous = read(RUNTIME_CHECK / "Lock.json")
    receipt = read(RUNTIME_CHECK / "Verification_01.json")
    assert receipt["status"] == "completed_runtime_evidence_rechecked"
    assert receipt["native_bounded_pairs"] == 48 and receipt["test_clips_used"] is False
    paths = {ROOT / name for name in previous["sha256"]}
    paths.update(HERE.glob("*.py"))
    paths.update([RUNTIME_CHECK / "Lock.json", RUNTIME_CHECK / "Verification_01.json"])
    for name, digest in previous["sha256"].items():
        assert sha(ROOT / name) == digest, name
    population = [json.loads(line) for line in (INPUTS / "Development.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(population) == 1082 and len({row["clip_id"] for row in population}) == 1082
    assert all(row["split"] == "development" and row["language"] == "en" for row in population)
    assert max(row["frames"] for row in population) == 153
    cache = Path("C:/Users/thien/Lip_Research_Backups/English_Transfer_20261027/Development_Encoded_01")
    assert not cache.exists() and shutil.disk_usage(cache.parent).free > 5 * 2**30
    expected_bytes = 4 * sum(row["frames"] for row in population) * 768 * 4
    assert expected_bytes < 2**30
    document = {"utc": utc(), "status": "source_development_encoding_only",
                "models": previous["models"], "clips_per_model": 1082,
                "source_frames_per_model": sum(row["frames"] for row in population),
                "state_shape": "[source_frames,768]", "state_dtype": "float32",
                "estimated_uncompressed_state_bytes": expected_bytes,
                "cache_root": str(cache), "maximum_seconds": 7200,
                "minimum_available_ram_gib": 3, "minimum_free_gpu_mib": 3000,
                "probes": previous["probes"], "replay_decoder": "native_ctc0.1_beam40_penalty0_fullcap",
                "accuracy_claimed": False, "test_inference_authorized": False, "paid_allocation": False,
                "sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}}
    save(HERE / "Lock.json", document)
    print({"status": document["status"], "models": len(document["models"]),
           "clips_per_model": 1082, "state_bytes": expected_bytes, "lock_sha256": sha(HERE / "Lock.json")})


if __name__ == "__main__":
    main()
