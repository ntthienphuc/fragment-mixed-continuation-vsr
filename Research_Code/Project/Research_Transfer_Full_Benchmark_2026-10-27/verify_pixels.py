"""Decode every full benchmark video; verify both native input conventions on CPU."""

import hashlib
import json
import sys
import time

from common import ENGINE, HERE, ROOT, lines, read, save, sha, utc

sys.path.insert(0, str(ENGINE))
from data_io import load_pixels


def main():
    plan = read(HERE / "Plan.json")
    for name, expected in plan["frozen_sha256"].items():
        assert sha(ROOT / name) == expected, name
    output = HERE / "Pixels_01"
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    cache_pixels = {}
    for model, source in plan["development_reuse"].items():
        from pathlib import Path
        path = Path(source["cache_folder"]) / "Index.jsonl"
        assert sha(path) == source["cache_index_sha256"]
        cache_pixels[model] = {row["clip_id"]: row["pixels_sha256"] for row in lines(path)}
    checks = 0
    replays = 0
    counts = {"development": 0, "assessment": 0}
    with (output / "Index.jsonl").open("x", encoding="utf-8") as stream:
        for filename in ("Development.jsonl", "Assessment.jsonl"):
            for row in lines(HERE / filename):
                rgb = load_pixels(row)
                gray_row = {**row, "input": {**row["input"], "color_space": "gray"}}
                gray = load_pixels(gray_row)
                rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                gray_sha = hashlib.sha256(gray.tobytes()).hexdigest()
                for model, expected in cache_pixels.items():
                    if row["clip_id"] in expected:
                        assert gray_sha == expected[row["clip_id"]], (model, row["clip_id"])
                        replays += 1
                record = {"clip_id": row["clip_id"], "source_id": row["source_id"],
                          "split": row["split"], "frames": len(gray),
                          "media_sha256": row["input"]["sha256"],
                          "gray_sha256": gray_sha, "rgb_sha256": rgb_sha}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                counts[row["split"]] += 1
                checks += 2
                if sum(counts.values()) % 128 == 0:
                    print({"utc": utc(), "decoded_clips": sum(counts.values()),
                           "seconds": time.monotonic() - began}, flush=True)
    assert counts == {"development": 768, "assessment": 1167}
    assert checks == 3870 and replays == 2655
    save(output / "Verification.json", {
        "utc": utc(), "status": "all_full_videos_decoded_in_both_input_conventions",
        "clips": counts, "decoded_arrays_checked": checks,
        "existing_model_clip_pixel_hashes_reproduced": replays,
        "seconds": time.monotonic() - began, "index_sha256": sha(output / "Index.jsonl"),
        "plan_sha256": sha(HERE / "Plan.json"), "code_sha256": sha(__file__),
        "gpu_used": False, "recognition_run": False, "reference_transcripts_changed": False,
    })
    print({"status": "passed", "decoded_arrays": checks, "cache_pixel_checks": replays}, flush=True)


if __name__ == "__main__":
    main()
