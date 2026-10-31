"""Describe source/target duration shifts without reading recognition outcomes."""

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE))
from experiment_core import cohort
from mechanism_common import read, sha


def statistics(frames):
    values = np.asarray(frames, dtype=np.int64)
    assert len(values) and (values > 0).all()
    seconds = values / 25
    return {
        "observations": len(values), "frames": int(values.sum()), "mean_seconds": float(seconds.mean()),
        "median_seconds": float(np.median(seconds)), "p95_seconds": float(np.quantile(seconds, .95)),
        "minimum_seconds": float(seconds.min()), "maximum_seconds": float(seconds.max()),
        "under_2_seconds": int((seconds < 2).sum()), "over_10_seconds": int((seconds > 10).sum()),
    }


def main():
    readiness = read(HERE / "Prepared_Plan.json")
    completion = read(HERE / "Complete.json")
    assert sha(HERE / "Prepared_Plan.json") == completion["prepared_plan_sha256"]
    plan = read(PHASE / "Plan.json")
    rows = []
    provenance = {"source_prepared_plan": sha(HERE / "Prepared_Plan.json"), "target_plan": sha(PHASE / "Plan.json")}
    schedule_dir = PHASE / "Augmentation_Controls_01/Matched_Schedule_01"
    schedule_summary = read(schedule_dir / "Summary.json")
    for code, target_code in (("vlr", "vi"), ("lrs2", "en")):
        pixel_manifest = HERE / (code + "_Pixel_Manifest.jsonl")
        assert sha(pixel_manifest) == completion["corpora"][code]["pixel_manifest_sha256"]
        pixels = [json.loads(line) for line in pixel_manifest.read_text().splitlines()]
        rows.append({"language": target_code, "population": "source_test", "unit": "unique_clip",
                     **statistics([row["frames"] for row in pixels])})
        target = cohort(plan, target_code, "assessment")
        rows.append({"language": target_code, "population": "target_assessment", "unit": "unique_clip",
                     **statistics([row["frames"] for row in target])})
        schedule = schedule_dir / (code + ".jsonl.gz")
        expected = next(item for item in schedule_summary["datasets"] if item["dataset"] == code)
        assert sha(schedule) == expected["schedule_sha256"]
        frames = []
        ids = set()
        with gzip.open(schedule, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                view = row["views"]["full"]
                frames.append(view["stop"] - view["start"])
                ids.add(row["clip_id"])
        assert len(frames) == 48000
        rows.append({"language": target_code, "population": "full_video_train_presentations",
                     "unit": "presentation_including_repeats", "unique_clips": len(ids), **statistics(frames)})
        provenance[code + "_training_schedule"] = sha(schedule)
    query_counts = {}
    for code, prefix in (("vlr", "VLR_"), ("lrs2", "EN_")):
        models = [item for name, item in readiness["models"].items() if name.startswith(prefix)]
        clips = readiness["corpora"][code]["clips"]
        query_counts[code] = {"encodes": clips * len(models),
                              "decoder_queries": clips * sum(len(item["execution_candidates"]) for item in models)}
    result = {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "duration_distributions_recounted",
        "rows": rows, "inputs_sha256": provenance, "future_full_plan_query_counts": query_counts,
        "recognition_outcomes_read": False, "causal_explanation_established": False,
        "source_tests_reused": True, "script_sha256": sha(Path(__file__)),
    }
    with (HERE / "Duration_Comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
