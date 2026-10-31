"""Check exhaustive assignments and actual gray-video decoding without loading a model."""

import argparse
from pathlib import Path

from common import HERE, ROOT, read, save, sha, utc
from data_tasks import Destination, positions, candidates


def main(arguments):
    summary = {}
    plan = read(HERE / "Plan.json")
    assert sha(ROOT / plan["source"]["assignment_plan"]) == plan["upstream_lock_sha256"][plan["source"]["assignment_plan"]]
    original = read(ROOT / plan["source"]["assignment_plan"])
    coverage = [position for task in plan["source"]["assignment_tasks"]
                for position in original["tasks"][task]["positions"]]
    assert sorted(coverage) == list(range(11307))
    summary["source_assignment"] = {"clips": len(coverage), "shards": 16, "original_assignment_preserved": True}
    for stage in ("development", "assessment"):
        dataset = Destination(stage, arguments.destination_root, ROOT)
        groups = [positions(dataset, f"{stage}_{shard:02}", "population") for shard in range(8)]
        assert sorted(index for group in groups for index in group) == list(range(len(dataset)))
        tested = sorted({0, min(range(len(dataset)), key=lambda index: dataset.metadata(index)["frames"])})
        checks = []
        for index in tested:
            row, pixels = dataset.load(index)
            assert len(pixels) == row["frames"]
            checks.append({"position": index, "clip_id": row["clip_id"], "frames": len(pixels),
                           "pixel_sha256": row["pixel_sha256"]})
        summary[stage] = {"clips": len(dataset), "shards": 8, "actual_gray_pixel_checks": checks}
    assert candidates("assessment", "common", "unbound", HERE) == ["beam_c0.1"]
    assert candidates("development", "grid", "unbound", HERE) == plan["secondary_candidate_order"]
    rejected = []
    for stage, readout in (("assessment", "grid"), ("source", "grid"), ("development", "common")):
        try:
            candidates(stage, readout, "unbound", HERE)
        except AssertionError:
            rejected.append(stage + ":" + readout)
        else:
            raise AssertionError("Wrong readout stage accepted")
    save(arguments.output, {"utc": utc(), "status": "complete_assignments_and_real_video_pixels_checked",
         "summary": summary, "invalid_readout_routes_rejected": rejected,
         "assessment_plan_sha256": sha(HERE / "Plan.json"), "model_inference": False})
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
