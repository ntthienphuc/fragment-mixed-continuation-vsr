"""Independently recount every record and replay fixed whole-video inputs."""

import argparse
import json
import math
from pathlib import Path
import sys

from assessment_common import HERE, check
from retention_common import ROOT, decode, read, require, runtime, save, sha, source, utc


def main(arguments):
    plan, task = check(arguments.task)
    output = arguments.output.resolve()
    require(output.is_relative_to(HERE / "Runs"), "Unexpected assessment output")
    start = read(output / "Start.json")
    require(start["plan_sha256"] == sha(HERE / "Plan.json"), "Plan changed after execution")
    require(start["device"] == arguments.device, "Independent replay requires the original device type")
    complete = read(output / "Complete.json")
    require(complete["status"] == "complete_pending_independent_verification", "Task not complete")
    require(complete["task"] == arguments.task and start["task"] == arguments.task, "Wrong task")
    require(sha(output / "Predictions.jsonl") == complete["predictions_sha256"], "Predictions changed")
    require(sha(output / "Identity.json") == complete["identity_sha256"], "Identity changed")
    records = []
    with (output / "Predictions.jsonl").open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n") and bool(line.strip()), "Incomplete JSONL")
            records.append(json.loads(line))
    require(len(records) == complete["clips"] == len(task["positions"]), "Coverage differs")
    sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
    from verify_probe import counts

    with source(task["corpus"]) as dataset:
        for position, record in zip(task["positions"], records):
            metadata = dataset.rows[position]
            require(record["position"] == position, "Position differs")
            for field in ("clip_id", "source_id", "frames"):
                require(record[field] == metadata[field], "Input identity differs: " + field)
            require(record["model"] == task["model"] and record["corpus"] == task["corpus"], "Wrong state/corpus")
            encoded = record["encoding"]
            require(encoded["frames"] == encoded["state_frames"] == metadata["frames"], "Clip truncated")
            require(encoded["model_id"] == task["model"], "Encoder changed")
            require(encoded["precision"] == "float32" and encoded["transform"] == "phase5_native_gray",
                    "Representation changed")
            require(encoded["pixel_sha256"] == dataset.pixels[metadata["clip_id"]]["pixel_sha256"], "Pixels changed")
            require(list(record["predictions"]) == task["candidates"], "Readout set changed")
            for candidate, predicted in record["predictions"].items():
                require(counts(metadata["text"], predicted["text"]) == predicted["counts"], "Edit counts differ")
                classes = 2002 if task["corpus"] == "vlr" else 5049
                require(all(type(token) is int and 0 < token < classes - 1 for token in predicted["tokens"]),
                        "Invalid language token")
                if candidate.startswith("beam_"):
                    require(predicted["cap"] == metadata["frames"] and math.isfinite(predicted["score"]),
                            "Beam horizon/score differs")
                    trace = predicted["trace"]
                    require(trace["beam_size"] == 40 and trace["penalty"] == 0, "Decoder size differs")
                    require(trace["ctc_weight"] == float(candidate.removeprefix("beam_c")), "Decoder weight differs")
                    require(trace["implementation"] == "upstream_BatchBeamSearch", "Wrong beam")
        replay_offsets = sorted({0, len(records) // 3, 2 * len(records) // 3, len(records) - 1})
        instance = runtime(task["model"], arguments.device)
        identity = read(output / "Identity.json")
        for key in ("registry_entry", "asset_sha256", "source_files", "torch", "numpy", "device"):
            require(instance.identity[key] == identity[key], "Replay runtime differs: " + key)
        comparisons = []
        for offset in replay_offsets:
            expected = records[offset]
            metadata, frames = dataset.load(expected["position"])
            encoded = instance.encode(frames)
            require(encoded["metadata"]["pixel_sha256"] == expected["encoding"]["pixel_sha256"], "Replay pixels differ")
            for candidate, prediction in expected["predictions"].items():
                actual = decode(instance, encoded, candidate, task["candidates"])
                require(actual["tokens"] == prediction["tokens"] and actual["text"] == prediction["text"],
                        "Fresh whole-video replay differs")
                error = abs(actual["score"] - prediction["score"]) if "score" in actual else 0.0
                require(error <= 1e-4, "Same-device score drift")
                comparisons.append({"position": expected["position"], "candidate": candidate, "score_error": error})
            del encoded
    save(output / "Verification.json", {
        "utc": utc(), "status": "all_records_recounted_and_native_replay_passed",
        "task": arguments.task, "clips": len(records), "fresh_pixel_replays": len(replay_offsets),
        "comparisons": comparisons, "predictions_sha256": sha(output / "Predictions.jsonl"),
        "identity_sha256": sha(output / "Identity.json"), "plan_sha256": sha(HERE / "Plan.json"),
        "complete_population_receipt": False,
    })
    print({"status": "verified", "task": arguments.task, "clips": len(records)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    main(parser.parse_args())
