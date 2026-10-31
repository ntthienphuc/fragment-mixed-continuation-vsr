"""Run a locked whole-clip shard, retaining every prediction and original denominator."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from assessment_common import HERE, check
from retention_common import ENGINE, ROOT, decode, read, require, runtime, save, sha, source, utc


def main(arguments):
    plan, task = check(arguments.task)
    require(arguments.attempt.isidentifier(), "Invalid attempt")
    proof_path = arguments.probe.resolve()
    require(proof_path.is_relative_to(HERE.parent), "Probe must be in this source-retention phase")
    proof = read(proof_path / "Complete.json")
    start = read(proof_path / "Start.json")
    require(proof["status"] == "resource_probe_replayed_not_population_evaluation", "Probe incomplete")
    require(proof["corpus"] == task["corpus"] and start["device"] == arguments.device, "Wrong resource probe")
    require(sha(proof_path / "Predictions.jsonl") == proof["predictions_sha256"], "Probe predictions changed")
    require(sha(proof_path / "Identity.json") == proof["identity_sha256"], "Probe identity changed")
    require(start["plan_sha256"] == plan["parent_plan_sha256"], "Probe uses different code")
    output = HERE / "Runs" / arguments.attempt / arguments.task
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + plan["maximum_task_seconds"]
    save(output / "Start.json", {
        "utc": utc(), "pid": os.getpid(), "task": arguments.task, "device": arguments.device,
        "plan_sha256": sha(HERE / "Plan.json"), "probe_receipt_sha256": sha(proof_path / "Complete.json"),
        "deadline_unix": deadline, "purpose": "complete_population_source_retention",
    })
    try:
        import torch
        sys.path.insert(0, str(ENGINE))
        from confirmation_common import edit_counts

        instance = runtime(task["model"], arguments.device)
        save(output / "Identity.json", instance.identity)
        if instance.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(instance.device)
        with source(task["corpus"]) as dataset:
            with (output / "Predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                began = time.monotonic()
                for ordinal, position in enumerate(task["positions"], 1):
                    require(time.time() < deadline, "Task deadline expired; preserve incomplete attempt")
                    metadata, frames = dataset.load(position)
                    encoded = instance.encode(frames)
                    predictions = {}
                    for candidate in task["candidates"]:
                        require(time.time() < deadline, "Decoder deadline expired")
                        value = decode(instance, encoded, candidate, task["candidates"])
                        value["counts"] = edit_counts(metadata["text"], value["text"])
                        predictions[candidate] = value
                    record = {
                        "position": position, "clip_id": metadata["clip_id"],
                        "source_id": metadata["source_id"], "frames": len(frames),
                        "model": task["model"], "corpus": task["corpus"], "encoding": encoded["metadata"],
                        "predictions": predictions, "utc": utc(),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    del encoded
                    if ordinal % 20 == 0 or ordinal == len(task["positions"]):
                        elapsed = time.monotonic() - began
                        print({"utc": utc(), "task": arguments.task, "clips": ordinal,
                               "total": len(task["positions"]), "seconds": elapsed,
                               "eta_seconds": elapsed * (len(task["positions"]) - ordinal) / ordinal}, flush=True)
        peak = torch.cuda.max_memory_allocated(instance.device) / 2**20 if arguments.device == "cuda" else None
        save(output / "Complete.json", {
            "utc": utc(), "status": "complete_pending_independent_verification", "task": arguments.task,
            "clips": len(task["positions"]), "predictions_sha256": sha(output / "Predictions.jsonl"),
            "identity_sha256": sha(output / "Identity.json"), "seconds": time.time() - started,
            "peak_gpu_mib": peak, "population_complete": False,
        })
    except BaseException:
        save(output / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--probe", required=True, type=Path)
    main(parser.parse_args())
