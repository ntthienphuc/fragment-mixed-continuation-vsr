"""Run one bounded task only after reviewed admission and, for test, decoder locking."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from admission import checked, require
from common import HERE, ROOT, lines, read, save, sha, utc
from task_design import candidates_for, selected_task


def append(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main(task_name, attempt, device, threads):
    lock_path = HERE / "Execution_Lock.json"
    require(lock_path.exists(), "Execution code is not locked; no model or output was created")
    lock = read(lock_path)
    task, rows = selected_task(task_name, lock)
    plan, admission, selection = checked(task["stage"], task["model"])
    require(attempt.isidentifier(), "Attempt must be a simple identifier")
    candidates = candidates_for(task["stage"], task["model"], plan, selection)
    folder = HERE / "Runs" / attempt / task_name
    require(not folder.exists(), "Preserve the previous task attempt; do not overwrite")
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "Start.json", {
        "utc": utc(), "pid": os.getpid(), "task": task_name, "model": task["model"],
        "stage": task["stage"], "execution_lock_sha256": sha(lock_path),
        "admission_sha256": sha(HERE / "Admission.json"), "deadline_unix": admission["deadline_unix"],
        "selection_sha256": sha(HERE / "Decoder_Selection.json") if selection else None,
    })
    began = time.monotonic()
    try:
        import torch
        from runtime_bridge import create_runtime, decode, pixels_for

        sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
        from verify_probe import counts

        pixels_path = HERE / "Pixels_01/Index.jsonl"
        pixels_proof = read(HERE / "Pixels_01/Verification.json")
        require(sha(pixels_path) == pixels_proof["index_sha256"], "Pixel evidence changed")
        pixel_index = {row["clip_id"]: row for row in lines(pixels_path)}
        runtime = create_runtime(task["model"], device=device, threads=threads)
        save(folder / "Identity.json", runtime.identity)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        for position, row in rows:
            require(time.time() < admission["deadline_unix"], "Deadline reached; preserve partial attempt")
            encoded = runtime.encode(pixels_for(row, task["model"], pixel_index[row["clip_id"]]))
            predictions = {}
            for candidate in candidates:
                require(time.time() < admission["deadline_unix"], "Deadline reached during decoder grid")
                result = decode(runtime, encoded, candidate, plan)
                predictions[candidate] = {**result, "counts": counts(row["reference"], result["text"])}
            append(folder / "Predictions.jsonl", {
                "index": position, "model": task["model"], "split": task["stage"],
                "clip_id": row["clip_id"], "source_id": row["source_id"], "frames": row["frames"],
                "media_sha256": row["input"]["sha256"], "encoding": encoded["metadata"],
                "predictions": predictions, "utc": utc(),
            })
            del encoded
        peak = torch.cuda.max_memory_allocated(runtime.device) / 2**20 if runtime.device.type == "cuda" else None
        save(folder / "Complete.json", {
            "utc": utc(), "status": "complete_pending_independent_verification", "task": task_name,
            "clips": len(rows), "candidate_order": candidates, "seconds": time.monotonic() - began,
            "predictions_sha256": sha(folder / "Predictions.jsonl"), "peak_gpu_mib": peak,
            "execution_lock_sha256": sha(lock_path), "plan_sha256": sha(HERE / "Plan.json"),
        })
        print({"task": task_name, "status": "pending_verification", "clips": len(rows)}, flush=True)
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    main(arguments.task, arguments.attempt, arguments.device, arguments.threads)
