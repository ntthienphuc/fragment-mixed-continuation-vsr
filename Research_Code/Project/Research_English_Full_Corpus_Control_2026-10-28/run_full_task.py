"""Decode every clip in one frozen F0 shard, retaining incomplete attempts."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from execution_common import checked, require
from experiment_core import candidates, decode, native_bridge, pixels
from mechanism_common import ENGINE, HERE, ROOT, save, sha, utc


def main(task_name, attempt, admission_path, device):
    plan, task, rows, admission, selection, locations = checked(admission_path, task_name, device)
    require(attempt.isidentifier(), "Invalid attempt name")
    output = HERE / "Runs" / attempt / task_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    selected = candidates(plan, task, selection)
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "started_unix": started,
         "task": task_name, "purpose": admission["purpose"], "device": device,
         "admission_path": admission_path.relative_to(ROOT).as_posix(),
         "admission_sha256": sha(admission_path), "deadline_unix": admission["deadline_unix"],
         "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
         "selection_sha256": sha(HERE / "Decoder_Selection.json") if selection else None})
    try:
        import torch

        sys.path.insert(0, str(ENGINE))
        from confirmation_common import edit_counts

        runtime = native_bridge().create_runtime(task["model"], device=device, threads=admission["threads"])
        save(output / "Identity.json", runtime.identity)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        began = time.monotonic()
        with (output / "Predictions.jsonl").open("x", encoding="utf-8") as stream:
            for number, (position, row) in enumerate(rows, 1):
                require(time.time() < admission["deadline_unix"], "Clip deadline expired")
                encoded = runtime.encode(pixels(row, locations))
                predictions = {}
                for candidate in selected:
                    require(time.time() < admission["deadline_unix"], "Decoder deadline expired")
                    prediction = decode(runtime, encoded, candidate, plan)
                    prediction["counts"] = edit_counts(row["reference"], prediction["text"])
                    predictions[candidate] = prediction
                record = {"index": position, "clip_id": row["clip_id"], "source_id": row["source_id"],
                          "frames": row["frames"], "model": task["model"], "split": task["stage"],
                          "media_sha256": row["input"]["sha256"], "encoding": encoded["metadata"],
                          "predictions": predictions, "utc": utc()}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                del encoded
                if number % 10 == 0 or number == len(rows):
                    elapsed = time.monotonic() - began
                    print({"utc": utc(), "task": task_name, "clips": number, "total": len(rows),
                           "seconds": elapsed, "eta_seconds": elapsed * (len(rows) - number) / number}, flush=True)
        peak = torch.cuda.max_memory_allocated(runtime.device) / 2**20 if runtime.device.type == "cuda" else None
        save(output / "Complete.json", {"utc": utc(), "status": "complete_pending_independent_verification",
             "task": task_name, "model": task["model"], "split": task["stage"], "purpose": admission["purpose"],
             "clips": len(rows), "candidate_order": selected, "peak_gpu_mib": peak, "seconds": time.time() - started,
             "predictions_sha256": sha(output / "Predictions.jsonl"),
             "identity_sha256": sha(output / "Identity.json"), "plan_sha256": sha(HERE / "Plan.json"),
             "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    except BaseException as error:
        save(output / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    arguments = parser.parse_args()
    main(arguments.task, arguments.attempt, arguments.admission.resolve(), arguments.device)
