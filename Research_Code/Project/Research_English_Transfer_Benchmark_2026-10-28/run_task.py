"""Run a registered full-clip task, preserving partial attempts on any failure."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from admission import checked
from english_common import ENGINE, HERE, ROOT, read, require, save, sha, utc
from task_design import candidates_for, selected_task


def main(task_name, attempt, admission_path, device, threads):
    execution = read(HERE / "Execution_Lock.json")
    task, rows = selected_task(task_name, execution)
    plan, admission, selection, locations = checked(task["stage"], task["model"], admission_path, task_name)
    require(attempt.isidentifier(), "Invalid attempt name")
    output = HERE / "Runs" / attempt / task_name
    require(not output.exists(), "Existing attempt must be preserved")
    output.mkdir(parents=True, exist_ok=False)
    candidates = candidates_for(task["stage"], task["model"], plan, selection)
    started = time.time()
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "started_unix": started,
                                  "task": task_name, "admission_path": admission_path.relative_to(ROOT).as_posix(),
                                  "admission_sha256": sha(admission_path), "deadline_unix": admission["deadline_unix"],
                                  "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
                                  "selection_sha256": sha(HERE / "Decoder_Selection.json") if selection else None})
    cache = None
    try:
        import torch
        from runtime_bridge import DevelopmentCache, create_runtime, decode, pixels_for

        sys.path.insert(0, str(ENGINE))
        from confirmation_common import edit_counts

        runtime = create_runtime(task["model"], device, threads)
        save(output / "Identity.json", runtime.identity)
        if task["stage"] == "development":
            cache = DevelopmentCache(task["model"], locations, runtime)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        began = time.monotonic()
        with (output / "Predictions.jsonl").open("x", encoding="utf-8") as stream:
            for number, (position, row) in enumerate(rows, 1):
                require(time.time() < admission["deadline_unix"], "Task deadline reached")
                encoded = cache.encoded(position, row) if cache else runtime.encode(pixels_for(row, locations))
                predictions = {}
                for candidate in candidates:
                    require(time.time() < admission["deadline_unix"], "Decoder deadline reached")
                    prediction = decode(runtime, encoded, candidate, plan)
                    predictions[candidate] = {**prediction, "counts": edit_counts(row["reference"], prediction["text"])}
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
                                        "task": task_name, "model": task["model"], "split": task["stage"],
                                        "clips": len(rows), "candidate_order": candidates, "peak_gpu_mib": peak,
                                        "seconds": time.time() - started,
                                        "predictions_sha256": sha(output / "Predictions.jsonl"),
                                        "identity_sha256": sha(output / "Identity.json"),
                                        "plan_sha256": sha(HERE / "Plan.json"),
                                        "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    except BaseException as error:
        save(output / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    main(arguments.task, arguments.attempt, arguments.admission.resolve(), arguments.device, arguments.threads)
