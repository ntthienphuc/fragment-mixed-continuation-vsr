"""Run one complete task without using reference text for recognition."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import HERE, ROOT, read, save, sha, utc
from data_tasks import open_data, positions, candidates
from execution import checked


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("task")
    value.add_argument("--model", required=True)
    value.add_argument("--binding", type=Path, required=True)
    value.add_argument("--admission", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", choices=("cpu", "cuda"), required=True)
    value.add_argument("--mode", choices=("fixture", "probe", "population"), required=True)
    value.add_argument("--readout", choices=("grid", "common", "selected"), required=True)
    value.add_argument("--selection", type=Path)
    value.add_argument("--destination-root", type=Path)
    return value


def main(arguments):
    admission = checked(arguments)
    output = arguments.output.resolve()
    assert output.is_relative_to(HERE) and output != HERE
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    deadline = min(admission["deadline_unix"], began + admission["maximum_task_seconds"])
    stage = arguments.task.rsplit("_", 1)[0]
    choices = candidates(stage, arguments.readout, arguments.model, arguments.binding, arguments.selection)
    start = {"utc": utc(), "pid": os.getpid(), "task": arguments.task, "model": arguments.model,
             "stage": stage, "mode": arguments.mode, "readout": arguments.readout, "device": arguments.device,
             "candidates": choices, "assessment_plan_sha256": sha(HERE / "Plan.json"),
             "code_lock_sha256": sha(HERE / "Code_Lock.json"), "admission_sha256": sha(arguments.admission),
             "registry_sha256": sha(arguments.binding / "model_registry.json"), "deadline_unix": deadline,
             "selection_sha256": sha(arguments.selection) if arguments.selection else None,
             "destination_root": str(arguments.destination_root) if arguments.destination_root else None}
    save(output / "Start.json", start)
    try:
        import torch
        sys.path.insert(0, str(ROOT / "Research_Server_Confirmation_2026-10-19"))
        from confirmation_common import edit_counts
        from runtime_bridge import create_runtime, decode

        runtime = create_runtime(arguments.binding, arguments.model, device=arguments.device)
        save(output / "Identity.json", runtime.identity)
        if arguments.device == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        with open_data(stage, arguments.destination_root) as dataset:
            selected = positions(dataset, arguments.task, arguments.mode)
            with (output / "Predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for ordinal, position in enumerate(selected, 1):
                    assert time.time() < deadline, "Deadline reached; retain incomplete attempt"
                    row, pixels = dataset.load(position)
                    encoded = runtime.encode(pixels)
                    results = {}
                    for candidate in choices:
                        assert time.time() < deadline, "Decoder deadline reached"
                        prediction = decode(runtime, encoded, candidate, choices)
                        prediction["counts"] = edit_counts(row["reference"], prediction["text"])
                        results[candidate] = prediction
                    record = {"utc": utc(), "position": position, "clip_id": row["clip_id"],
                              "source_id": row["source_id"], "frames": row["frames"], "model": arguments.model,
                              "stage": stage, "mode": arguments.mode, "encoding": encoded["metadata"],
                              "predictions": results}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    del encoded
                    if ordinal % 20 == 0 or ordinal == len(selected):
                        elapsed = time.time() - began
                        print({"task": arguments.task, "clips": ordinal, "total": len(selected),
                               "elapsed_seconds": elapsed, "eta_seconds": elapsed *
                               (len(selected) - ordinal) / ordinal}, flush=True)
        peak = torch.cuda.max_memory_allocated(runtime.device) / 2**20 if arguments.device == "cuda" else None
        save(output / "Complete.json", {"utc": utc(), "status": "complete_pending_independent_verification",
             "task": arguments.task, "model": arguments.model, "mode": arguments.mode,
             "clips": len(selected), "seconds": time.time() - began, "peak_gpu_mib": peak,
             "predictions_sha256": sha(output / "Predictions.jsonl"),
             "identity_sha256": sha(output / "Identity.json"), "start_sha256": sha(output / "Start.json"),
             "complete_population_receipt": False})
    except BaseException:
        save(output / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main(parser().parse_args())
