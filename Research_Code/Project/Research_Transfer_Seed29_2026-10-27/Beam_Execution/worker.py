"""Run one disjoint frozen GPU task; preserve attempts and per-clip predictions."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from common import ENGINE, HERE, ROOT, SOURCE, STORE, configure, lines, plan, read, save, sha, utc
from portable_runtime import DecoderRuntime, clear_attention

sys.path.insert(0, str(ENGINE))
from confirmation_common import edit_counts


def check_execution():
    lock = read(HERE / "Execution_Lock.json")
    assert lock["plan_sha256"] == sha(HERE / "Plan.json")
    for name, expected in lock["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for model, expected in lock["cache_index_sha256"].items():
        assert sha(STORE / "Cache" / model / "Index.jsonl") == expected, model
    return lock


def destination(attempt, task_id):
    base = (HERE / "Runs" / attempt).resolve()
    assert base.parent == (HERE / "Runs").resolve()
    path = (base / task_id).resolve()
    assert path.parent == base
    return path


def main(arguments):
    configure()
    frozen = plan()
    lock = check_execution()
    task = lock["tasks"][arguments.task]
    model = task["model"]
    folder = destination(arguments.attempt, arguments.task)
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    save(folder / "Start.json", {"utc": utc(), "pid": os.getpid(), "task": arguments.task,
                                 "deadline": arguments.deadline, "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    try:
        assert time.time() < arguments.deadline
        runtime = DecoderRuntime(model)
        save(folder / "Identity.json", runtime.identity)
        torch.cuda.reset_peak_memory_stats()
        rows = lines(SOURCE / "Cohort.jsonl")
        cache = STORE / "Cache" / model
        index = lines(cache / "Index.jsonl")
        with (folder / "Predictions.jsonl").open("x", encoding="utf-8") as stream:
            for completed, position in enumerate(task["positions"], 1):
                assert time.time() < arguments.deadline
                row = rows[position]
                item = index[position]
                assert row["clip_id"] == item["clip_id"] == frozen["clip_ids"][position]
                path = cache / item["file"]
                assert sha(path) == item["sha256"]
                with np.load(path, allow_pickle=False) as values:
                    states = torch.from_numpy(values["full_states"].copy()).to(runtime.device)
                    old_logp = values["full_logp"].copy()
                with torch.inference_mode():
                    logp = runtime.model.ctc.log_softmax(states.unsqueeze(0))[0].cpu().numpy()
                error = float(np.max(np.abs(logp - old_logp)))
                assert error <= frozen["ctc_logp_tolerance"], error
                outputs = {}
                for weight in frozen["beam_weights"]:
                    assert time.time() < arguments.deadline
                    prediction = runtime.decode(states, weight, cap=row["frames"], beam_size=40, penalty=0)
                    prediction["counts"] = edit_counts(row["reference"], prediction["text"])
                    outputs[f"beam_c{weight:.1f}"] = prediction
                    clear_attention(runtime.model)
                record = {"utc": utc(), "index": position, "clip_id": row["clip_id"],
                          "source_id": row["source_id"], "frames": row["frames"],
                          "cache_sha256": item["sha256"], "pixels_sha256": item["pixels_sha256"],
                          "maximum_ctc_error": error, "predictions": outputs}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if completed % 8 == 0 or completed == len(task["positions"]):
                    print(json.dumps({"utc": utc(), "task": arguments.task, "completed": completed,
                                      "total": len(task["positions"]), "seconds": time.monotonic() - start}), flush=True)
        save(folder / "Complete.json", {
            "status": "completed_pending_verification", "utc": utc(), "clips": len(task["positions"]),
            "task": arguments.task, "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
            "predictions_sha256": sha(folder / "Predictions.jsonl"), "seconds": time.monotonic() - start,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        })
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args())
