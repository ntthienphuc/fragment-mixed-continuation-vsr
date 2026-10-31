"""Independently recount every row and replay deterministic samples from original pixels."""

import argparse
from pathlib import Path
import sys
import time

from admission import checked, require
from common import ENGINE, HERE, lines, read, save, sha, utc
from task_design import candidates_for, selected_task


def verify_records(rows, predictions, model, stage, candidates, pixel_index):
    sys.path.insert(0, str(ENGINE))
    from confirmation_common import edit_counts

    require(len(rows) == len(predictions), "Incomplete task")
    checks = 0
    for (position, source), predicted in zip(rows, predictions):
        require(predicted["index"] == position, "Task order differs")
        require(predicted["model"] == model and predicted["split"] == stage, "Model/split differs")
        for field in ("clip_id", "source_id", "frames"):
            require(source[field] == predicted[field], "Row identity differs: " + field)
        require(predicted["media_sha256"] == source["input"]["sha256"], "Video differs")
        color = "rgb" if model == "VC_NATIVE" else "gray"
        metadata = predicted["encoding"]
        require(metadata["pixel_sha256"] == pixel_index[source["clip_id"]][color + "_sha256"],
                "Decoded pixels differ")
        require(metadata["color_space"] == color and metadata["precision"] == "float32", "Input convention differs")
        require(metadata["frames"] == source["frames"] == metadata["state_frames"], "Frame count differs")
        require(set(predicted["predictions"]) == set(candidates), "Extra or missing decoder")
        for candidate in candidates:
            result = predicted["predictions"][candidate]
            require(edit_counts(source["reference"], result["text"]) == result["counts"], "Edit recount differs")
            if candidate.startswith("beam_c"):
                require(result["cap"] == source["frames"], "Beam cap differs from full-clip input")
                require(result["trace"]["beam_size"] == 40 and result["trace"]["penalty"] == 0,
                        "Beam search settings differ")
            checks += 1
    return checks


def main(task_name, folder, device, threads):
    lock_path = HERE / "Execution_Lock.json"
    task, rows = selected_task(task_name, read(lock_path))
    plan, admission, selection = checked(task["stage"], task["model"])
    candidates = candidates_for(task["stage"], task["model"], plan, selection)
    complete = read(folder / "Complete.json")
    require(complete["execution_lock_sha256"] == sha(lock_path), "Worker lock differs")
    require(complete["task"] == task_name and complete["clips"] == len(rows), "Wrong completion receipt")
    path = folder / "Predictions.jsonl"
    require(complete["predictions_sha256"] == sha(path), "Predictions changed after completion")
    predictions = lines(path)
    pixels_path = HERE / "Pixels_01/Index.jsonl"
    require(sha(pixels_path) == read(HERE / "Pixels_01/Verification.json")["index_sha256"], "Pixels index changed")
    pixel_index = {row["clip_id"]: row for row in lines(pixels_path)}
    checks = verify_records(rows, predictions, task["model"], task["stage"], candidates, pixel_index)
    from runtime_bridge import create_runtime, decode, pixels_for

    runtime = create_runtime(task["model"], device=device, threads=threads)
    worker_identity = read(folder / "Identity.json")
    require(worker_identity["asset_sha256"] == runtime.identity["asset_sha256"], "Replay assets differ")
    replays = []
    for index in sorted({0, len(rows) - 1}):
        require(time.time() < admission["deadline_unix"], "Verification deadline reached")
        _, row = rows[index]
        encoded = runtime.encode(pixels_for(row, task["model"], pixel_index[row["clip_id"]]))
        for candidate in candidates:
            actual = decode(runtime, encoded, candidate, plan)
            expected = predictions[index]["predictions"][candidate]
            require(actual["tokens"] == expected["tokens"] and actual["text"] == expected["text"],
                    "Fresh-pixel recognition replay differs")
            score_error = abs(actual["score"] - expected["score"]) if "score" in expected else 0
            require(score_error <= 1e-4, "Fresh-pixel score replay differs")
            replays.append({"clip_id": row["clip_id"], "decoder": candidate, "score_error": score_error})
    save(folder / "Verification.json", {
        "status": "passed", "utc": utc(), "task": task_name, "model": task["model"], "split": task["stage"],
        "clips": len(rows), "independent_count_checks": checks, "fresh_pixel_replays": replays,
        "predictions_sha256": sha(path), "execution_lock_sha256": sha(lock_path),
        "checkpoint_sha256": runtime.entry["checkpoint_sha256"], "candidate_order": candidates,
        "plan_sha256": sha(HERE / "Plan.json"),
    })
    print({"task": task_name, "status": "passed", "count_checks": checks, "replays": len(replays)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    main(arguments.task, arguments.folder, arguments.device, arguments.threads)
