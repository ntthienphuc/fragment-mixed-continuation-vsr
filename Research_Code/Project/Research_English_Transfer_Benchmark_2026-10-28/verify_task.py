"""Recount every prediction and replay fixed task members from original pixels."""

import argparse
from pathlib import Path
import sys
import time

from admission import checked, validate_budget
from english_common import HERE, ROOT, lines, read, require, save, sha, utc
from task_design import candidates_for, selected_task

sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
from verify_probe import counts


def verify_records(rows, predictions, task, candidates, source_index=None):
    require(len(rows) == len(predictions), "Incomplete task predictions")
    checks = 0
    for (position, original), prediction in zip(rows, predictions):
        require(prediction["index"] == position, "Prediction order changed")
        require(prediction["model"] == task["model"] and prediction["split"] == task["stage"], "Wrong model/split")
        for name in ("clip_id", "source_id", "frames"):
            require(prediction[name] == original[name], "Wrong input identity: " + name)
        require(prediction["media_sha256"] == original["input"]["sha256"], "Wrong input packet")
        metadata = prediction["encoding"]
        require(metadata["model_id"] == task["model"], "Wrong encoding model")
        require(metadata["frames"] == metadata["state_frames"] == original["frames"], "Frame count changed")
        require(metadata["color_space"] == "gray" and metadata["precision"] == "float32", "Input convention changed")
        require(metadata["transform"] == "phase5_native_gray", "Transform changed")
        if task["stage"] == "development":
            expected = source_index[position]
            require(expected["clip_id"] == original["clip_id"], "Wrong cache position")
            require(metadata["state_sha256"] == expected["state_sha256"], "Wrong cached states")
            require(metadata["pixel_sha256"] == expected["encoding"]["pixel_sha256"], "Wrong source pixels")
        else:
            require(metadata["pixel_sha256"] == original["input"]["pixels_sha256"], "Wrong author-path pixels")
        require(set(prediction["predictions"]) == set(candidates), "Missing or extra decoder")
        for name, result in prediction["predictions"].items():
            require(counts(original["reference"], result["text"]) == result["counts"], "Independent recount differs")
            require(all(type(token) is int and 0 <= token < 5049 for token in result["tokens"]), "Invalid token")
            if name.startswith("beam_c"):
                require(result["cap"] == original["frames"], "Wrong search horizon")
                require(result["trace"]["beam_size"] == 40 and result["trace"]["penalty"] == 0, "Beam differs")
                require(result["trace"]["ctc_weight"] == float(name.removeprefix("beam_c")), "Weight differs")
            checks += 1
    return checks


def main(task_name, folder, admission_path, device, threads):
    task, rows = selected_task(task_name, read(HERE / "Execution_Lock.json"))
    plan, admission, selection, locations = checked(task["stage"], task["model"], admission_path, task_name)
    candidates = candidates_for(task["stage"], task["model"], plan, selection)
    complete = read(folder / "Complete.json")
    require(complete["task"] == task_name and complete["clips"] == len(rows), "Incomplete worker receipt")
    require(complete["plan_sha256"] == sha(HERE / "Plan.json"), "Wrong worker plan")
    require(complete["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong worker implementation")
    require(complete["predictions_sha256"] == sha(folder / "Predictions.jsonl"), "Predictions changed")
    require(complete["identity_sha256"] == sha(folder / "Identity.json"), "Worker identity changed")
    start = read(folder / "Start.json")
    original_admission = ROOT / start["admission_path"]
    require(sha(original_admission) == start["admission_sha256"], "Original admission changed")
    validate_budget(read(original_admission), start["started_unix"])
    predictions = lines(folder / "Predictions.jsonl")
    source_index = lines(HERE / "Source_Caches" / (task["model"] + ".jsonl")) if task["stage"] == "development" else None
    checked_counts = verify_records(rows, predictions, task, candidates, source_index)
    from runtime_bridge import create_runtime, decode, pixels_for

    runtime = create_runtime(task["model"], device, threads)
    require(read(folder / "Identity.json")["asset_sha256"] == runtime.identity["asset_sha256"], "Replay assets differ")
    for predicted in predictions:
        for value in predicted["predictions"].values():
            require(runtime.text(value["tokens"]) == value["text"], "Token/text conversion differs")
    replays = []
    for number in sorted({0, len(rows) - 1}):
        require(time.time() < admission["deadline_unix"], "Replay budget expired")
        _, row = rows[number]
        encoded = runtime.encode(pixels_for(row, locations))
        require(encoded["metadata"]["pixel_sha256"] == predictions[number]["encoding"]["pixel_sha256"],
                "Fresh-pixel input differs")
        for candidate in candidates:
            require(time.time() < admission["deadline_unix"], "Decoder replay budget expired")
            actual = decode(runtime, encoded, candidate, plan)
            expected = predictions[number]["predictions"][candidate]
            require(actual["tokens"] == expected["tokens"] and actual["text"] == expected["text"], "Replay differs")
            error = abs(actual["score"] - expected["score"]) if "score" in expected else 0
            require(error <= 1e-4, "Native score replay differs")
            replays.append({"clip_id": row["clip_id"], "candidate": candidate, "score_error": error})
        del encoded
    save(folder / "Verification.json", {"utc": utc(), "status": "passed", "task": task_name,
                                        "model": task["model"], "split": task["stage"], "clips": len(rows),
                                        "independent_counts_checked": checked_counts, "fresh_pixel_replays": replays,
                                        "candidate_order": candidates, "checkpoint_sha256": runtime.entry["checkpoint_sha256"],
                                        "plan_sha256": sha(HERE / "Plan.json"),
                                        "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
                                        "predictions_sha256": sha(folder / "Predictions.jsonl"),
                                        "verification_admission_sha256": sha(admission_path)})
    print({"task": task_name, "status": "passed", "counts": checked_counts, "replays": len(replays)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    main(arguments.task, arguments.folder, arguments.admission.resolve(), arguments.device, arguments.threads)
