"""Recount the entire shard and independently re-encode fixed endpoints."""

import argparse
from pathlib import Path
import time

from execution_common import checked, require, validate_budget
from experiment_core import candidates, decode, native_bridge, pixels
from mechanism_common import HERE, ROOT, lines, read, save, sha, utc
from verify_full_records import same_backend_replay, verify_records


def main(task_name, folder, admission_path, device):
    plan, task, rows, admission, selection, locations = checked(admission_path, task_name, device)
    folder = folder.resolve()
    folder.relative_to((HERE / "Runs").resolve())
    require(folder.name == task_name, "Wrong task folder")
    require(not (folder / "Failure.json").exists(), "Failed attempt cannot be verified")
    complete = read(folder / "Complete.json")
    start = read(folder / "Start.json")
    selected = candidates(plan, task, selection)
    require(complete["status"] == "complete_pending_independent_verification", "Task not complete")
    require(complete["task"] == start["task"] == task_name, "Task identity differs")
    require(complete["model"] == task["model"] and complete["split"] == task["stage"], "Wrong complete identity")
    require(complete["purpose"] == start["purpose"] == admission["purpose"], "Task purpose differs")
    require(complete["clips"] == len(rows) and complete["candidate_order"] == selected, "Wrong task coverage")
    require(complete["plan_sha256"] == sha(HERE / "Plan.json"), "Wrong complete plan")
    require(complete["execution_lock_sha256"] == start["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"),
            "Execution changed between task and verification")
    require(complete["predictions_sha256"] == sha(folder / "Predictions.jsonl"), "Predictions changed")
    require(complete["identity_sha256"] == sha(folder / "Identity.json"), "Identity changed")
    require(start["device"] == device, "Verifier must use the worker backend")
    require(start["selection_sha256"] == (sha(HERE / "Decoder_Selection.json") if selection else None),
            "Worker selection differs")
    original_admission = ROOT / start["admission_path"]
    require(sha(original_admission) == start["admission_sha256"], "Worker admission changed")
    validate_budget(read(original_admission), start["started_unix"])
    predicted = lines(folder / "Predictions.jsonl")
    independent = verify_records(rows, predicted, task, selected)
    runtime = native_bridge().create_runtime(task["model"], device=device, threads=admission["threads"])
    identity = read(folder / "Identity.json")
    require(identity["strict_state_load"] and identity["asset_sha256"] == runtime.identity["asset_sha256"],
            "Worker and verifier assets differ")
    require(identity["selected_training_step"] == runtime.identity["selected_training_step"] == 6000,
            "Checkpoint endpoint changed")
    for record in predicted:
        for result in record["predictions"].values():
            require(runtime.text(result["tokens"]) == result["text"], "Token/text conversion differs")
    replays = []
    for number in sorted({0, len(rows) - 1}):
        require(time.time() < admission["deadline_unix"], "Replay deadline expired")
        position, row = rows[number]
        encoded = runtime.encode(pixels(row, locations))
        require(encoded["metadata"]["pixel_sha256"] == predicted[number]["encoding"]["pixel_sha256"],
                "Fresh pixels differ")
        for candidate in selected:
            require(time.time() < admission["deadline_unix"], "Replay decoder deadline expired")
            actual = decode(runtime, encoded, candidate, plan)
            error = same_backend_replay(predicted[number]["predictions"][candidate], actual)
            replays.append({"position": position, "candidate": candidate, "score_error": error,
                            "tokens_and_text_exact": True})
        del encoded
    save(folder / "Verification.json", {"utc": utc(), "status": "passed", "task": task_name,
         "purpose": admission["purpose"], "model": task["model"], "split": task["stage"], "clips": len(rows),
         "independent_counts_checked": independent, "fresh_pixel_replays": replays, "candidate_order": selected,
         "checkpoint_sha256": runtime.entry["checkpoint_sha256"], "plan_sha256": sha(HERE / "Plan.json"),
         "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
         "predictions_sha256": sha(folder / "Predictions.jsonl"), "complete_sha256": sha(folder / "Complete.json"),
         "start_sha256": sha(folder / "Start.json"), "identity_sha256": sha(folder / "Identity.json"),
         "verification_admission_sha256": sha(admission_path), "verifier_sha256": sha(__file__)})
    print({"utc": utc(), "task": task_name, "status": "passed", "recounts": independent,
           "fresh_pixel_replays": len(replays)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    arguments = parser.parse_args()
    main(arguments.task, arguments.folder, arguments.admission.resolve(), arguments.device)
