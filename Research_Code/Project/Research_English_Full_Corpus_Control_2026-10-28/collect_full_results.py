"""Require full explicitly named verified populations; reject integration output."""

import argparse
import math
from pathlib import Path

from execution_common import checked_execution, checked_selection, require
from experiment_core import candidates, cohort, task_rows
from mechanism_common import HERE, ROOT, lines, read, save, sha, utc
from verify_full_records import verify_records


def collect(stage, manifest_path, plan, selection=None):
    manifest = read(manifest_path)
    expected = {name for name, task in plan["tasks"].items() if task["stage"] == stage}
    require(manifest["stage"] == stage and set(manifest["tasks"]) == expected, "Incomplete task manifest")
    require(manifest["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong manifest code")
    records = {}
    inputs = {manifest_path.resolve().relative_to(ROOT).as_posix(): sha(manifest_path)}
    registry = read(HERE / "model_registry.json")["models"]
    for name in sorted(expected):
        folder = (ROOT / manifest["tasks"][name]["path"]).resolve()
        folder.relative_to((HERE / "Runs").resolve())
        require(folder.name == name and not (folder / "Failure.json").exists(), "Wrong or failed task")
        proof_path = folder / "Verification.json"
        require(sha(proof_path) == manifest["tasks"][name]["verification_sha256"], "Verification changed")
        proof = read(proof_path)
        task, rows = task_rows(plan, name)
        selected = candidates(plan, task, selection)
        require(proof["status"] == "passed" and proof["task"] == name, "Unverified task")
        require(proof["purpose"] == "full_evaluation", "Integration evidence cannot enter the benchmark")
        require(proof["model"] == task["model"] and proof["split"] == stage, "Wrong model/stage")
        require(proof["plan_sha256"] == sha(HERE / "Plan.json"), "Wrong verification plan")
        require(proof["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong verification code")
        require(proof["candidate_order"] == selected and proof["clips"] == len(rows), "Wrong decoder coverage")
        require(proof["checkpoint_sha256"] == registry[task["model"]]["checkpoint_sha256"], "Wrong checkpoint")
        require(proof["verifier_sha256"] == sha(HERE / "verify_full_task.py"), "Verifier changed")
        for filename, field in (("Predictions.jsonl", "predictions_sha256"), ("Complete.json", "complete_sha256"),
                                ("Start.json", "start_sha256"), ("Identity.json", "identity_sha256")):
            require(sha(folder / filename) == proof[field], "Verified file changed: " + filename)
            inputs[(folder / filename).relative_to(ROOT).as_posix()] = sha(folder / filename)
        predictions = lines(folder / "Predictions.jsonl")
        checked = verify_records(rows, predictions, task, selected)
        require(checked == proof["independent_counts_checked"], "Incomplete recount")
        require(len(proof["fresh_pixel_replays"]) == 2 * len(selected), "Incomplete fresh-pixel replay")
        expected_replays = {(rows[index][0], candidate) for index in (0, len(rows) - 1) for candidate in selected}
        replays = proof["fresh_pixel_replays"]
        require({(value["position"], value["candidate"]) for value in replays} == expected_replays,
                "Replay scope differs")
        require(all(value["tokens_and_text_exact"] is True and math.isfinite(value["score_error"])
                    and 0 <= value["score_error"] <= 1e-4 for value in replays), "Replay did not pass")
        for predicted in predictions:
            index = predicted["index"]
            require(index not in records, "Duplicated clip")
            records[index] = predicted
        inputs[proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
    population = cohort(plan, "en", stage)
    require(set(records) == set(range(len(population))), "Incomplete F0 population")
    return [records[index] for index in range(len(population))], inputs


def build_manifest(stage, output):
    plan, lock = checked_execution()
    tasks = {}
    for name, task in plan["tasks"].items():
        if task["stage"] != stage:
            continue
        matches = list((HERE / "Runs").glob("*/" + name + "/Verification.json"))
        require(len(matches) == 1, "Need one verified attempt or an explicitly reviewed manifest: " + name)
        tasks[name] = {"path": matches[0].parent.relative_to(ROOT).as_posix(),
                       "verification_sha256": sha(matches[0])}
    require(len(tasks) == 8, "Incomplete stage")
    save(output, {"utc": utc(), "status": "full_F0_stage_explicitly_named", "stage": stage,
                  "execution_lock_sha256": sha(HERE / "Execution_Lock.json"), "tasks": tasks})
    selection = checked_selection(plan) if stage == "assessment" else None
    collect(stage, output, plan, selection)
    print({"stage": stage, "tasks": len(tasks), "status": "complete_population_verified"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["development", "assessment"])
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build_manifest(arguments.stage, arguments.output.resolve())
