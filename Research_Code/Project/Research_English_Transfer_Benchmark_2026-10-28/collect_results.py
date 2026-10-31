"""Collect a complete explicitly named task set; never select the best retry."""

import argparse
from pathlib import Path

from english_common import HERE, ROOT, lines, read, require, save, sha, utc
from task_design import candidates_for, cohort, selected_task
from verify_task import verify_records


def collect(stage, manifest_path, plan, selection=None):
    execution = read(HERE / "Execution_Lock.json")
    manifest = read(manifest_path)
    expected_names = {name for name, task in execution["tasks"].items() if task["stage"] == stage}
    require(manifest["stage"] == stage and set(manifest["tasks"]) == expected_names, "Incomplete task manifest")
    require(manifest["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong manifest lock")
    records = {model: {} for model in plan["models"]}
    inputs = {manifest_path.resolve().relative_to(ROOT.resolve()).as_posix(): sha(manifest_path)}
    registry = read(HERE / "model_registry.json")["models"]
    cache_indices = {model: lines(HERE / "Source_Caches" / (model + ".jsonl")) for model in plan["models"]}
    for name in sorted(expected_names):
        path = ROOT / manifest["tasks"][name]["path"]
        proof_path = path / "Verification.json"
        proof = read(proof_path)
        require(sha(proof_path) == manifest["tasks"][name]["verification_sha256"], "Verification changed")
        task, rows = selected_task(name, execution)
        candidates = candidates_for(stage, task["model"], plan, selection)
        require(proof["status"] == "passed" and proof["task"] == name, "Task not verified")
        require(proof["plan_sha256"] == sha(HERE / "Plan.json"), "Wrong verification plan")
        require(proof["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong verification code")
        require(proof["checkpoint_sha256"] == registry[task["model"]]["checkpoint_sha256"], "Wrong checkpoint")
        require(proof["candidate_order"] == candidates and proof["clips"] == len(rows), "Wrong candidate scope")
        predictions_path = path / "Predictions.jsonl"
        require(proof["predictions_sha256"] == sha(predictions_path), "Predictions changed")
        predictions = lines(predictions_path)
        checks = verify_records(rows, predictions, task, candidates, cache_indices[task["model"]])
        require(checks == proof["independent_counts_checked"], "Independent count coverage differs")
        for predicted in predictions:
            index = predicted["index"]
            require(index not in records[task["model"]], "Duplicated clip")
            records[task["model"]][index] = predicted
        inputs[predictions_path.relative_to(ROOT).as_posix()] = sha(predictions_path)
        inputs[proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
    population = cohort(stage)
    for model in plan["models"]:
        require(set(records[model]) == set(range(len(population))), "Incomplete model population")
        records[model] = [records[model][index] for index in range(len(population))]
    return records, inputs


def build_manifest(stage, output):
    require(not output.exists(), "Manifest already exists")
    execution = read(HERE / "Execution_Lock.json")
    tasks = {}
    for name, task in execution["tasks"].items():
        if task["stage"] != stage:
            continue
        matches = list((HERE / "Runs").glob("*/" + name + "/Verification.json"))
        require(len(matches) == 1, "Need exactly one verified attempt, or an explicit reviewed manifest: " + name)
        tasks[name] = {"path": matches[0].parent.relative_to(ROOT).as_posix(),
                       "verification_sha256": sha(matches[0])}
    require(len(tasks) == 32, "Incomplete stage")
    save(output, {"utc": utc(), "status": "all_stage_tasks_explicitly_named", "stage": stage,
                  "execution_lock_sha256": sha(HERE / "Execution_Lock.json"), "tasks": tasks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["development", "assessment"])
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build_manifest(arguments.stage, arguments.output)
