"""Complete all development populations before selecting any test decoder."""

import json

from admission import require, verify_plan
from common import HERE, ROOT, lines, read, save, sha, utc
from reuse_development import prior_predictions
from select_decoder import choose
from task_design import selected_task
from verify_task import verify_records


def main():
    plan = verify_plan()
    lock_path = HERE / "Execution_Lock.json"
    lock = read(lock_path)
    require(lock["plan_sha256"] == sha(HERE / "Plan.json"), "Execution plan changed")
    for name, expected in lock["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Execution code changed: " + name)
    pixel_path = HERE / "Pixels_01/Index.jsonl"
    require(sha(pixel_path) == read(HERE / "Pixels_01/Verification.json")["index_sha256"], "Pixel identity changed")
    pixel_index = {row["clip_id"]: row for row in lines(pixel_path)}
    cohort = lines(HERE / "Development.jsonl")
    registry = read(HERE / "model_registry.json")["models"]
    registry.update(read(HERE / "Native_Reference_Registry.json")["models"])
    assembled = {}
    evidence_by_model = {}
    for model in [*plan["matched_models"], plan["practical_reference"]]:
        records, evidence = ({}, {}) if model == "VC_NATIVE" else prior_predictions(model)
        for task_name, task in lock["tasks"].items():
            if task["stage"] != "development" or task["model"] != model:
                continue
            matches = list((HERE / "Runs").glob("*/" + task_name + "/Verification.json"))
            require(len(matches) == 1, "Exactly one verified complete attempt required: " + task_name)
            proof_path = matches[0]
            proof = read(proof_path)
            require(proof["status"] == "passed" and proof["model"] == model, "Invalid task proof")
            require(proof["execution_lock_sha256"] == sha(lock_path), "Wrong task execution lock")
            require(proof["checkpoint_sha256"] == registry[model]["checkpoint_sha256"], "Wrong checkpoint")
            path = proof_path.parent / "Predictions.jsonl"
            require(proof["predictions_sha256"] == sha(path), "Predictions changed")
            _, rows = selected_task(task_name, lock)
            predicted = lines(path)
            checked = verify_records(rows, predicted, model, "development", plan["candidate_order"], pixel_index)
            require(proof["independent_count_checks"] == checked, "Recount differs from proof")
            require(len(proof["fresh_pixel_replays"]) == 22, "Fresh runtime replay incomplete")
            for row in predicted:
                require(row["clip_id"] not in records, "Duplicate reused/new clip")
                records[row["clip_id"]] = row
            for item in (path, proof_path):
                evidence[item.relative_to(ROOT).as_posix()] = sha(item)
        complete_rows = [records[row["clip_id"]] for row in cohort]
        require(len(records) == len(cohort) == 768, "Incomplete or extra development clips")
        checked = choose(cohort, complete_rows, plan["candidate_order"])
        require(checked["independent_prediction_counts_checked"] == 8448, "Incomplete decoder grid")
        assembled[model] = complete_rows
        evidence_by_model[model] = evidence
    folder = HERE / "Development_Bundle_01"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "complete_verified_development_bundle", "utc": utc(),
                "plan_sha256": sha(HERE / "Plan.json"), "models": {}}
    for model, rows in assembled.items():
        path = folder / (model + ".jsonl")
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        proof_path = folder / (model + "_Verification.json")
        save(proof_path, {"status": "passed", "utc": utc(), "model": model, "split": "development",
                          "clips": 768, "independent_count_checks": 8448, "candidate_order": plan["candidate_order"],
                          "plan_sha256": sha(HERE / "Plan.json"), "predictions_sha256": sha(path),
                          "checkpoint_sha256": registry[model]["checkpoint_sha256"],
                          "input_sha256": evidence_by_model[model]})
        manifest["models"][model] = {
            "predictions_path": path.relative_to(ROOT).as_posix(), "predictions_sha256": sha(path),
            "verification_path": proof_path.relative_to(ROOT).as_posix(), "verification_sha256": sha(proof_path),
        }
    save(folder / "Manifest.json", manifest)
    print({"status": "complete_development_bundle_verified", "clips_per_model": 768, "models": 6}, flush=True)


if __name__ == "__main__":
    main()
