"""Validate execution contracts using actual source-grid records and deliberately malformed cases."""

from copy import deepcopy
import argparse
import math
from pathlib import Path

from execution_common import check_scope, require, runtime_device_matches, selected_rows, validate_budget
from experiment_core import cohort, task_rows
from mechanism_common import HERE, read, save, sha, utc, validate_inputs
from select_full_decoder import calibration_provenance, choose
from verify_full_records import same_backend_replay, verify_records


def rejected(label, action, output):
    try:
        action()
    except (AssertionError, ValueError, KeyError):
        output.append(label)
        return
    raise AssertionError("Malformed input accepted: " + label)


def main(output_name="Execution_Verification_01.json"):
    plan = validate_inputs()
    registry = read(HERE / "model_registry.json")["models"]
    rejected_cases = []
    for actual, requested in (("cpu", "cpu"), ("cuda", "cuda"), ("cuda:0", "cuda")):
        assert runtime_device_matches(actual, requested)
    for actual, requested in (("cuda:1", "cuda"), ("cpu", "cuda"), ("cuda:0", "cpu"), ("cuda:bad", "cuda")):
        rejected("runtime_device_" + actual + "_" + requested,
                 lambda: require(runtime_device_matches(actual, requested), "Wrong runtime device"), rejected_cases)
    observed_devices = []
    for model in ("VLR_T2_LAST", "VLR_T4_LAST"):
        path = HERE.parent / "Research_Vast_Mechanism_Execution_2026-10-28/Source_Grid" / model / "Complete.json"
        value = read(path)
        assert value["status"] == "three_source_inputs_full_grid_verified"
        assert value["independent_count_checks"] == 33 and runtime_device_matches(value["device"], "cuda")
        observed_devices.append({"model": model, "device": value["device"], "proof_sha256": sha(path)})
    assert Path(output_name).name == output_name and output_name.endswith(".json")
    metadata = calibration_provenance(plan)
    assert metadata["destination_calibration_labels_used"] is True
    assert metadata["destination_calibration_labels_used_by_language"] == {"vi": True, "en": False}
    for name, regimes in (
        ("missing_vi_regime", {"en": "source_corpus_development"}),
        ("unknown_en_regime", {"vi": "target_corpus_development", "en": "unspecified"}),
        ("misreported_vi_regime", {"vi": "source_corpus_development", "en": "source_corpus_development"}),
    ):
        rejected(name, lambda: calibration_provenance({**plan, "calibration_regime": regimes}), rejected_cases)
    checks = 0
    for model in plan["models"]:
        task_name = "source_integration_" + model
        task, rows = selected_rows(plan, task_name)
        directory = HERE / "Source_Grid_CPU_01/Proofs" / model
        complete = read(directory / "Complete.json")
        assert complete["plan_sha256"] == sha(HERE / "Plan.json")
        assert complete["checkpoint_sha256"] == registry[model]["checkpoint_sha256"]
        records = []
        for position, row in rows:
            name = "Probe_" + str(position) + ".json"
            assert sha(directory / name) == complete["probe_sha256"][name]
            probe = read(directory / name)
            records.append({"index": position, "clip_id": row["clip_id"], "source_id": row["source_id"],
                            "frames": row["frames"], "model": model, "split": "development",
                            "media_sha256": row["input"]["sha256"], "encoding": probe["encoding"],
                            "predictions": probe["predictions"]})
        selected = plan["candidate_order"]
        classes = registry[model]["classes"]
        checks += verify_records(rows, records, task, selected, classes)
        source_selection = choose([row for _, row in rows], records, selected)
        assert source_selection["clips"] == 3 and source_selection["independent_counts_checked"] == 33
        expected_errors = {candidate: sum(row["predictions"][candidate]["counts"]["errors"] for row in records)
                           for candidate in selected}
        assert source_selection["decoder"] == min(selected, key=lambda name: expected_errors[name])
        for change in ("index", "model", "media", "pixels", "vocabulary", "grid", "blank", "partial"):
            broken = deepcopy(records)
            wrong_classes = classes
            if change == "index":
                broken[0]["index"] += 1
            elif change == "model":
                broken[0]["model"] = "wrong"
            elif change == "media":
                broken[0]["media_sha256"] = "0" * 64
            elif change == "pixels":
                broken[0]["encoding"]["pixel_sha256"] = "not_a_hash"
            elif change == "vocabulary":
                wrong_classes = 5049 if classes == 2002 else 2002
            elif change == "grid":
                broken[0]["predictions"].pop(selected[0])
            elif change == "blank":
                broken[0]["predictions"][selected[0]]["tokens"] = [0]
            else:
                broken.pop()
            rejected(model + "_" + change,
                     lambda: verify_records(rows, broken, task, selected, wrong_classes), rejected_cases)
        contaminated = [dict(row, split="assessment") for _, row in rows]
        rejected(model + "_target_calibration", lambda: choose(contaminated, records, selected), rejected_cases)
        admission = {"status": "bounded_mechanism_execution_admitted", "device": "cpu", "models": plan["models"],
                     "tasks": [task_name], "stages": ["development"], "purpose": "source_integration"}
        check_scope(admission, task_name, task, "cpu", plan)
        admission["purpose"] = "full_evaluation"
        rejected(model + "_integration_as_full", lambda: check_scope(admission, task_name, task, "cpu", plan),
                 rejected_cases)
    groups = {}
    for name, task in plan["tasks"].items():
        _, rows = task_rows(plan, name)
        groups.setdefault((task["model"], task["stage"]), []).extend(position for position, _ in rows)
        admission = {"status": "bounded_mechanism_execution_admitted", "device": "cuda", "models": plan["models"],
                     "tasks": [name], "stages": [task["stage"]], "purpose": "full_evaluation"}
        check_scope(admission, name, task, "cuda", plan)
        rejected(name + "_wrong_device", lambda: check_scope(admission, name, task, "cpu", plan), rejected_cases)
    for (model, stage), indices in groups.items():
        size = len(cohort(plan, registry[model]["language"], stage))
        assert sorted(indices) == list(range(size))
    budget = {"admitted_unix": 1000, "maximum_seconds": 3600, "deadline_unix": 4600,
              "device": "cuda", "threads": 2, "execution_location": "server", "paid_allocation": True,
              "usd_per_hour": .77, "maximum_cost_usd": 1, "observed_balance_usd": 10,
              "reserved_balance_usd": 2, "provider_guard_verified": True, "backup_restore_verified": True}
    validate_budget(budget, 1001)
    for field, wrong in (("maximum_seconds", 0), ("usd_per_hour", 1), ("maximum_cost_usd", .01),
                         ("observed_balance_usd", 1), ("provider_guard_verified", False),
                         ("deadline_unix", math.inf), ("threads", 4)):
        invalid = {**budget, field: wrong}
        rejected("budget_" + field, lambda: validate_budget(invalid, 1001), rejected_cases)
    rejected("expired_budget", lambda: validate_budget(budget, 4600), rejected_cases)
    original = {"tokens": [1], "text": "a", "score": -1.0}
    assert same_backend_replay(original, original) == 0
    rejected("replay_score", lambda: same_backend_replay(original, {**original, "score": -1.01}), rejected_cases)
    rejected("replay_tokens", lambda: same_backend_replay(original, {**original, "tokens": [2]}), rejected_cases)
    scripts = ("execution_common.py", "run_full_task.py", "verify_full_records.py", "verify_full_task.py",
               "collect_full_results.py", "select_full_decoder.py", "test_execution.py")
    result = {"utc": utc(), "status": "passed", "actual_source_prediction_checks": checks,
              "models": plan["models"], "complete_partitions_checked": len(groups),
              "rejected_cases": rejected_cases, "new_inference": False, "assessment_predictions_read": False,
              "GPU_admission_granted": False, "plan_sha256": sha(HERE / "Plan.json"),
              "scripts": {name: sha(HERE / name) for name in scripts},
              "observed_gpu_device_records": observed_devices,
              "calibration_provenance_checked": metadata}
    save(HERE / output_name, result)
    print({"status": "passed", "source_records_checked": checks, "malformed_cases_rejected": len(rejected_cases)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Execution_Verification_01.json")
    main(parser.parse_args().output)
