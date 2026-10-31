"""Use real source outputs to test rejection of altered scientific and resource inputs."""

from copy import deepcopy
import math

from execution_common import VALIDATION_TASK, check_scope, selected_rows, validate_budget
from experiment_core import candidates, independent_counts
from mechanism_common import HERE, read, save, sha, utc
from select_full_decoder import choose
from verify_full_records import same_backend_replay, verify_records


def rejected(action):
    try:
        action()
    except (ValueError, AssertionError, KeyError, TypeError):
        return True
    raise AssertionError("Malformed evidence was accepted")


def fixture():
    plan = read(HERE / "Plan.json")
    task, rows = selected_rows(plan, VALIDATION_TASK)
    folder = HERE / "Source_Grid_CPU_01/Proofs/EN_F0_LAST"
    receipt = read(folder / "Complete.json")
    assert receipt["plan_sha256"] == sha(HERE / "Plan.json")
    records = []
    for position, row in rows:
        path = folder / ("Probe_" + str(position) + ".json")
        assert sha(path) == receipt["probe_sha256"][path.name]
        original = read(path)
        records.append({"index": position, "clip_id": row["clip_id"], "source_id": row["source_id"],
                        "frames": row["frames"], "model": task["model"], "split": task["stage"],
                        "media_sha256": row["input"]["sha256"], "encoding": original["encoding"],
                        "predictions": original["predictions"]})
    return plan, task, rows, records


def main():
    plan, task, rows, records = fixture()
    order = candidates(plan, task)
    assert verify_records(rows, records, task, order) == 33
    malformed = 0
    changes = [
        ("index", -1), ("model", "EN_T1_LAST"), ("split", "assessment"),
        ("clip_id", "wrong_clip"), ("source_id", "wrong_source"), ("frames", 999),
        ("media_sha256", "0" * 64),
    ]
    for field, value in changes:
        bad = deepcopy(records)
        bad[0][field] = value
        rejected(lambda: verify_records(rows, bad, task, order))
        malformed += 1
    for field, value in (("model_id", "EN_T0_LAST"), ("precision", "float16"), ("color_space", "rgb"),
                         ("transform", "different"), ("frames", 1), ("state_frames", 2), ("pixel_sha256", "")):
        bad = deepcopy(records)
        bad[0]["encoding"][field] = value
        rejected(lambda: verify_records(rows, bad, task, order))
        malformed += 1
    for field, value in (("tokens", [0]), ("tokens", [5048]), ("seconds", float("nan")),
                         ("seconds", -1), ("cap", 1), ("score", float("inf"))):
        bad = deepcopy(records)
        bad[0]["predictions"]["beam_c0.1"][field] = value
        rejected(lambda: verify_records(rows, bad, task, order))
        malformed += 1
    for field, value in (("beam_size", 1), ("penalty", 1), ("ctc_weight", .5), ("implementation", "other")):
        bad = deepcopy(records)
        bad[0]["predictions"]["beam_c0.1"]["trace"][field] = value
        rejected(lambda: verify_records(rows, bad, task, order))
        malformed += 1
    bad = deepcopy(records)
    bad[0]["predictions"]["beam_c0.1"]["counts"]["errors"] += 1
    rejected(lambda: verify_records(rows, bad, task, order))
    bad = deepcopy(records)
    del bad[0]["predictions"][order[-1]]
    rejected(lambda: verify_records(rows, bad, task, order))
    rejected(lambda: verify_records(rows, records[:-1], task, order))
    rejected(lambda: verify_records(rows, list(reversed(records)), task, order))
    malformed += 4

    population = [row for position, row in rows]
    choice = choose(population, records, order)
    assert choice["clips"] == 3 and choice["independent_counts_checked"] == 33
    ties = deepcopy(records)
    for predicted, row in zip(ties, population):
        for result in predicted["predictions"].values():
            result["text"] = ""
            result["counts"] = independent_counts(row["reference"], "")
    assert choose(population, ties, order)["decoder"] == order[0]
    rejected(lambda: choose(population, records[:-1], order))
    rejected(lambda: choose(population, [records[0]] * 3, order))
    rejected(lambda: choose([{**row, "split": "assessment"} for row in population], records, order))
    rejected(lambda: choose(population, records, order + [order[0]]))
    malformed += 4

    budget = {"admitted_unix": 100.0, "maximum_seconds": 1800.0, "deadline_unix": 1900.0,
              "execution_location": "local", "paid_allocation": False, "device": "cpu", "threads": 2}
    validate_budget(budget, 101)
    rejected(lambda: validate_budget(budget, 99))
    rejected(lambda: validate_budget(budget, 1900))
    for field, value in (("deadline_unix", 2000.0), ("maximum_seconds", float("inf")),
                         ("threads", 8), ("paid_allocation", True), ("device", "other")):
        rejected(lambda field=field, value=value: validate_budget({**budget, field: value}, 101))
    malformed += 7
    paid = {**budget, "execution_location": "server", "paid_allocation": True,
            "usd_per_hour": .83, "maximum_cost_usd": .5, "observed_balance_usd": 6,
            "reserved_balance_usd": 2.5, "provider_guard_verified": True, "backup_restore_verified": True}
    validate_budget(paid, 101)
    for field, value in (("usd_per_hour", 1.29), ("reserved_balance_usd", 0), ("maximum_cost_usd", .1),
                         ("observed_balance_usd", 2), ("provider_guard_verified", False)):
        rejected(lambda field=field, value=value: validate_budget({**paid, field: value}, 101))
    malformed += 5
    scope = {**budget, "status": "bounded_F0_execution_admitted", "models": ["EN_F0_LAST"],
             "tasks": [VALIDATION_TASK], "stages": ["development"], "purpose": "source_integration"}
    check_scope(scope, VALIDATION_TASK, task, "cpu")
    for field, value in (("purpose", "full_evaluation"), ("stages", ["assessment"]),
                         ("models", ["EN_T0_LAST"]), ("device", "cuda")):
        rejected(lambda field=field, value=value: check_scope({**scope, field: value}, VALIDATION_TASK, task, "cpu"))
    malformed += 4
    result = records[-1]["predictions"]["beam_c0.1"]
    assert same_backend_replay(result, deepcopy(result)) == 0
    for field, value in (("tokens", [1]), ("text", "wrong text"), ("score", result["score"] + .01)):
        rejected(lambda field=field, value=value: same_backend_replay(result, {**result, field: value}))
    malformed += 3
    names = ("execution_common.py", "run_full_task.py", "verify_full_records.py", "verify_full_task.py",
             "collect_full_results.py", "select_full_decoder.py", "test_full_execution.py")
    receipt = {"utc": utc(), "status": "passed", "source_prediction_records": 3,
               "real_prediction_count_checks": 33, "malformed_cases_rejected": malformed,
               "selection_tie_policy_checked": True, "runtime_model_inference_performed": False,
               "plan_sha256": sha(HERE / "Plan.json"), "scripts": {name: sha(HERE / name) for name in names}}
    save(HERE / "Execution_Verification_01.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
