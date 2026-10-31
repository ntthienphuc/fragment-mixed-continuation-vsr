"""Real-evidence reuse, blocked CLI execution, and finite resource/split controls."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from admission import checked, validate_budget, validate_parent, validate_selection, verify_plan
from common import HERE, PARENT, lines, read, save, sha, utc
from reuse_development import prior_predictions
from runtime_bridge import claim_family, collapse, decode
from task_design import build_tasks, candidates_for, cohort_for
from task_queue import execute


def rejects(function):
    try:
        function()
    except (ValueError, RuntimeError, KeyError, AssertionError):
        return 1
    raise AssertionError("Unsafe input was accepted")


def main():
    output = HERE / "Execution_Tests_01"
    output.mkdir(exist_ok=False)
    plan = verify_plan()
    tasks = build_tasks(plan)
    assert len(tasks) == 96
    for stage in ("development", "assessment"):
        for model in [*plan["matched_models"], plan["practical_reference"]]:
            chosen = [task for task in tasks.values() if task["stage"] == stage and task["model"] == model]
            positions = [position for task in chosen for position in task["positions"]]
            assert sorted(positions) == list(range(len(cohort_for(stage, model))))
    assert sum(len(task["positions"]) for task in tasks.values() if task["stage"] == "development") == 1953
    assert sum(len(task["positions"]) for task in tasks.values() if task["stage"] == "assessment") == 7002
    rejected = rejects(lambda: checked("development", "VLR_B0"))
    process = subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "run_task.py"),
                              "development_VLR_B0_00", "--attempt", "Forbidden_01"],
                             capture_output=True, text=True, encoding="utf-8", timeout=30, cwd=HERE)
    assert process.returncode != 0 and "Execution code is not locked" in process.stderr
    assert not (HERE / "Runs/Forbidden_01").exists()
    (output / "Blocked_Worker.stderr.log").write_text(process.stderr, encoding="utf-8")
    rejected += 1
    now = time.time()
    admission = {"admitted_unix": now - 10, "maximum_execution_seconds": 3600,
                 "deadline_unix": now + 3590, "execution_location": "server", "usd_per_hour": 0.6,
                 "maximum_cost_usd": 0.6, "observed_balance_usd": 2, "reserved_balance_usd": 1,
                 "provider_guard_verified": True, "backup_restore_verified": True}
    validate_budget(admission, now)
    for field, wrong in (("deadline_unix", now + 7200), ("usd_per_hour", 1.1), ("maximum_cost_usd", 2),
                         ("reserved_balance_usd", 0), ("provider_guard_verified", False),
                         ("backup_restore_verified", False), ("deadline_unix", float("nan"))):
        rejected += rejects(lambda field=field, wrong=wrong: validate_budget({**admission, field: wrong}, now))
    rejected += rejects(lambda: validate_budget(admission, now + 3600))
    models = [*plan["matched_models"], plan["practical_reference"]]
    selection = {"status": "all_development_decoders_locked_before_test", "plan_sha256": "plan",
                 "assessment_manifest_sha256": "test", "test_predictions_read": False,
                 "models": {model: {"clips": 768, "reference_units": 15919,
                                     "independent_prediction_counts_checked": 8448,
                                     "decoder": plan["candidate_order"][0]} for model in models}}
    assert validate_selection(selection, models[0], plan, "plan", "test") == "beam_c0.1"
    contaminated = {**selection, "test_predictions_read": True}
    rejected += rejects(lambda: validate_selection(contaminated, models[0], plan, "plan", "test"))
    incomplete = copy.deepcopy(selection)
    incomplete["models"][models[0]]["clips"] = 531
    rejected += rejects(lambda: validate_selection(incomplete, models[0], plan, "plan", "test"))
    assert candidates_for("assessment", models[0], plan, selection) == ["beam_c0.1"]
    rejected += rejects(lambda: candidates_for("assessment", models[0], plan, None))
    claim_family("standard")
    rejected += rejects(lambda: claim_family("vicocktail"))
    actual_checks = 0
    for model in ("VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST"):
        records, _ = prior_predictions(model)
        assert len(records) == 531 and all(len(row["predictions"]) == 11 for row in records.values())
        actual_checks += len(records) * 11
    cache = Path(plan["development_reuse"]["VLR_B0"]["cache_folder"])
    index = lines(cache / "Index.jsonl")
    expected = {row["clip_id"]: row for row in lines(PARENT / "Greedy_01/VLR_B0/Predictions.jsonl")}
    greedy_checks = 0
    for row in index[::53]:
        with np.load(cache / row["file"], allow_pickle=False) as packet:
            logp = packet["full_logp"].copy()
        for bias in plan["blank_biases"]:
            scores = logp.copy()
            scores[:, 0] += bias
            tokens = collapse(scores.argmax(axis=1))
            assert tokens == expected[row["clip_id"]]["predictions"]["greedy_b" + str(bias)]["tokens"]
            greedy_checks += 1
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    killed = False
    began = time.monotonic()
    try:
        execute(command, output / "Deadline_Child", time.time() + 0.5)
    except TimeoutError:
        killed = True
    assert killed and time.monotonic() - began < 30
    save(output / "Verification.json", {
        "status": "passed", "utc": utc(), "task_count": len(tasks), "new_development_model_clips": 1953,
        "assessment_model_clips": 7002, "invalid_inputs_rejected": rejected,
        "historical_prediction_counts_checked": actual_checks, "real_cache_greedy_token_replays": greedy_checks,
        "blocked_worker_created_no_output": True, "owned_child_stopped_at_deadline": True,
        "new_model_inference_run": False, "native_runtime_preflight_pending": True,
        "source_sha256": {path.name: sha(path) for path in HERE.glob("*.py")},
    })
    print({"status": "passed", "invalid_rejected": rejected, "historical_counts": actual_checks,
           "greedy_replays": greedy_checks}, flush=True)


if __name__ == "__main__":
    main()
