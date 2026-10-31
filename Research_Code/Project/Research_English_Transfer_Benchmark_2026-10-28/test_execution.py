"""Exercise actual data coverage, selection, scoring and fail-closed boundaries on CPU."""

import argparse
from collections import Counter
import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from admission import validate_budget, validate_selection
from analyze_complete import paired_bootstrap
from english_common import ENGINE, HERE, ROOT, lines, read, require, save, sha, utc
from select_decoder import choose
from task_design import build_tasks, candidates_for
from verify_task import counts, verify_records

sys.path.insert(0, str(ENGINE))
from confirmation_common import edit_counts


def rejected(action):
    try:
        action()
    except (ValueError, KeyError, AssertionError):
        return 1
    raise AssertionError("Invalid input was accepted")


def main(output):
    require(not output.exists(), "Test output already exists")
    output.mkdir(parents=True, exist_ok=False)
    plan = read(HERE / "Plan.json")
    tasks = build_tasks(plan)
    require(len(tasks) == 64, "Wrong total task count")
    populations = {stage: lines(HERE / (stage.capitalize() + ".jsonl"))
                   for stage in ("development", "assessment")}
    for stage, population in populations.items():
        for model in plan["models"]:
            indices = [index for task in tasks.values() if task["stage"] == stage and task["model"] == model
                       for index in task["positions"]]
            require(Counter(indices) == Counter(range(len(population))), "Coverage mismatch")
    candidates = plan["candidate_order"]
    population = populations["development"]
    choices = {}
    fixtures = {}
    for model_index, model in enumerate(plan["models"]):
        winner = candidates[model_index]
        fixtures[model] = []
        for row in population:
            predictions = {}
            for candidate in candidates:
                text = row["reference"] if candidate == winner else row["reference"] + " __synthetic_extra__"
                predictions[candidate] = {"text": text, "counts": edit_counts(row["reference"], text)}
            fixtures[model].append({"clip_id": row["clip_id"], "source_id": row["source_id"],
                                    "model": model, "split": "development", "predictions": predictions})
        choices[model] = choose(population, fixtures[model], candidates)
        require(choices[model]["decoder"] == winner, "Wrong synthetic decoder choice")
        require(choices[model]["reference_units"] == 7866, "Wrong denominator")
        require(choices[model]["independent_counts_checked"] == 11902, "Incomplete selection check")
    selection = {"status": "all_source_development_decoders_locked_before_test", "models": choices,
                 "test_predictions_read": False}
    validate_selection(selection, plan)
    for model in plan["models"]:
        expected = list(dict.fromkeys([choices[model]["decoder"], "beam_c0.1"]))
        require(candidates_for("assessment", model, plan, selection) == expected, "Anchor reuse mismatch")
    blocked = rejected(lambda: candidates_for("assessment", plan["models"][0], plan, None))
    bad = copy.deepcopy(selection)
    bad["test_predictions_read"] = True
    blocked += rejected(lambda: validate_selection(bad, plan))
    bad = copy.deepcopy(selection)
    bad["models"].pop(plan["models"][-1])
    blocked += rejected(lambda: validate_selection(bad, plan))
    bad = copy.deepcopy(selection)
    bad["models"][plan["models"][0]]["clips"] -= 1
    blocked += rejected(lambda: validate_selection(bad, plan))
    model = plan["models"][0]
    rows = population[:2]
    good = copy.deepcopy(fixtures[model][:2])
    for candidate in candidates:
        good[0]["predictions"][candidate] = copy.deepcopy(good[0]["predictions"][candidates[0]])
        good[1]["predictions"][candidate] = copy.deepcopy(good[1]["predictions"][candidates[0]])
    require(choose(rows, good, candidates)["decoder"] == candidates[0], "Tie order changed")
    require(choose(rows, good, list(reversed(candidates)))["decoder"] == candidates[-1], "Tie rule is not explicit")
    bad = copy.deepcopy(good)
    bad[0]["split"] = "assessment"
    blocked += rejected(lambda: choose(rows, bad, candidates))
    bad = copy.deepcopy(good)
    bad[1] = copy.deepcopy(bad[0])
    blocked += rejected(lambda: choose(rows, bad, candidates))
    bad = copy.deepcopy(good)
    bad[0]["predictions"][candidates[0]]["counts"]["errors"] += 1
    blocked += rejected(lambda: choose(rows, bad, candidates))
    baseline = {"admitted_unix": 100, "deadline_unix": 3700, "maximum_seconds": 3600,
                "execution_location": "local", "paid_allocation": False}
    validate_budget(baseline, 101)
    blocked += rejected(lambda: validate_budget(baseline, 3700))
    for field, value in (("maximum_seconds", float("nan")), ("deadline_unix", 3800), ("paid_allocation", True)):
        blocked += rejected(lambda field=field, value=value: validate_budget({**baseline, field: value}, 101))
    cloud = {**baseline, "execution_location": "server", "paid_allocation": True, "usd_per_hour": .6,
             "maximum_cost_usd": .6, "observed_balance_usd": 5, "reserved_balance_usd": 1,
             "provider_guard_verified": True, "backup_restore_verified": True}
    validate_budget(cloud, 101)
    for field, value in (("usd_per_hour", 1), ("reserved_balance_usd", 0), ("observed_balance_usd", .5),
                         ("maximum_cost_usd", .5), ("provider_guard_verified", False),
                         ("observed_balance_usd", float("inf"))):
        blocked += rejected(lambda field=field, value=value: validate_budget({**cloud, field: value}, 101))
    source_index = lines(HERE / "Source_Caches" / (model + ".jsonl"))
    original = population[0]
    decoder = candidates[0]
    predicted = {"index": 0, "model": model, "split": "development", "clip_id": original["clip_id"],
                 "source_id": original["source_id"], "frames": original["frames"],
                 "media_sha256": original["input"]["sha256"],
                 "encoding": {**source_index[0]["encoding"], "state_sha256": source_index[0]["state_sha256"]},
                 "predictions": {decoder: {"text": "", "tokens": [], "counts": counts(original["reference"], ""),
                                           "cap": original["frames"],
                                           "trace": {"beam_size": 40, "penalty": 0, "ctc_weight": .1}}}}
    task = {"model": model, "stage": "development"}
    require(verify_records([(0, original)], [predicted], task, [decoder], source_index) == 1, "Verifier fixture failed")
    for field, value in (("media_sha256", "0" * 64), ("model", "EN_T1_LAST"), ("index", 1)):
        bad = {**predicted, field: value}
        blocked += rejected(lambda bad=bad: verify_records([(0, original)], [bad], task, [decoder], source_index))
    bad = copy.deepcopy(predicted)
    bad["encoding"]["pixel_sha256"] = "0" * 64
    blocked += rejected(lambda: verify_records([(0, original)], [bad], task, [decoder], source_index))
    words = np.array([2, 3, 5], dtype=np.int64)
    pairs = {"primary": ([1, 2, 4], [2, 1, 2]), "anchor": ([2, 1, 2], [1, 2, 4])}
    samples = paired_bootstrap(pairs, words, 31, 20261028)
    generator = np.random.default_rng(20261028)
    expected = []
    differences = [-1, 1, 2]
    for _ in range(31):
        indices = generator.integers(0, 3, size=3)
        expected.append(100 * sum(differences[index] for index in indices) / sum(int(words[index]) for index in indices))
    require(np.allclose(samples["primary"], expected, rtol=0, atol=1e-12), "Bootstrap scalar replay differs")
    require(np.allclose(samples["anchor"], -samples["primary"], rtol=0, atol=1e-12), "Readouts use different draws")
    from local_resources import stop_owned

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(.15)
    require(process.poll() is None, "Owned process was not live")
    stop_owned(process)
    require(process.poll() is not None, "Owned process survived termination")
    save(output / "Verification.json", {"utc": utc(), "status": "cpu_execution_checks_passed",
                                         "tasks_coverage_checked": 64, "source_prediction_counts_checked": 47608,
                                         "invalid_cases_rejected": blocked, "scalar_bootstrap_checks": 31,
                                         "real_owned_child_termination": True, "gpu_inference_performed": False,
                                         "plan_sha256": sha(HERE / "Plan.json"),
                                         "code_sha256": {path.name: sha(path) for path in sorted(HERE.glob("*.py"))}})
    print({"status": "passed", "coverage": 64, "counts": 47608, "rejected": blocked, "bootstrap": 31})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
