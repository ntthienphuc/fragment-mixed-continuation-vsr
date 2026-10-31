"""Actual complete cohorts, malformed selection cases and independent CTC/count checks."""

from copy import deepcopy
from itertools import product
import sys

from mechanism_common import ENGINE, HERE, ROOT, read, save, sha, utc
from experiment_core import candidates, check_prediction, cohort, decode, independent_counts, independent_greedy
from experiment_core import task_rows, validate_choices


def main():
    import numpy as np

    output = HERE / "Core_Verification_01.json"
    assert not output.exists()
    plan = read(HERE / "Plan.json")
    registry = read(HERE / "model_registry.json")["models"]
    groups = {}
    for name, task in plan["tasks"].items():
        actual, records = task_rows(plan, name)
        assert actual == task
        groups.setdefault((task["model"], task["stage"]), []).extend(position for position, row in records)
    for (model, stage), positions in groups.items():
        rows = cohort(plan, registry[model]["language"], stage)
        assert sorted(positions) == list(range(len(rows)))
    selection = {"status": "all_mechanism_development_decoders_locked_before_assessment",
                 "test_predictions_read": False, "models": {}}
    for index, model in enumerate(plan["models"]):
        entry = plan["cohorts"][registry[model]["language"]]["development"]
        selection["models"][model] = {"clips": entry["clips"], "reference_units": entry["reference_units"],
             "independent_counts_checked": entry["clips"] * len(plan["candidate_order"]),
             "decoder": "beam_c0.1" if index % 2 == 0 else "greedy_b-1.0"}
    validate_choices(plan, selection)
    for task in plan["tasks"].values():
        chosen = candidates(plan, task, selection if task["stage"] == "assessment" else None)
        if task["stage"] == "development":
            assert chosen == plan["candidate_order"]
        else:
            selected = selection["models"][task["model"]]["decoder"]
            assert chosen == ([selected] if selected == plan["anchor"] else [selected, plan["anchor"]])
    rejected = []
    for change in ("missing_model", "partial_population", "wrong_denominator", "partial_grid", "test_tuning"):
        broken = deepcopy(selection)
        model = plan["models"][0]
        if change == "missing_model":
            del broken["models"][model]
        elif change == "partial_population":
            broken["models"][model]["clips"] -= 1
        elif change == "wrong_denominator":
            broken["models"][model]["reference_units"] -= 1
        elif change == "partial_grid":
            broken["models"][model]["independent_counts_checked"] -= 1
        else:
            broken["test_predictions_read"] = True
        try:
            validate_choices(plan, broken)
        except AssertionError:
            rejected.append(change)
        else:
            raise AssertionError(change + " accepted")
    class FixtureRuntime:
        blank_id = 0
        eos_id = 4

        @staticmethod
        def text(tokens):
            return " ".join(str(token) for token in tokens)

    runtime = FixtureRuntime()
    cases = 0
    for labels in product(range(5), repeat=4):
        logp = np.full((4, 5), -4, dtype=np.float32)
        for frame, label in enumerate(labels):
            logp[frame, label] = 0
        for candidate in [item for item in plan["candidate_order"] if item.startswith("greedy")]:
            result = decode(runtime, {"logp": logp}, candidate, plan)
            bias = float(candidate.removeprefix("greedy_b"))
            assert result["tokens"] == independent_greedy(logp, bias, 0, 4)
            cases += 1
    sys.path.insert(0, str(ENGINE))
    from confirmation_common import edit_counts

    strings = ["", "a", "a b", "a a b", "b a", "a b c", "HÒA  bình", "ho\u0300a bình"]
    for reference, hypothesis in product(strings, repeat=2):
        assert independent_counts(reference, hypothesis) == edit_counts(reference, hypothesis)
    source = {"reference": "a b", "frames": 14}
    prediction = {"tokens": [1, 2], "text": "a b", "counts": edit_counts("a b", "a b"),
                  "seconds": .1, "search_calls": 0}
    check_prediction(source, prediction, "greedy_b0.0", 2002)
    for change in ("token_zero", "token_eos", "noninteger_token", "wrong_count", "nonfinite_seconds"):
        broken = deepcopy(prediction)
        if change == "token_zero":
            broken["tokens"] = [0]
        elif change == "token_eos":
            broken["tokens"] = [2001]
        elif change == "noninteger_token":
            broken["tokens"] = [True]
        elif change == "wrong_count":
            broken["counts"]["errors"] = 1
        else:
            broken["seconds"] = float("nan")
        try:
            check_prediction(source, broken, "greedy_b0.0", 2002)
        except AssertionError:
            rejected.append(change)
        else:
            raise AssertionError(change + " accepted")
    result = {"utc": utc(), "status": "passed", "complete_model_stage_groups": len(groups),
              "independent_greedy_cases": cases, "independent_edit_cases": len(strings)**2,
              "rejected_cases": rejected, "actual_model_inference": False,
              "plan_sha256": sha(HERE / "Plan.json"),
              "scripts": {name: sha(HERE / name) for name in ("experiment_core.py", "test_experiment_core.py")}}
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
