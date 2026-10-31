"""Independent complete-cohort and identity checks before any new recognition."""

from copy import deepcopy
import json
from pathlib import Path
import zipfile

from mechanism_common import HERE, ROOT, lines, read, save, sha, utc, validate_inputs


def validate_structure(plan, cohorts, registry):
    expected = {"VLR_T2_LAST": "vi", "VLR_T4_LAST": "vi", "EN_T2_LAST": "en", "EN_T4_LAST": "en"}
    assert set(plan["models"]) == set(expected) == set(registry["models"])
    assert len(plan["tasks"]) == 64
    assert plan["all_model_selections_before_assessment"] is True
    assert plan["new_gpu_work_admitted"] is False
    assert plan["new_target_predictions_read"] is False
    assert plan["beam_size"] == 40 and plan["penalty"] == 0
    assert plan["precision"] == "float32" and plan["external_lm"] is False
    assert plan["anchor"] == "beam_c0.1" and plan["cap"] == "complete_encoded_frame_count"
    candidates = ["beam_c0.1", "beam_c0.5", "beam_c0.9", "beam_c1.0", "greedy_b0.0",
                  "greedy_b-0.5", "greedy_b0.5", "greedy_b-1.0", "greedy_b1.0", "greedy_b-2.0", "greedy_b2.0"]
    assert plan["candidate_order"] == candidates
    counts = {("vi", "development"): 768, ("vi", "assessment"): 1167,
              ("en", "development"): 1082, ("en", "assessment"): 2854}
    for (language, stage), rows in cohorts.items():
        assert len(rows) == counts[(language, stage)]
        assert len({row["clip_id"] for row in rows}) == len(rows)
        assert all(row["language"] == language and row["split"] == stage for row in rows)
        assert all(isinstance(row["frames"], int) and row["frames"] > 0 for row in rows)
        assert plan["cohorts"][language][stage]["clips"] == len(rows)
        assert plan["cohorts"][language][stage]["reference_units"] == sum(
            len(row["reference"].split()) for row in rows
        )
    for task in plan["tasks"].values():
        assert task["model"] in expected and task["language"] == expected[task["model"]]
        assert task["stage"] in {"development", "assessment"}
    coverage = {}
    for model, language in expected.items():
        entry = registry["models"][model]
        assert entry["language"] == language
        assert entry["classes"] == (2002 if language == "vi" else 5049)
        assert entry["training_identity"]["method"] == model.split("_")[1]
        assert entry["training_identity"]["seed"] == 17 and entry["selected_training_step"] == 6000
        for stage in ("development", "assessment"):
            tasks = [task for task in plan["tasks"].values()
                     if task["model"] == model and task["stage"] == stage]
            assert len(tasks) == 8
            positions = [position for task in tasks for position in task["positions"]]
            assert all(type(position) is int for position in positions)
            assert sorted(positions) == list(range(len(cohorts[(language, stage)]))), (model, stage)
            coverage[model + ":" + stage] = len(positions)
    return coverage


def main():
    output = HERE / "Preparation_Verification_02.json"
    assert not output.exists()
    plan = validate_inputs()
    registry = read(HERE / "model_registry.json")
    cohorts = {}
    for language, stages in plan["cohorts"].items():
        for stage, entry in stages.items():
            path = ROOT / entry["path"]
            assert sha(path) == entry["sha256"]
            cohorts[(language, stage)] = lines(path)
    coverage = validate_structure(plan, cohorts, registry)
    rejected = []
    for name in ("missing_position", "duplicate_position", "negative_position", "wrong_language",
                 "wrong_model", "changed_grid", "changed_penalty", "early_gpu_admission"):
        broken = deepcopy(plan)
        task = broken["tasks"]["development_VLR_T2_LAST_00"]
        if name == "missing_position":
            task["positions"].pop()
        elif name == "duplicate_position":
            task["positions"].append(task["positions"][0])
        elif name == "negative_position":
            task["positions"][0] = -1
        elif name == "wrong_language":
            task["language"] = "en"
        elif name == "wrong_model":
            task["model"] = "VLR_T0_LAST"
        elif name == "changed_grid":
            broken["candidate_order"] = broken["candidate_order"][:-1]
        elif name == "changed_penalty":
            broken["penalty"] = 1
        else:
            broken["new_gpu_work_admitted"] = True
        try:
            validate_structure(broken, cohorts, registry)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError("Malformed plan accepted: " + name)
    with zipfile.ZipFile(HERE / "Preparation_01/Rejected_Snapshot.zip") as archive:
        previous = json.loads(archive.read("Plan.json"))
    try:
        validate_structure(previous, cohorts, registry)
    except AssertionError:
        rejected.append("actual_initial_parent_delta_partition")
    else:
        raise AssertionError("Actual incomplete preparation was accepted")
    result = {"utc": utc(), "status": "passed", "coverage": coverage, "rejected_cases": rejected,
              "input_files_rehashed": len(plan["input_sha256"]), "plan_sha256": sha(HERE / "Plan.json"),
              "script_sha256": sha(__file__), "new_recognition_run": False, "GPU_admission_granted": False}
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
