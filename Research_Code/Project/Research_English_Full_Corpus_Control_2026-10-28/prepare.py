"""Freeze a separate complete-cohort F0 follow-up from existing source and state evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "Research_English_Transfer_Benchmark_2026-10-28"
MECHANISM = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    assert not (HERE / "Plan.json").exists()
    input_hashes = {}

    def bind(path):
        name = path.relative_to(ROOT).as_posix()
        if name not in input_hashes:
            input_hashes[name] = sha(path)
        return input_hashes[name]

    readiness_path = MECHANISM / "Augmentation_Controls_01/F0_Readiness.json"
    readiness = read(readiness_path)
    assert readiness["status"] == "file_hash_and_archived_training_identity_verified"
    entry = readiness["registry_entry"]
    assert entry["training_identity"]["cohort"] == "full"
    assert entry["training_identity"]["method"] == "T0"
    assert entry["selected_training_step"] == 6000 and entry["classes"] == 5049
    for field in ("checkpoint", "base_checkpoint", "tokenizer", "units", "config"):
        assert bind(ROOT / entry[field]) == entry[field + "_sha256"]
    original = read(PARENT / "model_registry.json")
    registry = {key: value for key, value in original.items() if key not in {"models", "pending_models"}}
    registry["status"] = "separate_full_corpus_control_prepared_not_admitted"
    registry["models"] = {"EN_F0_LAST": entry}
    save(HERE / "model_registry.json", registry)
    parent_plan = read(PARENT / "Plan.json")
    parent_lock = read(PARENT / "Execution_Lock.json")
    for name, digest in parent_lock["code_sha256"].items():
        assert bind(ROOT / name) == digest
    cohorts = {"en": {}}
    probe_positions = {}
    for stage in ("development", "assessment"):
        path = PARENT / (stage.title() + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == parent_plan[stage + "_clips"]
        cohorts["en"][stage] = {"path": path.relative_to(ROOT).as_posix(), "sha256": bind(path),
                                "clips": len(rows), "reference_units": sum(len(row["reference"].split()) for row in rows)}
        if stage == "development":
            ordered = sorted(range(len(rows)), key=lambda index: (rows[index]["frames"], rows[index]["clip_id"]))
            probe_positions["en"] = [ordered[index] for index in (0, len(rows) // 2, len(rows) - 1)]
    tasks = {}
    for name, task in parent_lock["tasks"].items():
        if task["model"] == "EN_T0_LAST":
            key = name.replace("EN_T0_LAST", "EN_F0_LAST")
            tasks[key] = {**deepcopy(task), "model": "EN_F0_LAST", "language": "en", "parent_task": name}
    assert len(tasks) == 16
    for stage in ("development", "assessment"):
        positions = [position for task in tasks.values() if task["stage"] == stage for position in task["positions"]]
        assert sorted(positions) == list(range(cohorts["en"][stage]["clips"]))
    copies = {}
    for name in ("mechanism_common.py", "runtime_bridge.py", "experiment_core.py",
                 "test_experiment_core.py", "source_grid_probe.py"):
        target = HERE / name
        assert not target.exists()
        shutil.copy2(MECHANISM / name, target)
        assert sha(target) == sha(MECHANISM / name)
        copies[name] = {"source": (MECHANISM / name).relative_to(ROOT).as_posix(), "sha256": bind(target)}
    save(HERE / "Copied_Source.json", copies)
    for path in (readiness_path, HERE / "prepare.py", HERE / "Protocol.md", HERE / "Copied_Source.json",
                 HERE / "model_registry.json", PARENT / "Plan.json", PARENT / "Execution_Lock.json"):
        bind(path)
    plan = {"utc": datetime.now(timezone.utc).isoformat(), "status": "full_corpus_control_prepared_not_admitted",
            "models": ["EN_F0_LAST"], "tasks": tasks, "cohorts": cohorts,
            "source_probe_positions": probe_positions, "candidate_order": parent_plan["candidate_order"],
            "anchor": parent_plan["anchor"], "beam_size": 40, "penalty": 0, "precision": "float32",
            "external_lm": False, "cap": "complete_encoded_frame_count", "normalization": "NFC_lower_whitespace",
            "calibration_regime": {"en": "source_corpus_development"},
            "all_model_selections_before_assessment": True, "new_training": False,
            "new_gpu_work_admitted": False, "new_target_predictions_read": False,
            "parent_completed_before_comparative_assessment": PARENT.name,
            "comparison_kind": "practical_full_training_split_reference_not_same_population_mechanism_control",
            "input_sha256": input_hashes}
    save(HERE / "Plan.json", plan)
    print({"model": "EN_F0_LAST", "tasks": len(tasks), "cohorts": cohorts,
           "plan_sha256": sha(HERE / "Plan.json"), "GPU_admission": False})


if __name__ == "__main__":
    main()
