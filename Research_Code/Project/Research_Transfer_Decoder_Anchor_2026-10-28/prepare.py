"""Freeze a secondary decoder comparison before expanded assessment starts."""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VI = ROOT / "Research_Transfer_Full_Benchmark_2026-10-27"
EN = ROOT / "Research_English_Transfer_Readiness_2026-10-27"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def native_defaults(path):
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    functions = [item for item in parsed.body if isinstance(item, ast.FunctionDef)
                 and item.name == "get_beam_search_decoder"]
    assert len(functions) == 1
    arguments = functions[0].args
    defaults = {argument.arg: ast.literal_eval(value)
                for argument, value in zip(arguments.args[-len(arguments.defaults):], arguments.defaults)}
    assert defaults["ctc_weight"] == 0.1 and defaults["beam_size"] == 40
    assert defaults["penalty"] == 0 and defaults["lm_weight"] == 0
    return defaults


def shards(population):
    assignments = [[] for _ in range(8)]
    totals = [0] * 8
    for index in sorted(range(len(population)), key=lambda value: (-population[value]["frames"], value)):
        lane = min(range(8), key=lambda value: (totals[value], value))
        assignments[lane].append(index)
        totals[lane] += population[index]["frames"]
    return assignments


def main():
    assert not (HERE / "Plan.json").exists()
    factory = ROOT / "Research_Phase_05_2026-10-13/server_2026-10-14/upstream_decoder_reference.py"
    native_defaults(factory)
    english_receipt = EN / "Runtime_Validation/Verification_01.json"
    assert read(english_receipt)["native_bounded_pairs"] == 48
    assert read(english_receipt)["test_clips_used"] is False
    sys.path.insert(0, str(VI / "Server_Operations"))
    from cloud import remote
    configuration = read(VI / "Server_Operations/Server_Config.json")
    base = configuration["remote_base"]
    monitor = base + "/project/" + VI.name + "/Server_Operations/remote_monitor.py"
    live = json.loads(remote([base + "/venv/bin/python", "-B", monitor, "status"]))
    assert live["Pipeline_Status.json"]["stage"] == "development"
    assessment = [task for task in live["tasks"].values() if task["stage"] == "assessment"]
    assert len(assessment) == 48
    assert all(task["Start.json"] is None and task["clips_written"] == 0 for task in assessment)
    assert live["Failure.json"] is None and live["Lifecycle_Complete.json"] is None
    save(HERE / "Preassessment_Server_Status.json", live)
    manifests = {"vi": VI / "Assessment.jsonl", "en": EN / "Transfer_Inputs_01/Assessment.jsonl"}
    populations = {language: rows(path) for language, path in manifests.items()}
    assert len(populations["vi"]) == 1167 and len(populations["en"]) == 2854
    registries = {"vi": read(VI / "model_registry.json")["models"],
                  "en": read(EN / "Transfer_Inputs_01/model_registry.json")["models"]}
    registries["vi"].update(read(VI / "Native_Reference_Registry.json")["models"])
    models = {"vi": ["VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST", "VLR_T0_29_LAST", "VLR_T1_29_LAST",
                     "VC_NATIVE"], "en": ["EN_B0", "EN_T0_LAST", "EN_T1_LAST", "EN_B1"]}
    tasks = {}
    for language, population in populations.items():
        assert len({row["clip_id"] for row in population}) == len(population)
        assert all(row["language"] == language and row["frames"] > 0 for row in population)
        assert all(model in registries[language] for model in models[language])
        for model in models[language]:
            for number, positions in enumerate(shards(population)):
                name = f"anchor_{model}_{number:02d}"
                tasks[name] = {"language": language, "model": model, "positions": positions,
                               "clip_ids": [population[position]["clip_id"] for position in positions]}
        for model in models[language]:
            covered = [position for task in tasks.values() if task["model"] == model
                       for position in task["positions"]]
            assert Counter(covered) == Counter(range(len(population)))
    inputs = [HERE / "Protocol.md", Path(__file__), factory, english_receipt,
              EN / "Runtime_Validation/Lock.json", EN / "Transfer_Inputs_01/model_registry.json",
              VI / "model_registry.json", VI / "Native_Reference_Registry.json", VI / "Execution_Lock.json",
              VI / "Plan.json", *manifests.values(), HERE / "Preassessment_Server_Status.json"]
    plan = {"utc": datetime.now(timezone.utc).isoformat(), "status": "secondary_protocol_frozen_not_admitted",
            "decoder": {"name": "beam_c0.1", "ctc_weight": 0.1, "beam_size": 40, "penalty": 0,
                        "external_lm": False, "cap": "complete_encoded_frame_count", "precision": "float32"},
            "model_order": models, "tasks": tasks,
            "maximum_additional_predictions": {"vi": 7002, "en": 11416},
            "current_primary_changed": False, "assessment_predictions_read": False,
            "earlier_project_test_use": "ViCocktail test was used in earlier project phases",
            "source_uncertainty": {"vi": "20 video sources, paired bootstrap 10000 seed 20261027",
                                   "en": "source map unavailable; do not infer from numeric clip IDs"},
            "reuse_rule": "Exact decoder/input/model identity and complete independent verification only",
            "gpu_execution_admitted": False, "paid_allocation_admitted": False,
            "resource_requirement": "Separate finite admission after measured ETA; existing guard unchanged",
            "sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}}
    save(HERE / "Plan.json", plan)
    assert len(tasks) == 80
    verification = {"utc": datetime.now(timezone.utc).isoformat(), "status": "plan_coverage_and_inputs_verified",
                    "tasks": len(tasks), "predictions": sum(len(task["positions"]) for task in tasks.values()),
                    "model_counts": {key: len(value) for key, value in models.items()},
                    "assessment_tasks_not_started": len(assessment), "plan_sha256": sha(HERE / "Plan.json")}
    assert verification["predictions"] == 18418
    for name, digest in plan["sha256"].items():
        assert sha(ROOT / name) == digest
    save(HERE / "Verification.json", verification)
    backup_root = Path("C:/Users/thien/Lip_Research_Backups/Transfer_Full_Benchmark_20261027")
    archive_path = backup_root / ("Common_Decoder_Anchor_" + str(time.time_ns()) + ".zip")
    expected = {}
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(HERE.iterdir()):
            if path.is_file():
                expected[path.name] = sha(path)
                archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(expected)
        for name, digest in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    save(HERE / "Backup.json", {"status": "sha256_readback_passed", "path": str(archive_path),
                               "sha256": sha(archive_path), "files": expected})
    print(verification)


if __name__ == "__main__":
    main()
