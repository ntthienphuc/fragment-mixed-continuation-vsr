"""Explicit bounded execution admission, separate from immutable scientific design."""

import os
from pathlib import Path
import time

from common import HERE, ROOT, read, sha


def checked(arguments):
    lock = read(HERE / "Code_Lock.json")
    assert lock["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    for name, expected in lock["code_sha256"].items():
        assert sha(HERE / name) == expected, "Execution code differs: " + name
    for name, expected in lock["external_code_sha256"].items():
        assert sha(ROOT / name) == expected, "External execution code differs: " + name
    admission = read(arguments.admission)
    assert admission["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    assert admission["code_lock_sha256"] == sha(HERE / "Code_Lock.json")
    assert admission["model"] == arguments.model and admission["mode"] == arguments.mode
    assert admission["device"] == arguments.device
    assert arguments.task in admission["tasks"] and admission["readout"] == arguments.readout
    assert admission["registry_sha256"] == sha(arguments.binding / "model_registry.json")
    assert time.time() < admission["deadline_unix"]
    assert 0 < admission["maximum_task_seconds"] <= 14400
    registry = read(arguments.binding / "model_registry.json")
    entry = registry["models"][arguments.model]
    if arguments.mode == "fixture":
        assert admission["status"] == "cpu_worker_integration_admitted"
        assert arguments.device == "cpu" and arguments.model == "WW_RECOVERY_RUNTIME_CHECK"
        assert arguments.task == "source_fixture" and arguments.readout == "common"
        assert entry["selected_training_step"] == 3000
        assert admission["maximum_task_seconds"] <= 900
        return admission
    assert arguments.model in read(HERE / "Plan.json")["models"]
    binding = read(arguments.binding / "Binding.json")
    assert binding["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    assert binding["registry_sha256"] == admission["registry_sha256"]
    assert entry["selected_training_step"] == 6000 and entry["training_identity"]["selection"] == "fixed_final_step"
    assert binding["checkpoint_sha256"] == entry["checkpoint_sha256"]
    if arguments.device == "cuda":
        assert str(admission["cuda_visible_device"]) == os.environ.get("CUDA_VISIBLE_DEVICES")
    if arguments.mode == "probe":
        assert admission["status"] == "bounded_final_model_resource_probe_admitted"
        assert admission["maximum_task_seconds"] <= 1800
        return admission
    assert arguments.mode == "population" and admission["status"] == "matched_population_assessment_admitted"
    probe = read(Path(admission["resource_probe"]) / "Verification.json")
    assert probe["mode"] == "probe" and probe["device"] == arguments.device
    assert probe["model"] == arguments.model and probe["readout"] == arguments.readout
    assert probe["stage"] == arguments.task.rsplit("_", 1)[0]
    assert probe["registry_sha256"] == admission["registry_sha256"]
    assert probe["code_lock_sha256"] == sha(HERE / "Code_Lock.json")
    assert probe["status"] == "all_records_recounted_and_fresh_replay_passed"
    return admission
