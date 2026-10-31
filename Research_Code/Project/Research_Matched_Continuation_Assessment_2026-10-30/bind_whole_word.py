"""Bind only a verified final whole-word export; recovery checkpoints cannot enter assessment."""

import argparse
import hashlib
import json
from pathlib import Path

from common import HERE, ROOT, read, save, sha, utc


def validate_identity(identity, step, proof, plan):
    assert step == 6000
    assert proof["status"] == "complete_whole_word_training_independently_verified_and_losslessly_exported"
    assert proof["training"]["updates"] == 6000
    assert proof["training"]["exposure"]["presentations"] == 48000
    assert proof["native_strict_load"] and proof["source_checkpoint_stable"]
    assert proof["export"]["all_tensors_byte_identical"] and not proof["export"]["optimizer_included"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert digest == proof["identity_sha256"]
    assert digest == plan["reference_windows_training_identity_sha256"]["whole_word"]
    assert identity["native_graph"] == "T0" and identity["objective"] == "matched_whole_word"
    assert identity["selection"] == "fixed_final_step" and identity["steps"] == 6000
    return digest


def main(arguments):
    root = arguments.root.resolve()
    checkpoint = arguments.checkpoint.resolve()
    assert checkpoint.is_relative_to(root)
    proof = read(arguments.verified)
    backup = read(arguments.backup)
    assert backup["status"] == "complete_training_export_and_resume_state_backed_up"
    assert backup["all_updates_and_diagnostics_recounted_locally"]
    assert backup["model_sha256"] == proof["export"]["sha256"] == sha(checkpoint)
    assert backup["checkpoint_sha256"] == proof["checkpoint"]["sha256"]
    assert checkpoint.stat().st_size == proof["export"]["bytes"]
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    plan = read(HERE / "Plan.json")
    identity_hash = validate_identity(payload["identity"], payload["step"], proof, plan)
    assert payload["format"] == "lossless-model-state-v1" and not payload["optimizer_included"]
    assert payload["source_checkpoint_sha256"] == proof["checkpoint"]["sha256"]
    assert payload["model_tensor_sha256"] == proof["export"]["model_tensor_sha256"]
    assert len(payload["model"]) == 767
    registry = read(root / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")
    entry = dict(registry["models"]["VLR_T0_LAST"])
    entry.update({"checkpoint": checkpoint.relative_to(root).as_posix(), "checkpoint_sha256": sha(checkpoint),
                  "training_identity": payload["identity"], "selected_training_step": 6000, "native_graph": "T0"})
    assert entry["base_checkpoint_sha256"] == payload["identity"]["initializer_sha256"]
    registry["models"] = {"MATCHED_WW17": entry}
    registry["status"] = "final_checkpoint_bound_resource_probe_and_assessment_admission_pending"
    arguments.output.mkdir(parents=True, exist_ok=False)
    save(arguments.output / "model_registry.json", registry)
    save(arguments.output / "Binding.json", {"utc": utc(), "model": "MATCHED_WW17",
         "training_identity_sha256": identity_hash, "assessment_plan_sha256": sha(HERE / "Plan.json"),
         "registry_sha256": sha(arguments.output / "model_registry.json"),
         "verification_sha256": sha(arguments.verified), "local_backup_receipt_sha256": sha(arguments.backup),
         "checkpoint_sha256": entry["checkpoint_sha256"], "recognition_run_admitted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
