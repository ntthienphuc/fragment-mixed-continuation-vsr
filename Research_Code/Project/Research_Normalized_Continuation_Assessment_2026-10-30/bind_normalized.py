"""Bind a completed, independently verified and backed-up Linux arm to the frozen assessment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, read, save, sha, utc

ALIASES = {"full": "NORMALIZED_FC17", "fixed_time": "NORMALIZED_FT17"}


def validate(arm, identity, step, verification, backup, lock):
    assert arm in ALIASES and identity == lock["identities"][arm]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert digest == lock["identity_sha256"][arm] == verification["identity_sha256"]
    assert step == 6000 and identity["selection"] == "fixed_final_step"
    assert identity["objective"] == "matched_normalized_" + arm and identity["native_graph"] == "T0"
    assert verification["status"] == "normalized_training_records_checkpoint_and_strict_load_verified"
    assert verification["arm"] == arm and verification["native_strict_load"]
    assert verification["batchnorm_buffers_unchanged"] and verification["source_checkpoint_stable"]
    assert verification["records"]["training"]["updates"] == 6000
    assert verification["records"]["training"]["exposure"]["presentations"] == 48000
    proof = backup["proof"]
    assert proof["status"] == "terminal_training_files_packaged" and proof["arm"] == arm
    assert proof["identity_sha256"] == digest and proof["updates"] == 6000
    assert proof["training_endpoint_reached"] and proof["independent_training_audit_passed"]
    assert not proof["independent_training_audit_required"]
    assert verification["checkpoint_sha256"] == proof["files"]["checkpoint"]["sha256"]
    assert backup["verified_members"] > 0
    for name in ("model", "checkpoint", "receipts"):
        assert backup["local_files"][name]["sha256"] == proof["files"][name]["sha256"]
        assert backup["local_files"][name]["bytes"] == proof["files"][name]["bytes"]
    return digest


def main(arguments):
    root = arguments.root.resolve()
    checkpoint = arguments.checkpoint.resolve()
    assert checkpoint.is_relative_to(root)
    phase = root / HERE.name
    native = root / "Research_Vast_Source_2026-10-29/Normalized_Production_01"
    if not native.exists():
        native = root / "Normalized_Production_01"
    lock = read(native / "Training_Verification_01/Lock.json")
    proof = read(arguments.verified)
    backup = read(arguments.backup)
    assert sha(checkpoint) == backup["proof"]["files"]["model"]["sha256"]
    assert checkpoint.stat().st_size == backup["proof"]["files"]["model"]["bytes"]
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    digest = validate(arguments.arm, payload["identity"], payload["step"], proof, backup, lock)
    assert payload["format"] == "lossless-model-state-v1" and not payload["optimizer_included"]
    assert payload["source_checkpoint_sha256"] == proof["checkpoint_sha256"]
    assert payload["model_tensor_sha256"] == backup["proof"]["files"]["model"]["model_tensor_sha256"]
    assert len(payload["model"]) == 767
    registry = read(root / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")
    entry = dict(registry["models"]["VLR_T0_LAST"])
    entry.update({"checkpoint": checkpoint.relative_to(root).as_posix(), "checkpoint_sha256": sha(checkpoint),
                  "training_identity": payload["identity"], "selected_training_step": 6000, "native_graph": "T0"})
    assert entry["base_checkpoint_sha256"] == payload["identity"]["initializer_sha256"]
    alias = ALIASES[arguments.arm]
    registry["models"] = {alias: entry}
    registry["status"] = "native_final_model_bound_actual_resource_probe_pending"
    arguments.output.mkdir(parents=True, exist_ok=False)
    save(arguments.output / "model_registry.json", registry)
    save(arguments.output / "Binding.json", {"utc": utc(), "model": alias, "arm": arguments.arm,
         "training_identity_sha256": digest, "assessment_plan_sha256": sha(phase / "Plan.json"),
         "registry_sha256": sha(arguments.output / "model_registry.json"),
         "verification_sha256": sha(arguments.verified), "local_backup_receipt_sha256": sha(arguments.backup),
         "checkpoint_sha256": entry["checkpoint_sha256"], "binding_code_sha256": sha(__file__),
         "cross_runtime_substitution": False, "recognition_run_admitted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=tuple(ALIASES), required=True)
    for name in ("checkpoint", "verified", "backup", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
