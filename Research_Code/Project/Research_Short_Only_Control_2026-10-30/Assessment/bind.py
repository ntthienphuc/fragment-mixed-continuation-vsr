"""Bind only the verified, locally backed-up fixed-endpoint short-only model."""

from pathlib import Path
import sys

from common import HERE, ROOT, read, save, sha, utc


def main():
    phase = HERE.parent
    proof = read(phase / "Training_Verification.json")
    package = read(phase / "Package/Complete.json")
    backup = read(phase / "Training_Backup_Verified.json")
    assert proof["status"] == "native_training_records_checkpoint_and_strict_load_verified"
    assert proof["updates"] == 6000 and proof["exposure"]["presentations"] == 48000
    assert proof["native_strict_load"] and proof["batchnorm_buffers_unchanged"]
    assert backup["status"] == "all_training_files_locally_hash_verified"
    for name, item in package["files"].items():
        assert backup["files"][name]["sha256"] == item["sha256"]
        assert backup["files"][name]["bytes"] == item["bytes"]
    checkpoint = Path(package["files"]["model"]["path"])
    assert checkpoint.is_relative_to(ROOT) and sha(checkpoint) == package["files"]["model"]["sha256"]
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["identity"]["objective"] == "matched_short_only" and payload["step"] == 6000
    assert payload["source_checkpoint_sha256"] == proof["checkpoint_sha256"]
    assert payload["format"] == "lossless-model-state-v1" and not payload["optimizer_included"]
    assert len(payload["model"]) == 767
    registry = read(ROOT / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")
    entry = dict(registry["models"]["VLR_T0_LAST"])
    entry.update(checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(checkpoint),
                 training_identity=payload["identity"], selected_training_step=6000, native_graph="T0")
    assert entry["base_checkpoint_sha256"] == payload["identity"]["initializer_sha256"]
    registry["models"] = {"MATCHED_SO17": entry}
    registry["status"] = "short_only_fixed_endpoint_bound_probes_pending"
    output = HERE / "Binding"
    output.mkdir(exist_ok=False)
    save(output / "model_registry.json", registry)
    save(output / "Binding.json", {"utc": utc(), "model": "MATCHED_SO17",
         "training_identity_sha256": proof["identity_sha256"], "assessment_plan_sha256": sha(HERE / "Plan.json"),
         "registry_sha256": sha(output / "model_registry.json"), "checkpoint_sha256": sha(checkpoint),
         "local_backup_receipt_sha256": sha(phase / "Training_Backup_Verified.json"),
         "verification_sha256": sha(phase / "Training_Verification.json"), "binding_code_sha256": sha(__file__)})


if __name__ == "__main__":
    main()
