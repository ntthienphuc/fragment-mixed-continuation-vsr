"""Verify archived seed-29 identities and current checkpoint bytes without inference."""

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "Research_Seed_Replication_2026-10-15"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def structure(state):
    return {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in state.items()}


def main():
    torch.set_num_threads(1)
    output = HERE / "Input_Check_01"
    output.mkdir(exist_ok=False)
    preservation_path = SOURCE / "Preservation_vlr_29.json"
    record = read(preservation_path)
    local_record_path = SOURCE / "Local_Preservation_vlr_29.json"
    local_record = read(local_record_path)
    assert record["status"] == local_record["status"] == "passed"
    assert record["seed"] == local_record["seed"] == 29
    assert len(record["completed"]) == 4
    expected_pairs = {(method, variant) for method in ("T0", "T1") for variant in ("best", "last")}
    assert {(row["method"], row["variant"]) for row in record["completed"]} == expected_pairs
    seed17_path = ROOT / "Research_Transfer_Initialization_2026-10-27/model_registry.json"
    seed17 = read(seed17_path)["models"]
    source_model_path = ROOT / seed17["VLR_T0_LAST"]["checkpoint"]
    source_model = torch.load(source_model_path, map_location="cpu", weights_only=True, mmap=True)
    expected_structure = structure(source_model["model"])
    assert len(expected_structure) == 767
    del source_model
    gc.collect()
    results = []
    for row in record["completed"]:
        model_path = SOURCE / "Model_States" / row["run"] / (row["variant"] + ".pt")
        actual_hash = digest(model_path)
        assert actual_hash == row["sha256"], str(model_path)
        assert model_path.stat().st_size == row["bytes"]
        local_proof = [item for item in local_record["models"]
                       if item["run"] == row["run"] and item["variant"] == row["variant"]]
        assert len(local_proof) == 1 and local_proof[0]["sha256"] == actual_hash
        payload = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
        assert payload["identity"] == row["identity"]
        assert payload["step"] == row["step"]
        assert payload["source_checkpoint_sha256"] == row["original_sha256"]
        assert payload["model_tensor_sha256"] == row["model_tensor_sha256"]
        assert structure(payload["model"]) == expected_structure
        assert len(payload["model"]) == row["tensor_count"]
        identity = dict(payload["identity"])
        assert identity.pop("seed") == 29
        peer = dict(seed17["VLR_" + row["method"] + "_LAST"]["training_identity"])
        assert peer.pop("seed") == 17
        assert identity == peer, "Training identity differs beyond the training seed"
        completed_path = SOURCE / "runs" / row["run"] / "completed.json"
        completed = read(completed_path)
        assert completed["status"] == "completed" and completed["updates"] == 6000
        assert completed["identity"] == payload["identity"]
        assert completed["overlap"] == {"clip_id": 0, "source_id": 0, "video_sha256": 0}
        if row["variant"] == "last":
            assert row["step"] == 6000
        item = {"method": row["method"], "variant": row["variant"], "step": row["step"],
                "checkpoint": model_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": actual_hash,
                "original_sha256": row["original_sha256"], "tensors": len(payload["model"]),
                "tensor_structure_matches_seed17": True, "training_identity_matches_except_seed": True,
                "completed_record_sha256": digest(completed_path),
                "historical_tensor_equality_verified": row["tensor_equality_verified"],
                "original_tensor_equality_recomputed_now": False}
        results.append(item)
        print(json.dumps({"checked": len(results), "method": row["method"], "variant": row["variant"]}),
              flush=True)
        del payload
        gc.collect()
    save(output / "Models.json", results)
    save(output / "Summary.json", {
        "status": "four_archived_checkpoints_currently_verified_no_inference",
        "utc": datetime.now(timezone.utc).isoformat(), "models": len(results), "seed": 29,
        "primary_pair": {"T0": 6000, "T1": 6000},
        "source_selected_pair": {"T0": 5000, "T1": 5500},
        "preservation_sha256": digest(preservation_path), "local_preservation_sha256": digest(local_record_path),
        "seed17_registry_sha256": digest(seed17_path), "models_sha256": digest(output / "Models.json"),
        "protocol_sha256": digest(HERE / "Protocol.md"), "code_sha256": digest(Path(__file__)),
        "gpu_job_started": False, "destination_predictions_opened": False,
        "new_training_required_for_this_seed": False, "independent_new_data_replication": False,
        "current_running_matrix_changed": False,
    })
    print(json.dumps({"status": "passed", "verified_models": len(results), "gpu_inference": False}), flush=True)


if __name__ == "__main__":
    main()
