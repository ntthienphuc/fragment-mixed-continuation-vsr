"""Check saved baseline lineage and exact VLR initializer tensor export on CPU."""

import torch

from common import HERE, ROOT, save, sha, utc


def main():
    output = HERE / "Lineage_01"
    output.mkdir(exist_ok=False)
    original = ROOT / "experiments/_archive_sources/pavir_2026_08/03_models/w0/vlr_w0.ckpt"
    exported = ROOT / "Server_Rescue_2026-10-15/Assets/VLR/pretrained.pt"
    native = ROOT / "experiments/_archive_sources/pavir_2026_08/03_models/w0/vicocktail_w0.ckpt"
    expected_original = "57a6e8086c7b3a9452ab92f97e76265914ce670e920f6598b9e2f2a44cc79816"
    expected_export = "b38177ae98f1f3cbea1a4c183c2bca804cf0a576e99abd23aab85ae56f413280"
    expected_native = "6a41d0a3acf15075a0e2f75ead2215b5b8fcc38697885de8ee851eb3ff576c00"
    assert sha(original) == expected_original
    assert sha(exported) == expected_export
    assert sha(native) == expected_native
    payload = torch.load(original, map_location="cpu", weights_only=True, mmap=True)
    state = torch.load(exported, map_location="cpu", weights_only=True, mmap=True)
    native_payload = torch.load(native, map_location="cpu", weights_only=True, mmap=True)
    original_state = payload["state_dict"]
    mapped = {key.removeprefix("model."): value for key, value in original_state.items()
              if key.startswith("model.")}
    assert set(mapped) == set(state)
    torch.set_num_threads(2)
    for key in mapped:
        assert torch.equal(mapped[key], state[key]), key
    fields = (
        "exp_name", "root_dir", "train_file", "val_file", "test_file",
        "pretrained_model_path", "transfer_frontend", "transfer_encoder",
        "max_epochs", "ctc_weight", "precision", "ckpt_path",
    )
    result = {
        "status": "baseline_lineage_and_current_export_verified", "utc": utc(),
        "original_sha256": expected_original, "export_sha256": expected_export,
        "current_tensor_equalities": len(mapped), "native_sha256": expected_native,
        "vlr": {"epoch": payload["epoch"], "step": payload["global_step"],
                "declared_config": {key: payload["hyper_parameters"].get(key) for key in fields}},
        "native_vicocktail": {
            "epoch": native_payload["epoch"], "step": native_payload["global_step"],
            "declared_config": {key: native_payload["hyper_parameters"].get(key) for key in fields}},
        "historical_source_code_path": (
            "experiments/_archive_sources/lip_final_2026_07/02_PROJECT_MAIN/"
            "main_code_results_phase/scripts/train_auto_avsr_full.sh"),
        "metadata_is_declared_training_lineage_not_proof_of_all_pretraining_samples": True,
        "upstream_sample_overlap_independently_excluded": False,
        "new_inference": False,
    }
    result["historical_source_code_sha256"] = sha(ROOT / result["historical_source_code_path"])
    result["inspection_code_sha256"] = sha(__file__)
    save(output / "Summary.json", result)
    print({"status": result["status"], "tensor_equalities": len(mapped),
           "vlr_declared_initializer": result["vlr"]["declared_config"]["pretrained_model_path"]}, flush=True)


if __name__ == "__main__":
    main()
