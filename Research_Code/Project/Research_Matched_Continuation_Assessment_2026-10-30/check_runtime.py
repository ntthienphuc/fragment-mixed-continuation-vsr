"""Compare the new graph binding with the unchanged loader using a real recovery model, CPU only."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from common import HERE, ROOT, read, save, sha, utc, load_module
from runtime_bridge import create_runtime, decode


def main():
    began = time.monotonic()
    output = HERE / "Runtime_Check_01"
    output.mkdir(exist_ok=False)
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "device": "cpu",
                                "purpose": "runtime_equivalence_not_population_recognition"})
    private = ROOT.parent
    recovery = private / "Recovery_Snapshots/20261029T174340Z/last.pt"
    expected = "d758745a96cd04a5a45fadfe632a74238d1b36c6f10a4f3d399e8b5ba4741fc2"
    assert sha(recovery) == expected
    sys.path.insert(0, str(private / "WholeWord_Finalization_01"))
    from finalize import export_model, tensor_digest
    import torch

    torch.set_num_threads(2)
    payload = torch.load(recovery, map_location="cpu", weights_only=True)
    assert payload["step"] == 3000 and payload["identity"]["native_graph"] == "T0"
    assert "method" not in payload["identity"] and payload["identity"]["objective"] == "matched_whole_word"
    expected_identity = read(HERE / "Plan.json")["reference_windows_training_identity_sha256"]["whole_word"]
    assert hashlib.sha256(json.dumps(payload["identity"], sort_keys=True).encode()).hexdigest() == expected_identity
    model_path = output / "model.pt"
    export = export_model(payload, expected, model_path)
    registry = read(ROOT / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")
    entry = dict(registry["models"]["VLR_T0_LAST"])
    entry.update({"checkpoint": model_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": export["sha256"],
                  "native_graph": "T0", "training_identity": payload["identity"], "selected_training_step": 3000})
    registry["models"] = {"WW_RECOVERY_RUNTIME_CHECK": entry}
    registry["status"] = "recovery_model_runtime_check_only_not_final_assessment"
    save(output / "model_registry.json", registry)
    del payload
    gc.collect()
    runtime = create_runtime(output, "WW_RECOVERY_RUNTIME_CHECK", device="cpu")
    assert tensor_digest(runtime.recognizer.state_dict()) == export["model_tensor_sha256"]
    source_path = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Source_Retention_Runtime_01/source_loader.py"
    loader = load_module("matched_runtime_probe_source", source_path)
    with loader.open_source("vlr") as source:
        metadata, pixels = source.load(7617)
    choices = ["beam_c0.1", "greedy_b0.0"]
    encoded = runtime.encode(pixels)
    first = {candidate: decode(runtime, encoded, candidate, choices) for candidate in choices}
    states = encoded["states"].clone()
    logp = encoded["logp"].copy()
    identity = runtime.identity
    del encoded, runtime
    gc.collect()
    original = sys.modules["matched_original_runtime"]
    baseline = original.Runtime("WW_RECOVERY_RUNTIME_CHECK", device="cpu", threads=2, root=ROOT)
    assert baseline.recognizer.method == "B0"
    assert tensor_digest(baseline.recognizer.state_dict()) == export["model_tensor_sha256"]
    encoded = baseline.encode(pixels)
    assert torch.equal(encoded["states"], states)
    import numpy as np

    assert np.array_equal(encoded["logp"], logp)
    for candidate in choices:
        second = decode(baseline, encoded, candidate, choices)
        assert second["tokens"] == first[candidate]["tokens"] and second["text"] == first[candidate]["text"]
        if "score" in second:
            assert second["score"] == first[candidate]["score"]
    assert not torch.cuda.is_initialized()
    save(output / "Identity.json", identity)
    save(output / "Predictions.json", {"clip_id": metadata["clip_id"], "frames": len(pixels),
                                       "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                                       "predictions": first, "reference_provided_to_recognizer": False})
    save(output / "Complete.json", {"utc": utc(), "status": "cpu_real_recovery_runtime_equivalence_passed",
                                    "training_step": 3000, "model_export": export,
                                    "native_graph": "T0", "old_loader_graph": "B0",
                                    "model_tensors_identical": True, "encoded_values_exact": True,
                                    "ctc_values_exact": True, "decoded_tokens_text_and_score_exact": True,
                                    "source_position": 7617, "clips": 1, "cuda_initialized": False,
                                    "seconds": time.monotonic() - began, "final_assessment_admitted": False})
    print(json.dumps(read(output / "Complete.json")), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save(HERE / "Runtime_Check_01/Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise
