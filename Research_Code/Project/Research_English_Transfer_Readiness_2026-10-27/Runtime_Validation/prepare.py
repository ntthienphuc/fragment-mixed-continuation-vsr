"""Freeze source-development runtime probes before loading any English recognizer."""

import json

from common import ENGINE, HERE, INPUTS, PHASE, ROOT, read, save, sha, utc


def main():
    proof_path = PHASE / "Strict_Input_Check_01/Verification.json"
    proof = read(proof_path)
    assert proof["status"] == "passed" and proof["model_inference"] is False
    assert proof["populations"]["Development"]["clips"] == 1082
    assert proof["populations"]["Assessment"]["clips"] == 2854
    assert proof["adapter_sha256"] == sha(PHASE / "strict_inputs.py")
    registry = read(INPUTS / "model_registry.json")
    models = ["EN_B0", "EN_T0_LAST", "EN_T1_LAST", "EN_B1"]
    assert set(registry["models"]) == set(models)
    source = ROOT / "Research_Phase_05_2026-10-13"
    paths = set(HERE.glob("*.py"))
    paths.update((source / "vendor/auto_avsr").rglob("*.py"))
    paths.update((source / "phase5").rglob("*.py"))
    paths.update(ENGINE / name for name in ("native_runtime.py", "confirmation_common.py", "data_io.py"))
    paths.update(ROOT / registry[name] for name in
                 ("factory_source", "safe_search_source", "horizon_source", "alignment_source"))
    paths.update([INPUTS / "model_registry.json", INPUTS / "Development.jsonl",
                  PHASE / "strict_inputs.py", proof_path])
    for entry in registry["models"].values():
        for field in ("checkpoint", "base_checkpoint", "tokenizer", "units", "config"):
            if field in entry:
                path = ROOT / entry[field]
                assert sha(path) == entry[field + "_sha256"], field
                paths.add(path)
    population = [json.loads(line) for line in
                  (INPUTS / "Development.jsonl").read_text(encoding="utf-8").splitlines()]
    ordered = sorted(population, key=lambda row: (row["frames"], row["clip_id"]))
    probes = [{"clip_id": ordered[index]["clip_id"], "frames": ordered[index]["frames"]}
              for index in (0, len(ordered) // 2, len(ordered) - 1)]
    value = {
        "status": "local_runtime_preflight_only", "utc": utc(), "models": models,
        "probes": probes, "probe_selection": "shortest, median and longest source-development duration",
        "ctc_weights": [0.1, 0.5, 0.9, 1.0], "beam_size": 40, "penalty": 0,
        "maximum_seconds": 2700, "maximum_concurrent_models": 1,
        "minimum_available_ram_gib": 3, "minimum_free_gpu_mib": 3000,
        "test_inference_authorized": False, "new_training_authorized": False, "paid_allocation": False,
        "score_tolerance": 1e-5, "criterion": "equal token IDs and text for native/bounded full-cap decoding",
        "sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)},
    }
    save(HERE / "Lock.json", value)
    print({"status": value["status"], "files": len(paths), "probes": probes}, flush=True)


if __name__ == "__main__":
    main()
