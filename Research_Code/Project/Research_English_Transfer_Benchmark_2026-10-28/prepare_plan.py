"""Bind every model and clip before any benchmark recognition or selection."""

from pathlib import Path

from english_common import ANCHOR, HERE, READY, ROOT, exact_copy, lines, read, require, save, sha, utc


def main():
    require(not (HERE / "Plan.json").exists(), "Plan already sealed")
    models = ["EN_B0", "EN_T0_LAST", "EN_T1_LAST", "EN_B1"]
    source = READY / "Transfer_Inputs_01"
    registry = read(source / "model_registry.json")["models"]
    require(set(registry) == set(models), "Model set changed")
    for model in models:
        entry = registry[model]
        require(entry["classes"] == 5049 and entry["language"] == "en", "Wrong language head")
        require(sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"], "Checkpoint changed")
    for model in models[1:3]:
        require(registry[model]["selected_training_step"] == 6000, "Wrong adaptation step")
        require(registry[model]["training_identity"]["seed"] == 17, "Wrong training seed")
        require(registry[model]["base_checkpoint_sha256"] == registry["EN_B0"]["checkpoint_sha256"],
                "Shared initializer mismatch")
    population = {stage: lines(source / (stage + ".jsonl")) for stage in ("Development", "Assessment")}
    expected = {"Development": (1082, 7866), "Assessment": (2854, 48037)}
    for stage, cohort in population.items():
        require(len(cohort) == expected[stage][0], "Wrong population size")
        require(len({row["clip_id"] for row in cohort}) == len(cohort), "Duplicate clips")
        require(all(row["split"] == stage.lower() and row["language"] == "en" for row in cohort), "Wrong split")
        require(sum(len(row["reference"].split()) for row in cohort) == expected[stage][1], "Word count changed")
    require(all(row["source_id"] is None for row in population["Assessment"]), "Unexpected source mapping")
    encoding = READY / "Development_Encoding"
    encoding_receipt = read(encoding / "Verification_01.json")
    require(encoding_receipt["packets"] == 4328 and encoding_receipt["test_clips_used"] is False,
            "Source feature verification incomplete")
    require(read(READY / "Runtime_Validation/Verification_01.json")["native_bounded_pairs"] == 48,
            "English runtime not verified")
    require(read(ANCHOR / "Plan.json")["model_order"]["en"] == models, "Anchor model set changed")
    for filename in ("model_registry.json", "Development.jsonl", "Assessment.jsonl"):
        exact_copy(source / filename, HERE / filename)
    caches = {}
    locations = {"status": "local_locations_require_asset_verification", "states": {}}
    source_files = [HERE / "Protocol.md", Path(__file__), HERE / "english_common.py",
                    HERE / "model_registry.json", HERE / "Development.jsonl", HERE / "Assessment.jsonl",
                    encoding / "Verification_01.json", encoding / "Lock.json",
                    READY / "Runtime_Validation/Verification_01.json", READY / "Runtime_Validation/Lock.json",
                    READY / "Strict_Input_Check_01/Verification.json", READY / "strict_inputs.py",
                    ANCHOR / "Plan.json", ANCHOR / "Protocol.md", ANCHOR / "Verification.json"]
    for model in models:
        proof_path = encoding / "Local_01" / model / "Complete.json"
        proof = read(proof_path)
        require(sha(proof_path) == encoding_receipt["model_caches"][model]["proof_sha256"], "Cache proof changed")
        archive = Path(proof["archive"])
        require(sha(archive) == proof["archive_sha256"], "Source cache changed")
        index_source = encoding / "Local_01" / model / "Index.jsonl"
        index_target = HERE / "Source_Caches" / (model + ".jsonl")
        exact_copy(index_source, index_target)
        require(sha(index_target) == proof["index_sha256"], "Source index changed")
        caches[model] = {"archive_sha256": proof["archive_sha256"], "index_sha256": proof["index_sha256"],
                         "index": index_target.relative_to(ROOT).as_posix(), "clips": proof["clips"],
                         "model_id": model, "checkpoint_sha256": registry[model]["checkpoint_sha256"]}
        locations["states"][model] = str(archive)
        source_files.extend([proof_path, index_target])
    wild_paths = [Path(row["input"]["path"]) for row in population["Assessment"]]
    require(len({path.parent for path in wild_paths}) == 1, "Unexpected cache layout")
    require(len({path.name for path in wild_paths}) == len(wild_paths), "Duplicate WildVSR packets")
    locations["wild_root"] = str(wild_paths[0].parent)
    save(HERE / "Source_Caches.json", caches)
    save(HERE / "Local_Locations.json", locations)
    source_files.append(HERE / "Source_Caches.json")
    candidates = ["beam_c0.1", "beam_c0.5", "beam_c0.9", "beam_c1.0", "greedy_b0.0", "greedy_b-0.5",
                  "greedy_b0.5", "greedy_b-1.0", "greedy_b1.0", "greedy_b-2.0", "greedy_b2.0"]
    plan = {"utc": utc(), "status": "complete_english_benchmark_prepared_before_recognition",
            "models": models, "matched_models": models[:3], "practical_reference": "EN_B1",
            "development_clips": 1082, "assessment_clips": 2854,
            "development_reference_words": 7866, "assessment_reference_words": 48037,
            "candidate_order": candidates, "anchor": "beam_c0.1", "beam_size": 40, "penalty": 0,
            "cap": "complete_encoded_frame_count", "precision": "float32",
            "normalization": "NFC_lower_whitespace", "bootstrap_draws": 10000, "bootstrap_seed": 20261028,
            "uncertainty_unit": "released clip; original source grouping unavailable",
            "development_predictions": 47608, "primary_assessment_predictions": 11416,
            "maximum_assessment_predictions_with_anchor": 22832,
            "assessment_predictions_read": False, "new_training_admitted": False, "gpu_work_admitted": False,
            "frozen_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in source_files}}
    save(HERE / "Plan.json", plan)
    print({"status": plan["status"], "models": models, "plan_sha256": sha(HERE / "Plan.json")})


if __name__ == "__main__":
    main()
