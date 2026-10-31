"""Freeze the second-seed final pair without admitting or launching inference."""

from copy import deepcopy
import json

from common import COHORT, HERE, PRIOR, ROOT, STORE, lines, read, save, sha, utc


def main():
    proof = read(HERE / "Input_Check_01/Summary.json")
    assert proof["status"] == "four_archived_checkpoints_currently_verified_no_inference"
    assert proof["models_sha256"] == sha(HERE / "Input_Check_01/Models.json")
    assert proof["protocol_sha256"] == sha(HERE / "Protocol.md")
    records = read(HERE / "Input_Check_01/Models.json")
    preservation = read(ROOT / "Research_Seed_Replication_2026-10-15/Preservation_vlr_29.json")
    source = read(PRIOR / "Plan.json")
    original_registry = read(PRIOR / "model_registry.json")
    registry = deepcopy(original_registry)
    registry["status"] = "seed29_final_pair_prepared_not_admitted"
    registry["models"] = {}
    registry["pending_models"] = []
    models = []
    for method in ("T0", "T1"):
        name = "VLR_" + method + "_29_LAST"
        record = next(row for row in records if row["method"] == method and row["variant"] == "last")
        original = next(row for row in preservation["completed"]
                        if row["method"] == method and row["variant"] == "last")
        assert sha(ROOT / record["checkpoint"]) == record["checkpoint_sha256"] == original["sha256"]
        entry = deepcopy(original_registry["models"]["VLR_" + method + "_LAST"])
        entry.update(checkpoint=record["checkpoint"], checkpoint_sha256=record["checkpoint_sha256"],
                     training_identity=original["identity"], selected_training_step=6000)
        registry["models"][name] = entry
        models.append(name)
    registry_text = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    assert len(registry_text.encode()) < 10000
    with (HERE / "model_registry.json").open("x", encoding="utf-8") as stream:
        stream.write(registry_text)
    files = [HERE / name for name in ("common.py", "runtime_core.py", "encode.py", "verify.py",
                                      "prepare_inference.py", "Protocol.md", "Inference_Design.md",
                                      "model_registry.json", "Input_Check_01/Models.json",
                                      "Input_Check_01/Summary.json")]
    rows = lines(COHORT)
    assert len(rows) == 531 and sha(COHORT) == source["cohort_sha256"]
    plan = {
        "status": "prepared_before_seed29_destination_inference", "utc": utc(), "models": models,
        "clips": 531, "sources": 40, "reference_words": sum(len(row["reference"].split()) for row in rows),
        "cohort_sha256": sha(COHORT), "parent_initialization_plan_sha256": sha(PRIOR / "Plan.json"),
        "frozen_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        "runtime_origin_sha256": sha(PRIOR / "runtime_core.py"), "store": str(STORE),
        "probe_positions": source["probe_positions"], "prior_cache_index_sha256": source["prior_cache_index_sha256"],
        "candidate_order": source["candidate_order"], "blank_biases": source["blank_biases"],
        "beam_weights": source["beam_weights"], "source_folds": source["source_folds"],
        "expected_final_prediction_count": 11682, "new_training": False,
        "gpu_admitted": False, "independent_new_data": False, "novelty_established": False,
        "primary_pair": "T0 versus T1 at fixed final step 6000, training seed 29",
    }
    assert plan["reference_words"] == 13887
    plan_text = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    assert len(registry_text.encode()) + len(plan_text.encode()) < 15000
    with (HERE / "Plan.json").open("x", encoding="utf-8") as stream:
        stream.write(plan_text)
    save(HERE / "Preparation_Receipt.json", {
        "status": "prepared_not_admitted", "utc": utc(), "plan_sha256": sha(HERE / "Plan.json"),
        "registry_sha256": sha(HERE / "model_registry.json"), "models": models,
        "checkpoint_hashes_checked": 2, "total_primary_predictions_required": 11682,
        "active_initialization_matrix_changed": False, "new_gpu_inference": False,
    })
    print({"status": "prepared_not_admitted", "models": models, "required_predictions": 11682}, flush=True)


if __name__ == "__main__":
    main()
