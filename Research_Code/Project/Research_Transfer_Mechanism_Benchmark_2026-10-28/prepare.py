"""Freeze model identities, complete cohorts and matched decoder tasks."""

from copy import deepcopy

from mechanism_common import DIAGNOSTIC, EN, ENGINE, HERE, ROOT, VI, lines, logical, read, save, sha, utc


def main():
    assert not (HERE / "Plan.json").exists()
    inspected = read(DIAGNOSTIC / "Inspection_01/Registry.json")
    assert inspected["status"] == "passed" and len(inspected["models"]) == 4
    for name, digest in inspected["scripts"].items():
        assert sha(DIAGNOSTIC / name) == digest
    parents = {"vi": VI, "en": EN}
    registries = {language: read(folder / "model_registry.json") for language, folder in parents.items()}
    registry = {key: value for key, value in registries["en"].items() if key not in {"models", "pending_models"}}
    registry["status"] = "mechanism_extension_prepared_no_assessment_admission"
    registry["models"] = {}
    hashes = {}

    def bind(path):
        name = logical(path)
        if name not in hashes:
            hashes[name] = sha(path)
        return hashes[name]

    for field in ("factory_source", "safe_search_source", "horizon_source", "alignment_source"):
        assert registries["vi"][field] == registries["en"][field]
        bind(ROOT / registry[field])
    for path in (DIAGNOSTIC / "Inspection_01/Registry.json", DIAGNOSTIC / "Replay_01/Verification.json",
                 HERE / "Protocol.md", HERE / "prepare.py", HERE / "mechanism_common.py"):
        bind(path)
    original_vi = ROOT / "Research_Transfer_Initialization_2026-10-27/runtime_core.py"
    assert bind(original_vi) == bind(ENGINE / "native_runtime.py")
    bind(ENGINE / "data_io.py")
    bind(ROOT / "Research_English_Transfer_Readiness_2026-10-27/strict_inputs.py")
    bind(VI / "Pixels_01/Index.jsonl")
    bind(VI / "Pixels_01/Verification.json")
    tasks = {}
    cohorts = {}
    probes = {}
    parent_plans = {}
    for language, folder in parents.items():
        parent_plans[language] = read(folder / "Plan.json")
        execution = read(folder / "Execution_Lock.json")
        for name in ("Plan.json", "Execution_Lock.json", "model_registry.json"):
            bind(folder / name)
        for name, digest in execution["code_sha256"].items():
            assert bind(ROOT / name) == digest, name
        cohorts[language] = {}
        for stage in ("development", "assessment"):
            path = folder / (stage.title() + ".jsonl")
            records = lines(path)
            assert len(records) == parent_plans[language][stage + "_clips"]
            assert len({row["clip_id"] for row in records}) == len(records)
            cohorts[language][stage] = {"path": logical(path), "sha256": bind(path), "clips": len(records),
                                        "reference_units": sum(len(row["reference"].split()) for row in records)}
            if stage == "development":
                ordered = sorted(range(len(records)), key=lambda index: (records[index]["frames"],
                                                                          records[index]["clip_id"]))
                probes[language] = [ordered[index] for index in (0, len(ordered) // 2, len(ordered) - 1)]
    assert parent_plans["vi"]["candidate_order"] == parent_plans["en"]["candidate_order"]
    model_order = []
    for record in inspected["models"]:
        language = "vi" if record["dataset"] == "vlr" else "en"
        assert record["dataset"] in {"vlr", "lrs2"} and record["method"] in {"T2", "T4"}
        assert record["seed"] == 17 and record["step"] == 6000
        assert record["same_identity_as_T1_except_method"] and record["same_T1_parameter_schema"]
        prefix = "VLR" if language == "vi" else "EN"
        original = prefix + "_T1_LAST"
        model = prefix + "_" + record["method"] + "_LAST"
        assert model not in registry["models"]
        entry = deepcopy(registries[language]["models"][original])
        entry.update({"checkpoint": record["path"].replace("\\", "/"),
                      "checkpoint_sha256": record["sha256"], "training_identity": record["training_identity"],
                      "selected_training_step": 6000})
        assert entry["training_identity"]["method"] == record["method"]
        for field in ("checkpoint", "base_checkpoint", "config", "tokenizer", "units"):
            assert bind(ROOT / entry[field]) == entry[field + "_sha256"]
        registry["models"][model] = entry
        model_order.append(model)
        execution = read(parents[language] / "Execution_Lock.json")
        for name, task in execution["tasks"].items():
            if task["model"] != original:
                continue
            new_name = name.replace(original, model)
            tasks[new_name] = {**task, "model": model, "language": language, "parent_task": name}
    assert len(tasks) == 64
    save(HERE / "model_registry.json", registry)
    bind(HERE / "model_registry.json")
    plan = {"utc": utc(), "status": "complete_mechanism_extension_prepared_not_admitted",
            "models": model_order, "tasks": tasks, "cohorts": cohorts, "source_probe_positions": probes,
            "candidate_order": parent_plans["en"]["candidate_order"], "anchor": "beam_c0.1",
            "beam_size": 40, "penalty": 0, "precision": "float32", "external_lm": False,
            "cap": "complete_encoded_frame_count", "normalization": "NFC_lower_whitespace",
            "calibration_regime": {"vi": "target_corpus_development", "en": "source_corpus_development"},
            "all_model_selections_before_assessment": True, "new_training": False,
            "new_gpu_work_admitted": False, "new_target_predictions_read": False,
            "parent_vietnamese_result_already_known": True,
            "required_completed_parents": [VI.name, EN.name, "Research_Transfer_Decoder_Anchor_2026-10-28"],
            "input_sha256": hashes}
    save(HERE / "Plan.json", plan)
    print({"models": model_order, "tasks": len(tasks), "status": plan["status"]})


if __name__ == "__main__":
    main()
