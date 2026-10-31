"""Freeze the complete benchmark population without reading recognition outcomes."""

import argparse
import json
from pathlib import Path

from common import ENGINE, HERE, PARENT, READINESS, ROOT, SEED29, lines, read, save, sha, utc


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def compact(row, label_hashes):
    keys = ("clip_id", "corpus", "language", "split", "source_id", "speaker_id", "frames",
            "fps", "reference", "metric", "input", "prior_exposure", "transcript_provenance")
    result = {key: row[key] for key in keys}
    result["metadata"] = {key: row["metadata"][key] for key in (
        "full_clip_eligible", "all_video_frames_decoded", "source_id_rule",
        "source_label_member", "native_reference_normalization")}
    label_sha = row["metadata"].get("source_label_sha256")
    if label_sha is None:
        label_sha = label_hashes[row["clip_id"]]
    assert isinstance(label_sha, str) and len(label_sha) == 64
    result["metadata"]["source_label_sha256"] = label_sha
    return result


def check_population(name, expected, label_hashes):
    path = ENGINE / "manifests" / ("VC_" + name + ".jsonl")
    assert sha(path) == expected["manifest_sha256"]
    rows = lines(path)
    assert len(rows) == expected["rows"]
    assert len({row["clip_id"] for row in rows}) == len(rows)
    assert len({row["source_id"] for row in rows}) == expected["sources"]
    assert sum(len(row["reference"].split()) for row in rows) == expected["reference_words"]
    for row in rows:
        assert row["language"] == "vi" and row["corpus"] == "VC" and row["split"] == name
        assert row["metadata"]["full_clip_eligible"] and row["metadata"]["all_video_frames_decoded"]
        assert row["frames"] >= 1 and row["fps"] == 25 and row["reference"].strip()
        assert sha(ROOT / row["input"]["path"]) == row["input"]["sha256"], row["clip_id"]
    return [compact(row, label_hashes) for row in rows]


def cache_map(model, phase, subset, output):
    store = Path(read(phase / "Plan.json")["store"])
    folder = store / "Cache" / model
    proof_path = folder / "Verification.json"
    proof = read(proof_path)
    assert proof["status"] == "passed" and proof["clips"] == len(subset) == 531
    index_path = folder / "Index.jsonl"
    assert proof["index_sha256"] == sha(index_path)
    records = lines(index_path)
    assert len(records) == 531
    for row, record in zip(subset, records):
        assert row["clip_id"] == record["clip_id"]
        assert row["source_id"] == record["source_id"] and row["frames"] == record["frames"]
        assert row["input"]["sha256"] == record["media_sha256"]
        assert sha(folder / record["file"]) == record["sha256"]
    value = {"source_phase": phase.relative_to(ROOT).as_posix(), "model": model,
             "cache_folder": str(folder), "cache_index_sha256": sha(index_path),
             "cache_verification_sha256": sha(proof_path),
             "requires_complete_matrix_receipt_before_prediction_reuse": True,
             "cache_files_currently_rehashed": len(records)}
    save(output / (model + "_Cache.json"), value)
    return value


def main(attempt):
    assert attempt.startswith("Inputs_") and attempt.removeprefix("Inputs_").isdigit()
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    readiness = read(READINESS / "Check_01/Summary.json")
    assert readiness["status"] == "inputs_verified_no_new_recognition"
    assert readiness["checks_sha256"] == sha(READINESS / "Check_01/Checks.json")
    label_hashes = {row["clip_id"]: row["label_sha256"]
                    for row in read(READINESS / "Check_01/Checks.json")["labels"]}
    development = check_population("development", readiness["development"], label_hashes)
    assessment = check_population("assessment", readiness["assessment"], label_hashes)
    for field in ("clip_id", "source_id"):
        assert not {row[field] for row in development} & {row[field] for row in assessment}
    assert not {row["input"]["sha256"] for row in development} & {
        row["input"]["sha256"] for row in assessment}
    subset_path = ROOT / "Research_Interval_Transfer_2026-10-27/Cohort.jsonl"
    subset = lines(subset_path)
    by_id = {row["clip_id"]: row for row in development}
    for row in subset:
        original = by_id[row["clip_id"]]
        for field in ("source_id", "frames", "fps", "reference", "input"):
            assert row[field] == original[field], (row["clip_id"], field)
    existing = {row["clip_id"] for row in subset}
    assert len(existing) == 531 and existing == {
        row["clip_id"] for row in development if row["frames"] >= 82}
    missing = [row for row in development if row["clip_id"] not in existing]
    assert len(missing) == 237
    for name, rows in (("Development", development), ("Development_New", missing), ("Assessment", assessment)):
        write_rows(HERE / (name + ".jsonl"), rows)
    registry = read(PARENT / "model_registry.json")
    registry["status"] = "full_benchmark_inputs_prepared_no_inference_admission"
    registry["models"].update(read(SEED29 / "model_registry.json")["models"])
    assert len(registry["models"]) == 5
    reuse = {}
    for model, entry in registry["models"].items():
        assert entry["classes"] == 2002 and entry["language"] == "vi"
        assert sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
        if model != "VLR_B0":
            assert entry["selected_training_step"] == 6000
        phase = SEED29 if "_29_" in model else PARENT
        reuse[model] = cache_map(model, phase, subset, output)
    native = read(ENGINE / "VC_Model_Registry.json")
    native_entry = native["models"]["VC_NATIVE"]
    assert sha(ROOT / native_entry["checkpoint"]) == native_entry["checkpoint_sha256"]
    for entry in [*registry["models"].values(), native_entry]:
        for field in ("tokenizer", "units"):
            assert sha(ROOT / entry[field]) == entry[field + "_sha256"]
    save(HERE / "model_registry.json", registry)
    save(HERE / "Native_Reference_Registry.json", native)
    lineage = read(HERE / "Lineage_01/Summary.json")
    assert lineage["status"] == "baseline_lineage_and_current_export_verified"
    assert lineage["current_tensor_equalities"] == 767
    frozen = read(SEED29 / "Plan.json")
    plan = {
        "status": "full_benchmark_prepared_before_expanded_inference", "utc": utc(),
        "matched_models": list(registry["models"]), "practical_reference": "VC_NATIVE",
        "development_clips": len(development), "assessment_clips": len(assessment),
        "development_sources": 43, "assessment_sources": 20,
        "development_reference_units": 15919, "assessment_reference_units": 30091,
        "metric": "WER using normalized Vietnamese whitespace tokens; often syllables",
        "candidate_order": frozen["candidate_order"], "beam_weights": frozen["beam_weights"],
        "blank_biases": frozen["blank_biases"], "beam_size": 40, "penalty": 0,
        "selection": "minimum pooled development edit errors; frozen order resolves ties",
        "test_selection_policy": "one locked development-selected decoder per model",
        "bootstrap": {"unit": "video_source", "draws": 10000, "seed": 20261027,
                      "same_source_draws_across_two_fixed_training_seeds": True},
        "duration_diagnostics": {"short_frames_lt": 82, "not_primary_endpoint": True},
        "development_reuse": reuse,
        "additional_matched_model_clip_encodings": 5 * len(missing),
        "native_reference_development_encodings": len(development),
        "new_development_predictions": (5 * len(missing) + len(development)) * 11,
        "planned_assessment_predictions": 6 * len(assessment),
        "mandatory_admission": "complete seed29 matrix review, runtime verification, bounded resource decision",
        "new_gpu_work_admitted": False, "test_outcomes_read": False,
        "independent_new_test": False, "new_human_review": False,
        "upstream_pretraining_sample_overlap_excluded": False,
        "frozen_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in (
            HERE / "Protocol.md", HERE / "Development.jsonl", HERE / "Development_New.jsonl",
            HERE / "Assessment.jsonl", HERE / "model_registry.json", HERE / "Native_Reference_Registry.json",
            HERE / "Lineage_01/Summary.json", READINESS / "Check_01/Summary.json",
            READINESS / "Check_01/Checks.json", subset_path,
            PARENT / "runtime_core.py", ENGINE / "vicocktail_runtime.py", ENGINE / "native_runtime.py",
            ENGINE / "data_io.py", ENGINE / "confirmation_common.py", HERE / "common.py", Path(__file__))},
    }
    save(HERE / "Plan.json", plan)
    save(output / "Summary.json", {
        "status": "complete_population_model_and_cache_inputs_verified", "utc": utc(),
        "plan_sha256": sha(HERE / "Plan.json"), "media_hashes": len(development) + len(assessment),
        "cache_hashes": 5 * len(subset), "matched_checkpoint_hashes": 5,
        "new_development_predictions": plan["new_development_predictions"],
        "planned_assessment_predictions": plan["planned_assessment_predictions"],
        "recognition_run": False, "test_predictions_read": False,
    })
    print({"status": "complete_inputs_verified", "new_dev_predictions": plan["new_development_predictions"],
           "planned_test_predictions": plan["planned_assessment_predictions"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    arguments = parser.parse_args()
    main(arguments.attempt)
