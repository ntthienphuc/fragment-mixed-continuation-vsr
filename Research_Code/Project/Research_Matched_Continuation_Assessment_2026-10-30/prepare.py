"""Freeze comparisons before any new matched checkpoint is scored."""

from common import HERE, ROOT, read, rows, save, sha, utc


def main():
    transfer = ROOT / "Research_Transfer_Full_Benchmark_2026-10-27"
    retention = ROOT / "Research_Source_Retention_2026-10-29"
    controls = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Augmentation_Controls_01"
    source = read(retention / "Plan.json")["cohorts"]["vlr"]
    source_tasks = read(retention / "Assessment_01/Plan.json")["tasks"]
    coverage = [position for index in range(16)
                for position in source_tasks[f"VLR_B0_{index:02}"]["positions"]]
    assert sorted(coverage) == list(range(11307))
    populations = {}
    for stage, filename, count, words, groups in (
            ("development", "Development.jsonl", 768, 15919, 43),
            ("assessment", "Assessment.jsonl", 1167, 30091, 20)):
        cohort = rows(transfer / filename)
        assert len(cohort) == len({item["clip_id"] for item in cohort}) == count
        assert sum(len(item["reference"].split()) for item in cohort) == words
        assert len({item["source_id"] for item in cohort}) == groups
        populations[stage] = {"manifest": (transfer / filename).relative_to(ROOT).as_posix(),
                              "sha256": sha(transfer / filename), "clips": count,
                              "reference_units": words, "source_groups": groups}
    controls_plan = read(controls / "Portable_Training_01/Production_01/Plan.json")
    upstream = read(transfer / "Plan.json")
    code_paths = ["Research_Transfer_Initialization_2026-10-27/runtime_core.py",
                  "Research_English_Transfer_Benchmark_2026-10-28/runtime_bridge.py",
                  "Research_English_Transfer_Benchmark_2026-10-28/english_common.py",
                  "Research_Phase_05_2026-10-13/phase5/model.py",
                  "Research_Server_Confirmation_2026-10-19/confirmation_common.py"]
    locks = [retention / "Plan.json", retention / "Assessment_01/Plan.json",
             transfer / "Plan.json", transfer / "Pixels_01/Verification.json",
             controls / "Portable_Training_01/Production_01/Plan.json"]
    plan = {
        "utc": utc(), "status": "prospective_matched_assessment_prepared_no_gpu_admission",
        "question": "Does fixed-time fragment mixing improve whole-utterance transfer beyond whole-word mixing?",
        "models": {"MATCHED_FC17": "full", "MATCHED_FT17": "fixed_time", "MATCHED_WW17": "whole_word"},
        "reference_windows_training_identity_sha256": controls_plan["identity_sha256"],
        "new_host_binding": "Require separately admitted execution identity; preserve numerical protocol and record runtime",
        "training_endpoint": {"updates": 6000, "presentations": 48000, "seed": 17,
                              "shared_fallbacks": 317, "selection": "fixed_final_step"},
        "runtime_source": code_paths[0],
        "runtime_code_sha256": {name: sha(ROOT / name) for name in code_paths},
        "upstream_lock_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in locks},
        "source": {"corpus": "VLR", "clips": source["clips"], "reference_units": source["reference_units"],
                   "frames": source["total_frames"], "source_groups": 61,
                   "assignment_plan": "Research_Source_Retention_2026-10-29/Assessment_01/Plan.json",
                   "assignment_tasks": [f"VLR_B0_{index:02}" for index in range(16)]},
        "destination": {"corpus": "ViCocktail", **populations,
                        "pixel_index": "Research_Transfer_Full_Benchmark_2026-10-27/Pixels_01/Index.jsonl",
                        "pixel_index_sha256": sha(transfer / "Pixels_01/Index.jsonl")},
        "primary_decoder": "beam_c0.1", "primary_decoder_selected_before_new_checkpoint_assessment": True,
        "secondary_candidate_order": upstream["candidate_order"],
        "secondary_selection": "Minimize complete destination-development errors; ties use frozen candidate order",
        "secondary_source_decoder": "Use the same selected destination-development decoder without source-test tuning",
        "beam_size": 40, "penalty": 0, "precision": "float32", "external_lm": False,
        "horizon": "full_encoded_length", "primary_common_decoder_must_not_be_replaced_by_selected_decoder": True,
        "report_each_corpus_separately": True, "bootstrap_source_groups": True, "bootstrap_resamples": 10000,
        "historical_FM_is_matched_fixed_time_arm": False, "historical_primary_decoder_reused_for_new_model": False,
        "cross_runtime_training_is_exact_hardware_match": False, "reused_assessment_is_new_blind_test": False,
        "per_run_device_probe_required": True, "final_checkpoint_and_backup_required": True,
        "recognition_run_admitted": False,
    }
    save(HERE / "Plan.json", plan)
    print({"source_clips": source["clips"], "destination_clips": populations["assessment"]["clips"],
           "models": list(plan["models"]), "primary_decoder": plan["primary_decoder"], "gpu_run": False})


if __name__ == "__main__":
    main()
