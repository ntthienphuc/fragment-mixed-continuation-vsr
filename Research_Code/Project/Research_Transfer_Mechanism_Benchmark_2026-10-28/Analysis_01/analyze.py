"""Report both prespecified readouts only after every new and parent model is verified."""

import argparse
import csv
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent
ROOT = SCIENCE.parent
sys.path.insert(0, str(SCIENCE))

from gather import gather
from mechanism_common import read, save, sha, utc
from transfer_statistics import describe, require


def main(manifest, output):
    output.resolve().relative_to(HERE.resolve())
    require(not output.exists(), "Preserve previous analysis output")
    lock_path = HERE / "Lock.json"
    lock = read(lock_path)
    require(lock["status"] == "mechanism_analysis_frozen_before_new_assessment", "Analysis not frozen")
    require(lock["plan_sha256"] == sha(SCIENCE / "Plan.json"), "Analysis plan changed")
    require(lock["execution_lock_sha256"] == sha(SCIENCE / "Execution_Lock.json"), "Execution changed")
    for group in ("script_sha256", "prerequisite_sha256"):
        for name, digest in lock[group].items():
            require(sha(ROOT / name) == digest, "Analysis prerequisite changed: " + name)
    populations, matrices, export, inputs = gather(manifest, lock)
    tables = {}
    for language, views in matrices.items():
        population = populations[language]
        sources = [row["source_id"] for row in population] if language == "vi" else None
        parameters = lock["bootstrap"][language]
        if language == "vi":
            require(len(set(sources)) == 20 and len(population) == 1167, "Wrong Vietnamese population")
        else:
            require(len(population) == 2854, "Wrong English population")
        tables[language] = {}
        for view, records in views.items():
            tables[language][view] = describe(records, [tuple(pair) for pair in lock["pairs"][language]],
                                             sources, **parameters)
    require(len(export) == 53960, "Incomplete bilingual model/readout/clip matrix")
    result = {"utc": utc(), "status": "complete_mechanism_full_transfer_comparison", "tables": tables,
              "new_training": False, "new_states_training_seed": 17,
              "new_states_selected_training_step": 6000,
              "calibration_regime": read(SCIENCE / "Plan.json")["calibration_regime"],
              "primary_readout": "development_selected", "secondary_readout": "fixed_beam_c0.1",
              "test_column_selection": False, "interval_paper_changed": False,
              "inference": "one ordinary full-clip recognition pass per model",
              "reference_models_use_different_training_resources": True,
              "independent_source_groups_available_for_english": False,
              "pointwise_intervals_are_conditional_on_fitted_states": True,
              "model_readout_clip_records": len(export), "input_sha256": inputs,
              "analysis_lock_sha256": sha(lock_path)}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "Summary.json", result)
    with (output / "Per_Clip_Counts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    for name, digest in inputs.items():
        require(sha(ROOT / name) == digest, "Evidence changed during analysis")
    save(SCIENCE / "Completion_Receipt.json", {
        "utc": utc(), "status": "complete_mechanism_transfer_verified", "assessment_tasks": 32,
        "development_tasks_required": 32, "new_models": 4, "model_readout_clip_records": len(export),
        "summary_sha256": sha(output / "Summary.json"),
        "per_clip_counts_sha256": sha(output / "Per_Clip_Counts.csv"),
        "analysis_lock_sha256": sha(lock_path), "paper_changed": False,
    })
    print({"status": result["status"], "model_readout_clip_records": len(export)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.manifest.resolve(), arguments.output.resolve())
