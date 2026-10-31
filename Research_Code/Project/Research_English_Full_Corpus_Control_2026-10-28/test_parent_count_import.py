"""Exercise the full 2,854-clip count-table contract with explicitly synthetic evidence."""

from copy import deepcopy
import csv

import analyze_full_f0 as analysis
from full_analysis_core import FIELDS, total_counts
from mechanism_common import HERE, ROOT, save, sha, utc


def build_case(folder, rows, tables, models):
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "Synthetic_Test_Only.json", {"synthetic": True, "recognition_experiment": False})
    save(folder / "Plan.json", {"models": models, "synthetic": True})
    save(folder / "Execution_Lock.json", {"code_sha256": {}})
    table = folder / "Per_Clip_Counts.csv"
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = folder / "Summary.json"
    save(summary_path, {"status": "complete_verified_full_wildvsr_matrix", "synthetic": True,
         "assessment_clips": 2854, "assessment_reference_words": 48037, "models": tables,
         "plan_sha256": sha(folder / "Plan.json"), "execution_lock_sha256": sha(folder / "Execution_Lock.json"),
         "input_sha256": {table.relative_to(ROOT).as_posix(): sha(table)}})
    save(folder / "Completion_Receipt.json", {"status": "english_full_benchmark_verified", "synthetic": True,
         "summary_sha256": sha(summary_path), "per_clip_counts_sha256": sha(table)})
    return summary_path


def main():
    output = HERE / "Analysis_Tests_01"
    output.mkdir(exist_ok=False)
    population = [{"clip_id": "synthetic_" + str(index), "frames": 90 + index % 40,
                   "reference": " ".join(["synthetic"] * (16 if index < 481 else 17))}
                  for index in range(2854)]
    assert sum(len(row["reference"].split()) for row in population) == 48037
    models = ["EN_B0", "EN_B1", "EN_T0_LAST", "EN_T1_LAST"]
    rows = []
    tables = {}
    for model_index, model in enumerate(models):
        tables[model] = {}
        for readout in ("primary", "anchor"):
            counts = []
            for index, row in enumerate(population):
                value = {"errors": index % 4 + model_index % 2, "substitutions": index % 4,
                         "deletions": model_index % 2, "insertions": 0,
                         "units": len(row["reference"].split())}
                counts.append(value)
                rows.append({"model": model, "readout": readout, "decoder": "beam_c0.1",
                             "clip_id": row["clip_id"], "frames": row["frames"], **value})
            tables[model][readout] = {"decoder": "beam_c0.1", "clips": 2854, **total_counts(counts)}
    parent = analysis.EN
    try:
        valid = output / "Valid"
        summary = build_case(valid, rows, tables, models)
        analysis.EN = valid
        imported, restored, inputs = analysis.parent_counts(summary, population)
        assert set(imported) == set(models)
        assert all(len(imported[model][readout]) == 2854 for model in models for readout in ("primary", "anchor"))
        mutations = [
            ("clip_id", "synthetic_1"), ("clip_id", "missing_clip"), ("frames", 999),
            ("decoder", "greedy_b0.0"), ("units", 99), ("errors", 99),
        ]
        failures = []
        for index, (field, value) in enumerate(mutations):
            changed = list(rows)
            changed[0] = {**changed[0], field: value}
            folder = output / ("Reject_" + str(index))
            summary = build_case(folder, changed, tables, models)
            analysis.EN = folder
            try:
                analysis.parent_counts(summary, population)
            except ValueError as error:
                failures.append({"field": field, "reason": str(error)})
            else:
                raise AssertionError("Corrupt parent count population accepted: " + field)
        missing = output / "Missing_Row"
        summary = build_case(missing, rows[:-1], tables, models)
        analysis.EN = missing
        try:
            analysis.parent_counts(summary, population)
        except ValueError as error:
            failures.append({"field": "missing_row", "reason": str(error)})
        else:
            raise AssertionError("Incomplete population accepted")
    finally:
        analysis.EN = parent
    receipt = {"utc": utc(), "status": "passed", "synthetic_only": True, "real_assessment_predictions_read": False,
               "valid_count_rows": len(rows), "models": 4, "clips_per_readout": 2854,
               "independently_rehashed_malformed_cases_rejected": failures,
               "scripts": {name: sha(HERE / name) for name in
                           ("analyze_full_f0.py", "full_analysis_core.py", "test_parent_count_import.py")}}
    save(HERE / "Parent_Count_Import_Verification_01.json", receipt)
    print({"status": "passed", "synthetic_count_rows": len(rows), "malformed_cases": len(failures)})


if __name__ == "__main__":
    main()
