"""Reproduce completed parent results and reject incomplete or corrupted comparison tables."""

from copy import deepcopy
import csv
from pathlib import Path
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent
ROOT = SCIENCE.parent
sys.path.insert(0, str(SCIENCE))

from experiment_core import cohort
from mechanism_common import read, save, sha, utc
from parent_tables import load_csv, parents
from transfer_statistics import FIELDS, checked_arrays, describe


def rejected(name, action, cases):
    try:
        action()
    except (ValueError, AssertionError, KeyError):
        cases.append(name)
        return
    raise AssertionError("Malformed table accepted: " + name)


def main():
    plan = read(SCIENCE / "Plan.json")
    populations = {language: cohort(plan, language, "assessment") for language in ("vi", "en")}
    records, settings, inputs, summaries = parents(populations)
    checked_parent_rows = sum(len(values) for table in records.values() for values in table.values())
    assert checked_parent_rows == 42544
    reproduced = []
    cases = []
    for language, seed in (("vi", 20261027), ("en", 20261028)):
        prefix = "VLR" if language == "vi" else "EN"
        left = prefix + "_T0_LAST"
        right = prefix + "_T1_LAST"
        for view in ("primary", "anchor"):
            values = {model: [[count[field] for field in FIELDS] for count in records[language][(view, model)]]
                      for model in (left, right)}
            sources = [row["source_id"] for row in populations[language]] if language == "vi" else None
            result = describe(values, [(left, right)], sources,
                              "video_source" if language == "vi" else "released_clip", 10000, seed)
            actual = result["comparisons"][left + "_minus_" + right]
            if language == "vi":
                phase_name = ("Research_Transfer_Full_Benchmark_2026-10-27" if view == "primary"
                              else "Research_Transfer_Decoder_Anchor_2026-10-28")
                expected = summaries[phase_name]["pairs"][left + "_minus_" + right]
                expected_gain = expected["gain_pp"]
                expected_ci = expected["source_ci95_pp"]
            else:
                original = read(ROOT / "Research_English_Transfer_Benchmark_2026-10-28/Assessment_Analysis_01/Summary.json")
                expected = original["training_comparisons"][view]
                expected_gain = expected["T0_minus_T1_wer_pp"]
                expected_ci = expected["conditional_clip_bootstrap_ci95_pp"]
            assert abs(actual["gain_pp"] - expected_gain) < 1e-10
            assert np.allclose(actual["conditional_ci95_pp"], expected_ci, atol=1e-12, rtol=0)
            reproduced.append({"language": language, "readout": view,
                               "gain_pp": actual["gain_pp"], "ci95": actual["conditional_ci95_pp"]})
    synthetic = {"left": [[2, 1, 1, 0, 4], [1, 0, 0, 1, 3]],
                 "right": [[2, 1, 1, 0, 4], [1, 0, 0, 1, 3]]}
    result = describe(synthetic, [("left", "right")], None, "released_clip", 100, 17)
    assert result["comparisons"]["left_minus_right"]["conditional_ci95_pp"] == [0.0, 0.0]
    for name, malformed in (
        ("missing_clip", {"left": synthetic["left"], "right": synthetic["right"][:1]}),
        ("negative_count", {"left": [[-1, -1, 0, 0, 4]], "right": [[0, 0, 0, 0, 4]]}),
        ("different_denominator", {"left": [[0, 0, 0, 0, 4]], "right": [[0, 0, 0, 0, 3]]}),
        ("bad_decomposition", {"left": [[2, 0, 0, 0, 4]], "right": [[0, 0, 0, 0, 4]]}),
        ("fractional_counts", {"left": [[0.5, 0.5, 0, 0, 4]], "right": [[0, 0, 0, 0, 4]]}),
    ):
        rejected(name, lambda: checked_arrays(malformed), cases)
    rejected("invented_english_groups",
             lambda: describe(synthetic, [("left", "right")], ["a", "b"], "released_clip", 10, 17), cases)
    tiny = [{"clip_id": "a", "source_id": "source_a", "frames": 50, "reference": "a b"},
            {"clip_id": "b", "source_id": "source_b", "frames": 25, "reference": "b"}]
    original_rows = [dict(model="T0", readout="anchor", decoder="beam_c0.1", clip_id=row["clip_id"],
                          source_id=row["source_id"], frames=row["frames"], errors=0, substitutions=0,
                          deletions=0, insertions=0, units=len(row["reference"].split())) for row in tiny]
    with tempfile.TemporaryDirectory(prefix="lip_mechanism_analysis_") as temporary:
        path = Path(temporary) / "counts.csv"
        for change in ("valid", "duplicate", "missing", "source", "units", "decoder", "frames", "errors"):
            rows = deepcopy(original_rows)
            if change == "duplicate":
                rows.append(rows[0])
            elif change == "missing":
                rows.pop()
            elif change == "source":
                rows[0]["source_id"] = "wrong"
            elif change == "units":
                rows[0]["units"] = 3
            elif change == "decoder":
                rows[0]["decoder"] = "beam_c0.5"
            elif change == "frames":
                rows[0]["frames"] += 1
            elif change == "errors":
                rows[0]["errors"] = 1
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(original_rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if change == "valid":
                parsed, _ = load_csv(path, tiny)
                assert len(parsed[("anchor", "T0")]) == 2
            else:
                rejected("csv_" + change, lambda: load_csv(path, tiny), cases)
    scripts = ["transfer_statistics.py", "parent_tables.py", "gather.py", "analyze.py", "test_analysis.py"]
    result = {"utc": utc(), "status": "passed", "completed_parent_csv_rows_checked": checked_parent_rows,
              "parent_WER_and_bootstrap_reproductions": reproduced, "rejected_cases": cases,
              "zero_gain_control_exact": True, "new_T2_T4_predictions_read": False,
              "input_sha256": inputs, "plan_sha256": sha(SCIENCE / "Plan.json"),
              "scripts": {name: sha(HERE / name) for name in scripts}}
    save(HERE / "Verification_01.json", result)
    print({"status": "passed", "parent_rows": checked_parent_rows,
           "bootstrap_reproductions": len(reproduced), "rejected_cases": len(cases)}, flush=True)


if __name__ == "__main__":
    main()
