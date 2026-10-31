"""Recount original completed per-clip tables and export a portable numerical supplement."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "Data"
FILES = [
    ("VI", "primary", "Research_Transfer_Full_Benchmark_2026-10-27/Assessment_Analysis_01"),
    ("VI", "common", "Research_Transfer_Decoder_Anchor_2026-10-28/Analysis_01/Results_01"),
    ("EN", None, "Research_English_Transfer_Benchmark_2026-10-28/Assessment_Analysis_01"),
    ("EN", None, "Research_English_Full_Corpus_Control_2026-10-28/Assessment_Analysis_01"),
]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from completed_results import active, gate
    expanded = active()
    inputs = list(FILES)
    if expanded:
        matrix = gate()
        inputs.append((None, None, matrix.relative_to(PROJECT).as_posix()))
    observations = {}
    input_hashes = {}
    for language, regime, folder in inputs:
        path = PROJECT / folder / "Per_Clip_Counts.csv"
        input_hashes[path.relative_to(PROJECT).as_posix()] = digest(path)
        for row in read(path):
            current_regime = regime or ("common" if row["readout"] == "anchor" else "primary")
            record = {
                "language": language or row["language"].upper(),
                "model_id": row["model"],
                "decoder_regime": current_regime,
                "decoder": row["decoder"],
                "clip_id": row["clip_id"],
                "source_id": row.get("source_id", ""),
                "frames": int(row["frames"]),
                "reference_units": int(row["units"]),
                "errors": int(row["errors"]),
                "substitutions": int(row["substitutions"]),
                "deletions": int(row["deletions"]),
                "insertions": int(row["insertions"]),
            }
            key = (record["model_id"], current_regime, record["clip_id"])
            if key in observations:
                assert observations[key] == record, key
            observations[key] = record
    totals = defaultdict(Counter)
    cohorts = defaultdict(dict)
    fields = ["reference_units", "errors", "substitutions", "deletions", "insertions"]
    for record in observations.values():
        key = (record["model_id"], record["decoder_regime"])
        assert record["errors"] == sum(record[field] for field in fields[2:])
        totals[key].update({field: record[field] for field in fields})
        totals[key]["clips"] += 1
        cohorts[key][record["clip_id"]] = (record["frames"], record["reference_units"])
    results = read(DATA / "Results.csv")
    assert len(results) == len(totals) == (30 if expanded else 22)
    for row in results:
        key = (row["model_id"], row["decoder_regime"])
        for field in fields + ["clips"]:
            assert totals[key][field] == int(row[field]), (key, field)
        expected = 100 * totals[key]["errors"] / totals[key]["reference_units"]
        assert abs(expected - float(row["wer_percent"])) < 1e-9, key
        reference = "VLR_B0" if row["language"] == "VI" else "EN_B0"
        assert cohorts[key] == cohorts[(reference, "primary")], key
    output = DATA / "Per_Clip_Errors.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(observations.values()))))
        writer.writeheader()
        writer.writerows(observations[key] for key in sorted(observations))
    reopened = read(output)
    assert len(reopened) == (58628 if expanded else 42544)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_original_per_clip_recount_and_cohort_identity",
        "unique_prediction_records": len(reopened),
        "aggregate_result_rows_checked": len(results),
        "completed_only": True,
        "input_sha256": input_hashes,
        "per_clip_export_sha256": digest(output),
        "results_sha256": digest(DATA / "Results.csv"),
        "verifier_sha256": digest(Path(__file__)),
    }
    (DATA / "Verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ["status", "unique_prediction_records"]}))


if __name__ == "__main__":
    main()
