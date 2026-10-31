"""Validate all completed parent CSV records before reusing matched control counts."""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent
ROOT = SCIENCE.parent
sys.path.insert(0, str(SCIENCE))

from mechanism_common import read, sha
from transfer_statistics import FIELDS, require


def load_csv(path, population, fixed_readout=None):
    by_id = {row["clip_id"]: row for row in population}
    require(len(by_id) == len(population), "Duplicated reference clips")
    records = defaultdict(dict)
    decoders = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            view = fixed_readout or row["readout"]
            require(view in {"primary", "anchor"}, "Unknown readout")
            key = (view, row["model"])
            identity = row["clip_id"]
            require(identity in by_id and identity not in records[key], "Wrong or duplicated clip")
            source = by_id[identity]
            require(int(row["frames"]) == source["frames"], "Frame length changed")
            if "source_id" in row:
                require(row["source_id"] == source["source_id"], "Source membership changed")
            counts = {field: int(row[field]) for field in FIELDS}
            require(all(value >= 0 for value in counts.values()), "Negative count")
            require(counts["errors"] == sum(counts[field] for field in FIELDS[1:4]), "Bad edit decomposition")
            require(counts["units"] == len(source["reference"].split()), "Reference denominator changed")
            if view == "anchor":
                require(row["decoder"] == "beam_c0.1", "Fixed decoder changed")
            records[key][identity] = counts
            decoders[key].add(row["decoder"])
    require(bool(records), "No parent records")
    for key, values in records.items():
        require(set(values) == set(by_id), "Incomplete parent population")
        require(len(decoders[key]) == 1, "Model-specific decoder varies over assessment clips")
    ordered = {key: [values[row["clip_id"]] for row in population] for key, values in records.items()}
    return ordered, {key: next(iter(values)) for key, values in decoders.items()}


def load_parent(folder, population, status, summary_name, csv_field, fixed_readout=None):
    receipt_path = folder / "Completion_Receipt.json"
    receipt = read(receipt_path)
    require(receipt["status"] == status, "Parent not complete")
    summary_path = folder / summary_name / "Summary.json"
    csv_path = folder / summary_name / "Per_Clip_Counts.csv"
    require(sha(summary_path) == receipt["summary_sha256"], "Parent summary changed")
    require(sha(csv_path) == receipt[csv_field], "Parent counts changed")
    summary = read(summary_path)
    for name, digest in summary["input_sha256"].items():
        require(sha(ROOT / name) == digest, "Parent evidence changed: " + name)
    records, decoders = load_csv(csv_path, population, fixed_readout)
    expected_keys = {(view, model) for model in summary["models"]
                     for view in ([fixed_readout] if fixed_readout else ["primary", "anchor"])}
    require(set(records) == expected_keys, "Parent model/readout coverage differs")
    for (view, model), rows in records.items():
        totals = Counter()
        for row in rows:
            totals.update(row)
        descriptor = summary["models"][model]
        if fixed_readout:
            expected = descriptor["counts"]
            expected_wer = descriptor["wer"]
        else:
            descriptor = descriptor[view]
            expected = {field: descriptor[field] for field in FIELDS}
            expected_wer = descriptor["wer_percent"]
        require(dict(totals) == expected, "Parent summary recount differs")
        require(abs(100 * totals["errors"] / totals["units"] - expected_wer) < 1e-10, "Parent WER differs")
    inputs = {path.relative_to(ROOT).as_posix(): sha(path) for path in (receipt_path, summary_path, csv_path)}
    return records, decoders, inputs, summary


def parents(populations):
    specifications = [
        ("vi", "Research_Transfer_Full_Benchmark_2026-10-27", "complete_full_benchmark_verified",
         "Assessment_Analysis_01", "per_clip_sha256", "primary"),
        ("vi", "Research_Transfer_Decoder_Anchor_2026-10-28", "complete_vi_anchor_verified",
         "Analysis_01/Results_01", "per_clip_sha256", "anchor"),
        ("en", "Research_English_Full_Corpus_Control_2026-10-28", "F0_complete_control_verified",
         "Assessment_Analysis_01", "per_clip_counts_sha256", None),
    ]
    result = {"vi": {}, "en": {}}
    settings = {"vi": {}, "en": {}}
    inputs = {}
    summaries = {}
    for language, name, status, folder, count_field, view in specifications:
        records, decoders, evidence, summary = load_parent(
            ROOT / name, populations[language], status, folder, count_field, view
        )
        require(not (set(result[language]) & set(records)), "Duplicated parent table")
        result[language].update(records)
        settings[language].update(decoders)
        inputs.update(evidence)
        summaries[name] = summary
    return result, settings, inputs, summaries
