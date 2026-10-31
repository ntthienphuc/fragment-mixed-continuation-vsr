"""Summarize complete source cohorts without choosing another test decoder."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from collect import HERE, ROOT, digest, metadata, model, plans, require

FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")
BOOTSTRAPS = 10000


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def candidate(parent, name, regime):
    return parent["models"][name]["frozen_transfer_decoder"] if regime == "primary" else "beam_c0.1"


def totals(records, decoder):
    result = {field: sum(row["predictions"][decoder]["counts"][field] for row in records) for field in FIELDS}
    require(result["errors"] == sum(result[field] for field in FIELDS[1:4]), "Invalid error decomposition")
    result["wer_percent"] = 100 * result["errors"] / result["units"]
    return result


def paired(left, right, left_decoder, right_decoder, label):
    grouped = defaultdict(lambda: [0, 0])
    for baseline, proposed in zip(left, right):
        require(baseline["clip_id"] == proposed["clip_id"], "Unpaired clips")
        first = baseline["predictions"][left_decoder]["counts"]
        second = proposed["predictions"][right_decoder]["counts"]
        require(first["units"] == second["units"], "Reference denominator differs")
        grouped[baseline["source_id"]][0] += first["errors"] - second["errors"]
        grouped[baseline["source_id"]][1] += first["units"]
    values = np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(("20261029|" + label).encode()).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    draws = []
    for start in range(0, BOOTSTRAPS, 200):
        size = min(200, BOOTSTRAPS - start)
        sampled = generator.integers(0, len(values), size=(size, len(values)))
        summed = values[sampled].sum(axis=1)
        draws.extend((100 * summed[:, 0] / summed[:, 1]).tolist())
    low, high = np.quantile(draws, [0.025, 0.975], method="linear").tolist()
    result = {
        "reference_minus_comparison_pp": 100 * int(values[:, 0].sum()) / int(values[:, 1].sum()),
        "ci95_low_pp": low, "ci95_high_pp": high,
        "groups": len(values), "bootstrap_unit": "archived_source_id", "resamples": BOOTSTRAPS,
        "seed": seed, "conditional_on_fitted_states": True, "multiplicity_adjusted": False,
    }
    return result


def pairs(corpus):
    prefix = "EN" if corpus == "lrs2" else "VLR"
    combinations = [
        (prefix + "_T0_LAST", prefix + "_T1_LAST", "matched_full_vs_fragment"),
        (prefix + "_B0", prefix + "_T0_LAST", "initial_vs_full"),
        (prefix + "_B0", prefix + "_T1_LAST", "initial_vs_fragment"),
        (prefix + "_T0_LAST", prefix + "_T2_LAST", "full_vs_self_teacher"),
        (prefix + "_T0_LAST", prefix + "_T4_LAST", "full_vs_context_preserving"),
        (prefix + "_T1_LAST", prefix + "_T2_LAST", "fragment_vs_self_teacher"),
        (prefix + "_T1_LAST", prefix + "_T4_LAST", "fragment_vs_context_preserving"),
    ]
    if corpus == "vlr":
        combinations.extend([
            ("VLR_T0_29_LAST", "VLR_T1_29_LAST", "matched_full_vs_fragment_seed29"),
            ("VLR_B0", "VLR_T0_29_LAST", "initial_vs_full_seed29"),
            ("VLR_B0", "VLR_T1_29_LAST", "initial_vs_fragment_seed29"),
        ])
    else:
        combinations.extend([
            ("EN_T0_LAST", "EN_F0_LAST", "different_training_population"),
            ("EN_B0", "EN_B1", "different_training_resources"),
        ])
    return combinations


def main(arguments):
    output = arguments.output.resolve()
    require(output.is_relative_to(HERE) and output != HERE, "Write analysis under a new attempt folder")
    output.mkdir(parents=True, exist_ok=False)
    parent, assessment = plans()
    corpus = arguments.corpus
    cohort = parent["cohorts"][corpus]
    names = arguments.model or cohort["models"]
    require(len(names) == len(set(names)) and set(names).issubset(cohort["models"]), "Invalid model selection")
    rows, pixels = metadata(corpus)
    status = []
    completed = {}
    inputs = []
    for name in names:
        state, records, provenance = model(corpus, name, parent, assessment, rows, pixels)
        status.append(state)
        inputs.extend(provenance)
        if records is not None:
            completed[name] = records
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "corpus": corpus, "models": status,
        "all_requested_models_complete": len(completed) == len(names),
        "whole_corpus_matrix_complete": set(completed) == set(cohort["models"]),
        "population_clips": len(rows), "source_groups": len({row["source_id"] for row in rows}),
        "inputs": inputs,
        "code_sha256": {path.name: digest(path.read_bytes()) for path in HERE.glob("*.py")},
        "protocol_sha256": digest((HERE / "Protocol.md").read_bytes()),
        "numpy": np.__version__, "python": sys.version,
    }
    save(output / "Readiness.json", receipt)
    if len(completed) != len(names):
        print(json.dumps({"status": "waiting_for_complete_verified_models", "models": status}))
        return 2
    summaries = []
    comparisons = []
    for name, records in completed.items():
        for regime in ("primary", "common"):
            decoder = candidate(parent, name, regime)
            counted = totals(records, decoder)
            require(counted["units"] == cohort["reference_units"], "Population denominator differs")
            summaries.append({"model": name, "regime": regime, "decoder": decoder,
                              "clips": len(records), **counted})
    for reference, comparison, role in pairs(corpus):
        if reference not in completed or comparison not in completed:
            continue
        for regime in ("primary", "common"):
            left_decoder = candidate(parent, reference, regime)
            right_decoder = candidate(parent, comparison, regime)
            label = "|".join([corpus, reference, comparison, regime])
            estimate = paired(completed[reference], completed[comparison], left_decoder, right_decoder, label)
            comparisons.append({"reference": reference, "comparison": comparison, "role": role,
                                "regime": regime, "reference_decoder": left_decoder,
                                "comparison_decoder": right_decoder, **estimate})
    paper = HERE / "Inputs_01/Destination_Results.csv"
    require(digest(paper.read_bytes()) ==
            "ae4d7bbfa4927b58d5a6150500e8d16c0b9e84012d95a678d92ea7aaf4e9a681",
            "Frozen destination table changed")
    with paper.open(encoding="utf-8-sig", newline="") as stream:
        destination = {(row["model_id"], row["decoder_regime"]): row for row in csv.DictReader(stream)}
    domain_pairs = []
    for row in summaries:
        target = destination[(row["model"], row["regime"])]
        require(target["decoder"] == row["decoder"], "Source/destination readout differs")
        domain_pairs.append({"model": row["model"], "regime": row["regime"], "decoder": row["decoder"],
                             "source_corpus": corpus, "source_wer": row["wer_percent"],
                             "source_errors": row["errors"], "source_units": row["units"],
                             "destination_corpus": target["assessment"],
                             "destination_wer": float(target["wer_percent"]),
                             "destination_errors": int(target["errors"]),
                             "destination_units": int(target["reference_units"])})
    save(output / "Summary.json", {
        **receipt, "status": "requested_complete_populations_independently_recounted",
        "source_test_readouts_retuned": False, "results": summaries, "comparisons": comparisons,
        "source_and_destination": domain_pairs, "destination_table_sha256": digest(paper.read_bytes()),
        "novelty_established": False, "manuscript_updated": False,
    })
    print(json.dumps({"status": "complete_population_summary_verified", "corpus": corpus,
                      "models": names, "results": summaries,
                      "whole_corpus_matrix_complete": receipt["whole_corpus_matrix_complete"]}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("lrs2", "vlr"), required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
