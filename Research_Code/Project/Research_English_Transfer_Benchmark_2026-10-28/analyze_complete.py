"""Summarize only the complete, verified WildVSR matrix under fixed readouts."""

import argparse
import csv
from pathlib import Path

import numpy as np

from admission import validate_selection
from collect_results import collect
from english_common import HERE, ROOT, lines, read, require, save, sha, utc

COUNT_FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def sum_counts(values):
    totals = {field: sum(item[field] for item in values) for field in COUNT_FIELDS}
    require(totals["errors"] == totals["substitutions"] + totals["deletions"] + totals["insertions"], "Bad errors")
    require(totals["units"] > 0, "Empty denominator")
    return {**totals, "wer_percent": 100 * totals["errors"] / totals["units"]}


def paired_bootstrap(pairs, words, draws, seed):
    words = np.asarray(words, dtype=np.int64)
    require(words.ndim == 1 and len(words) > 0 and np.all(words > 0), "Invalid clip denominators")
    differences = {name: np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)
                   for name, (left, right) in pairs.items()}
    require(all(value.shape == words.shape for value in differences.values()), "Unpaired sample vectors")
    output = {name: [] for name in differences}
    generator = np.random.default_rng(seed)
    remaining = draws
    while remaining:
        count = min(256, remaining)
        indices = generator.integers(0, len(words), size=(count, len(words)))
        denominators = words[indices].sum(axis=1)
        for name, values in differences.items():
            output[name].extend((100 * values[indices].sum(axis=1) / denominators).tolist())
        remaining -= count
    return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}


def main(manifest_path, output):
    require(not output.exists(), "Preserve earlier analysis")
    plan = read(HERE / "Plan.json")
    for name, expected in plan["frozen_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen input changed")
    execution = read(HERE / "Execution_Lock.json")
    for name, expected in execution["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Execution changed")
    selection = read(HERE / "Decoder_Selection.json")
    validate_selection(selection, plan)
    require(selection["plan_sha256"] == sha(HERE / "Plan.json"), "Selection uses a different plan")
    require(selection["assessment_manifest_sha256"] == sha(HERE / "Assessment.jsonl"), "Assessment changed")
    require(selection["selection_code_sha256"] == sha(HERE / "select_decoder.py"), "Selection code changed")
    for name, expected in selection["input_sha256"].items():
        require(sha(ROOT / name) == expected, "Source selection evidence changed")
    records, inputs = collect("assessment", manifest_path, plan, selection)
    population = lines(HERE / "Assessment.jsonl")
    require(len(population) == 2854, "Incomplete assessment")
    words = np.asarray([len(row["reference"].split()) for row in population], dtype=np.int64)
    require(int(words.sum()) == 48037, "Assessment denominator changed")
    groups = {"below_82_frames": [index for index, row in enumerate(population) if row["frames"] < 82],
              "at_least_82_frames": [index for index, row in enumerate(population) if row["frames"] >= 82]}
    tables = {}
    errors = {}
    export = []
    for model in plan["models"]:
        tables[model] = {}
        errors[model] = {}
        settings = {"primary": selection["models"][model]["decoder"], "anchor": plan["anchor"]}
        for readout, decoder in settings.items():
            values = [row["predictions"][decoder]["counts"] for row in records[model]]
            totals = sum_counts(values)
            require(totals["units"] == 48037, "Wrong model denominator")
            strata = {name: {"clips": len(indices), **sum_counts([values[index] for index in indices])}
                      for name, indices in groups.items() if indices}
            tables[model][readout] = {"decoder": decoder, "clips": 2854, **totals, "duration_strata": strata}
            errors[model][readout] = [value["errors"] for value in values]
            for row, value in zip(population, values):
                export.append({"model": model, "readout": readout, "decoder": decoder,
                               "clip_id": row["clip_id"], "frames": row["frames"],
                               **{field: value[field] for field in COUNT_FIELDS}})
    pairs = {readout: (errors["EN_T0_LAST"][readout], errors["EN_T1_LAST"][readout])
             for readout in ("primary", "anchor")}
    draws = paired_bootstrap(pairs, words, plan["bootstrap_draws"], plan["bootstrap_seed"])
    comparisons = {}
    for readout in ("primary", "anchor"):
        left = tables["EN_T0_LAST"][readout]
        right = tables["EN_T1_LAST"][readout]
        differences = {field: right[field] - left[field] for field in COUNT_FIELDS[:-1]}
        comparisons[readout] = {"T0_minus_T1_wer_pp": left["wer_percent"] - right["wer_percent"],
                                "T1_minus_T0_counts": differences,
                                "conditional_clip_bootstrap_ci95_pp": np.quantile(draws[readout], [.025, .975]).tolist(),
                                "resampling_unit": "released clip; not independently reconstructed source groups"}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "Per_Clip_Counts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    inputs[(HERE / "Decoder_Selection.json").relative_to(ROOT).as_posix()] = sha(HERE / "Decoder_Selection.json")
    summary = {"utc": utc(), "status": "complete_verified_full_wildvsr_matrix", "assessment_clips": 2854,
               "assessment_reference_words": 48037, "source_development_clips": 1082, "models": tables,
               "training_comparisons": comparisons, "training_seeds": [17], "new_training": False,
               "uncertainty_note": "Clip bootstrap is descriptive conditional on the released clips; source map absent",
               "decoder_calibration": "LRS2 source development only; no destination calibration",
               "normalization": "NFC_lower_whitespace", "bootstrap_draws": plan["bootstrap_draws"],
               "bootstrap_seed": plan["bootstrap_seed"], "input_sha256": inputs,
               "plan_sha256": sha(HERE / "Plan.json"), "execution_lock_sha256": sha(HERE / "Execution_Lock.json")}
    save(output / "Summary.json", summary)
    save(HERE / "Completion_Receipt.json", {"utc": utc(), "status": "english_full_benchmark_verified",
                                           "models": 4, "assessment_tasks": 32, "source_tasks_required": 32,
                                           "primary_model_clip_predictions": 11416,
                                           "summary_sha256": sha(output / "Summary.json"),
                                           "per_clip_counts_sha256": sha(output / "Per_Clip_Counts.csv"),
                                           "paper_or_novelty_claim_updated": False})
    print(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.manifest.resolve(), arguments.output)
