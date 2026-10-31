"""Compare complete F0 predictions with the verified English transfer matrix."""

import argparse
import csv
from pathlib import Path

import numpy as np

from collect_full_results import collect
from execution_common import checked_execution, checked_selection, require
from experiment_core import cohort
from full_analysis_core import FIELDS, bootstrap, total_counts
from mechanism_common import EN, HERE, ROOT, read, save, sha, utc


def parent_counts(summary_path, population):
    receipt_path = EN / "Completion_Receipt.json"
    receipt = read(receipt_path)
    require(receipt["status"] == "english_full_benchmark_verified", "Parent comparison incomplete")
    require(receipt["summary_sha256"] == sha(summary_path), "Parent summary changed")
    table_path = summary_path.parent / "Per_Clip_Counts.csv"
    require(receipt["per_clip_counts_sha256"] == sha(table_path), "Parent count table changed")
    summary = read(summary_path)
    require(summary["status"] == "complete_verified_full_wildvsr_matrix", "Parent matrix incomplete")
    require(summary["assessment_clips"] == 2854 and summary["assessment_reference_words"] == 48037,
            "Parent population changed")
    require(summary["plan_sha256"] == sha(EN / "Plan.json"), "Parent plan changed")
    require(summary["execution_lock_sha256"] == sha(EN / "Execution_Lock.json"), "Parent code changed")
    for name, digest in read(EN / "Execution_Lock.json")["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Parent implementation changed")
    for name, digest in summary["input_sha256"].items():
        require(sha(ROOT / name) == digest, "Parent evidence changed")
    models = read(EN / "Plan.json")["models"]
    require(set(summary["models"]) == set(models), "Missing parent model")
    by_id = {row["clip_id"]: row for row in population}
    result = {model: {readout: {} for readout in ("primary", "anchor")} for model in models}
    with table_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            readout = row["readout"]
            require(model in result and readout in result[model], "Unknown parent result")
            clip = row["clip_id"]
            require(clip in by_id and clip not in result[model][readout], "Wrong or duplicated parent clip")
            require(int(row["frames"]) == by_id[clip]["frames"], "Parent duration changed")
            require(row["decoder"] == summary["models"][model][readout]["decoder"], "Parent decoder changed")
            counts = {field: int(row[field]) for field in FIELDS}
            require(counts["units"] == len(by_id[clip]["reference"].split()), "Parent reference units changed")
            result[model][readout][clip] = counts
    for model in models:
        for readout, values in result[model].items():
            require(set(values) == set(by_id), "Incomplete parent model population")
            ordered = [values[row["clip_id"]] for row in population]
            totals = total_counts(ordered)
            for field, value in totals.items():
                require(abs(value - summary["models"][model][readout][field]) < 1e-9, "Parent summary disagrees")
            result[model][readout] = ordered
    inputs = {path.relative_to(ROOT).as_posix(): sha(path) for path in (receipt_path, summary_path, table_path)}
    return result, summary, inputs


def main(manifest, parent_summary, output):
    plan, execution = checked_execution()
    policy_path = HERE / "Analysis_Lock.json"
    policy = read(policy_path)
    require(policy["status"] == "F0_analysis_prespecified", "Analysis not fixed")
    require(policy["plan_sha256"] == sha(HERE / "Plan.json"), "Analysis plan changed")
    require(policy["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Worker changed")
    for name, digest in policy["scripts"].items():
        require(sha(HERE / name) == digest, "Analysis source changed")
    require(policy["verification_sha256"] == sha(HERE / "Analysis_Verification_01.json"), "Analysis checks changed")
    selection = checked_selection(plan)
    records, inputs = collect("assessment", manifest, plan, selection)
    population = cohort(plan, "en", "assessment")
    require(len(population) == len(records) == 2854, "Incomplete assessment")
    words = [len(row["reference"].split()) for row in population]
    require(sum(words) == 48037, "Wrong assessment denominator")
    values, parent, parent_inputs = parent_counts(parent_summary, population)
    inputs.update(parent_inputs)
    values["EN_F0_LAST"] = {}
    decoders = {model: {readout: value["decoder"] for readout, value in table.items()}
                for model, table in parent["models"].items()}
    decoders["EN_F0_LAST"] = {"primary": selection["models"]["EN_F0_LAST"]["decoder"], "anchor": plan["anchor"]}
    for readout, decoder in decoders["EN_F0_LAST"].items():
        values["EN_F0_LAST"][readout] = [record["predictions"][decoder]["counts"] for record in records]
    tables = {}
    export = []
    for model, readouts in values.items():
        tables[model] = {}
        for readout, counts in readouts.items():
            totals = total_counts(counts)
            require(totals["units"] == 48037, "Model denominator changed")
            strata = {}
            for name, condition in (("below_82_frames", lambda value: value < 82),
                                    ("at_least_82_frames", lambda value: value >= 82)):
                indices = [index for index, row in enumerate(population) if condition(row["frames"])]
                if indices:
                    strata[name] = {"clips": len(indices), **total_counts([counts[index] for index in indices])}
            tables[model][readout] = {"decoder": decoders[model][readout], "clips": 2854,
                                      **totals, "duration_strata": strata}
            for row, count in zip(population, counts):
                export.append({"model": model, "readout": readout, "decoder": decoders[model][readout],
                               "clip_id": row["clip_id"], "frames": row["frames"],
                               **{field: count[field] for field in FIELDS}})
    pairs = {}
    for readout in ("primary", "anchor"):
        for control in ("EN_T0_LAST", "EN_T1_LAST"):
            key = "F0_minus_" + control + "_" + readout
            pairs[key] = ([row["errors"] for row in values["EN_F0_LAST"][readout]],
                          [row["errors"] for row in values[control][readout]])
    samples = bootstrap(pairs, words, policy["bootstrap_draws"], policy["bootstrap_seed"])
    comparisons = {}
    for key, (left, right) in pairs.items():
        comparisons[key] = {"wer_difference_pp": 100 * (sum(left) - sum(right)) / sum(words),
                            "conditional_clip_bootstrap_ci95_pp": np.quantile(samples[key], [.025, .975]).tolist(),
                            "positive_difference_favors": "the named T0 or T1 control"}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "Per_Clip_Counts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    for path in (policy_path, HERE / "Decoder_Selection.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    save(output / "Summary.json", {"utc": utc(), "status": "complete_F0_practical_control_verified",
         "models": tables, "comparisons": comparisons, "primary_question": policy["primary_question"],
         "assessment_clips": 2854, "reference_words": 48037, "training_seed": 17,
         "same_training_population_control": False, "new_training": False,
         "bootstrap_draws": policy["bootstrap_draws"], "bootstrap_seed": policy["bootstrap_seed"],
         "resampling_unit": "released clip; source grouping unavailable",
         "input_sha256": inputs, "plan_sha256": sha(HERE / "Plan.json"),
         "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    save(HERE / "Completion_Receipt.json", {"utc": utc(), "status": "F0_complete_control_verified",
         "source_clips": 1082, "assessment_clips": 2854, "assessment_tasks": 8,
         "summary_sha256": sha(output / "Summary.json"),
         "per_clip_counts_sha256": sha(output / "Per_Clip_Counts.csv"), "paper_or_novelty_claim_updated": False})
    print(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("parent_summary", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.manifest.resolve(), arguments.parent_summary.resolve(), arguments.output.resolve())
