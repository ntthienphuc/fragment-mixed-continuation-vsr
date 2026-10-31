"""Score the complete, development-locked test; never select a decoder from test results."""

import csv

import numpy as np

from admission import require, validate_selection, verify_plan
from common import HERE, ROOT, lines, read, save, sha, utc
from task_design import selected_task
from verify_task import verify_records

FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")
PAIRS = (("VLR_T0_LAST", "VLR_T1_LAST"), ("VLR_T0_29_LAST", "VLR_T1_29_LAST"))


def describe_source_counts(values, models, sources, weights):
    require(values.shape == (len(models), len(sources), 5), "Source array dimensions differ")
    units = values[0, :, 4]
    require(np.all(units > 0), "Empty source reference")
    require(np.all(values[:, :, 4] == units[None, :]), "Model test populations differ")
    require(np.all(values[:, :, 0] == values[:, :, 1:4].sum(axis=-1)), "S/D/I decomposition differs")
    denominator = weights @ units
    require(np.all(denominator > 0), "Invalid bootstrap draw")
    totals = values.sum(axis=1)
    wers = 100 * totals[:, 0] / totals[:, 4]
    output = {"models": {model: {"wer": float(wer), "counts": dict(zip(FIELDS, total.tolist()))}
                          for model, wer, total in zip(models, wers, totals)}, "pairs": {}}
    gains = []
    observed = []
    scalar_checks = 0
    for left, right in PAIRS:
        first = models.index(left)
        second = models.index(right)
        differences = values[first, :, 0] - values[second, :, 0]
        draws = 100 * (weights @ differences) / denominator
        actual = float(wers[first] - wers[second])
        scalar = []
        for draw in weights[:101]:
            errors = sum(int(weight) * int(error) for weight, error in zip(draw, differences))
            words = sum(int(weight) * int(count) for weight, count in zip(draw, units))
            scalar.append(100 * errors / words)
        require(np.allclose(draws[:101], scalar, atol=1e-12, rtol=0), "Bootstrap scalar recount differs")
        scalar_checks += len(scalar)
        source_gain = 100 * differences / units
        output["pairs"][left + "_minus_" + right] = {
            "gain_pp": actual, "source_ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
            "source_gain_pp": dict(zip(sources, source_gain.tolist())),
            "unweighted_mean_source_gain_pp": float(source_gain.mean()),
            "sources_better_worse_equal": [int((differences > 0).sum()), int((differences < 0).sum()),
                                           int((differences == 0).sum())],
        }
        gains.append(draws)
        observed.append(actual)
    output["fixed_two_seed_mean"] = {
        "gain_pp": float(np.mean(observed)),
        "source_ci95_pp": np.quantile(np.mean(gains, axis=0), [.025, .975]).tolist(),
        "scope": "Source uncertainty conditional on the two fitted seeds and locked decoders; not seed-population CI",
        "same_source_draws_for_both_seeds": True,
    }
    output["independent_scalar_bootstrap_checks"] = scalar_checks
    return output


def collect(plan, selection, lock):
    models = [*plan["matched_models"], plan["practical_reference"]]
    cohort = lines(HERE / "Assessment.jsonl")
    sources = sorted({row["source_id"] for row in cohort})
    require(len(cohort) == 1167 and len(sources) == 20, "Official test population differs")
    records = {model: {} for model in models}
    evidence = {}
    pixels = {row["clip_id"]: row for row in lines(HERE / "Pixels_01/Index.jsonl")}
    registry = read(HERE / "model_registry.json")["models"]
    registry.update(read(HERE / "Native_Reference_Registry.json")["models"])
    checked = 0
    tasks = 0
    for name, task in lock["tasks"].items():
        if task["stage"] != "assessment":
            continue
        matches = list((HERE / "Runs").glob("*/" + name + "/Verification.json"))
        require(len(matches) == 1, "Exactly one complete verified test attempt required: " + name)
        proof_path = matches[0]
        proof = read(proof_path)
        path = proof_path.parent / "Predictions.jsonl"
        require(proof["status"] == "passed" and proof["task"] == name, "Test task not verified")
        require(proof["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Wrong execution lock")
        require(proof["predictions_sha256"] == sha(path), "Test predictions changed")
        model = task["model"]
        require(proof["checkpoint_sha256"] == registry[model]["checkpoint_sha256"], "Wrong checkpoint")
        require(len(proof["fresh_pixel_replays"]) == 2, "Fresh-pixel verification missing")
        start = read(proof_path.parent / "Start.json")
        require(start["selection_sha256"] == sha(HERE / "Decoder_Selection.json"), "Worker used another selection")
        _, rows = selected_task(name, lock)
        predicted = lines(path)
        checked += verify_records(rows, predicted, model, "assessment",
                                  [selection["models"][model]["decoder"]], pixels)
        for row in predicted:
            require(row["clip_id"] not in records[model], "Duplicate test record")
            records[model][row["clip_id"]] = row
        for item in (path, proof_path):
            evidence[item.relative_to(ROOT).as_posix()] = sha(item)
        tasks += 1
    expected_ids = {row["clip_id"] for row in cohort}
    require(all(set(value) == expected_ids for value in records.values()), "Incomplete test matrix")
    require(checked == 7002 and tasks == 48, "Test matrix verification incomplete")
    return models, sources, cohort, records, evidence


def main():
    plan = verify_plan()
    lock_path = HERE / "Execution_Lock.json"
    selection_path = HERE / "Decoder_Selection.json"
    lock = read(lock_path)
    for name, expected in lock["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen execution code changed: " + name)
    selection = read(selection_path)
    validate_selection(selection, "VLR_B0", plan, sha(HERE / "Plan.json"), sha(HERE / "Assessment.jsonl"))
    for name, expected in selection["input_sha256"].items():
        require(sha(ROOT / name) == expected, "Selection input changed: " + name)
    models, sources, cohort, records, evidence = collect(plan, selection, lock)
    values = np.zeros((len(models), len(sources), 5), dtype=np.int64)
    strata = {label: np.zeros((len(models), 5), dtype=np.int64) for label in ("short_lt82", "long_ge82")}
    per_clip = []
    for model_index, model in enumerate(models):
        candidate = selection["models"][model]["decoder"]
        for source in cohort:
            row = records[model][source["clip_id"]]
            result = row["predictions"][candidate]
            counts = np.array([result["counts"][field] for field in FIELDS], dtype=np.int64)
            values[model_index, sources.index(source["source_id"])] += counts
            stratum = "short_lt82" if source["frames"] < 82 else "long_ge82"
            strata[stratum][model_index] += counts
            per_clip.append({"model": model, "clip_id": source["clip_id"], "source_id": source["source_id"],
                             "frames": source["frames"], "decoder": candidate, **dict(zip(FIELDS, counts.tolist())),
                             "encode_seconds": row["encoding"]["encode_seconds"],
                             "decode_seconds": result["seconds"]})
    require(np.all(values[:, :, 4].sum(axis=1) == 30091), "Reference token totals differ")
    bootstrap = plan["bootstrap"]
    random = np.random.default_rng(bootstrap["seed"])
    weights = random.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), bootstrap["draws"])
    result = describe_source_counts(values, models, sources, weights)
    result.update({"status": "complete_full_test_with_development_locked_decoders", "utc": utc(),
                   "clips": 1167, "sources": 20, "models_order": models, "source_counts": values.tolist(),
                   "source_order": sources, "independent_prediction_count_checks": 7002,
                   "test_decoder_sweep": False, "new_hidden_test": False,
                   "strata_secondary_counts": {key: value.tolist() for key, value in strata.items()},
                   "plan_sha256": sha(HERE / "Plan.json"), "selection_sha256": sha(selection_path),
                   "execution_lock_sha256": sha(lock_path), "input_sha256": evidence})
    folder = HERE / "Assessment_Analysis_01"
    folder.mkdir(exist_ok=False)
    save(folder / "Summary.json", result)
    with (folder / "Per_Clip_Counts.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    for name, digest in evidence.items():
        require(sha(ROOT / name) == digest, "Input changed during analysis")
    save(HERE / "Completion_Receipt.json", {
        "status": "complete_full_benchmark_verified", "utc": utc(), "models": models,
        "clips_per_model": 1167, "independent_count_checks": 7002,
        "summary_sha256": sha(folder / "Summary.json"), "per_clip_sha256": sha(folder / "Per_Clip_Counts.csv"),
        "decoder_selection_sha256": sha(selection_path), "paper_changed": False,
    })
    print({"status": result["status"], "pairs": result["pairs"]}, flush=True)


if __name__ == "__main__":
    main()
