"""Complete fixed-decoder Vietnamese results with the parent's source bootstrap."""

import csv
from pathlib import Path

import numpy as np

from collect_anchor import FULL, PHASE, ROOT, collect, read, sha
from anchor_core import save, utc
from analyze_assessment import FIELDS, describe_source_counts


HERE = Path(__file__).resolve().parent


def build_tables(population, records, models):
    assert len(population) == 1167
    assert set(records) == set(models) and len(models) == 6
    assert all(set(values) == set(range(1167)) for values in records.values())
    sources = sorted({row["source_id"] for row in population})
    assert len(sources) == 20
    values = np.zeros((len(models), len(sources), 5), dtype=np.int64)
    strata = {name: np.zeros((len(models), 5), dtype=np.int64) for name in ("short_lt82", "long_ge82")}
    exported = []
    for model_index, model in enumerate(models):
        for index, source in enumerate(population):
            predicted = records[model][index]
            assert predicted["clip_id"] == source["clip_id"]
            count = predicted["predictions"]["beam_c0.1"]["counts"]
            assert all(type(count[field]) is int and count[field] >= 0 for field in FIELDS)
            assert count["errors"] == count["substitutions"] + count["deletions"] + count["insertions"]
            assert count["units"] == len(source["reference"].split())
            vector = np.array([count[field] for field in FIELDS], dtype=np.int64)
            values[model_index, sources.index(source["source_id"])] += vector
            stratum = "short_lt82" if source["frames"] < 82 else "long_ge82"
            strata[stratum][model_index] += vector
            exported.append({"model": model, "clip_id": source["clip_id"], "source_id": source["source_id"],
                             "frames": source["frames"], "decoder": "beam_c0.1", **count})
    assert np.all(values[:, :, 4].sum(axis=1) == 30091)
    return sources, values, strata, exported


def main():
    lock = read(HERE / "Analysis_Lock.json")
    assert lock["status"] == "vi_anchor_analysis_fixed_before_complete_matrix"
    for name, digest in lock["scripts"].items():
        assert sha(HERE / name) == digest
    assert lock["plan_sha256"] == sha(PHASE / "Plan.json")
    assert lock["primary_completion_sha256"] == sha(FULL / "Completion_Receipt.json")
    assert lock["parent_analysis_sha256"] == sha(FULL / "analyze_assessment.py")
    assert lock["tests_sha256"] == sha(HERE / lock["tests_path"])
    plan, population, records, inputs, manifest = collect()
    models = plan["model_order"]["vi"]
    sources, values, strata, exported = build_tables(population, records, models)
    bootstrap = lock["bootstrap"]
    assert bootstrap == read(FULL / "Plan.json")["bootstrap"]
    generator = np.random.default_rng(bootstrap["seed"])
    weights = generator.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), bootstrap["draws"])
    result = describe_source_counts(values, models, sources, weights)
    receipt = read(FULL / "Completion_Receipt.json")
    summary_path = FULL / "Assessment_Analysis_01/Summary.json"
    assert receipt["summary_sha256"] == sha(summary_path)
    selected = read(summary_path)
    comparison = {model: {"selected_decoder_wer": selected["models"][model]["wer"],
                          "fixed_decoder_wer": result["models"][model]["wer"]} for model in models}
    inputs[summary_path.relative_to(ROOT).as_posix()] = sha(summary_path)
    result.update({"utc": utc(), "status": "complete_vi_fixed_decoder_matrix", "clips_per_model": 1167,
                   "reference_units": 30091, "models_order": models, "source_order": sources,
                   "source_counts": values.tolist(), "independent_count_checks": 7002,
                   "source_group_is_verified_speaker": False, "new_hidden_test": False,
                   "decoder": plan["decoder"], "bootstrap": bootstrap,
                   "selected_and_fixed_columns": comparison,
                   "strata_secondary_counts": {name: item.tolist() for name, item in strata.items()},
                   "input_sha256": inputs, "analysis_lock_sha256": sha(HERE / "Analysis_Lock.json")})
    output = HERE / "Results_01"
    output.mkdir(exist_ok=False)
    save(output / "Summary.json", result)
    save(output / "Task_Manifest.json", manifest)
    with (output / "Per_Clip_Counts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, "Evidence changed during analysis"
    save(PHASE / "Completion_Receipt.json", {"utc": utc(), "status": "complete_vi_anchor_verified",
         "models": models, "clips_per_model": 1167, "independent_count_checks": 7002,
         "summary_sha256": sha(output / "Summary.json"), "per_clip_sha256": sha(output / "Per_Clip_Counts.csv"),
         "analysis_lock_sha256": sha(HERE / "Analysis_Lock.json"), "paper_changed": False})
    print({"status": result["status"], "pairs": result["pairs"]}, flush=True)


if __name__ == "__main__":
    main()
