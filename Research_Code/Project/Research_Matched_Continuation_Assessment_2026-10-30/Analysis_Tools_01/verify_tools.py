"""Exercise aggregation on real historical counts and explicit non-scientific transport fixtures."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from collect_population import HERE, PHASE, ROOT, check_shard, collect, population_metadata
from common import read, save, sha, utc
from paired_statistics import FIELDS, paired, totals


def rejected(function):
    try:
        function()
    except (AssertionError, ValueError, FileNotFoundError, KeyError):
        return True
    raise RuntimeError("Invalid evidence was accepted")


def historical_statistics():
    folder = ROOT / "Research_Transfer_Full_Benchmark_2026-10-27/Assessment_Analysis_01"
    historical = read(folder / "Summary.json")
    records = {}
    with (folder / "Per_Clip_Counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            counts = {field: int(row[field]) for field in FIELDS}
            records.setdefault(row["model"], []).append({"clip_id": row["clip_id"],
                "source_id": row["source_id"], "predictions": {"historical": {"counts": counts}}})
    for model, predictions in records.items():
        result = totals(predictions, "historical")
        expected = historical["models"][model]
        assert all(result[field] == expected["counts"][field] for field in FIELDS)
        assert abs(result["wer_percent"] - expected["wer"]) < 1e-12
    left = records["VLR_T0_LAST"]
    right = records["VLR_T1_LAST"]
    result = paired(left, list(reversed(right)), "historical", "historical", "historical_regression")
    sources = result["source_order"]
    left_by_source = [sum(row["predictions"]["historical"]["counts"]["errors"]
                          for row in left if row["source_id"] == source) for source in sources]
    right_by_source = [sum(row["predictions"]["historical"]["counts"]["errors"]
                           for row in right if row["source_id"] == source) for source in sources]
    units = [sum(row["predictions"]["historical"]["counts"]["units"]
                 for row in left if row["source_id"] == source) for source in sources]
    random = np.random.default_rng(result["bootstrap_seed"])
    indices = random.integers(0, len(sources), size=(10000, len(sources)))
    weights = np.asarray([np.bincount(row, minlength=len(sources)) for row in indices], dtype=np.int64)
    differences = np.asarray(left_by_source) - np.asarray(right_by_source)
    draws = 100 * (weights @ differences) / (weights @ np.asarray(units))
    bounds = np.quantile(draws, [.025, .975]).tolist()
    assert bounds == [result["ci95_low_pp"], result["ci95_high_pp"]]
    assert abs(result["reference_minus_comparison_pp"] - historical["pairs"][
        "VLR_T0_LAST_minus_VLR_T1_LAST"]["gain_pp"]) < 1e-12
    assert rejected(lambda: paired(left[:-1], right, "historical", "historical", "bad"))
    altered = deepcopy(right)
    altered[0]["source_id"] = "incorrect_group"
    assert rejected(lambda: paired(left, altered, "historical", "historical", "bad"))
    altered = deepcopy(right)
    altered[0]["predictions"]["historical"]["counts"]["units"] += 1
    assert rejected(lambda: paired(left, altered, "historical", "historical", "bad"))
    return {"models_reproduced": list(records), "clips_per_model": len(left),
            "historical_counts_sha256": sha(folder / "Per_Clip_Counts.csv"),
            "historical_summary_sha256": sha(folder / "Summary.json"),
            "independent_group_bootstrap_draws": len(draws), "invalid_pairing_cases_rejected": 3}


def shard_transport(output):
    archive_path = PHASE / "Worker_Integration_Receipts_02.zip"
    with zipfile.ZipFile(archive_path) as archive:
        prefix = "Worker_Integration_02/Run/"
        documents = {name: json.loads(archive.read(prefix + name)) for name in
                     ("Start.json", "Complete.json", "Verification.json", "Identity.json")}
        record = json.loads(archive.read(prefix + "Predictions.jsonl"))
    dataset, _ = population_metadata("source")
    source = dataset.metadata(record["position"])
    model = record["model"]
    binding = output / "Synthetic_Binding"
    binding.mkdir()
    save(binding / "model_registry.json", {"models": {model: documents["Identity.json"]["registry_entry"]}})
    folder = output / "Synthetic_Transport_Only"
    folder.mkdir()
    save(folder / "Fixture_Label.json", {"scientific_result": False,
                                         "purpose": "Transport test derived from one verified CPU recovery clip"})

    class SingleClip:
        stage = "assessment"

        def __len__(self):
            return 1

        def metadata(self, position):
            assert position == 0
            return source

    for value in (documents["Start.json"], documents["Verification.json"]):
        value.update({"task": "assessment_00", "stage": "assessment", "mode": "population",
                      "registry_sha256": sha(binding / "model_registry.json")})
    documents["Complete.json"].update({"task": "assessment_00", "mode": "population"})
    documents["Identity.json"]["registry_sha256"] = sha(binding / "model_registry.json")
    record.update({"position": 0, "stage": "assessment", "mode": "population"})
    documents["Verification.json"]["replays"][0]["position"] = 0

    def write_case(values, prediction):
        for name in ("Start.json", "Identity.json"):
            (folder / name).write_text(json.dumps(values[name]), encoding="utf-8")
        (folder / "Predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        for name, field in (("Predictions.jsonl", "predictions_sha256"), ("Identity.json", "identity_sha256")):
            values["Complete.json"][field] = sha(folder / name)
            values["Verification.json"][field] = sha(folder / name)
        values["Complete.json"]["start_sha256"] = sha(folder / "Start.json")
        for name in ("Complete.json", "Verification.json"):
            (folder / name).write_text(json.dumps(values[name]), encoding="utf-8")

    def check():
        return check_shard(folder, "assessment_00", model, "assessment", "common", ["beam_c0.1"],
                           binding, None, SingleClip())

    write_case(deepcopy(documents), record)
    checked, _, _ = check()
    assert len(checked) == 1
    mutations = {
        "fixture_mode": ("Verification.json", "mode", "fixture"),
        "wrong_code": ("Start.json", "code_lock_sha256", "wrong"),
        "missing_replay": ("Verification.json", "replays", []),
        "wrong_count_total": ("Verification.json", "independent_count_checks", 0),
        "wrong_model": ("Identity.json", "model_id", "another_model"),
        "unverified": ("Verification.json", "status", "pending"),
    }
    for name, (file_name, field, value) in mutations.items():
        altered = deepcopy(documents)
        altered[file_name][field] = value
        write_case(altered, record)
        assert rejected(check), name
    wrong_counts = deepcopy(record)
    wrong_counts["predictions"]["beam_c0.1"]["counts"]["errors"] += 1
    write_case(deepcopy(documents), wrong_counts)
    assert rejected(check)
    write_case(deepcopy(documents), record)
    (folder / "Predictions.jsonl").write_text(json.dumps(record) + "\n ", encoding="utf-8")
    assert rejected(check)
    write_case(deepcopy(documents), record)
    assert rejected(lambda: collect("assessment", model, "common", binding, folder))
    return {"real_cpu_receipt_sha256": sha(archive_path), "positive_transport_cases": 1,
            "invalid_cases_rejected": [*mutations, "incorrect_edit_count", "modified_stream",
                                       "recovery_model_in_population"], "scientific_result": False}


def main(arguments):
    output = arguments.output.resolve()
    assert output.is_relative_to(HERE.resolve())
    output.mkdir(exist_ok=False)
    populations = {}
    for stage in ("source", "development", "assessment"):
        data, spec = population_metadata(stage)
        populations[stage] = {"clips": len(data), "units": spec["reference_units"],
                              "groups": spec["source_groups"]}
    result = {"utc": utc(), "status": "analysis_tools_regression_passed", "populations": populations,
              "statistics": historical_statistics(), "transport": shard_transport(output),
              "complete_new_model_population_tested": False, "new_recognition_result": False,
              "code_sha256": {path.name: sha(path) for path in HERE.glob("*.py")}}
    save(output / "Complete.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
