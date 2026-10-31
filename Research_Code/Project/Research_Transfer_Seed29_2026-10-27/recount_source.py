"""Recount the archived, fixed-greedy seed-29 source-domain comparison."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "Research_Seed_Replication_2026-10-15"
COUNT_CODE = ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py"
sys.path.insert(0, str(COUNT_CODE.parent))
from verify_probe import counts, normalized

FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def main():
    output = HERE / "Source_Recount_01"
    output.mkdir(exist_ok=False)
    receipt = read(SOURCE / "Results_Readback_vlr_29.json")
    archive_path = SOURCE / "Results_vlr_29.zip"
    assert digest(archive_path) == receipt["sha256"]
    manifest_path = ROOT / "Research_Server_Confirmation_2026-10-19/manifests/VLR_development.jsonl"
    manifest = rows(manifest_path)
    references = {row["clip_id"]: row for row in manifest}
    assert len(references) == len(manifest) == 11578
    source_ids = sorted({row["source_id"] for row in manifest})
    source_index = {source: index for index, source in enumerate(source_ids)}
    values = np.zeros((2, len(source_ids), len(FIELDS)), dtype=np.int64)
    file_hashes = {}
    checked = 0
    models = read(HERE / "Input_Check_01/Models.json")
    with zipfile.ZipFile(archive_path) as archive:
        for model_index, method in enumerate(("T0", "T1")):
            directory = SOURCE / "evaluation" / ("vlr_" + method + "_29_full")
            model = next(row for row in models if row["method"] == method and row["variant"] == "best")
            for name in ("predictions.jsonl", "summary.json"):
                path = directory / name
                matches = [member for member in archive.namelist()
                           if member.endswith("evaluation/" + directory.name + "/" + name)]
                assert len(matches) == 1
                actual = digest(path)
                assert hashlib.sha256(archive.read(matches[0])).hexdigest() == actual
                file_hashes[path.relative_to(ROOT).as_posix()] = actual
            summary = read(directory / "summary.json")
            assert summary["decoder"] == "greedy" and summary["view"] == "full"
            assert summary["checkpoint_sha256"] == model["original_sha256"]
            assert summary["checkpoint_selected_step"] == model["step"]
            assert summary["subset_limit"] == summary["inference_failed"] == 0
            predictions = rows(directory / "predictions.jsonl")
            assert len(predictions) == len(references)
            assert len({row["clip_id"] for row in predictions}) == len(references)
            assert {row["clip_id"] for row in predictions} == set(references)
            for row in predictions:
                expected = references[row["clip_id"]]
                assert row["source_id"] == expected["source_id"]
                assert normalized(row["reference"]) == normalized(expected["reference"])
                assert row["input_frames"] == expected["frames"]
                assert row["method"] == method and row["status"] == "ok"
                assert row["decoder"] == "greedy" and row["view"] == "full"
                measured = counts(row["reference"], row["text"])
                measured.pop("metric")
                assert measured == row["scores"]["word"], row["clip_id"]
                values[model_index, source_index[row["source_id"]]] += [measured[key] for key in FIELDS]
                checked += 1
            total = values[model_index].sum(axis=0)
            assert all(total[index] == summary["metrics"]["word"][field]
                       for index, field in enumerate(FIELDS))
            print(json.dumps({"method": method, "independent_predictions_recounted": len(predictions)}),
                  flush=True)
    assert np.array_equal(values[0, :, 4], values[1, :, 4])
    random = np.random.default_rng(20261027)
    samples = random.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)), size=10000)
    units = samples @ values[0, :, 4]
    draws = 100 * (samples @ (values[0, :, 0] - values[1, :, 0])) / units
    for index, weights in enumerate(samples[:101]):
        error_delta = sum(int(weight) * int(left - right)
                          for weight, left, right in zip(weights, values[0, :, 0], values[1, :, 0]))
        word_total = sum(int(weight) * int(count) for weight, count in zip(weights, values[0, :, 4]))
        assert abs(draws[index] - 100 * error_delta / word_total) < 1e-12
    totals = values.sum(axis=1)
    rates = 100 * totals[:, 0] / totals[:, 4]
    summary = {
        "status": "archived_source_predictions_independently_recounted",
        "utc": datetime.now(timezone.utc).isoformat(), "dataset": "VLR", "training_seed": 29,
        "clips": len(references), "source_groups": len(source_ids), "decoder": "fixed_greedy",
        "view": "full_clip", "selected_steps": {"T0": 5000, "T1": 5500},
        "wer_percent": dict(zip(("T0", "T1"), rates.tolist())),
        "T0_minus_T1_pp": float(rates[0] - rates[1]),
        "paired_source_ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
        "counts": {method: dict(zip(FIELDS, total.tolist()))
                   for method, total in zip(("T0", "T1"), totals)},
        "independent_prediction_checks": checked, "scalar_bootstrap_checks": 101,
        "bootstrap_draws": 10000, "bootstrap_seed": 20261027,
        "source_groups_are_not_independently_verified_speakers": True,
        "input_hashes": file_hashes, "source_archive_sha256": receipt["sha256"],
        "manifest_sha256": digest(manifest_path), "code_sha256": digest(Path(__file__)),
        "independent_count_code_sha256": digest(COUNT_CODE),
        "new_gpu_inference": False, "destination_predictions_read": False,
        "decoder_calibration_comparable_to_destination_matrix": False,
        "equal_checkpoint_steps": False, "retrospective_reanalysis": True,
    }
    save(output / "Summary.json", summary)
    save(output / "Source_Counts.json", {source: values[:, index].tolist()
                                         for index, source in enumerate(source_ids)})
    print(json.dumps({key: summary[key] for key in ("status", "wer_percent", "T0_minus_T1_pp",
                                                   "paired_source_ci95_pp")}), flush=True)


if __name__ == "__main__":
    main()
