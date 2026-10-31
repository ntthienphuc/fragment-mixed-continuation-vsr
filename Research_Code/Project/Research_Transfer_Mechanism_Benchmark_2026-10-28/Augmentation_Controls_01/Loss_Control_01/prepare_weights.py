"""Build an exact training-only scalar schedule without importing torch or reading assessment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPLAY = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"
READY = HERE.parent / "Readiness_02"
sys.path.insert(0, str(ROOT / "Research_Phase_05_2026-10-13"))
from phase5.text import Tokenizer


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def expected(dataset):
    if dataset == "vlr":
        source = ROOT / "Research_Transfer_Training_Exposure_2026-10-27/Replay_03/Summary.json"
        return source, read(source)["seeds"]["17"]["final_exposure"]
    source = ROOT / "Research_English_Training_Scope_2026-10-28/Replay_01/Summary.json"
    return source, read(source)["totals"]


def build(dataset, parent, ready_proof, archive, output):
    trace_path = REPLAY / (dataset + "_trace.jsonl.gz")
    short_path = READY / (dataset + "_short_slots.jsonl.gz")
    assert sha(trace_path) == parent["trace_sha256"]
    assert sha(short_path) == ready_proof["schedule_sha256"]
    needed = {row["clip_id"] for row in records(trace_path)}
    assert len(needed) == parent["unique_clips_seen"]
    directory = ("Research_Transfer_Full_Benchmark_2026-10-27" if dataset == "vlr"
                 else "Research_English_Transfer_Benchmark_2026-10-28")
    registry_path = ROOT / directory / "model_registry.json"
    prefix = "VLR" if dataset == "vlr" else "EN"
    model = read(registry_path)["models"][prefix + "_B0"]
    for key in ["tokenizer", "units"]:
        assert sha(ROOT / model[key]) == model[key + "_sha256"]
    language = "vi" if dataset == "vlr" else "en"
    tokenizer = Tokenizer(ROOT / model["tokenizer"], ROOT / model["units"], language)
    metadata = {}
    manifest_digest = hashlib.sha256()
    with archive.open(f"Phase05/prepared/{dataset}/train/manifest.jsonl") as stream:
        for raw in stream:
            manifest_digest.update(raw)
            row = json.loads(raw)
            assert row["split"] == "train"
            if row["clip_id"] in needed:
                assert row["train_eligible"] and row["full_eligible"]
                assert row["clip_id"] not in metadata
                ids = tokenizer.encode(row["text"])
                metadata[row["clip_id"]] = {"tokens": len(ids), "frames": row["frames"],
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
    assert set(metadata) == needed and manifest_digest.hexdigest() == parent["manifest_sha256"]
    short = iter(records(short_path))
    totals = Counter()
    odd_weights = []
    schedule = output / (dataset + "_weights.jsonl.gz")
    with gzip.open(schedule, "xt", encoding="utf-8") as destination:
        for index, row in enumerate(records(trace_path)):
            assert row["sample"] == index and row["full"] is (index % 2 == 0)
            full = metadata[row["clip_id"]]
            assert full["frames"] == row["clip_frames"]
            crop_tokens = full["tokens"]
            target_sha = row["target_sha256"]
            if row["full"]:
                assert target_sha == full["text_sha256"]
            if not row["full"]:
                crop = next(short)
                assert crop["sample"] == index and crop["clip_id"] == row["clip_id"]
                assert crop["original_target_sha256"] == target_sha
                crop_tokens = len(crop["original_native_tokens"])
            numerator = crop_tokens + 1
            denominator = full["tokens"] + 1
            assert numerator > 1 and denominator > 1
            if not row["full"]:
                odd_weights.append(numerator / denominator)
                totals["odd_ratios_above_one"] += numerator > denominator
                totals["odd_ratios_equal_one"] += numerator == denominator
            totals.update(presentations=1, full_target_tokens=full["tokens"],
                          t1_target_tokens=crop_tokens, full_attention_positions=denominator,
                          t1_attention_positions=numerator, full_video_frames=full["frames"])
            totals["original_full_slots" if row["full"] else "original_crop_slots"] += 1
            item = {"sample": index, "clip_id": row["clip_id"], "split": "train", "seed": 17,
                    "original_full_slot": row["full"], "input_start": 0, "input_stop": full["frames"],
                    "full_target_sha256": full["text_sha256"], "t1_target_sha256": target_sha,
                    "full_native_tokens": full["tokens"], "t1_native_tokens": crop_tokens,
                    "weight_numerator": numerator, "weight_denominator": denominator}
            destination.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    assert next(short, None) is None
    source, prior = expected(dataset)
    assert totals["presentations"] == 48000
    assert totals["original_full_slots"] == totals["original_crop_slots"] == 24000
    for side, method in [("full", "T0"), ("t1", "T1")]:
        for field in ["target_tokens", "attention_positions"]:
            assert totals[side + "_" + field] == prior[method][field]
    assert totals["full_video_frames"] == prior["T0"]["input_frames"]
    return {"dataset": dataset, "totals": dict(totals), "unique_original_videos": len(needed),
            "odd_weight_mean": statistics.mean(odd_weights), "odd_weight_median": statistics.median(odd_weights),
            "odd_weight_minimum": min(odd_weights), "odd_weight_maximum": max(odd_weights),
            "schedule_sha256": sha(schedule), "trace_sha256": sha(trace_path),
            "original_short_schedule_sha256": sha(short_path), "source_manifest_sha256": manifest_digest.hexdigest(),
            "expected_exposure_sha256": sha(source), "registry_sha256": sha(registry_path),
            "tokenizer_sha256": model["tokenizer_sha256"], "units_sha256": model["units_sha256"]}


def main(attempt):
    assert attempt.isidentifier() and attempt.startswith("Preparation_")
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    parent = read(REPLAY / "Summary.json")
    proof = read(REPLAY / "Verification.json")
    assert proof["status"] == "passed" and proof["replay_sha256"] == sha(REPLAY / "Summary.json")
    ready = read(READY / "Verification.json")
    assert ready["status"] == "complete_training_schedule_verified"
    assert ready["summary_sha256"] == sha(READY / "Summary.json")
    ready_by_dataset = {item["dataset"]: item for item in ready["datasets"]}
    results = []
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        for item in parent["datasets"]:
            result = build(item["dataset"], item, ready_by_dataset[item["dataset"]], archive, output)
            results.append(result)
            print(json.dumps(result), flush=True)
    save(output / "Summary.json", {"utc": datetime.now(timezone.utc).isoformat(),
         "status": "weights_prepared_pending_independent_verification", "datasets": results,
         "protocol_sha256": sha(HERE / "Protocol.md"), "script_sha256": sha(Path(__file__)),
         "training_admitted": False, "assessment_predictions_read": False, "gpu_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="Preparation_01")
    main(parser.parse_args().attempt)
