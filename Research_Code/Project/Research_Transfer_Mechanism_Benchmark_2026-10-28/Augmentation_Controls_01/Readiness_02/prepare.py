"""Prepare every training slot, with an explicit shared full-view fallback."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

from word_span import choose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPLAY = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"
PHASE = ROOT / "Research_Phase_05_2026-10-13"
sys.path.insert(0, str(PHASE))
from phase5.text import Tokenizer


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def distribution(values):
    ordered = sorted(values)
    return {"sum": sum(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": ordered[0], "maximum": ordered[-1], "p95": ordered[(95 * len(ordered) - 1) // 100]}


def build(dataset, parent, archive):
    path = REPLAY / (dataset + "_trace.jsonl.gz")
    assert sha(path) == parent["trace_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    assert len(trace) == 48000
    needed = {row["clip_id"] for row in trace if not row["full"]}
    metadata = {}
    checksum = hashlib.sha256()
    fields = ("clip_id", "frames", "words", "text", "frame_path", "frames_sha256")
    with archive.open(f"Phase05/prepared/{dataset}/train/manifest.jsonl") as stream:
        for raw in stream:
            checksum.update(raw)
            row = json.loads(raw)
            assert row["split"] == "train"
            if row["clip_id"] in needed:
                assert row["train_eligible"] and row["full_eligible"] and row["clip_id"] not in metadata
                metadata[row["clip_id"]] = {field: row[field] for field in fields}
    assert checksum.hexdigest() == parent["manifest_sha256"] and set(metadata) == needed
    language = "vi" if dataset == "vlr" else "en"
    registry_name = ("Research_Transfer_Full_Benchmark_2026-10-27" if dataset == "vlr"
                     else "Research_English_Transfer_Benchmark_2026-10-28")
    registry_path = ROOT / registry_name / "model_registry.json"
    name = "VLR_B0" if dataset == "vlr" else "EN_B0"
    model = read(registry_path)["models"][name]
    for field in ("tokenizer", "units"):
        assert sha(ROOT / model[field]) == model[field + "_sha256"]
    tokenizer = Tokenizer(ROOT / model["tokenizer"], ROOT / model["units"], language)
    counts = Counter()
    lengths = []
    token_lengths = []
    parent_tokens = []
    matched_tokens = []
    destination = HERE / (dataset + "_short_slots.jsonl.gz")
    with gzip.open(destination, "xt", encoding="utf-8") as output:
        for index, sample in enumerate(trace):
            assert sample["sample"] == index and sample["full"] == (index % 2 == 0)
            if sample["full"]:
                counts["unchanged_full_slots"] += 1
                counts["unchanged_full_frames"] += sample["clip_frames"]
                continue
            row = metadata[sample["clip_id"]]
            lower = sample["core_start"] / 25
            upper = sample["core_stop"] / 25
            old_text = " ".join(word["text"] for word in row["words"] if
                                (word["end"] > lower and word["start"] < upper
                                 and word["end"] > word["start"])
                                or (word["start"] == word["end"] and lower <= word["start"] < upper)).strip()
            assert hashlib.sha256(old_text.encode()).hexdigest() == sample["target_sha256"]
            old_ids = tokenizer.encode(old_text)
            selected = choose(row["words"], row["frames"], sample["core_start"], sample["core_stop"],
                              row["text"], tokenizer.encode)
            fallback = selected["view"] == "shared_full_fallback"
            matched = {"start": 0 if fallback else sample["core_start"],
                       "stop": row["frames"] if fallback else sample["core_stop"],
                       "text": row["text"] if fallback else old_text,
                       "native_tokens": selected["native_tokens"] if fallback else old_ids}
            length = selected["stop"] - selected["start"]
            counts[selected["view"]] += 1
            counts["short_slots"] += 1
            counts["same_input_interval_as_original_T1"] += (selected["start"], selected["stop"]) == (
                sample["core_start"], sample["core_stop"])
            counts["same_text_as_original_T1"] += selected["text"] == old_text
            counts["longer_than_75_frames"] += length > 75
            counts["ctc_candidates_rejected"] += selected["earlier_ctc_or_empty_candidates"]
            counts["matched_T1_short_slot_frames"] += matched["stop"] - matched["start"]
            lengths.append(length)
            token_lengths.append(len(selected["native_tokens"]))
            parent_tokens.append(len(old_ids))
            matched_tokens.append(len(matched["native_tokens"]))
            result = {"sample": index, "clip_id": row["clip_id"], "clip_frames": row["frames"],
                      "original_start": sample["core_start"], "original_stop": sample["core_stop"],
                      "original_target_sha256": sample["target_sha256"], "original_native_tokens": old_ids,
                      "frame_path": row["frame_path"], "frames_sha256": row["frames_sha256"],
                      "matched_T1": matched, **selected}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    assert counts["short_slots"] == counts["unchanged_full_slots"] == 24000
    assert counts["word_aligned"] + counts["shared_full_fallback"] == 24000
    return {"dataset": dataset, "counts": dict(counts), "input_frames_on_short_slots": distribution(lengths),
            "native_tokens_on_short_slots": distribution(token_lengths),
            "original_T1_tokens_on_same_slots": distribution(parent_tokens),
            "matched_T1_tokens_on_same_slots": distribution(matched_tokens),
            "schedule_sha256": sha(destination), "source_manifest_sha256": checksum.hexdigest(),
            "trace_sha256": sha(path), "registry_sha256": sha(registry_path),
            "tokenizer_sha256": model["tokenizer_sha256"], "units_sha256": model["units_sha256"]}


def main():
    assert not (HERE / "Summary.json").exists()
    parent = read(REPLAY / "Summary.json")
    proof = read(REPLAY / "Verification.json")
    assert proof["status"] == "passed" and proof["replay_sha256"] == sha(REPLAY / "Summary.json")
    results = []
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        for item in parent["datasets"]:
            result = build(item["dataset"], item, archive)
            results.append(result)
            print(json.dumps(result), flush=True)
    value = {"utc": datetime.now(timezone.utc).isoformat(), "status": "metadata_pending_independent_verification",
             "datasets": results, "training_admitted": False, "media_payloads_verified": False,
             "assessment_predictions_used": False, "protocol_sha256": sha(HERE / "Protocol.md"),
             "scripts": {name: sha(HERE / name) for name in ("prepare.py", "word_span.py")}}
    with (HERE / "Summary.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
