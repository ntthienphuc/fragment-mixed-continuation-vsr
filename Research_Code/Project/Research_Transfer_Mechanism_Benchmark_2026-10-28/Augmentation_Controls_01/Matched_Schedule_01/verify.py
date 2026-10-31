"""Rebuild targets from original training metadata and recount the shared comparison."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
CONTROLS = HERE.parent
ROOT = HERE.parents[2]
WORD = CONTROLS / "Bounded_Word_Control_01"
LOSS = CONTROLS / "Loss_Control_01/Preparation_01"
REPLAY = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def target(text, start, stop, encode):
    tokens = encode(text)
    minimum = len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))
    assert minimum <= stop - start
    return {"start": start, "stop": stop, "target_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "native_token_count": len(tokens)}


def reconstruct(dataset, original, archive):
    encoder_path = CONTROLS / "Loss_Control_01/verify_weights.py"
    original_receipt = read(LOSS / "Verification.json")
    assert sha(encoder_path) == original_receipt["script_sha256"]
    spec = importlib.util.spec_from_file_location("matched_native_tokenizer", encoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encode = module.native_encoder(dataset, original)
    trace_path = REPLAY / (dataset + "_trace.jsonl.gz")
    assert sha(trace_path) == original["trace_sha256"]
    trace = list(records(trace_path))
    assert len(trace) == 48000
    slots = defaultdict(list)
    for position, row in enumerate(trace):
        assert row["sample"] == position and row["full"] is (position % 2 == 0)
        slots[row["clip_id"]].append(position)
    assert len(slots) == original["unique_original_videos"]
    spans = {row["sample"]: row for row in records(WORD / (dataset + "_short_slots.jsonl.gz"))}
    assert set(spans) == set(range(1, 48000, 2))
    expected = {}
    manifest_hash = hashlib.sha256()
    seen = set()
    with archive.open(f"Phase05/prepared/{dataset}/train/manifest.jsonl") as stream:
        for raw in stream:
            manifest_hash.update(raw)
            source = json.loads(raw)
            assert source["split"] == "train"
            clip_id = source["clip_id"]
            if clip_id not in slots:
                continue
            assert clip_id not in seen and source["train_eligible"] and source["full_eligible"]
            seen.add(clip_id)
            full = target(source["text"], 0, source["frames"], encode)
            for position in slots[clip_id]:
                archived = trace[position]
                assert archived["clip_frames"] == source["frames"]
                fallback = False
                fixed = dict(full)
                word = dict(full)
                if position % 2:
                    span = spans[position]
                    assert span["clip_id"] == clip_id and span["sample"] == position
                    start, stop = archived["core_start"], archived["core_stop"]
                    assert stop - start == 50
                    pieces = []
                    for item in source["words"]:
                        left, right = item["start"], item["end"]
                        intersects = left < right and right > start / 25 and left < stop / 25
                        point_inside = left == right and start / 25 <= left < stop / 25
                        if intersects or point_inside:
                            pieces.append(item["text"])
                    fixed = target(" ".join(pieces).strip(), start, stop, encode)
                    assert fixed["target_sha256"] == archived["target_sha256"]
                    fallback = span["view"] == "shared_full_fallback"
                    if fallback:
                        fixed = dict(full)
                    else:
                        assert span["view"] == "word_aligned"
                        words = source["words"][span["first_word"]:span["last_word"] + 1]
                        assert words and all(item["end"] > item["start"] for item in words)
                        start_word = math.floor(words[0]["start"] * 25 + 1e-8)
                        stop_word = math.ceil(max(item["end"] for item in words) * 25 - 1e-8)
                        assert 0 <= start_word < stop_word <= source["frames"]
                        assert 50 <= stop_word - start_word <= 82
                        word = target(" ".join(item["text"] for item in words).strip(),
                                      start_word, stop_word, encode)
                        assert start_word == span["start"] and stop_word == span["stop"]
                else:
                    assert full["target_sha256"] == archived["target_sha256"]
                expected[position] = (clip_id, fallback, {"full": full, "fixed_time": fixed, "whole_word": word})
    assert seen == set(slots) and set(expected) == set(range(48000))
    assert manifest_hash.hexdigest() == original["source_manifest_sha256"]
    return expected


def check_dataset(item, original, archive):
    dataset = item["dataset"]
    output = HERE / (dataset + ".jsonl.gz")
    assert sha(output) == item["schedule_sha256"]
    assert sha(LOSS / (dataset + "_weights.jsonl.gz")) == item["old_loss_schedule_sha256"]
    assert sha(WORD / (dataset + "_short_slots.jsonl.gz")) == item["bounded_schedule_sha256"]
    expected = reconstruct(dataset, original, archive)
    exposure = {name: Counter() for name in ("full", "fixed_time", "whole_word")}
    counts = Counter()
    fallbacks = []
    weighted_positions = Fraction(0)
    for position, pair in enumerate(itertools.zip_longest(records(output), range(48000))):
        row, marker = pair
        assert row is not None and marker is not None
        clip_id, fallback, views = expected[position]
        assert row["sample"] == position and row["clip_id"] == clip_id
        assert row["split"] == "train" and row["seed"] == 17
        assert row["original_full_slot"] is (position % 2 == 0)
        assert row["shared_full_fallback"] is fallback and row["views"] == views
        numerator = row["weighted_full"]["numerator"]
        denominator = row["weighted_full"]["denominator"]
        assert type(numerator) is int and type(denominator) is int
        assert numerator == views["fixed_time"]["native_token_count"] + 1
        assert denominator == views["full"]["native_token_count"] + 1
        if fallback or position % 2 == 0:
            assert numerator == denominator and len({json.dumps(view) for view in views.values()}) == 1
        if fallback:
            fallbacks.append(position)
        weighted_positions += Fraction(numerator, denominator) * denominator
        for name, view in views.items():
            exposure[name].update(presentations=1, input_frames=view["stop"] - view["start"],
                                  target_tokens=view["native_token_count"],
                                  attention_positions=view["native_token_count"] + 1)
        counts.update(presentations=1, shared_full_fallbacks=fallback,
                      original_full_slots=position % 2 == 0, original_short_slots=position % 2 == 1,
                      ratios_above_one=numerator > denominator, unit_ratios=numerator == denominator)
    assert dict(counts) == item["counts"] and exposure == item["exposure"]
    assert fallbacks == item["shared_fallback_samples"]
    assert weighted_positions.denominator == 1
    assert weighted_positions.numerator == exposure["fixed_time"]["attention_positions"]
    assert exposure["full"]["target_tokens"] == original["totals"]["full_target_tokens"]
    assert exposure["full"]["input_frames"] == original["totals"]["full_video_frames"]
    return {"dataset": dataset, "positions_checked": counts["presentations"],
            "fallbacks_checked": len(fallbacks), "weighted_unit_attention_sum": weighted_positions.numerator,
            "all_targets_rebuilt_from_training_manifest": True, "all_ctc_lengths_feasible": True,
            "schedule_sha256": sha(output)}


def main():
    summary = read(HERE / "Summary.json")
    assert summary["status"] == "matched_schedule_prepared" and summary["weighted_full_input_view"] == "full"
    for name, digest in summary["input_and_code_sha256"].items():
        assert sha(CONTROLS / name) == digest
    original = {item["dataset"]: item for item in read(LOSS / "Summary.json")["datasets"]}
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        checks = [check_dataset(item, original[item["dataset"]], archive) for item in summary["datasets"]]
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "matched_training_schedules_verified",
               "datasets": checks, "summary_sha256": sha(HERE / "Summary.json"),
               "verifier_sha256": sha(Path(__file__)), "training_launched": False,
               "assessment_predictions_read": False, "gpu_used": False}
    with (HERE / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
