"""Independently rebuild full/crop targets from train word timings and check every scalar."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

import sentencepiece

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPLAY = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def native_encoder(dataset, proof):
    directory = ("Research_Transfer_Full_Benchmark_2026-10-27" if dataset == "vlr"
                 else "Research_English_Transfer_Benchmark_2026-10-28")
    registry = ROOT / directory / "model_registry.json"
    assert sha(registry) == proof["registry_sha256"]
    name = "VLR_B0" if dataset == "vlr" else "EN_B0"
    model = read(registry)["models"][name]
    for key in ["tokenizer", "units"]:
        assert sha(ROOT / model[key]) == proof[key + "_sha256"]
    processor = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / model["tokenizer"]))
    units = {}
    for line in (ROOT / model["units"]).read_text(encoding="utf-8").splitlines():
        token, identifier = line.rsplit(maxsplit=1)
        units[token] = int(identifier)
    assert sorted(units.values()) == list(range(1, len(units) + 1))

    def encode(text):
        normalized = " ".join(unicodedata.normalize("NFC", text).lower().split())
        if dataset == "vlr":
            normalized = "".join(char if unicodedata.category(char)[0] in "LNM" else " "
                                 for char in normalized)
            normalized = " ".join(normalized.split())
        else:
            normalized = normalized.upper()
        pieces = processor.encode(normalized, out_type=int)
        identifiers = [units[processor.id_to_piece(piece)] for piece in pieces]
        assert identifiers and all(0 < token <= len(units) for token in identifiers)
        return identifiers

    return encode


def rebuild(dataset, proof, archive):
    trace_path = REPLAY / (dataset + "_trace.jsonl.gz")
    assert sha(trace_path) == proof["trace_sha256"]
    trace = []
    crops = defaultdict(list)
    for row in rows(trace_path):
        position = row["sample"]
        assert position == len(trace) and row["full"] is (position % 2 == 0)
        trace.append((row["clip_id"], row["clip_frames"], row["target_sha256"]))
        if not row["full"]:
            crops[row["clip_id"]].append((position, row["core_start"], row["core_stop"]))
    needed = {item[0] for item in trace}
    assert len(trace) == 48000 and len(needed) == proof["unique_original_videos"]
    encode = native_encoder(dataset, proof)
    full = {}
    short = {}
    checksum = hashlib.sha256()
    with archive.open(f"Phase05/prepared/{dataset}/train/manifest.jsonl") as stream:
        for raw in stream:
            checksum.update(raw)
            item = json.loads(raw)
            assert item["split"] == "train"
            if item["clip_id"] not in needed:
                continue
            assert item["train_eligible"] and item["full_eligible"]
            assert item["clip_id"] not in full
            identifiers = encode(item["text"])
            minimum = len(identifiers) + sum(a == b for a, b in zip(identifiers, identifiers[1:]))
            assert minimum <= item["frames"]
            full[item["clip_id"]] = (len(identifiers), item["frames"],
                                      hashlib.sha256(item["text"].encode()).hexdigest())
            for position, start, stop in crops[item["clip_id"]]:
                words = []
                for word in item["words"]:
                    left, right = word["start"], word["end"]
                    overlap = left < right and right > start / 25 and left < stop / 25
                    point = left == right and start / 25 <= left < stop / 25
                    if overlap or point:
                        words.append(word["text"])
                text = " ".join(words).strip()
                digest = hashlib.sha256(text.encode()).hexdigest()
                assert digest == trace[position][2]
                identifiers = encode(text)
                minimum = len(identifiers) + sum(a == b for a, b in zip(identifiers, identifiers[1:]))
                assert minimum <= stop - start
                short[position] = len(identifiers)
    assert checksum.hexdigest() == proof["source_manifest_sha256"]
    assert set(full) == needed and len(short) == 24000
    return trace, full, short


def verify_dataset(proof, output, archive):
    dataset = proof["dataset"]
    trace, full, short = rebuild(dataset, proof, archive)
    schedule = output / (dataset + "_weights.jsonl.gz")
    assert sha(schedule) == proof["schedule_sha256"]
    totals = Counter()
    weighted_positions = Fraction(0)
    count = 0
    for count, row in enumerate(rows(schedule), 1):
        index = count - 1
        original, frame_count, target_sha = trace[index]
        tokens, frames, full_sha = full[original]
        assert frames == frame_count
        crop_tokens = short[index] if index % 2 else tokens
        assert row["sample"] == index and row["clip_id"] == original
        assert row["split"] == "train" and row["seed"] == 17
        assert row["original_full_slot"] is (index % 2 == 0)
        assert row["input_start"] == 0 and row["input_stop"] == frames
        assert row["full_target_sha256"] == full_sha and row["t1_target_sha256"] == target_sha
        if index % 2 == 0:
            assert target_sha == full_sha
        assert row["full_native_tokens"] == tokens and row["t1_native_tokens"] == crop_tokens
        numerator, denominator = row["weight_numerator"], row["weight_denominator"]
        assert type(numerator) is int and type(denominator) is int
        assert numerator == crop_tokens + 1 and denominator == tokens + 1
        exact = Fraction(numerator, denominator) * (tokens + 1)
        assert exact == crop_tokens + 1
        weighted_positions += exact
        totals.update(presentations=1, full_target_tokens=tokens, t1_target_tokens=crop_tokens,
                      full_attention_positions=tokens + 1, t1_attention_positions=crop_tokens + 1,
                      full_video_frames=frames)
        totals["original_crop_slots" if index % 2 else "original_full_slots"] += 1
        if index % 2:
            totals["odd_ratios_above_one"] += numerator > denominator
            totals["odd_ratios_equal_one"] += numerator == denominator
    assert count == 48000 and dict(totals) == proof["totals"]
    assert weighted_positions.denominator == 1
    assert weighted_positions.numerator == totals["t1_attention_positions"]
    return {"dataset": dataset, "positions_independently_checked": count,
            "full_targets_checked": len(full), "crop_targets_rebuilt_from_word_timings": len(short),
            "weighted_unit_attention_sum": weighted_positions.numerator,
            "all_input_intervals_are_full_video": True, "schedule_sha256": sha(schedule)}


def main(attempt):
    assert attempt.isidentifier()
    output = HERE / attempt
    summary_path = output / "Summary.json"
    summary = read(summary_path)
    assert summary["status"] == "weights_prepared_pending_independent_verification"
    assert summary["protocol_sha256"] == sha(HERE / "Protocol.md")
    assert summary["script_sha256"] == sha(HERE / "prepare_weights.py")
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        checks = [verify_dataset(item, output, archive) for item in summary["datasets"]]
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "status": "all_original_targets_and_exact_weight_positions_verified", "datasets": checks,
              "summary_sha256": sha(summary_path), "script_sha256": sha(Path(__file__)),
              "real_gradients_verified": False, "training_admitted": False,
              "assessment_predictions_read": False, "gpu_used": False}
    with (output / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="Preparation_01")
    main(parser.parse_args().attempt)
