"""Separate archive/trace verification without importing the reconstruction code."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import unicodedata
import zipfile

import sentencepiece

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(attempt):
    output = HERE / attempt
    summary_path = output / "Summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["code_sha256"] == digest(HERE / "reconstruct.py")
    assert summary["protocol_sha256"] == digest(HERE / "Protocol.md")
    trace_path = output / "Trace.jsonl.gz"
    assert digest(trace_path) == summary["trace_sha256"]
    source = ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip"
    manifest_hash = hashlib.sha256()
    all_rows = []
    with zipfile.ZipFile(source) as archive:
        with archive.open("Phase05/prepared/lrs2/train/manifest.jsonl") as stream:
            for line in stream:
                manifest_hash.update(line)
                all_rows.append(json.loads(line))
    assert manifest_hash.hexdigest() == summary["training_receipt"]["manifest_sha256"]
    rows = [row for row in all_rows if row["train_eligible"]]
    observed = {"prepared": len(all_rows), "window_eligible": len(rows),
                "full_eligible": sum(bool(row["full_eligible"]) for row in all_rows),
                "shorter_than_82_frames": sum(row["frames"] < 82 for row in all_rows),
                "without_word_times": sum(len(row["words"]) == 0 for row in all_rows),
                "timing_invalid": sum(not row["timing_valid"] for row in all_rows)}
    for key, value in observed.items():
        assert summary["population"][key] == value, key
    observed["mean_prepared_seconds"] = sum(row["frames"] for row in all_rows) / (25 * len(all_rows))
    observed["mean_eligible_seconds"] = sum(row["frames"] for row in rows) / (25 * len(rows))
    observed["eligible_fraction"] = len(rows) / len(all_rows)
    for key, value in observed.items():
        assert abs(summary["population"][key] - value) < 1e-12, key
    flags = Counter(str((row["frames"] < 82, not bool(row["words"]), not row["timing_valid"]))
                    for row in all_rows)
    assert dict(flags) == summary["exclusion_flags_short_no_times_timing_invalid"]
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(ROOT / "Research_Phase_05_2026-10-13/assets/tokenizers/en.model"))
    totals = {method: Counter() for method in ("T0", "T1")}
    seen = Counter()
    sources = set()
    orders = {}
    for epoch in range(7):
        order = list(range(len(rows)))
        random.Random(17 + epoch).shuffle(order)
        orders[epoch] = order
    count = 0
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            record = json.loads(line)
            epoch, index = divmod(number, len(rows))
            row = rows[orders[epoch][index]]
            assert record["sample"] == number and record["epoch"] == epoch
            assert record["clip_id"] == row["clip_id"]
            seen[row["clip_id"]] += 1
            sources.add(row["source_id"])
            for method in ("T0", "T1"):
                full = method == "T0" or number % 2 == 0
                left, right = 0, row["frames"]
                text = row["text"]
                partial_left = False
                partial_right = False
                if not full:
                    identifier = "17|" + row["clip_id"] + "|" + str(number)
                    number_hash = int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16)
                    left = row["train_starts"][number_hash % len(row["train_starts"])]
                    right = left + 50
                    selected = []
                    for word in row["words"]:
                        start, stop = word["start"], word["end"]
                        lower, upper = left / 25, right / 25
                        overlaps = max(start, lower) < min(stop, upper)
                        point_inside = start == stop and lower <= start < upper
                        if overlaps or point_inside:
                            selected.append(word["text"])
                            partial_left |= start < lower < stop
                            partial_right |= start < upper < stop
                    text = " ".join(selected).strip()
                normalized = " ".join(unicodedata.normalize("NFC", text).lower().split()).upper()
                token_count = len(tokenizer.encode(normalized, out_type=int))
                expected = {"start": left, "stop": right, "samples": 1, "input_frames": right - left,
                            "target_tokens": token_count, "attention_positions": token_count + 1,
                            "target_words": len(normalized.split()), "full_samples": int(full),
                            "crop_samples": int(not full), "crop_partial_left": int(partial_left),
                            "crop_partial_right": int(partial_right),
                            "crop_partial_either": int(partial_left or partial_right)}
                assert record["views"][method] == expected, (number, method)
                totals[method].update({key: value for key, value in expected.items() if key not in ("start", "stop")})
            count += 1
    assert count == summary["sample_decisions"] == 48000
    assert summary["native_control_flow_checks"] == count * 2 == 96000
    assert len(seen) == summary["unique_clips_seen"] == 7291
    assert len(sources) == summary["unique_sources_seen"]
    histogram = {str(key): value for key, value in Counter(seen.values()).items()}
    assert histogram == summary["clip_exposure_histogram"]
    assert abs(count / len(rows) - summary["eligible_epoch_equivalents"]) < 1e-12
    for method in totals:
        assert dict(totals[method]) == summary["totals"][method], method
    for field, ratio in summary["T1_over_T0"].items():
        assert abs(totals["T1"][field] / totals["T0"][field] - ratio) < 1e-12
    verification = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
                    "summary_sha256": digest(summary_path), "verifier_sha256": digest(Path(__file__)),
                    "source_manifest_sha256": manifest_hash.hexdigest(), "sample_records": count,
                    "independent_view_checks": count * 2, "pixels_or_neural_calls": 0}
    with (output / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, indent=2)
        stream.write("\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    main(parser.parse_args().attempt)
