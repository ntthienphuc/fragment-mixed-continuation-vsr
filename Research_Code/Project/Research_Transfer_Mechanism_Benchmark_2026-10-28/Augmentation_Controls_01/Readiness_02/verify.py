"""Independently check real schedules and enumerate feasible alternatives on a fixed sample."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Research_Phase_05_2026-10-13"))
from phase5.text import Tokenizer


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_and_minimum(tokenizer, text):
    ids = tokenizer.encode(text)
    return ids, len(ids) + sum(ids[index] == ids[index - 1] for index in range(1, len(ids)))


def exhaustive(words, frames, lower, upper, tokenizer):
    valid = []
    for first in range(len(words)):
        for last in range(first, len(words)):
            segment = words[first:last + 1]
            if any(word["end"] <= word["start"] for word in segment):
                continue
            beginning = segment[0]["start"]
            ending = max(word["end"] for word in segment)
            if beginning >= upper / 25 or ending <= lower / 25:
                continue
            start = math.floor(beginning * 25 + 1e-8)
            stop = math.ceil(ending * 25 - 1e-8)
            if not 0 <= start < stop <= frames:
                continue
            text = " ".join(word["text"] for word in segment).strip()
            try:
                _, minimum = ids_and_minimum(tokenizer, text)
            except ValueError:
                continue
            if minimum > stop - start:
                continue
            priority = (abs(start - lower) + abs(stop - upper), abs(stop - start - 50),
                        abs(start + stop - lower - upper) / 2, start, first, last)
            valid.append(priority)
    return valid


def check_dataset(item, archive):
    dataset = item["dataset"]
    language = "vi" if dataset == "vlr" else "en"
    assets = ROOT / "Research_Phase_05_2026-10-13/assets/tokenizers"
    assert sha(assets / (language + ".model")) == item["tokenizer_sha256"]
    assert sha(assets / (language + "_units.txt")) == item["units_sha256"]
    tokenizer = Tokenizer(assets / (language + ".model"), assets / (language + "_units.txt"), language)
    path = HERE / (dataset + "_short_slots.jsonl.gz")
    assert sha(path) == item["schedule_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        schedule = [json.loads(line) for line in stream]
    assert [row["sample"] for row in schedule] == list(range(1, 48000, 2))
    wanted = {row["clip_id"] for row in schedule}
    metadata = {}
    digest = hashlib.sha256()
    with archive.open(f"Phase05/prepared/{dataset}/train/manifest.jsonl") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            assert row["split"] == "train"
            if row["clip_id"] in wanted:
                metadata[row["clip_id"]] = row
    assert digest.hexdigest() == item["source_manifest_sha256"] and set(metadata) == wanted
    counts = Counter()
    examples = []
    for record in schedule:
        original = metadata[record["clip_id"]]
        words = original["words"]
        assert original["frames"] == record["clip_frames"]
        assert original["frame_path"] == record["frame_path"]
        assert original["frames_sha256"] == record["frames_sha256"]
        lower = record["original_start"]
        upper = record["original_stop"]
        old_text = " ".join(word["text"] for word in words if
                            (word["end"] > lower / 25 and word["start"] < upper / 25
                             and word["end"] > word["start"])
                            or (word["start"] == word["end"] and lower / 25 <= word["start"] < upper / 25)).strip()
        assert hashlib.sha256(old_text.encode()).hexdigest() == record["original_target_sha256"]
        old_ids, _ = ids_and_minimum(tokenizer, old_text)
        assert old_ids == record["original_native_tokens"]
        start = record["start"]
        stop = record["stop"]
        assert 0 <= start < stop <= original["frames"]
        ids, minimum = ids_and_minimum(tokenizer, record["text"])
        assert ids == record["native_tokens"] and minimum == record["minimum_ctc_frames"] <= stop - start
        fallback = record["view"] == "shared_full_fallback"
        expected_matched = {"start": 0 if fallback else lower,
                            "stop": original["frames"] if fallback else upper,
                            "text": original["text"] if fallback else old_text,
                            "native_tokens": ids if fallback else old_ids}
        assert record["matched_T1"] == expected_matched
        if fallback:
            assert (start, stop, record["text"]) == (0, original["frames"], original["text"])
            examples.append({"clip_id": record["clip_id"], "sample": record["sample"]})
        else:
            assert record["view"] == "word_aligned"
            selected = words[record["first_word"]:record["last_word"] + 1]
            assert selected and all(word["end"] > word["start"] for word in selected)
            assert record["text"] == " ".join(word["text"] for word in selected).strip()
            assert start == math.floor(selected[0]["start"] * 25 + 1e-8)
            assert stop == math.ceil(max(word["end"] for word in selected) * 25 - 1e-8)
            counts["one_frame_word_spans"] += stop - start == 1
            counts["word_spans_below_25_frames"] += stop - start < 25
        counts[record["view"]] += 1
        counts["frames"] += stop - start
        counts["tokens"] += len(ids)
        counts["matched_T1_tokens"] += len(expected_matched["native_tokens"])
        counts["matched_T1_frames"] += expected_matched["stop"] - expected_matched["start"]
        counts["original_T1_tokens"] += len(old_ids)
        if record["sample"] % 127 == 1 or fallback:
            alternatives = exhaustive(words, original["frames"], lower, upper, tokenizer)
            if fallback:
                assert alternatives == []
            else:
                assert tuple(record["priority"]) == min(alternatives)
            counts["exhaustive_checks"] += 1
    assert counts["word_aligned"] + counts["shared_full_fallback"] == 24000
    for field in ("word_aligned", "shared_full_fallback"):
        assert counts[field] == item["counts"].get(field, 0)
    assert counts["frames"] == item["input_frames_on_short_slots"]["sum"]
    assert counts["tokens"] == item["native_tokens_on_short_slots"]["sum"]
    assert counts["matched_T1_tokens"] == item["matched_T1_tokens_on_same_slots"]["sum"]
    assert counts["original_T1_tokens"] == item["original_T1_tokens_on_same_slots"]["sum"]
    assert counts["matched_T1_frames"] == item["counts"]["matched_T1_short_slot_frames"]
    return {"dataset": dataset, "short_slots_checked": 24000, "counts": dict(counts),
            "fallback_cases": examples, "schedule_sha256": sha(path)}


def main():
    summary = json.loads((HERE / "Summary.json").read_text(encoding="utf-8"))
    assert summary["protocol_sha256"] == sha(HERE / "Protocol.md")
    for name, digest in summary["scripts"].items():
        assert sha(HERE / name) == digest
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        datasets = [check_dataset(item, archive) for item in summary["datasets"]]
    proof = {"utc": datetime.now(timezone.utc).isoformat(), "status": "complete_training_schedule_verified",
             "datasets": datasets, "summary_sha256": sha(HERE / "Summary.json"),
             "script_sha256": sha(Path(__file__).resolve()), "training_admitted": False,
             "pixel_and_gradient_verification_still_required": True}
    with (HERE / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
