"""Extend the frozen mixed schedule to nominal short-only without changing source order."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTROLS = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Augmentation_Controls_01"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def setup():
    sys.path.insert(0, str(ROOT / "Research_Phase_05_2026-10-13"))
    from phase5.text import Tokenizer
    from phase5.common import rank

    def training_anchor(row, sample_index):
        choices = row["train_starts"]
        position = int(rank(f"17|{row['clip_id']}|{sample_index}")[:16], 16) % len(choices)
        return choices[position]

    source = CONTROLS / "Bounded_Word_Control_01/word_span.py"
    spec = importlib.util.spec_from_file_location("locked_bounded_span", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = read(ROOT / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")["models"]["VLR_B0"]
    for field in ("tokenizer", "units"):
        assert sha(ROOT / model[field]) == model[field + "_sha256"]
    tokenizer = Tokenizer(ROOT / model["tokenizer"], ROOT / model["units"], "vi")
    matched = CONTROLS / "Matched_Schedule_01"
    proof = read(matched / "Verification.json")
    assert proof["status"] == "matched_training_schedules_verified"
    assert sha(matched / "Summary.json") == proof["summary_sha256"]
    entry = next(item for item in read(matched / "Summary.json")["datasets"] if item["dataset"] == "vlr")
    old_path = matched / "vlr.jsonl.gz"
    word_path = CONTROLS / "Bounded_Word_Control_01/vlr_short_slots.jsonl.gz"
    assert sha(old_path) == entry["schedule_sha256"]
    assert sha(word_path) == entry["bounded_schedule_sha256"]
    old = records(old_path)
    words = {item["sample"]: item for item in records(word_path)}
    needed = {item["clip_id"] for item in old}
    metadata = {}
    checksum = hashlib.sha256()
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        with archive.open("Phase05/prepared/vlr/train/manifest.jsonl") as stream:
            for line in stream:
                checksum.update(line)
                row = json.loads(line)
                if row["clip_id"] in needed:
                    assert row["train_eligible"] and row["split"] == "train"
                    metadata[row["clip_id"]] = row
    assert checksum.hexdigest() == "70ec83434982fd3e350ac43c51a20468f416d32b8bd48b254021d76b7f9e0dfb"
    assert len(needed) == len(old) == len(metadata) == 48000
    return old, words, metadata, tokenizer, module.choose, training_anchor, entry, model


def main():
    old, words, metadata, tokenizer, choose, training_anchor, entry, model = setup()
    output = HERE / "Schedule"
    output.mkdir(exist_ok=False)
    counts = Counter()
    exposure = Counter()
    schedule_path = output / "vlr_short_only.jsonl.gz"
    with gzip.open(schedule_path, "xt", encoding="utf-8") as stream:
        for index, previous in enumerate(old):
            assert previous["sample"] == index and previous["seed"] == 17
            row = metadata[previous["clip_id"]]
            anchor = training_anchor(row, index)
            if index % 2:
                selected = words[index]
                assert selected["original_start"] == anchor
                assert selected["clip_id"] == row["clip_id"]
                counts["preserved_odd_slots"] += 1
            else:
                selected = choose(row["words"], row["frames"], anchor, anchor + 50,
                                  row["text"], tokenizer.encode)
                counts["new_even_slots"] += 1
            text = selected["text"]
            tokens = tokenizer.encode(text)
            start, stop = selected["start"], selected["stop"]
            fallback = selected["view"] == "shared_full_fallback"
            assert tokens == selected["native_tokens"] and 0 <= start < stop <= row["frames"]
            minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
            assert tokens and minimum <= stop - start
            assert fallback or 50 <= stop - start <= 82
            target_sha = hashlib.sha256(text.encode()).hexdigest()
            if index % 2:
                assert previous["views"]["whole_word"] == {
                    "start": start, "stop": stop, "target_sha256": target_sha,
                    "native_token_count": len(tokens)}
            record = {"sample": index, "clip_id": row["clip_id"], "split": "train", "seed": 17,
                      "anchor_start": anchor, "anchor_stop": anchor + 50, "view": selected["view"],
                      "start": start, "stop": stop, "clip_frames": row["frames"],
                      "first_word": selected.get("first_word"), "last_word": selected.get("last_word"),
                      "target_sha256": target_sha, "native_tokens": tokens, "native_token_count": len(tokens),
                      "minimum_ctc_frames": minimum, "full_fallback": fallback}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts.update(presentations=1, full_fallbacks=fallback,
                          even_fallbacks=fallback and index % 2 == 0,
                          odd_fallbacks=fallback and index % 2 == 1,
                          full_pixel_extent=start == 0 and stop == row["frames"])
            exposure.update(input_frames=stop - start, target_tokens=len(tokens),
                            attention_positions=len(tokens) + 1)
    assert counts["odd_fallbacks"] == 317 and counts["presentations"] == 48000
    dependencies = [Path(__file__), HERE / "Protocol.md", CONTROLS / "Bounded_Word_Control_01/word_span.py",
                    CONTROLS / "Readiness_02/word_span.py", ROOT / "Research_Phase_05_2026-10-13/phase5/views.py"]
    result = {"utc": datetime.now(timezone.utc).isoformat(), "status": "short_only_schedule_prepared",
              "counts": dict(counts), "exposure": dict(exposure), "schedule_sha256": sha(schedule_path),
              "parent_schedule_sha256": entry["schedule_sha256"],
              "parent_word_schedule_sha256": entry["bounded_schedule_sha256"],
              "tokenizer_sha256": model["tokenizer_sha256"], "units_sha256": model["units_sha256"],
              "dependencies": {str(path.relative_to(ROOT)): sha(path) for path in dependencies},
              "assessment_predictions_read": False, "training_started": False}
    (output / "Summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
