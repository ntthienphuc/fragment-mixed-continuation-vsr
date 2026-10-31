"""Propagate frozen whole-word fallbacks through all future matched descriptors."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLS = HERE.parent
WORD = CONTROLS / "Bounded_Word_Control_01"
LOSS = CONTROLS / "Loss_Control_01/Preparation_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def descriptor(start, stop, digest, tokens):
    return {"start": start, "stop": stop, "target_sha256": digest, "native_token_count": tokens}


def build(dataset, old, bounded):
    weights = LOSS / (dataset + "_weights.jsonl.gz")
    spans = WORD / (dataset + "_short_slots.jsonl.gz")
    assert sha(weights) == old["schedule_sha256"]
    assert sha(spans) == bounded["schedule_sha256"]
    short = iter(rows(spans))
    totals = Counter()
    exposure = {name: Counter() for name in ("full", "fixed_time", "whole_word")}
    fallback_positions = []
    output = HERE / (dataset + ".jsonl.gz")
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for position, source in enumerate(rows(weights)):
            assert source["sample"] == position
            assert source["split"] == "train" and source["seed"] == 17
            assert source["original_full_slot"] is (position % 2 == 0)
            full = descriptor(0, source["input_stop"], source["full_target_sha256"], source["full_native_tokens"])
            fixed = dict(full)
            word = dict(full)
            fallback = False
            if position % 2:
                span = next(short)
                assert span["sample"] == position and span["clip_id"] == source["clip_id"]
                assert span["clip_frames"] == full["stop"]
                assert span["original_target_sha256"] == source["t1_target_sha256"]
                target = span["matched_T1"]
                fixed = descriptor(target["start"], target["stop"], text_sha(target["text"]),
                                   len(target["native_tokens"]))
                word = descriptor(span["start"], span["stop"], text_sha(span["text"]),
                                  len(span["native_tokens"]))
                fallback = span["view"] == "shared_full_fallback"
                if fallback:
                    assert fixed == full and word == full
                    fallback_positions.append(position)
                else:
                    assert span["view"] == "word_aligned"
                    assert fixed["stop"] - fixed["start"] == 50
                    assert 50 <= word["stop"] - word["start"] <= 82
            arms = {"full": full, "fixed_time": fixed, "whole_word": word}
            numerator = fixed["native_token_count"] + 1
            denominator = full["native_token_count"] + 1
            record = {"sample": position, "clip_id": source["clip_id"], "split": "train", "seed": 17,
                      "original_full_slot": position % 2 == 0, "shared_full_fallback": fallback,
                      "views": arms, "weighted_full": {"numerator": numerator, "denominator": denominator}}
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            for name, view in arms.items():
                exposure[name].update(presentations=1, input_frames=view["stop"] - view["start"],
                                      target_tokens=view["native_token_count"],
                                      attention_positions=view["native_token_count"] + 1)
            totals.update(presentations=1, shared_full_fallbacks=fallback,
                          original_full_slots=position % 2 == 0,
                          original_short_slots=position % 2 == 1,
                          ratios_above_one=numerator > denominator,
                          unit_ratios=numerator == denominator)
    assert next(short, None) is None
    assert totals["presentations"] == 48000 and totals["original_full_slots"] == 24000
    assert totals["shared_full_fallbacks"] == bounded["counts"]["shared_full_fallback"]
    assert exposure["full"]["input_frames"] == old["totals"]["full_video_frames"]
    assert exposure["full"]["target_tokens"] == old["totals"]["full_target_tokens"]
    return {"dataset": dataset, "counts": dict(totals), "exposure": exposure,
            "shared_fallback_samples": fallback_positions, "schedule_sha256": sha(output),
            "old_loss_schedule_sha256": sha(weights), "bounded_schedule_sha256": sha(spans)}


def main():
    assert not (HERE / "Summary.json").exists()
    original = read(LOSS / "Summary.json")
    original_proof = read(LOSS / "Verification.json")
    bounded = read(WORD / "Summary.json")
    bounded_proof = read(WORD / "Verification.json")
    assert original_proof["status"] == "all_original_targets_and_exact_weight_positions_verified"
    assert original_proof["summary_sha256"] == sha(LOSS / "Summary.json")
    assert bounded_proof["status"] == "all_bounded_slots_and_shared_fallbacks_verified"
    assert bounded_proof["summary_sha256"] == sha(WORD / "Summary.json")
    old_by_name = {item["dataset"]: item for item in original["datasets"]}
    results = [build(item["dataset"], old_by_name[item["dataset"]], item) for item in bounded["datasets"]]
    dependencies = [LOSS / "Summary.json", LOSS / "Verification.json", WORD / "Summary.json",
                    WORD / "Verification.json", HERE / "Protocol.md", Path(__file__).resolve()]
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "matched_schedule_prepared",
               "datasets": results, "input_and_code_sha256": {
                   path.relative_to(CONTROLS).as_posix(): sha(path) for path in dependencies},
               "weighted_full_input_view": "full", "assessment_predictions_read": False,
               "training_launched": False, "gpu_used": False}
    with (HERE / "Summary.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    for result in results:
        print(json.dumps({"dataset": result["dataset"], "counts": result["counts"],
                          "exposure": result["exposure"]}), flush=True)


if __name__ == "__main__":
    main()
