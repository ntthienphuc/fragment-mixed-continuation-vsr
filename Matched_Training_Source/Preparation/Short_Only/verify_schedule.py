"""Reconstruct every target, boundary, native slot and eligibility decision before training."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from prepare_schedule import HERE, ROOT, CONTROLS, read, records, setup, sha


def main():
    old, words, metadata, tokenizer, choose, anchor, entry, _ = setup()
    rows = records(HERE / "Schedule/vlr_short_only.jsonl.gz")
    summary = read(HERE / "Schedule/Summary.json")
    totals = Counter()
    exposure = Counter()
    for index, record in enumerate(rows):
        source = metadata[old[index]["clip_id"]]
        assert record["sample"] == index and record["clip_id"] == source["clip_id"]
        assert record["split"] == "train" and record["seed"] == 17
        start = anchor(source, index)
        assert (record["anchor_start"], record["anchor_stop"]) == (start, start + 50)
        selected = choose(source["words"], source["frames"], start, start + 50,
                          source["text"], tokenizer.encode)
        for field in ("start", "stop", "native_tokens", "view", "minimum_ctc_frames"):
            assert record[field] == selected[field], (index, field)
        if record["full_fallback"]:
            assert record["first_word"] is None and record["last_word"] is None
            text = source["text"]
        else:
            first, last = record["first_word"], record["last_word"]
            assert (first, last) == (selected["first_word"], selected["last_word"])
            text = " ".join(word["text"] for word in source["words"][first:last + 1]).strip()
            assert 50 <= record["stop"] - record["start"] <= 82
        assert hashlib.sha256(text.encode()).hexdigest() == record["target_sha256"]
        assert tokenizer.encode(text) == record["native_tokens"]
        if index % 2:
            prior = words[index]
            for field in ("start", "stop", "native_tokens", "view"):
                assert record[field] == prior[field]
        totals.update(presentations=1, full_fallbacks=record["full_fallback"],
                      even_fallbacks=record["full_fallback"] and index % 2 == 0,
                      odd_fallbacks=record["full_fallback"] and index % 2 == 1,
                      full_pixel_extent=record["start"] == 0 and record["stop"] == source["frames"])
        exposure.update(input_frames=record["stop"] - record["start"],
                        target_tokens=len(record["native_tokens"]),
                        attention_positions=len(record["native_tokens"]) + 1)
    assert len(rows) == 48000 and dict(exposure) == summary["exposure"]
    for field, value in totals.items():
        assert value == summary["counts"][field]
    proof = {"utc": datetime.now(timezone.utc).isoformat(), "status": "all_short_only_views_verified",
             "summary_sha256": sha(HERE / "Schedule/Summary.json"),
             "schedule_sha256": sha(HERE / "Schedule/vlr_short_only.jsonl.gz"),
             "views_reconstructed": len(rows), "odd_slots_identical_to_mixed": 24000,
             "counts": dict(totals), "exposure": dict(exposure),
             "verification_code_sha256": sha(Path(__file__)), "neural_updates": 0}
    (HERE / "Schedule/Verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
