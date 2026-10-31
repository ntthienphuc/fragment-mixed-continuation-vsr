"""Use only the locked short views; retain the original native loss implementation."""

import hashlib
from pathlib import Path

import numpy as np

from adapter import MatchedObjective, read, records, sha

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE.parent / "Schedule"


class ShortOnlyObjective(MatchedObjective):
    def __init__(self, dataset, arm):
        if dataset != "vlr" or arm != "short_only":
            raise ValueError("This extension admits one VLR short-only arm")
        proof = read(SCHEDULE / "Verification.json")
        summary = read(SCHEDULE / "Summary.json")
        if proof["status"] != "all_short_only_views_verified":
            raise ValueError("Short-only schedule verification is missing")
        if proof["summary_sha256"] != sha(SCHEDULE / "Summary.json"):
            raise ValueError("Schedule summary changed")
        path = SCHEDULE / "vlr_short_only.jsonl.gz"
        if sha(path) != summary["schedule_sha256"] or sha(path) != proof["schedule_sha256"]:
            raise ValueError("Short-only schedule changed")
        self.dataset = dataset
        self.arm = arm
        self.schedule = list(records(path))
        if len(self.schedule) != 48000:
            raise ValueError("Short-only presentation count changed")
        self.identity = {"dataset": dataset, "arm": arm, "schedule_sha256": sha(path),
                         "adapter_sha256": sha(Path(__file__)),
                         "verification_sha256": sha(SCHEDULE / "Verification.json"),
                         "native_loss_adapter_sha256": sha(Path(__import__("adapter").__file__))}

    def select(self, model, frames, row, sample_index, seed):
        if type(sample_index) is not int or not 0 <= sample_index < len(self.schedule):
            raise ValueError("Sample position outside the locked schedule")
        record = self.schedule[sample_index]
        if seed != 17 or row["split"] != "train" or model.method != "T0":
            raise ValueError("Native graph, split or seed changed")
        if record["clip_id"] != row["clip_id"] or record["sample"] != sample_index:
            raise ValueError("Source identity or order changed")
        if frames.dtype != np.uint8 or frames.shape != (row["frames"], 96, 96):
            raise ValueError("Native pixel shape or dtype changed")
        start, stop = record["start"], record["stop"]
        if not 0 <= start < stop <= len(frames):
            raise ValueError("Crop is outside video")
        fallback = record["full_fallback"]
        if fallback:
            if start != 0 or stop != len(frames):
                raise ValueError("Fallback must use the full native view")
            text = row["text"]
        else:
            selected = row["words"][record["first_word"]:record["last_word"] + 1]
            text = " ".join(word["text"] for word in selected).strip()
        if hashlib.sha256(text.encode()).hexdigest() != record["target_sha256"]:
            raise ValueError("Short-only target text changed")
        tokens = model.tokenizer.encode(text)
        if tokens != record["native_tokens"]:
            raise ValueError("Native tokenizer targets changed")
        minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        if not tokens or minimum > stop - start:
            raise ValueError("Empty or CTC-infeasible target")
        return frames[start:stop], text, 1, 1, fallback
