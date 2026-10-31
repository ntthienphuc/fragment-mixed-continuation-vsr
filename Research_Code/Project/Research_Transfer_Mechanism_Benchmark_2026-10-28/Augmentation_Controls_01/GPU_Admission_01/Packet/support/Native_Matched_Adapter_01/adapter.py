"""Schedule-bound input/target selection for the unchanged native recognizer."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE.parent / "Matched_Schedule_01"
WORD = HERE.parent / "Bounded_Word_Control_01"
ARMS = ("full", "fixed_time", "whole_word", "weighted_full")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


class MatchedObjective:
    def __init__(self, dataset, arm):
        if dataset not in ("vlr", "lrs2") or arm not in ARMS:
            raise ValueError("Unknown matched dataset or arm")
        proof = read(SCHEDULE / "Verification.json")
        if proof["status"] != "matched_training_schedules_verified":
            raise ValueError("Matched schedules have not passed verification")
        if sha(SCHEDULE / "Summary.json") != proof["summary_sha256"]:
            raise ValueError("Matched summary changed")
        summary = read(SCHEDULE / "Summary.json")
        entry = next(item for item in summary["datasets"] if item["dataset"] == dataset)
        path = SCHEDULE / (dataset + ".jsonl.gz")
        word_path = WORD / (dataset + "_short_slots.jsonl.gz")
        if sha(path) != entry["schedule_sha256"] or sha(word_path) != entry["bounded_schedule_sha256"]:
            raise ValueError("Matched schedule bytes changed")
        self.dataset = dataset
        self.arm = arm
        self.schedule = list(records(path))
        self.words = {row["sample"]: row for row in records(word_path)}
        if len(self.schedule) != 48000 or set(self.words) != set(range(1, 48000, 2)):
            raise ValueError("Incomplete matched schedule")
        self.identity = {"dataset": dataset, "arm": arm, "schedule_sha256": sha(path),
                         "word_schedule_sha256": sha(word_path), "adapter_sha256": sha(Path(__file__))}

    def select(self, model, frames, row, sample_index, seed):
        if type(sample_index) is not int or not 0 <= sample_index < len(self.schedule):
            raise ValueError("Sample position is outside the locked schedule")
        record = self.schedule[sample_index]
        if seed != record["seed"] or row["split"] != "train":
            raise ValueError("Only the locked seed and training rows are accepted")
        if record["clip_id"] != row["clip_id"] or record["sample"] != sample_index:
            raise ValueError("Training sample identity/order changed")
        if model.method != "T0":
            raise ValueError("Matched objective requires the unchanged T0 graph")
        if frames.dtype != np.uint8 or frames.shape != (row["frames"], 96, 96):
            raise ValueError("Unexpected native frame shape or dtype")
        arm = "full" if self.arm == "weighted_full" else self.arm
        view = record["views"][arm]
        if not 0 <= view["start"] < view["stop"] <= len(frames):
            raise ValueError("Matched interval is outside the video")
        full = record["original_full_slot"] or record["shared_full_fallback"] or arm == "full"
        if full:
            text = row["text"]
        elif arm == "fixed_time":
            from phase5.views import target_text

            text = target_text(row["words"], view["start"], view["stop"])
        else:
            span = self.words[sample_index]
            selected = row["words"][span["first_word"]:span["last_word"] + 1]
            text = " ".join(word["text"] for word in selected).strip()
        if hashlib.sha256(text.encode()).hexdigest() != view["target_sha256"]:
            raise ValueError("Matched target text changed")
        tokens = model.tokenizer.encode(text)
        if len(tokens) != view["native_token_count"]:
            raise ValueError("Native target token count changed")
        minimum = len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))
        if not tokens or minimum > view["stop"] - view["start"]:
            raise ValueError("CTC target is empty or infeasible")
        numerator, denominator = 1, 1
        if self.arm == "weighted_full":
            numerator = record["weighted_full"]["numerator"]
            denominator = record["weighted_full"]["denominator"]
        return frames[view["start"]:view["stop"]], text, numerator, denominator, full

    def loss(self, model, frames, row, sample_index, seed, budget):
        if budget not in ("past", "both"):
            raise ValueError("Unsupported native budget label")
        pixels, text, numerator, denominator, full = self.select(model, frames, row, sample_index, seed)
        encoded = model.encode(pixels)
        representation = model.representation(encoded, 0, len(pixels))
        native, details = model.asr_loss(representation, text)
        loss = native if numerator == denominator else native * (numerator / denominator)
        if not bool(loss.detach().isfinite()):
            raise FloatingPointError("Nonfinite matched objective")
        details.update(full=full, input_frames=len(pixels), target_frames=len(pixels),
                       matched_arm=self.arm, weight_numerator=numerator, weight_denominator=denominator)
        return loss, details
