"""Reconstruct the frozen English training schedule without neural computation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE = ROOT / "Research_Phase_05_2026-10-13"
sys.path.insert(0, str(PHASE))

from phase5.common import code_identity
from phase5.data import PreparedData
from phase5.objectives import sample_loss
from phase5.text import Tokenizer, reference_text, required_frames


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


class Frames:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, selection):
        assert isinstance(selection, slice) and selection.step is None
        assert 0 <= selection.start < selection.stop <= len(self)
        return Frames(self.start + selection.start, self.start + selection.stop)


class Recorder:
    def __init__(self, method):
        self.method = method
        self.intermediate = None
        self.last = None
        self.zero = torch.tensor(0.0)

    def encode(self, frames):
        self.last = {"start": frames.start, "stop": frames.stop}
        return frames

    def representation(self, frames, begin, end):
        assert begin == 0 and end == len(frames)
        return frames

    def asr_loss(self, encoded, text):
        self.last["text"] = text
        return self.zero, {}


def load_rows():
    archive_path = ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip"
    manifest_member = "Phase05/prepared/lrs2/train/manifest.jsonl"
    with zipfile.ZipFile(archive_path) as archive:
        receipt = json.loads(archive.read("Phase05/prepared/lrs2/train/receipt.json"))
        failures = archive.read("Phase05/prepared/lrs2/train/failures.jsonl")
        assert hashlib.sha256(failures).hexdigest() == receipt["failures_sha256"]
        digest = hashlib.sha256()
        rows = []
        with archive.open(manifest_member) as stream:
            for line in stream:
                digest.update(line)
                rows.append(json.loads(line))
    assert digest.hexdigest() == receipt["manifest_sha256"]
    assert len(rows) == receipt["prepared"] == 45838
    assert receipt["planned"] == len(rows) + receipt["failed"]
    assert len({row["clip_id"] for row in rows}) == len(rows)
    assert all(row["split"] == "train" for row in rows)
    return rows, receipt


def main(attempt):
    assert attempt.isidentifier() and attempt.startswith("Replay_")
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    started = utc()
    torch.set_num_threads(1)
    registry_path = ROOT / "Research_English_Transfer_Benchmark_2026-10-28/model_registry.json"
    models = json.loads(registry_path.read_text(encoding="utf-8-sig"))["models"]
    rows, receipt = load_rows()
    actual_code = code_identity()
    assert actual_code == receipt["code_sha256"]
    for method in ("T0", "T1"):
        identity = models[f"EN_{method}_LAST"]["training_identity"]
        for field in ("code_sha256", "config_sha256"):
            assert identity[field] == receipt[field]
        assert identity["train_manifest_sha256"] == receipt["manifest_sha256"]
        assert identity["cohort"] == "window" and identity["scope"] == "all"
        assert identity["seed"] == 17 and identity["budget"] == "both"
        assert identity["steps"] == 6000 and identity["accumulation"] == 8
        assert models[f"EN_{method}_LAST"]["selected_training_step"] == 6000
    eligible = [row for row in rows if row["train_eligible"]]
    assert len(eligible) == receipt["train_eligible"] == 7291
    assert all(row["core_frames"] == 50 for row in rows)
    population = Counter(prepared=len(rows), window_eligible=len(eligible))
    flags = Counter()
    for row in rows:
        population["full_eligible"] += int(row["full_eligible"])
        population["shorter_than_82_frames"] += int(row["frames"] < 82)
        population["without_word_times"] += int(not row["words"])
        population["timing_invalid"] += int(not row["timing_valid"])
        key = (row["frames"] < 82, not bool(row["words"]), not row["timing_valid"])
        flags[str(key)] += 1
    population["mean_prepared_seconds"] = sum(row["frames"] for row in rows) / 25 / len(rows)
    population["mean_eligible_seconds"] = sum(row["frames"] for row in eligible) / 25 / len(eligible)
    population["eligible_fraction"] = len(eligible) / len(rows)
    model_path = PHASE / "assets/tokenizers/en.model"
    units_path = PHASE / "assets/tokenizers/en_units.txt"
    assert sha(model_path) == models["EN_B0"]["tokenizer_sha256"]
    assert sha(units_path) == models["EN_B0"]["units_sha256"]
    tokenizer = Tokenizer(model_path, units_path, "en")
    data = PreparedData.__new__(PreparedData)
    data.rows = eligible
    lengths = {row["clip_id"]: row["frames"] for row in eligible}
    data.frames = lambda clip_id: Frames(0, lengths[clip_id])
    orders = {}
    for epoch in range(7):
        indices = list(range(len(eligible)))
        random.Random(17 + epoch).shuffle(indices)
        assert indices == data.order(epoch, 17)
        orders[epoch] = indices
    recorders = {method: Recorder(method) for method in ("T0", "T1")}
    totals = {method: Counter() for method in recorders}
    occurrences = Counter()
    source_samples = Counter()
    trace_path = output / "Trace.jsonl.gz"
    checks = 0
    with gzip.open(trace_path, "xt", encoding="utf-8") as trace:
        for number in range(48000):
            row, frames = data.sample(number, 17)
            epoch, position = divmod(number, len(eligible))
            assert row is eligible[orders[epoch][position]]
            occurrences[row["clip_id"]] += 1
            source_samples[row["source_id"]] += 1
            record = {"sample": number, "epoch": epoch, "clip_id": row["clip_id"], "views": {}}
            for method, model in recorders.items():
                full = method == "T0" or number % 2 == 0
                if full:
                    left, right, text = 0, row["frames"], row["text"]
                    partial_left = partial_right = 0
                else:
                    message = f"17|{row['clip_id']}|{number}"
                    choice = int(hashlib.sha256(message.encode()).hexdigest()[:16], 16)
                    left = row["train_starts"][choice % len(row["train_starts"])]
                    right = left + 50
                    lower, upper = left / 25, right / 25
                    words = [word for word in row["words"] if
                             (word["end"] > lower and word["start"] < upper
                              and word["end"] > word["start"])
                             or (word["start"] == word["end"] and lower <= word["start"] < upper)]
                    text = " ".join(word["text"] for word in words).strip()
                    partial_left = sum(word["start"] < lower < word["end"] for word in words)
                    partial_right = sum(word["start"] < upper < word["end"] for word in words)
                _, detail = sample_loss(model, frames, row, number, 17, "both")
                assert model.last == {"start": left, "stop": right, "text": text}
                assert detail == {"full": full, "input_frames": right - left, "target_frames": right - left}
                ids = tokenizer.encode(text)
                assert required_frames(ids) <= right - left
                canonical = reference_text(text, "en").upper()
                pieces = tokenizer.processor.encode(canonical, out_type=int)
                expected_ids = [tokenizer.mapping[tokenizer.processor.id_to_piece(piece)] for piece in pieces]
                assert ids == expected_ids
                counts = dict(samples=1, input_frames=right - left, target_tokens=len(ids),
                              attention_positions=len(ids) + 1, target_words=len(canonical.split()),
                              full_samples=int(full), crop_samples=int(not full),
                              crop_partial_left=int(partial_left > 0), crop_partial_right=int(partial_right > 0),
                              crop_partial_either=int(partial_left > 0 or partial_right > 0))
                totals[method].update(counts)
                record["views"][method] = {"start": left, "stop": right, **counts}
                checks += 1
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")
            if (number + 1) % 12000 == 0:
                print(json.dumps({"samples": number + 1}), flush=True)
    assert set(occurrences.values()) == {6, 7} and len(occurrences) == 7291
    ratios = {field: totals["T1"][field] / totals["T0"][field]
              for field in ("input_frames", "target_tokens", "attention_positions", "target_words")}
    summary = {"status": "complete_pending_independent_verification", "started_utc": started,
               "completed_utc": utc(), "training_receipt": receipt, "population": dict(population),
               "exclusion_flags_short_no_times_timing_invalid": dict(flags),
               "unique_clips_seen": len(occurrences), "unique_sources_seen": len(source_samples),
               "clip_exposure_histogram": dict(Counter(occurrences.values())),
               "sample_decisions": 48000, "eligible_epoch_equivalents": 48000 / len(eligible),
               "native_control_flow_checks": checks, "totals": {key: dict(value) for key, value in totals.items()},
               "T1_over_T0": ratios, "trace_sha256": sha(trace_path), "code_sha256": sha(Path(__file__)),
               "protocol_sha256": sha(HERE / "Protocol.md"), "registry_sha256": sha(registry_path),
               "phase5_code_identity": actual_code, "pixels_read": False, "neural_calls": 0,
               "gradient_or_optimizer_replay": False, "new_recognition": False}
    save(output / "Summary.json", summary)
    print(json.dumps({"population": dict(population), "ratios": ratios, "checks": checks}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    main(args.attempt)
