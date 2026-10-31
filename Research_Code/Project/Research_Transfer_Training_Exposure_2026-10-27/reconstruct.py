"""Reconstruct input/text scheduling; the recording model never runs a neural layer."""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

from common import HERE, PHASE5, ROOT, load_training, read, save, sha, utc

sys.path.insert(0, str(PHASE5))
from phase5.data import PreparedData
from phase5.objectives import sample_loss
from phase5.text import Tokenizer, reference_text, required_frames


class Frames:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, selection):
        assert isinstance(selection, slice) and selection.step is None
        start, stop = selection.start, selection.stop
        assert 0 <= start < stop <= len(self)
        return Frames(self.start + start, self.start + stop)


class RecordingRecognizer:
    def __init__(self, method):
        self.method = method
        self.intermediate = None
        self.zero = torch.tensor(0.0)
        self.last = None

    def encode(self, frames):
        return frames

    def representation(self, frames, start, stop):
        assert 0 <= start < stop <= len(frames)
        self.last = {"input": (frames.start, frames.stop), "core": (start, stop)}
        return frames

    def asr_loss(self, encoded, text):
        self.last["text"] = text
        return self.zero, {}


def manual_view(row, sample, seed, method):
    if method == "T0" or sample % 2 == 0:
        return 0, row["frames"], row["text"], True
    message = f"{seed}|{row['clip_id']}|{sample}"
    index = int(hashlib.sha256(message.encode()).hexdigest()[:16], 16) % len(row["train_starts"])
    start = row["train_starts"][index]
    stop = start + row["core_frames"]
    lower, upper = start / 25, stop / 25
    chosen = []
    for word in row["words"]:
        begin, end = word["start"], word["end"]
        if begin < end and end > lower and begin < upper:
            chosen.append(word["text"])
        elif begin == end and lower <= begin < upper:
            chosen.append(word["text"])
    return start, stop, " ".join(chosen).strip(), False


def main(attempt):
    torch.set_num_threads(1)
    assert attempt.isidentifier() and attempt.startswith("Replay_")
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    started = utc()
    rows, receipt = load_training()
    expected = read(ROOT / "Research_Transfer_Initialization_2026-10-27/model_registry.json")["models"]
    identity = expected["VLR_T0_LAST"]["training_identity"]
    assert receipt["manifest_sha256"] == identity["train_manifest_sha256"]
    assert receipt["code_sha256"] == identity["code_sha256"]
    assert receipt["config_sha256"] == identity["config_sha256"]
    model_path = PHASE5 / "assets/tokenizers/vi.model"
    units_path = PHASE5 / "assets/tokenizers/vi_units.txt"
    assert sha(model_path) == expected["VLR_B0"]["tokenizer_sha256"]
    assert sha(units_path) == expected["VLR_B0"]["units_sha256"]
    tokenizer = Tokenizer(model_path, units_path, "vi")
    data = PreparedData.__new__(PreparedData)
    data.rows = rows
    frame_lengths = {row["clip_id"]: row["frames"] for row in rows}
    data.frames = lambda clip_id: Frames(0, frame_lengths[clip_id])
    selected_steps = {17: {"T0": 3000, "T1": 3500}, 29: {"T0": 5000, "T1": 5500}}
    results = {}
    control_checks = 0
    for seed in (17, 29):
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        assert order == data.order(0, seed)
        assert len(set(order[:48000])) == 48000
        models = {method: RecordingRecognizer(method) for method in ("T0", "T1")}
        totals = {method: Counter() for method in models}
        source_frames = {method: Counter() for method in models}
        source_samples = {method: Counter() for method in models}
        token_histograms = {method: Counter() for method in models}
        digests = {method: hashlib.sha256() for method in models}
        selected = {}
        for sample in range(48000):
            row, frames = data.sample(sample, seed)
            assert row is rows[order[sample]]
            assert len(frames) == row["frames"] and row["core_frames"] == 50
            for method, model in models.items():
                left, right, text, is_full = manual_view(row, sample, seed, method)
                _, details = sample_loss(model, frames, row, sample, seed, "both")
                assert model.last == {"input": (left, right), "core": (0, right - left), "text": text}
                assert details == {"full": is_full, "input_frames": right - left, "target_frames": right - left}
                ids = tokenizer.encode(text)
                assert required_frames(ids) <= right - left
                independent_pieces = tokenizer.processor.encode(reference_text(text, "vi"), out_type=int)
                independent_ids = [tokenizer.mapping[tokenizer.processor.id_to_piece(piece)]
                                   for piece in independent_pieces]
                assert ids == independent_ids
                totals[method].update(samples=1, input_frames=right - left, target_tokens=len(ids),
                                      attention_positions=len(ids) + 1,
                                      target_words=len(reference_text(text, "vi").split()),
                                      full_samples=int(is_full), crop_samples=int(not is_full))
                source_frames[method][row["source_id"]] += right - left
                source_samples[method][row["source_id"]] += 1
                token_histograms[method].update(ids)
                trace = [sample, row["clip_id"], left, right, ids]
                digests[method].update((json.dumps(trace, separators=(",", ":")) + "\n").encode())
                control_checks += 1
                if sample + 1 == selected_steps[seed][method] * 8:
                    selected[method] = {"step": selected_steps[seed][method], "exposure": dict(totals[method])}
            if (sample + 1) % 12000 == 0:
                print(json.dumps({"seed": seed, "samples_per_method": sample + 1}), flush=True)
        assert source_samples["T0"] == source_samples["T1"]
        assert totals["T0"]["full_samples"] == 48000 and totals["T0"]["crop_samples"] == 0
        assert totals["T1"]["full_samples"] == totals["T1"]["crop_samples"] == 24000
        for method in models:
            assert sum(source_frames[method].values()) == totals[method]["input_frames"]
            assert sum(token_histograms[method].values()) == totals[method]["target_tokens"]
        ratios = {field: totals["T1"][field] / totals["T0"][field]
                  for field in ("samples", "input_frames", "target_tokens", "attention_positions", "target_words")}
        results[str(seed)] = {"final_steps": 6000, "final_exposure": {key: dict(value)
                              for key, value in totals.items()}, "T1_over_T0": ratios,
                              "source_selected_exposure": selected,
                              "schedule_sha256": {key: value.hexdigest() for key, value in digests.items()},
                              "source_sample_identity_matches": True, "sources_seen": len(source_samples["T0"])}
        save(output / ("Counts_" + str(seed) + ".json"), {
            "source_frames": {key: dict(value) for key, value in source_frames.items()},
            "source_samples": {key: dict(value) for key, value in source_samples.items()},
            "target_token_histograms": {key: dict(value) for key, value in token_histograms.items()},
        })
    code_paths = [PHASE5 / "phase5" / name for name in
                  ("data.py", "objectives.py", "views.py", "text.py", "common.py", "runtime.py", "training.py")]
    summary = {"status": "complete_deterministic_schedule_reconstruction", "started_utc": started,
               "completed_utc": utc(), "training_receipt": receipt, "seeds": results,
               "native_objective_control_flow_checks": control_checks,
               "neural_forward_calls": 0, "gradient_or_optimizer_replay": False,
               "pixels_read": False, "new_recognition": False,
               "training_code_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in code_paths},
               "code_sha256": sha(Path(__file__)), "helpers_sha256": sha(HERE / "common.py"),
               "protocol_sha256": sha(HERE / "Protocol.md"), "tokenizer_sha256": sha(model_path)}
    save(output / "Summary.json", summary)
    print(json.dumps({"status": summary["status"], "checks": control_checks,
                      "ratios": {seed: item["T1_over_T0"] for seed, item in results.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="Replay_01")
    arguments = parser.parse_args()
    try:
        main(arguments.attempt)
    except BaseException as error:
        destination = HERE / arguments.attempt / "Failure.json"
        if destination.parent.exists() and not destination.exists():
            save(destination, {"utc": utc(), "error": repr(error), "code_sha256": sha(HERE / "reconstruct.py")})
        raise
