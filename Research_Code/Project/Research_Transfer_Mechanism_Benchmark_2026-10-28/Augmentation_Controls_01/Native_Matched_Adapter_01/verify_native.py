"""Replay actual native forward losses for the matched objective, with no updates."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03"
SCHEDULE = HERE.parent / "Matched_Schedule_01"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def probe_inputs(adapter):
    positions = [0, 1]
    positions.append(next(row["sample"] for row in adapter.schedule if row["shared_full_fallback"]))
    eligible = [row for row in adapter.schedule if not row["original_full_slot"]
                and not row["shared_full_fallback"]]
    maximum = max(row["views"]["whole_word"]["stop"] - row["views"]["whole_word"]["start"]
                  for row in eligible)
    positions.append(next(row["sample"] for row in eligible if
                          row["views"]["whole_word"]["stop"] - row["views"]["whole_word"]["start"] == maximum))
    assert len(set(positions)) == 4
    wanted = {adapter.schedule[position]["clip_id"] for position in positions}
    metadata = {}
    checksum = hashlib.sha256()
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        with archive.open("Phase05/prepared/vlr/train/manifest.jsonl") as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                assert row["split"] == "train"
                if row["clip_id"] in wanted:
                    assert row["clip_id"] not in metadata
                    metadata[row["clip_id"]] = row
    spec = read(PARENT / "Inputs.json")
    assert checksum.hexdigest() == spec["identity"]["train_manifest_sha256"]
    assert set(metadata) == wanted
    return positions, metadata


def reject_invalid(adapter, model, pixels, row, position):
    rejected = []
    invalid = [("clip_identity", {**row, "clip_id": row["clip_id"] + "_wrong"}, 17),
               ("validation_split", {**row, "split": "val"}, 17), ("seed", row, 29)]
    for name, altered, seed in invalid:
        try:
            adapter.select(model, pixels, altered, position, seed)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("Invalid input was accepted: " + name)
    return rejected


def main(output):
    began = time.monotonic()
    output.mkdir(exist_ok=False)
    save(output / "Start.json", {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                                "device": "cpu", "maximum_seconds": 900, "optimizer_updates": 0})
    import torch
    from adapter import MatchedObjective, ARMS

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(PARENT / "source"))
    from phase5.common import code_identity
    from phase5.media import read_video
    from phase5.model import Recognizer
    from phase5.objectives import sample_loss as original_loss

    spec = read(PARENT / "Inputs.json")
    assert code_identity() == spec["identity"]["code_sha256"]
    config = spec["config"].copy()
    checkpoint = ROOT / "Server_Rescue_2026-10-15/Assets/VLR/pretrained.pt"
    config["checkpoint"] = str(checkpoint)
    assert sha(checkpoint) == config["checkpoint_sha256"]
    adapter = MatchedObjective("vlr", "full")
    positions, metadata = probe_inputs(adapter)
    save(output / "Plan.json", {"positions": positions, "schedule_identity": adapter.identity,
                               "arms": list(ARMS), "code_sha256": sha(Path(__file__)),
                               "protocol_sha256": sha(HERE / "Protocol.md")})
    torch.manual_seed(17)
    model = Recognizer(config, method="T0", device="cpu", scope="all")
    model.train()
    initial_state = state_digest(model)
    assert not torch.cuda.is_initialized()
    results = []
    rejections = []
    for position in positions:
        assert time.monotonic() - began < 900
        schedule = adapter.schedule[position]
        row = metadata[schedule["clip_id"]]
        media_root = ROOT / "data/official_raw/vlr/extracted/cloud_dataset_signer_split/data/train"
        path = (media_root / (row["clip_id"] + ".mp4")).resolve()
        assert path.is_relative_to(media_root.resolve()) and sha(path) == row["video_sha256"]
        pixels = read_video(path)
        assert pixels.shape == (row["frames"], 96, 96)
        pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        if row.get("pixels_sha256"):
            assert pixel_sha == row["pixels_sha256"]
        else:
            sys.path.insert(0, str(HERE.parent / "Readiness_02"))
            from verify_vlr_media import archived_cache_bytes

            assert hashlib.sha256(archived_cache_bytes(pixels)).hexdigest() == row["frames_sha256"]
        for arm in ARMS:
            assert time.monotonic() - began < 900
            adapter.arm = arm
            model.method = "T0"
            torch.manual_seed(1700 + position)
            with torch.no_grad():
                actual, actual_parts = adapter.loss(model, pixels, row, position, 17, "both")
            torch.manual_seed(1700 + position)
            reference_pixels = pixels
            reference_row = row
            reference_method = "T0"
            if arm == "fixed_time" and not schedule["shared_full_fallback"]:
                reference_method = "T1"
            if arm == "whole_word" and not schedule["original_full_slot"]:
                if not schedule["shared_full_fallback"]:
                    view = schedule["views"]["whole_word"]
                    span = adapter.words[position]
                    words = row["words"][span["first_word"]:span["last_word"] + 1]
                    reference_row = {**row, "text": " ".join(word["text"] for word in words).strip()}
                    reference_pixels = pixels[view["start"]:view["stop"]]
            model.method = reference_method
            with torch.no_grad():
                expected, expected_parts = original_loss(model, reference_pixels, reference_row, position, 17, "both")
                if arm == "weighted_full":
                    scalar = schedule["weighted_full"]
                    expected = expected * (scalar["numerator"] / scalar["denominator"])
            model.method = "T0"
            assert actual.detach().numpy().tobytes() == expected.detach().numpy().tobytes()
            assert all(actual_parts[key] == expected_parts[key] for key in ("ctc", "attention"))
            result = {"sample": position, "arm": arm, "frames": actual_parts["input_frames"],
                      "loss": float(actual.detach()), "ctc": actual_parts["ctc"],
                      "attention": actual_parts["attention"], "native_loss_bytes_exact": True,
                      "raw_sha256": row["video_sha256"], "pixels_sha256": pixel_sha,
                      "shared_full_fallback": schedule["shared_full_fallback"],
                      "elapsed_seconds": time.monotonic() - began}
            results.append(result)
            with (output / "Comparisons.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps({key: result[key] for key in ("sample", "arm", "frames", "elapsed_seconds")}), flush=True)
            del actual, expected
        rejections.extend(reject_invalid(adapter, model, pixels, row, position))
        del pixels
    assert len(results) == 16 and len(rejections) == 12
    assert state_digest(model) == initial_state and sha(checkpoint) == config["checkpoint_sha256"]
    assert code_identity() == spec["identity"]["code_sha256"] and not torch.cuda.is_initialized()
    save(output / "Complete.json", {"utc": datetime.now(timezone.utc).isoformat(),
         "status": "native_cpu_forward_adapter_comparisons_passed", "seconds": time.monotonic() - began,
         "comparisons": 16, "invalid_inputs_rejected": rejections, "positions": positions,
         "comparisons_sha256": sha(output / "Comparisons.jsonl"), "plan_sha256": sha(output / "Plan.json"),
         "initializer_sha256": config["checkpoint_sha256"], "native_code_identity": code_identity(),
         "state_unchanged": True, "trained_weights_saved": False, "optimizer_updates": 0,
         "gpu_used": False, "backward_or_gpu_readiness_proven": False,
         "strict_scaling_gate_overridden": False, "assessment_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="CPU_Forward_01")
    args = parser.parse_args()
    assert args.attempt.isidentifier()
    output = HERE / args.attempt
    try:
        main(output)
    except BaseException:
        if output.exists() and not (output / "Failure.json").exists():
            save(output / "Failure.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                          "error": traceback.format_exc()})
        raise
