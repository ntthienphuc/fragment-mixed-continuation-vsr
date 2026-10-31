"""Independently inspect the actual one-update CPU checkpoint without further training."""

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03"
OUTPUT = Path(sys.argv[1]).resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    began = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    complete = read(OUTPUT / "Complete.json")
    identity = read(OUTPUT / "Identity.json")
    assert complete["status"] == "bounded_integration_update_complete"
    assert complete["updates"] == 1 and complete["presentations"] == 8
    assert identity["objective"] == "matched_whole_word" and identity["precision"] == "fp32"
    assert identity["mode"] == "integration" and identity["device_type"] == "cpu"
    canonical = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert canonical == complete["identity_sha256"]
    checkpoint = OUTPUT / "last.pt"
    assert sha(checkpoint) == complete["checkpoint_sha256"]
    for name, digest in identity["code_sha256"].items():
        assert sha(HERE / name) == digest
    spec = read(NATIVE / "Inputs.json")
    sys.path.insert(0, str(NATIVE / "source"))
    from phase5.common import code_identity
    from phase5.model import Recognizer

    assert code_identity() == identity["native_code_identity"]
    config = spec["config"].copy()
    initializer = ROOT / "Server_Rescue_2026-10-15/Assets/VLR/pretrained.pt"
    assert sha(initializer) == identity["initializer_sha256"]
    config["checkpoint"] = str(initializer)
    original = Recognizer(config, method="T0", device="cpu", scope="all")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert payload["identity"] == identity and payload["step"] == 1
    assert payload["scaler"] == {}
    trace = [json.loads(line) for line in (OUTPUT / "Attempts/001.steps.jsonl").read_text().splitlines()]
    assert len(trace) == 1 and trace[0]["step"] == 1
    samples = trace[0]["samples"]
    assert len(samples) == 8
    schedule_path = HERE.parent / "Matched_Schedule_01/vlr.jsonl.gz"
    assert sha(schedule_path) == identity["schedule"]["schedule_sha256"]
    with gzip.open(schedule_path, "rt", encoding="utf-8") as stream:
        for position, sample in enumerate(samples):
            expected = json.loads(next(stream))
            view = expected["views"]["whole_word"]
            assert sample["sample"] == position == expected["sample"]
            assert sample["clip_id"] == expected["clip_id"] == spec["cases"][position]["clip_id"]
            assert spec["cases"][position]["split"] == "train"
            assert sample["input_frames"] == view["stop"] - view["start"]
            assert sample["target_frames"] == sample["input_frames"]
            assert sample["full"] == (expected["original_full_slot"] or expected["shared_full_fallback"])
            assert sample["weight_numerator"] == sample["weight_denominator"] == 1
    average = sum(0.1 * row["ctc"] + 0.9 * row["attention"] for row in samples) / 8
    assert math.isclose(average, trace[0]["loss"], abs_tol=1e-5)
    groups = payload["optimizer"]["param_groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["weight_decay"] == 0.01 and group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["fused"] is False
    assert math.isclose(group["lr"], 4e-9, rel_tol=1e-14)
    named = list(original.named_parameters())
    assert len(named) == len(group["params"])
    assert sum(parameter.numel() for unused, parameter in named) == 243357028
    states = payload["optimizer"]["state"]
    initial_state = original.state_dict()
    assert set(initial_state) == set(payload["model"])
    comparisons = []
    changed = 0
    gradient_square_sum = 0.0
    for (name, initial), index in zip(named, group["params"]):
        actual = payload["model"][name]
        assert torch.isfinite(actual).all()
        expected = initial.detach().clone()
        if index in states:
            moment = states[index]
            assert moment["step"].item() == 1
            first = moment["exp_avg"]
            second = moment["exp_avg_sq"]
            assert first.shape == second.shape == actual.shape
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
            assert (second >= 0).all()
            gradient_square_sum += (first.double() / 0.1).square().sum().item()
            denominator = second.sqrt().div_(math.sqrt(1 - 0.999)).add_(group["eps"])
            expected.mul_(1 - group["lr"] * group["weight_decay"])
            expected.addcdiv_(first, denominator, value=-(group["lr"] / (1 - 0.9)))
        difference = (actual - expected).abs().max().item()
        assert torch.equal(actual, expected), (name, difference)
        changed += int(torch.count_nonzero(actual != initial.detach()))
        comparisons.append({"name": name, "optimizer_state": index in states, "maximum_error": difference})
    buffers = dict(original.named_buffers())
    for name, initial in buffers.items():
        assert torch.equal(initial, payload["model"][name]), "Unexpected buffer change: " + name
    assert changed > 0 and math.sqrt(gradient_square_sum) <= 5.0001
    original.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW(original.parameters(), lr=2e-6, weight_decay=0.01, fused=False)
    optimizer.load_state_dict(payload["optimizer"])
    assert len(optimizer.state) == len(states)
    assert not torch.cuda.is_initialized()
    assert sha(initializer) == identity["initializer_sha256"]
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "actual_cpu_checkpoint_independently_verified",
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "identity_sha256": canonical,
        "native_parameters": 243357028,
        "parameter_tensors": len(named),
        "optimizer_states": len(states),
        "changed_parameter_elements": changed,
        "first_adamw_update_byte_exact": True,
        "reconstructed_clipped_gradient_norm": math.sqrt(gradient_square_sum),
        "unchanged_buffer_tensors": len(buffers),
        "schedule_samples_checked": 8,
        "strict_native_and_optimizer_load_passed": True,
        "cuda_initialized": False,
        "seconds": time.monotonic() - began,
        "verifier_sha256": sha(Path(__file__)),
        "gpu_training_readiness_proven": False,
        "recognition_benefit_proven": False,
        "comparisons": comparisons,
    }
    with (OUTPUT / "Independent_Verification_01.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
