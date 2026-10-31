"""Exercise dropout/RNG and optimizer trajectory across actual checkpoint roundtrips."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03/source"))

import torch
from phase5.common import seed_all
from phase5.training import learning_rate
from engine import update, save_checkpoint, restore_checkpoint


class Objective:
    def loss(self, model, pixels, row, position, seed, budget):
        assert row["clip_id"] == str(position) and seed == 17 and budget == "both"
        target = torch.tensor([position % 5])
        loss = 100 * torch.nn.functional.cross_entropy(model(pixels), target)
        return loss, {"input_frames": 1}


def data(first=0):
    for number in range(first, 16):
        pixels = torch.arange(12, dtype=torch.float32).reshape(1, 12) / (number + 1)
        yield {"clip_id": str(number)}, pixels


def create():
    seed_all(17)
    model = torch.nn.Sequential(torch.nn.Linear(12, 9), torch.nn.ReLU(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(9, 5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01, fused=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            equal_tree(first, second)
    else:
        assert left == right


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    objective = Objective()
    reference, reference_optimizer, _ = create()
    stream = data()
    for step in (1, 2):
        reference.train()
        reference_optimizer.zero_grad(set_to_none=True)
        for group in reference_optimizer.param_groups:
            group["lr"] = learning_rate(step, 6000, 2e-6)
        for micro in range(8):
            row, pixels = next(stream)
            loss, _ = objective.loss(reference, pixels, row, (step - 1) * 8 + micro, 17, "both")
            (loss / 8).backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0, error_if_nonfinite=True)
        assert float(norm) > 5.0
        reference_optimizer.step()
    expected_model = copy.deepcopy(reference.state_dict())
    expected_optimizer = copy.deepcopy(reference_optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    model, optimizer, scaler = create()
    first = update(model, optimizer, scaler, objective, data(), 1, "fp32", "cpu")
    assert math.isclose(first["lr"], 4e-9, rel_tol=1e-15) and first["gradient_norm"] > 5.0
    identity = {"objective": "test_checkpoint_roundtrip", "native_graph": "toy_only"}
    rejections = []
    with tempfile.TemporaryDirectory(prefix="lip_matched_resume_") as directory:
        path = Path(directory) / "state.pt"
        save_checkpoint(path, model, optimizer, scaler, 1, identity)
        torch.randn(200)
        resumed, resumed_optimizer, resumed_scaler = create()
        try:
            restore_checkpoint(path, resumed, resumed_optimizer, resumed_scaler, {"wrong": True})
        except ValueError:
            rejections.append("changed_identity")
        else:
            raise AssertionError("Changed identity accepted")
        assert restore_checkpoint(path, resumed, resumed_optimizer, resumed_scaler, identity) == 1
        update(resumed, resumed_optimizer, resumed_scaler, objective, data(8), 2, "fp32", "cpu")
        equal_tree(expected_model, resumed.state_dict())
        equal_tree(expected_optimizer, resumed_optimizer.state_dict())
        assert torch.equal(expected_rng, torch.get_rng_state())
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".tmp").write_bytes(b"interrupted")
        try:
            save_checkpoint(path, resumed, resumed_optimizer, resumed_scaler, 2, identity)
        except FileExistsError:
            rejections.append("interrupted_write_preserved")
        else:
            raise AssertionError("Interrupted write overwritten")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    for step in (0, 6001):
        try:
            update(model, optimizer, scaler, objective, data(), step, "fp32", "cpu")
        except ValueError:
            rejections.append("invalid_step_" + str(step))
        else:
            raise AssertionError("Invalid update accepted")
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "status": "optimizer_and_rng_resume_roundtrip_exact",
              "scope": "small dropout model; not a recognition result or GPU test",
              "native_learning_rate_used": True, "gradient_clipping_exercised": True,
              "rejected": rejections, "cuda_initialized": torch.cuda.is_initialized(),
              "engine_sha256": hashlib.sha256((HERE / "engine.py").read_bytes()).hexdigest(),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert not result["cuda_initialized"]
    with (HERE / "Resume_Verification.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
