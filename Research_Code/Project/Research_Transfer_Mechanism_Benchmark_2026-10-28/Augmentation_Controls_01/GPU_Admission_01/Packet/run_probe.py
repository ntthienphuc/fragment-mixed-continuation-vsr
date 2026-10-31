"""Isolated native CUDA resource/resume check; never produce a recognition claim."""

import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE / "support/Native_Matched_Adapter_01"))
sys.path.insert(0, str(HERE / "support/Matched_Training_Runner_01"))

import torch
from adapter import MatchedObjective
from engine import update, save_checkpoint, restore_checkpoint
from phase5.common import code_identity, seed_all
from phase5.model import Recognizer
from phase5.runtime import autocast
from probe_support import Inputs, complete_state, fingerprint, read, save, sha, verify_packet


def require_idle():
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    used = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if len(used) != 1 or used[0] > 1000:
        raise RuntimeError("Training probe requires one otherwise idle GPU")


def create(config):
    seed_all(17)
    model = Recognizer(config, method="T0", device="cuda", scope="all")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def run(args):
    began = time.monotonic()
    require_idle()
    plan = verify_packet(HERE)
    assert code_identity() == plan["native_code_identity"]
    assert sha(args.checkpoint) == plan["config"]["checkpoint_sha256"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_bf16_supported()
    deadline = time.time() + plan["maximum_seconds"]
    config = plan["config"].copy()
    config["checkpoint"] = str(Path(args.checkpoint).resolve())
    inputs = Inputs(HERE, plan)
    identity = {"plan_sha256": sha(HERE / "Plan.json"), "packet_manifest_sha256": sha(HERE / "Manifest.json"),
                "graph": "T0", "device": torch.cuda.get_device_name(), "precision": "bf16",
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "tf32": False,
                "checkpoint_sha256": sha(args.checkpoint), "native_code_identity": code_identity()}
    save(output / "Start.json", {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                                 "deadline_unix": deadline, "identity": identity})
    results = []
    resume_result = None
    for arm in plan["arms"]:
        if time.time() >= deadline:
            raise TimeoutError("Resource probe deadline reached")
        objective = MatchedObjective("vlr", arm)
        model, optimizer, scaler = create(config)
        buffers_before = fingerprint(dict(model.named_buffers()))
        initial_parameters = fingerprint(model.state_dict())
        arm_identity = {**identity, "arm": arm, "schedule": objective.identity, "purpose": "resource_probe_only"}
        torch.cuda.reset_peak_memory_stats()
        stream = inputs.stream(0, 16)
        for step in (1, 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = update(model, optimizer, scaler, objective, stream, step, "bf16", "cuda", deadline)
            torch.cuda.synchronize()
            result.update(arm=arm, update_seconds=time.perf_counter() - start,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            results.append(result)
            save(output / (arm + "_step_" + str(step) + ".json"), result)
            print({key: result[key] for key in ("arm", "step", "loss", "update_seconds", "peak_allocated_mib")},
                  flush=True)
            if arm == "whole_word" and step == 1:
                save_checkpoint(output / "Resume_Step1.pt", model, optimizer, scaler, step, arm_identity)
        assert fingerprint(dict(model.named_buffers())) == buffers_before
        assert fingerprint(model.state_dict()) != initial_parameters
        assert all(int(state["step"]) == 2 for state in optimizer.state.values())
        if arm == "whole_word":
            expected = complete_state(model, optimizer, scaler)
            del model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
            model, optimizer, scaler = create(config)
            assert restore_checkpoint(output / "Resume_Step1.pt", model, optimizer, scaler, arm_identity) == 1
            replay = update(model, optimizer, scaler, objective, inputs.stream(8, 16), 2, "bf16", "cuda", deadline)
            actual = complete_state(model, optimizer, scaler)
            assert actual == expected, "Native CUDA model/optimizer/RNG continuation differs"
            assert replay["samples"] == results[-1]["samples"] and replay["loss"] == results[-1]["loss"]
            resume_result = {"status": "exact_native_cuda_model_optimizer_rng_resume",
                             "expected_sha256": expected, "actual_sha256": actual,
                             "checkpoint_sha256": sha(output / "Resume_Step1.pt")}
            save(output / "Resume_Verification.json", resume_result)
        for position in sorted(set(plan["extreme_positions"].values())):
            if time.time() >= deadline:
                raise TimeoutError("Resource extreme deadline reached")
            optimizer.zero_grad(set_to_none=True)
            row, pixels = inputs.sample(position)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.train()
            with autocast("bf16", "cuda"):
                loss, details = objective.loss(model, pixels, row, position, 17, "both")
            (loss / 8).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            torch.cuda.synchronize()
            extreme = {"arm": arm, "sample": position, "optimizer_update": False,
                       "loss": float(loss.detach()), "gradient_norm": float(norm), **details,
                       "seconds": time.perf_counter() - start,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            save(output / (arm + "_extreme_" + str(position) + ".json"), extreme)
            del loss, pixels
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    assert resume_result is not None and sha(args.checkpoint) == identity["checkpoint_sha256"]
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "bounded_gpu_resource_resume_passed",
               "identity": identity, "seconds": time.monotonic() - began, "native_updates": 7,
               "recognition_benefit_proven": False, "full_training_admitted": False,
               "output_sha256": {path.name: sha(path) for path in sorted(output.glob("*.json"))},
               "checkpoint_sha256": resume_result["checkpoint_sha256"]}
    save(output / "Complete.json", receipt)
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    existed_before = Path(args.output).exists()
    try:
        run(args)
    except BaseException:
        target = Path(args.output)
        if not existed_before and target.is_dir() and not (target / "Failure.json").exists():
            save(target / "Failure.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                          "error": traceback.format_exc()})
        raise
