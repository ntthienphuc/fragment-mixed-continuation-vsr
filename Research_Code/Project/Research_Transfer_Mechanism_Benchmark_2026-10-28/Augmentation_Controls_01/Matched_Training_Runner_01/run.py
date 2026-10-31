"""Run a separately identified matched objective; CPU integration is the default limit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03"
ADAPTER = HERE.parent / "Native_Matched_Adapter_01"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(args):
    started = time.monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    if args.mode == "integration":
        if args.device != "cpu" or args.precision != "fp32" or args.fused_adam or args.stop_after != 1:
            raise ValueError("Initial integration is exactly one CPU FP32, unfused update")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not args.device.startswith("cuda") or args.precision != "bf16" or not args.fused_adam:
        raise ValueError("Long training must use the archived CUDA bf16/fused-Adam profile")
    import torch

    sys.path.insert(0, str(NATIVE / "source"))
    sys.path.insert(0, str(ADAPTER))
    from phase5.common import code_identity, seed_all
    from phase5.data import check_disjoint
    from phase5.evaluation import evaluate
    from phase5.model import Recognizer
    from phase5.runtime import autocast, samples
    from adapter import MatchedObjective
    from data_io import TrainingData, ValidationData
    from engine import update, save_checkpoint, restore_checkpoint

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_all(17)
    autocast(args.precision, args.device)
    spec_path = NATIVE / "Inputs.json"
    if sha(spec_path) != "ab7df96dcd97d658b3147b708cc2ae6c749e461720426de470eb0f78bb271fa1":
        raise ValueError("Native input specification changed")
    spec = read(spec_path)
    if code_identity() != spec["identity"]["code_sha256"]:
        raise ValueError("Archived native recognizer changed")
    forward = read(ADAPTER / "CPU_Forward_01/Complete.json")
    if forward["status"] != "native_cpu_forward_adapter_comparisons_passed":
        raise ValueError("Matched native forward check is missing")
    objective = MatchedObjective("vlr", args.arm)
    forward_plan_path = ADAPTER / "CPU_Forward_01/Plan.json"
    if sha(forward_plan_path) != forward["plan_sha256"]:
        raise ValueError("Native forward proof changed")
    forward_plan = read(forward_plan_path)
    for field in ("dataset", "adapter_sha256", "schedule_sha256", "word_schedule_sha256"):
        if objective.identity[field] != forward_plan["schedule_identity"][field]:
            raise ValueError("Objective differs from the verified native forward adapter")
    resume_proof = read(HERE / "Resume_Verification.json")
    if resume_proof["status"] != "optimizer_and_rng_resume_roundtrip_exact":
        raise ValueError("Optimizer/RNG resume has not been verified")
    if resume_proof["engine_sha256"] != sha(HERE / "engine.py"):
        raise ValueError("Update engine differs from verified resume test")
    config = spec["config"].copy()
    checkpoint = ROOT / "Server_Rescue_2026-10-15/Assets/VLR/pretrained.pt"
    config["checkpoint"] = str(checkpoint)
    if sha(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Initializer bytes changed")
    data = TrainingData(objective, spec)
    validation = ValidationData(spec)
    overlap = check_disjoint(data, validation)
    code = {name: sha(HERE / name) for name in ("data_io.py", "engine.py", "run.py", "Protocol.md")}
    identity = {
        "schema": 1, "native_graph": "T0", "objective": "matched_" + args.arm,
        "dataset": "vlr", "scope": "all", "schedule": objective.identity,
        "seed": 17, "steps": 6000, "accumulation": 8, "lr": 2e-6,
        "weight_decay": 0.01, "gradient_clip": 5.0, "eval_every": 500,
        "validation_limit": 512, "validation_view": "window", "selection": "fixed_final_step",
        "precision": args.precision, "fused_adam": args.fused_adam, "device_type": args.device,
        "threads": 2, "workers": 2, "mode": args.mode, "code_sha256": code,
        "native_code_identity": code_identity(), "initializer_sha256": config["checkpoint_sha256"],
        "train_manifest_sha256": data.manifest_sha256, "val_manifest_sha256": validation.manifest_sha256,
    }
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.mode == "train":
        admission = read(args.admission)
        if admission["status"] != "matched_gpu_training_admitted":
            raise ValueError("No measured GPU training admission")
        if admission["run_identity_sha256"] != identity_sha:
            raise ValueError("GPU admission is for a different run")
        deadline = datetime.fromisoformat(admission["deadline_utc"]).timestamp()
        if deadline <= time.time() or admission["maximum_updates"] != 6000:
            raise ValueError("Training admission expired or budget changed")
    else:
        deadline = time.time() + 1200
    output = Path(args.output).resolve()
    if args.resume:
        if read(output / "Identity.json") != identity or (output / "Complete.json").exists():
            raise ValueError("Run cannot be resumed under this identity")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "Identity.json", identity)
    attempts = output / "Attempts"
    attempts.mkdir(exist_ok=True)
    number = len(list(attempts.glob("*.start.json"))) + 1
    attempt = f"{number:03}"
    save(attempts / (attempt + ".start.json"), {
        "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "identity_sha256": identity_sha, "deadline_unix": deadline, "resume": args.resume,
    })
    model = Recognizer(config, method="T0", device=args.device, scope="all")
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad),
                                  lr=2e-6, weight_decay=0.01, fused=args.fused_adam)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step = restore_checkpoint(output / "last.pt", model, optimizer, scaler, identity) if args.resume else 0
    if step >= args.stop_after:
        raise ValueError("No additional admitted updates remain")
    stream = samples(data, step * 8, 48000, 17, workers=2)
    try:
        for step in range(step + 1, 6001):
            if time.time() >= deadline:
                raise TimeoutError("Finite execution deadline reached")
            result = update(model, optimizer, scaler, objective, stream, step, args.precision, args.device, deadline)
            result["seconds"] = time.monotonic() - started
            with (attempts / (attempt + ".steps.jsonl")).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({key: result[key] for key in ("step", "loss", "lr", "gradient_norm", "seconds")}),
                  flush=True)
            if step % 500 == 0 or step == 6000:
                _, metrics = evaluate(model, validation, context="both", limit=512, view="window")
                if metrics["inference_failed"] or not metrics["metrics"]["word"]["units"]:
                    raise RuntimeError("Source validation did not retain its full required coverage")
                save(attempts / (attempt + f".validation_{step:06}.json"), metrics)
            if step % 500 == 0 or step == args.stop_after or step == 6000:
                save_checkpoint(output / "last.pt", model, optimizer, scaler, step, identity)
            if step >= args.stop_after:
                break
    finally:
        stream.close()
    if sha(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Original initializer changed")
    for name, expected in code.items():
        if sha(HERE / name) != expected:
            raise ValueError("Runner changed while executing")
    if args.mode == "integration" and torch.cuda.is_initialized():
        raise RuntimeError("CPU integration unexpectedly initialized CUDA")
    status = "matched_training_checkpointed_incomplete"
    if step == 6000:
        status = "matched_training_complete"
    elif args.mode == "integration":
        status = "bounded_integration_update_complete"
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "status":
        status,
        "updates": step, "presentations": step * 8, "identity_sha256": identity_sha,
        "checkpoint_sha256": sha(output / "last.pt"), "overlap": overlap,
        "seconds": time.monotonic() - started, "initializer_file_unchanged": True,
        "gpu_training_readiness_proven": False, "recognition_benefit_proven": False,
        "parameter_counts": model.counts(),
    }
    terminal = "Complete.json" if step == 6000 or args.mode == "integration" else attempt + ".stopped.json"
    save(output / terminal, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("full", "fixed_time", "whole_word"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("integration", "train"), default="integration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=1)
    parser.add_argument("--admission")
    arguments = parser.parse_args()
    if not 1 <= arguments.stop_after <= 6000 or (arguments.mode == "train" and not arguments.admission):
        parser.error("A valid update cap and, for long training, a GPU admission are required")
    try:
        run(arguments)
    except BaseException:
        failure = Path(arguments.output) / ("Failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
        if failure.parent.is_dir():
            save(failure, {"utc": datetime.now(timezone.utc).isoformat(), "error": traceback.format_exc()})
        raise
