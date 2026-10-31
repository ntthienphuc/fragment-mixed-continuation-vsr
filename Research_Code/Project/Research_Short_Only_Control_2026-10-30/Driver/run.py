"""Inspect, integrate once on CPU, or execute explicitly admitted matched CUDA continuation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def selected_gpu_usage():
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected not in ("0", "1"):
        raise ValueError("Select exactly one physical GPU from this two-device host")
    result = subprocess.run(
        ["nvidia-smi", "--id=" + selected,
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
    if len(values) != 1:
        raise RuntimeError("The selected-device query did not return one physical GPU")
    return values


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    if args.mode != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from configuration import configure, admit, identity_hash, read, sha
    from loop import execute, save, run_lock

    config, objective, training, validation, identity, overlap = configure(args)
    identity_sha = identity_hash(identity)
    output = Path(args.output).resolve()
    if args.mode == "inspect":
        output.mkdir(parents=True, exist_ok=False)
        save(output / "Identity.json", identity)
        save(output / "Complete.json", {"status": "configuration_verified_without_training",
                                        "identity_sha256": identity_sha, "source_overlap": overlap,
                                        "production_admitted": False})
        print(json.dumps({"status": "configuration_verified_without_training", "identity_sha256": identity_sha}))
        return
    deadline = admit(args, identity)
    import torch
    from phase5.common import seed_all
    from phase5.model import Recognizer
    from engine import restore_checkpoint

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                   "device": "cpu", "python": sys.version}
    if args.mode == "train":
        usage = selected_gpu_usage()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA must expose exactly the selected device")
        if len(usage) != 1 or usage[0] > 1000 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("An otherwise idle bf16-capable GPU is required")
        environment["device"] = torch.cuda.get_device_name()
        probe = read(read(args.admission)["gpu_probe_receipt"])["identity"]
        if (environment["torch"], environment["cuda_build"], environment["device"]) != (
                probe["torch"], probe["cuda"], probe["device"]):
            raise ValueError("GPU environment differs from the admitted resource/resume check")
    if args.resume:
        if read(output / "Identity.json") != identity or (output / "Complete.json").exists():
            raise ValueError("Run identity changed or run already completed")
        if read(output / "Environment.json") != environment:
            raise ValueError("Exact resume requires the original execution environment")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "Identity.json", identity)
        save(output / "Environment.json", environment)
    with run_lock(output):
        attempts = output / "Attempts"
        attempts.mkdir(exist_ok=True)
        attempt = f"{len(list(attempts.glob('*.start.json'))) + 1:03}"
        save(attempts / (attempt + ".start.json"), {"utc": datetime.now(timezone.utc).isoformat(),
                                                  "pid": os.getpid(), "identity_sha256": identity_sha,
                                                  "deadline_unix": deadline, "resume": args.resume})
        try:
            seed_all(17)
            model = Recognizer(config, method="T0", device=identity["device"], scope="all")
            if model.counts()["trainable_parameters"] != 243357028:
                raise ValueError("Native trainable parameter count changed")
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01,
                                          fused=identity["fused_adam"])
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            first = restore_checkpoint(output / "last.pt", model, optimizer, scaler, identity) if args.resume else 0
            if first >= args.stop_after:
                raise ValueError("No further admitted update requested")
            result = execute(model, optimizer, scaler, objective, training, validation, identity,
                             output, attempt, first, args.stop_after, deadline)
            if sha(args.checkpoint) != identity["initializer_sha256"]:
                raise ValueError("Original initializer changed during execution")
            for name, expected in identity["code_sha256"].items():
                if sha(HERE / name) != expected:
                    raise ValueError("Driver source changed during execution")
            if args.mode == "integration" and torch.cuda.is_initialized():
                raise RuntimeError("CPU integration initialized CUDA")
            status = "matched_training_checkpointed_incomplete"
            if args.mode == "integration":
                status = "portable_driver_one_cpu_update_complete"
            elif result["fixed_final_endpoint_reached"]:
                status = "matched_training_complete"
            receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": status, **result,
                       "identity_sha256": identity_sha, "checkpoint_sha256": sha(output / "last.pt"),
                       "source_overlap": overlap, "parameter_counts": model.counts(),
                       "cuda_initialized": torch.cuda.is_initialized(), "recognition_benefit_proven": False}
            terminal = "Complete.json" if args.mode == "integration" or result["updates"] == 6000 else (
                "Attempts/" + attempt + ".stopped.json")
            save(output / terminal, receipt)
            print(json.dumps(receipt), flush=True)
        except BaseException:
            save(attempts / (attempt + ".failure.json"), {"utc": datetime.now(timezone.utc).isoformat(),
                                                        "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "integration", "train"), required=True)
    parser.add_argument("--arm", choices=("short_only",), required=True)
    for name in ("packet", "provider", "runtime", "spec", "input-proof", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stop-after", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--admission")
    args = parser.parse_args()
    if not 1 <= args.stop_after <= 6000 or (args.mode == "train" and not args.admission):
        parser.error("Use a valid update cap and an explicit admission for CUDA training")
    main(args)
