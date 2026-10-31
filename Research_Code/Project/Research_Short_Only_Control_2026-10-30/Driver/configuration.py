"""Bind the native runtime, portable source data and execution policy to one run identity."""

import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PACKET_SHA = "e9c8d65c05da01953c30c98e32d40c66a5e2555d393e57e08a972f455cbb8923"
SPEC_SHA = "ab7df96dcd97d658b3147b708cc2ae6c749e461720426de470eb0f78bb271fa1"
ENGINE_SHA = "d1deb74136312d7f9737649861fbe62301e4a8d42cf7c38932c4244c3ad6cba7"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def configure(args):
    packet = Path(args.packet).resolve()
    provider = Path(args.provider).resolve()
    if sha(packet / "Manifest.json") != PACKET_SHA or sha(args.spec) != SPEC_SHA:
        raise ValueError("Native packet or specification changed")
    sys.path.insert(0, str(packet))
    sys.path.insert(0, str(packet / "source"))
    sys.path.insert(0, str(packet / "support/Native_Matched_Adapter_01"))
    sys.path.insert(0, str(packet / "support/Matched_Training_Runner_01"))
    sys.path.insert(0, str(provider))
    from objective import ShortOnlyObjective
    from portable_data import InputPaths, TrainingData, ValidationData
    from phase5.common import code_identity
    from phase5.data import check_disjoint
    from phase5.text import reference_text
    from probe_support import verify_packet

    plan = verify_packet(packet)
    spec = read(args.spec)
    proof = read(args.input_proof)
    if proof["status"] != "portable_source_inputs_native_order_pixels_and_coverage_verified":
        raise ValueError("Portable input check is missing")
    if proof["provider_sha256"] != sha(provider / "portable_data.py"):
        raise ValueError("Provider differs from its real-data check")
    if proof["packet_manifest_sha256"] != PACKET_SHA or proof["native_spec_sha256"] != SPEC_SHA:
        raise ValueError("Portable check belongs to another native runtime")
    if code_identity() != plan["native_code_identity"] or code_identity() != proof["native_code_identity"]:
        raise ValueError("Native model graph changed")
    if sha(packet / "support/Matched_Training_Runner_01/engine.py") != ENGINE_SHA:
        raise ValueError("Native update/resume engine changed")
    paths = InputPaths(args.runtime, proof["identity"]["runtime_sha256"])
    if paths.identity != proof["identity"]:
        raise ValueError("Data paths differ from verified input identity")
    objective = ShortOnlyObjective("vlr", args.arm)
    training = TrainingData(objective, spec, paths)
    validation = ValidationData(spec, paths)
    overlap = check_disjoint(training, validation)
    selected = validation.selection(512)
    eligible = [row for row in selected if row["window_eligible"]]
    expected_validation = {
        "clip_ids": [row["clip_id"] for row in eligible], "requested_rows": 512,
        "scored_rows": len(eligible), "window_ineligible": len(selected) - len(eligible),
        "reference_units": sum(len(reference_text(row["window_reference"], "vi").split()) for row in eligible),
    }
    if len(eligible) != 504 or len(selected) != 512 or expected_validation["reference_units"] == 0:
        raise ValueError("Fixed source diagnostic population changed")
    config = plan["config"].copy()
    config["checkpoint"] = str(Path(args.checkpoint).resolve())
    if sha(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Original initializer changed")
    mode = "integration" if args.mode == "integration" else "train"
    if mode == "integration" and (args.arm != "short_only" or args.stop_after != 1 or args.resume):
        raise ValueError("Integration is one fresh short-only CPU update")
    code = {name: sha(HERE / name) for name in ("configuration.py", "loop.py", "run.py", "Protocol.md", "objective.py")}
    identity = {
        "schema": 1, "mode": mode, "native_graph": "T0", "objective": "matched_" + args.arm,
        "dataset": "vlr", "scope": "all", "seed": 17, "steps": 6000, "accumulation": 8,
        "lr": 2e-6, "weight_decay": 0.01, "gradient_clip": 5.0, "eval_every": 500,
        "selection": "fixed_final_step", "source_diagnostic": expected_validation,
        "schedule": objective.identity, "inputs": paths.identity, "input_proof_sha256": sha(args.input_proof),
        "native_code_identity": code_identity(), "packet_manifest_sha256": PACKET_SHA,
        "initializer_sha256": config["checkpoint_sha256"], "update_engine_sha256": ENGINE_SHA,
        "device": "cpu" if mode == "integration" else "cuda",
        "precision": "fp32" if mode == "integration" else "bf16", "fused_adam": mode != "integration",
        "tf32": False, "deterministic": True, "threads": 2, "workers": 2, "code_sha256": code,
    }
    return config, objective, training, validation, identity, overlap


def admit(args, identity):
    if args.mode == "inspect":
        raise ValueError("Inspection cannot admit training")
    if args.mode == "integration":
        return time.time() + 1200
    from datetime import datetime

    admission = read(args.admission)
    if admission["status"] != "matched_gpu_training_admitted":
        raise ValueError("Long training is not admitted")
    if admission["run_identity_sha256"] != identity_hash(identity):
        raise ValueError("Admission names another run identity")
    if admission["maximum_updates"] != 6000 or not 1 <= args.stop_after <= 6000:
        raise ValueError("Training update budget changed")
    if sha(admission["gpu_probe_receipt"]) != admission["gpu_probe_receipt_sha256"]:
        raise ValueError("GPU probe receipt changed")
    probe = read(admission["gpu_probe_receipt"])
    if probe["status"] != "bounded_gpu_resource_resume_passed" or probe["native_updates"] != 7:
        raise ValueError("Real CUDA update/resume/resource check did not pass")
    if probe["identity"]["packet_manifest_sha256"] != PACKET_SHA:
        raise ValueError("GPU probe used another native packet")
    if probe["identity"]["checkpoint_sha256"] != identity["initializer_sha256"]:
        raise ValueError("GPU probe used another initializer")
    resume_path = Path(admission["gpu_probe_receipt"]).parent / "Resume_Verification.json"
    if sha(resume_path) != probe["output_sha256"]["Resume_Verification.json"]:
        raise ValueError("GPU resume evidence changed")
    resume = read(resume_path)
    if resume["status"] != "exact_native_cuda_model_optimizer_rng_resume":
        raise ValueError("GPU native resume has not passed")
    if resume["actual_sha256"] != resume["expected_sha256"]:
        raise ValueError("GPU restored state differs")
    deadline = datetime.fromisoformat(admission["deadline_utc"]).timestamp()
    if not time.time() < deadline <= time.time() + 72 * 3600:
        raise ValueError("Expired or excessive execution deadline")
    return deadline
