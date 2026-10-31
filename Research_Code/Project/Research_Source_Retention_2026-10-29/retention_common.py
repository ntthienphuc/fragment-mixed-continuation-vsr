"""Preserve the original model, pixels, tokenizers and decoder implementation."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28"
READY = PARENT / "Source_Retention_Readiness_01"
INPUTS = PARENT / "Source_Retention_Runtime_01"
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
PREPARED_SHA256 = "57043031647da4c158d8b83b5680d4e2031dfbd8ec1d85af4d21c7521851e084"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def logical(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def prepared_plan():
    path = READY / "Prepared_Plan.json"
    require(sha(path) == PREPARED_SHA256, "Source preparation changed")
    value = read(path)
    for name, digest in value["prerequisites"].items():
        path = READY / name if "/" not in name else ROOT / name
        require(sha(path) == digest, "Source prerequisite changed: " + name)
    return value


def load_module(name, path):
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


def source(language):
    return load_module("retention_source_inputs", INPUTS / "source_loader.py").open_source(language)


def native_path(model):
    original = ROOT / "Research_Transfer_Initialization_2026-10-27/runtime_core.py"
    return original if model.startswith("VLR_") else ENGINE / "native_runtime.py"


def runtime(model, device):
    plan = prepared_plan()
    require(model in plan["models"], "Unknown frozen model")
    declaration = plan["models"][model]
    registry_path = ROOT / declaration["registry"]
    require(read(registry_path)["models"][model] == declaration["identity"], "Model entry changed")
    sys.path.insert(0, str(ENGINE))
    module = load_module("retention_native_" + model, native_path(model))
    module.HERE = registry_path.parent
    instance = module.Runtime(model, device=device, threads=2, root=ROOT)
    identity = instance.identity
    require(identity["registry_entry"] == declaration["identity"], "Loaded a different model")
    require(identity["strict_state_load"] and identity["precision"] == "float32", "Wrong state/precision")
    require(identity["tf32"] is False and identity["reference_used_for_inference"] is False,
            "Inference policy changed")
    expected_classes = 2002 if model.startswith("VLR_") else 5049
    require(identity["classes"] == expected_classes and instance.blank_id == 0, "Wrong vocabulary")
    require(instance.eos_id == expected_classes - 1, "Wrong EOS")
    return instance


def decode(instance, encoded, candidate, choices):
    parent = ROOT / "Research_English_Transfer_Benchmark_2026-10-28"
    original = list(sys.path)
    try:
        sys.path.insert(0, str(parent))
        module = load_module("retention_frozen_decoder", parent / "runtime_bridge.py")
    finally:
        sys.path[:] = original
    contract = {"candidate_order": choices, "beam_size": 40, "penalty": 0}
    return module.decode(instance, encoded, candidate, contract)


def checked_plan():
    value = read(HERE / "Plan.json")
    require(value["prepared_plan_sha256"] == PREPARED_SHA256, "Wrong preparation")
    prepared_plan()
    for name, digest in value["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Frozen code changed: " + name)
    for name, digest in value["prerequisite_sha256"].items():
        require(sha(ROOT / name) == digest, "Completion evidence changed: " + name)
    return value
