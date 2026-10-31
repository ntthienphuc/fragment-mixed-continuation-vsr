"""Bounded source-development encoding, separate from test recognition."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
RUNTIME_CHECK = PHASE / "Runtime_Validation"
INPUTS = PHASE / "Transfer_Inputs_01"
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def lock():
    value = read(HERE / "Lock.json")
    assert value["status"] == "source_development_encoding_only"
    assert value["test_inference_authorized"] is False and value["paid_allocation"] is False
    for name, digest in value["sha256"].items():
        assert sha(ROOT / name) == digest, name
    return value


def create_runtime(model):
    sys.path.insert(0, str(ENGINE))
    name = "english_development_encoder_runtime"
    assert name not in sys.modules
    specification = importlib.util.spec_from_file_location(name, ENGINE / "native_runtime.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.HERE = INPUTS
    runtime = module.Runtime(model, device="cuda", threads=2, root=ROOT)
    assert runtime.entry["language"] == "en" and runtime.entry["classes"] == 5049
    assert runtime.blank_id == 0 and runtime.eos_id == 5048
    assert runtime.identity["strict_state_load"] is True and runtime.identity["precision"] == "float32"
    assert runtime.identity["reference_used_for_inference"] is False
    return runtime
