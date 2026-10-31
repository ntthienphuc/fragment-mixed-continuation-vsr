"""Bind the existing native recognizer to the verified English model registry."""

import importlib.util
import sys

from common import ENGINE, INPUTS, ROOT, sha


def create_runtime(model):
    sys.path.insert(0, str(ENGINE))
    name = "english_transfer_native_runtime"
    assert name not in sys.modules, "Use a separate process for every model check"
    source = ENGINE / "native_runtime.py"
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.HERE = INPUTS
    runtime = module.Runtime(model, device="cuda", threads=2, root=ROOT)
    assert runtime.entry["language"] == "en" and runtime.entry["classes"] == 5049
    assert runtime.blank_id == 0 and runtime.eos_id == 5048
    assert runtime.identity["strict_state_load"] is True
    assert runtime.identity["precision"] == "float32"
    assert runtime.identity["reference_used_for_inference"] is False
    runtime.identity["english_runtime_bridge_sha256"] = sha(__file__)
    return runtime
