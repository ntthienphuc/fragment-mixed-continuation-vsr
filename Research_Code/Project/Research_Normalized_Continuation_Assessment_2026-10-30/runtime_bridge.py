"""Use unchanged inference, explicitly binding the native graph of the new training recipe."""

from importlib import import_module
import importlib.util
import inspect
from pathlib import Path
import sys

from common import ROOT, read, sha, load_module

_BINDING = None


def create_runtime(binding, model_id, device="cuda", root=ROOT):
    global _BINDING
    root = Path(root).resolve()
    binding = Path(binding).resolve()
    if _BINDING is not None and _BINDING != binding:
        raise RuntimeError("Use a fresh process for another model binding")
    _BINDING = binding
    plan = read(Path(__file__).parent / "Plan.json")
    for relative, expected in plan["runtime_code_sha256"].items():
        assert sha(root / relative) == expected, "Frozen inference code changed: " + relative
    sys.path.insert(0, str(root / "Research_Server_Confirmation_2026-10-19"))
    base = load_module("matched_original_runtime", root / plan["runtime_source"])
    base.HERE = binding

    class MatchedRuntime(base.Runtime):
        def _load_auto_avsr(self):
            import torch

            assert self.entry["state_field"] == "model"
            assert self.entry["native_graph"] == self.entry["training_identity"]["native_graph"] == "T0"
            source = self.path(self.entry["source"])
            base.claim_source(source)
            package_path = self.path(base.PHASE + "/phase5")
            package_name = "confirmation_frozen_phase5"
            if package_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    package_name, package_path / "__init__.py",
                    submodule_search_locations=[str(package_path)])
                package = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = package
                spec.loader.exec_module(package)
            recognizer = import_module(package_name + ".model").Recognizer
            config = read(self.path(self.entry["config"]))
            for field in ("source", "tokenizer", "units"):
                config[field] = str(self.path(self.entry[field]))
            config["checkpoint"] = str(self.path(self.entry["base_checkpoint"]))
            config["checkpoint_sha256"] = self.entry["base_checkpoint_sha256"]
            config["state_field"] = "raw"
            self.recognizer = recognizer(config, method="T0", device="cpu", scope="all")
            payload = torch.load(self.path(self.entry["checkpoint"]), map_location="cpu", weights_only=True)
            self.export_metadata = base.validate_export(payload, self.entry)
            self.recognizer.load_state_dict(payload["model"], strict=True)
            del payload
            self.recognizer.eval().requires_grad_(False).float().to(self.device)
            self.model = self.recognizer.native
            self.vocabulary = list(self.recognizer.tokenizer.tokens)
            assert self.vocabulary == base.read_vocabulary(self.path(self.entry["units"]))
            assert Path(inspect.getfile(type(self.model))).resolve().is_relative_to(source)
            self.factory_path = self.path(self.registry["factory_source"])

    runtime = MatchedRuntime(model_id, device=device, threads=2, root=root)
    assert runtime.recognizer.method == "T0" and runtime.recognizer.auxiliary is None
    assert runtime.recognizer.readout is None
    assert runtime.entry["classes"] == 2002 and runtime.blank_id == 0 and runtime.eos_id == 2001
    assert runtime.identity["strict_state_load"] and runtime.identity["precision"] == "float32"
    assert not runtime.identity["tf32"] and not runtime.identity["reference_used_for_inference"]
    runtime.identity["matched_bridge_sha256"] = sha(__file__)
    runtime.identity["native_graph_binding"] = "T0"
    runtime.identity["training_identity_modified"] = False
    for name in ("encode", "decode_native", "_load_search", "text"):
        assert getattr(MatchedRuntime, name) is getattr(base.Runtime, name)
    return runtime


def decode(runtime, encoded, candidate, choices, root=ROOT):
    parent = Path(root) / "Research_English_Transfer_Benchmark_2026-10-28"
    original = list(sys.path)
    try:
        sys.path.insert(0, str(parent))
        module = load_module("matched_unchanged_decoder", parent / "runtime_bridge.py")
    finally:
        sys.path[:] = original
    return module.decode(runtime, encoded, candidate, {"candidate_order": choices, "beam_size": 40, "penalty": 0})
