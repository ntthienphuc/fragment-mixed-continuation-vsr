"""Reuse the exact native graph and each parent's pixel reader for new control states."""

import hashlib
import importlib.util
import sys

from mechanism_common import ENGINE, HERE, ROOT, VI, lines, read, sha, validate_inputs


def create_runtime(model, device="cpu", threads=2):
    plan = validate_inputs()
    assert model in plan["models"]
    sys.path.insert(0, str(ENGINE))
    name = "mechanism_extension_native_runtime"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, ENGINE / "native_runtime.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        module.HERE = HERE
    runtime = sys.modules[name].Runtime(model, device=device, threads=threads, root=ROOT)
    expected_classes = 2002 if runtime.entry["language"] == "vi" else 5049
    assert runtime.entry["classes"] == expected_classes
    assert runtime.blank_id == 0 and runtime.eos_id == expected_classes - 1
    assert runtime.identity["strict_state_load"] and runtime.identity["precision"] == "float32"
    assert runtime.identity["reference_used_for_inference"] is False
    runtime.identity["mechanism_bridge_sha256"] = sha(__file__)
    return runtime


def pixels_for(row):
    sys.path.insert(0, str(ENGINE))
    if row["language"] == "en":
        ready = ROOT / "Research_English_Transfer_Readiness_2026-10-27"
        sys.path.insert(0, str(ready))
        from strict_inputs import load_pixels

        return load_pixels(row, root=ROOT)
    assert row["language"] == "vi"
    from data_io import load_pixels

    metadata = {key: value for key, value in row.get("metadata", {}).items() if key != "pixels_sha256"}
    item = {**row, "input": {**row["input"], "color_space": "gray"}, "metadata": metadata}
    pixels = load_pixels(item, root=ROOT)
    verification = read(VI / "Pixels_01/Verification.json")
    index = VI / "Pixels_01/Index.jsonl"
    assert sha(index) == verification["index_sha256"]
    matches = [value for value in lines(index) if value["clip_id"] == row["clip_id"]]
    assert len(matches) == 1
    receipt = matches[0]
    assert receipt["media_sha256"] == row["input"]["sha256"]
    assert receipt["gray_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    return pixels
