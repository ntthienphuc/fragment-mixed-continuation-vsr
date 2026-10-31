"""Use the existing recognizers without changing graph, vocabulary, or transforms."""

import hashlib
import importlib.util
import sys

from common import ENGINE, HERE, PARENT, ROOT, read, sha

_FAMILY = None


def family_for(model):
    if model == "VC_NATIVE":
        return "vicocktail"
    if model in read(HERE / "model_registry.json")["models"]:
        return "standard"
    raise ValueError(f"Unregistered model: {model}")


def claim_family(family):
    global _FAMILY
    if family not in {"standard", "vicocktail"}:
        raise ValueError("Unknown graph family")
    if _FAMILY is not None and _FAMILY != family:
        raise RuntimeError("Graph families require separate processes")
    _FAMILY = family


def create_runtime(model, device="cuda", threads=2):
    """Call only after admission; imports are deliberately lazy."""
    family = family_for(model)
    claim_family(family)
    sys.path.insert(0, str(ENGINE))
    if family == "vicocktail":
        from vicocktail_runtime import Runtime
        expected = read(HERE / "Native_Reference_Registry.json")
        assert read(ENGINE / "VC_Model_Registry.json") == expected
    else:
        name = "full_benchmark_standard_runtime"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, PARENT / "runtime_core.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            module.HERE = HERE
        Runtime = sys.modules[name].Runtime
    runtime = Runtime(model, device=device, threads=threads, root=ROOT)
    assert runtime.entry["classes"] == 2002 and runtime.blank_id == 0
    assert runtime.eos_id == 2001 and runtime.identity["strict_state_load"]
    assert runtime.identity["precision"] == "float32"
    assert runtime.identity["reference_used_for_inference"] is False
    runtime.identity["full_benchmark_bridge_sha256"] = sha(__file__)
    runtime.identity["full_benchmark_family"] = family
    return runtime


def pixels_for(row, model, pixel_receipt):
    sys.path.insert(0, str(ENGINE))
    from data_io import load_pixels

    color = "rgb" if family_for(model) == "vicocktail" else "gray"
    metadata = {key: value for key, value in row.get("metadata", {}).items() if key != "pixels_sha256"}
    item = {**row, "input": {**row["input"], "color_space": color}, "metadata": metadata}
    pixels = load_pixels(item, root=ROOT)
    actual = hashlib.sha256(pixels.tobytes()).hexdigest()
    assert pixel_receipt["clip_id"] == row["clip_id"]
    assert pixel_receipt["media_sha256"] == row["input"]["sha256"]
    assert pixel_receipt[color + "_sha256"] == actual
    return pixels


def collapse(path, blank=0, eos=2001):
    output = []
    previous = None
    for value in path:
        token = int(value)
        if token != previous and token not in {blank, eos}:
            output.append(token)
        previous = token
    return output


def decode(runtime, encoded, candidate, plan):
    """No reference is accepted. Native beam cap is the full encoded frame count."""
    import numpy as np
    import time

    assert candidate in plan["candidate_order"]
    if candidate.startswith("beam_c"):
        result = runtime.decode_native(
            encoded["states"], float(candidate.removeprefix("beam_c")),
            beam_size=plan["beam_size"], penalty=plan["penalty"],
        )
        assert result["cap"] == len(encoded["states"])
        return result
    began = time.monotonic()
    scores = encoded["logp"].copy()
    scores[:, runtime.blank_id] += float(candidate.removeprefix("greedy_b"))
    tokens = collapse(np.argmax(scores, axis=1), runtime.blank_id, runtime.eos_id)
    return {"text": runtime.text(tokens), "tokens": tokens,
            "seconds": time.monotonic() - began, "search_calls": 0}
