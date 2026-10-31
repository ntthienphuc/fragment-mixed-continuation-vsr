"""Native English inference with lossless source-cache reuse and verified raw pixels."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import time
import zipfile

from english_common import ENGINE, HERE, READY, ROOT, lines, read, require, root_path, sha


def create_runtime(model, device="cuda", threads=2):
    sys.path.insert(0, str(ENGINE))
    name = "english_full_benchmark_native_runtime"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, ENGINE / "native_runtime.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        module.HERE = HERE
    runtime = sys.modules[name].Runtime(model, device=device, threads=threads, root=ROOT)
    require(runtime.entry["language"] == "en" and runtime.entry["classes"] == 5049, "English head required")
    require(runtime.blank_id == 0 and runtime.eos_id == 5048, "Wrong special tokens")
    require(runtime.identity["strict_state_load"] is True, "Strict checkpoint load required")
    require(runtime.identity["precision"] == "float32" and runtime.identity["tf32"] is False, "Wrong precision")
    require(runtime.identity["reference_used_for_inference"] is False, "Reference cannot enter inference")
    runtime.identity["english_benchmark_bridge_sha256"] = sha(__file__)
    return runtime


def pixels_for(row, locations):
    sys.path.insert(0, str(READY))
    from strict_inputs import load_pixels

    if row["split"] == "development":
        require(row["input"]["kind"] == "zip_npz", "Wrong source input kind")
        return load_pixels(row, root=ROOT)
    require(row["split"] == "assessment" and row["input"]["kind"] == "npz_frames", "Wrong assessment input")
    name = Path(row["input"]["path"].replace("\\", "/")).name
    require(name == Path(row["clip_id"]).stem + ".npz", "Unexpected packet identity")
    resolved = root_path(locations["wild_root"]) / name
    declaration = {**row, "input": {**row["input"], "path": str(resolved)}}
    return load_pixels(declaration, root=ROOT)


class DevelopmentCache:
    def __init__(self, model, locations, runtime):
        self.model = model
        self.runtime = runtime
        entry = read(HERE / "Source_Caches.json")[model]
        require(entry["model_id"] == model, "Cache model mismatch")
        require(entry["checkpoint_sha256"] == runtime.entry["checkpoint_sha256"], "Cache checkpoint mismatch")
        path = root_path(locations["states"][model])
        require(sha(path) == entry["archive_sha256"], "Cache archive changed")
        index_path = ROOT / entry["index"]
        require(sha(index_path) == entry["index_sha256"], "Cache index changed")
        self.records = lines(index_path)
        require(len(self.records) == entry["clips"] == 1082, "Source cache incomplete")
        self.archive = zipfile.ZipFile(path)
        require(len(set(self.archive.namelist())) == len(self.archive.namelist()) == 1082, "Bad archive members")
        require(set(self.archive.namelist()) == {row["member"] for row in self.records}, "Cache member mismatch")

    def close(self):
        self.archive.close()

    def encoded(self, position, row):
        import numpy as np
        import torch

        began = time.monotonic()
        record = self.records[position]
        require(record["position"] == position and record["model"] == self.model, "Cache position mismatch")
        for name in ("clip_id", "source_id", "frames"):
            require(record[name] == row[name], "Cache input differs: " + name)
        require(record["source_payload_sha256"] == row["input"]["sha256"], "Cache pixels originate elsewhere")
        payload = self.archive.read(record["member"])
        require(hashlib.sha256(payload).hexdigest() == record["payload_sha256"], "Feature packet changed")
        values = np.load(io.BytesIO(payload), allow_pickle=False)
        require(values.shape == (row["frames"], 768) and values.dtype == np.float32, "Feature shape/dtype differs")
        require(np.isfinite(values).all(), "Nonfinite feature array")
        require(hashlib.sha256(values.tobytes()).hexdigest() == record["state_sha256"], "Feature values changed")
        states = torch.from_numpy(values.copy()).to(self.runtime.device)
        with torch.inference_mode(), torch.autocast(self.runtime.device.type, enabled=False):
            logp = self.runtime.model.ctc.log_softmax(states.unsqueeze(0))[0]
        require(logp.shape == (row["frames"], 5049) and logp.dtype == torch.float32, "English CTC shape differs")
        require(torch.isfinite(logp).all(), "Nonfinite CTC probabilities")
        metadata = {**record["encoding"], "source_encoder_cache": True,
                    "cache_restore_seconds": time.monotonic() - began,
                    "state_sha256": record["state_sha256"], "fresh_encoder_calls": 0}
        return {"states": states, "logp": logp.detach().cpu().numpy().copy(), "metadata": metadata}


def collapse(labels, blank, eos):
    tokens = []
    previous = None
    for value in labels:
        token = int(value)
        if token != previous and token not in {blank, eos}:
            tokens.append(token)
        previous = token
    return tokens


def decode(runtime, encoded, candidate, plan):
    require(candidate in plan["candidate_order"], "Unregistered decoder")
    if candidate.startswith("beam_c"):
        result = runtime.decode_native(encoded["states"], float(candidate.removeprefix("beam_c")),
                                       beam_size=plan["beam_size"], penalty=plan["penalty"])
        require(result["cap"] == len(encoded["states"]), "Search cap changed")
        return result
    import numpy as np

    require(candidate.startswith("greedy_b"), "Unknown decoder kind")
    began = time.monotonic()
    scores = encoded["logp"].copy()
    scores[:, runtime.blank_id] += float(candidate.removeprefix("greedy_b"))
    tokens = collapse(np.argmax(scores, axis=1), runtime.blank_id, runtime.eos_id)
    return {"tokens": tokens, "text": runtime.text(tokens), "seconds": time.monotonic() - began,
            "search_calls": 0}
