"""Reference-free encoding of the common initializer and equal-step controls."""

import argparse
import hashlib
import os
import sys
import time

from common import COHORT, ENGINE, HERE, ROOT, STORE, append, checked_plan, lines, read, save, sha, utc

sys.path.insert(0, str(ENGINE))
import numpy as np
import torch
from data_io import load_pixels
from runtime_core import Runtime


def old_indices():
    source = ROOT / "Research_Interval_Transfer_2026-10-27"
    folder = read(source / "Cache_Plan.json")["destination"]
    from pathlib import Path
    expected = read(HERE / "Plan.json")["prior_cache_index_sha256"]
    output = []
    for model in ("VLR_T0", "VLR_T1"):
        path = Path(folder) / model / "Index.jsonl"
        assert sha(path) == expected[model]
        output.append(lines(path))
    return output


def main(model, deadline):
    frozen = checked_plan()
    assert model in frozen["models"] and time.time() < deadline <= frozen["admission_deadline_unix"]
    folder = STORE / "Cache" / model
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "Start.json", {"utc": utc(), "pid": os.getpid(), "deadline": deadline,
                                 "plan_sha256": sha(HERE / "Plan.json")})
    began = time.monotonic()
    try:
        runtime = Runtime(model, device="cuda", threads=2)
        save(folder / "Identity.json", runtime.identity)
        save(folder / "Vocabulary.json", runtime.vocabulary)
        models = STORE / "Models"
        models.mkdir(parents=True, exist_ok=True)
        payload = {"decoder": {key: value.cpu().clone() for key, value in runtime.model.decoder.state_dict().items()},
                   "ctc": {key: value.cpu().clone() for key, value in runtime.model.ctc.state_dict().items()},
                   "vocabulary": runtime.vocabulary,
                   "source_checkpoint_sha256": runtime.entry["checkpoint_sha256"]}
        path = models / (model + ".pt")
        with path.open("xb") as stream:
            torch.save(payload, stream)
        check = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for name in ("decoder", "ctc"):
            assert set(check[name]) == set(payload[name])
            for key, value in payload[name].items():
                assert torch.equal(value, check[name][key]), key
        save(models / (model + ".json"), {"utc": utc(), "status": "tensor_readback_passed",
                                         "export_sha256": sha(path), "checkpoint_sha256":
                                         runtime.entry["checkpoint_sha256"]})
        del payload, check
        rows = lines(COHORT)
        prior = old_indices()
        torch.cuda.reset_peak_memory_stats()
        for index, row in enumerate(rows):
            assert time.time() < deadline, "Encoder deadline reached; preserve partial attempt"
            gray = {**row, "input": {**row["input"], "color_space": "gray"}}
            gray["metadata"] = {key: value for key, value in row["metadata"].items() if key != "pixels_sha256"}
            pixels = load_pixels(gray)
            pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
            for previous in prior:
                assert previous[index]["clip_id"] == row["clip_id"]
                assert previous[index]["pixels_sha256"] == pixel_hash
            value = runtime.encode(pixels)
            path = folder / f"Case_{index:04d}.npz"
            np.savez(path, full_states=value["states"].cpu().numpy(), full_logp=value["logp"])
            append(folder / "Index.jsonl", {"index": index, "clip_id": row["clip_id"],
                                          "source_id": row["source_id"], "frames": row["frames"],
                                          "file": path.name, "sha256": sha(path),
                                          "pixels_sha256": pixel_hash, "media_sha256": row["input"]["sha256"],
                                          "encoding": value["metadata"], "utc": utc()})
            del value
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                status = {"utc": utc(), "model": model, "completed": index + 1,
                          "total": len(rows), "seconds": time.monotonic() - began}
                append(HERE / "Progress.jsonl", status)
                print(status, flush=True)
        save(folder / "Complete.json", {"status": "complete_pending_verification", "utc": utc(),
                                         "clips": len(rows), "seconds": time.monotonic() - began,
                                         "index_sha256": sha(folder / "Index.jsonl"),
                                         "plan_sha256": sha(HERE / "Plan.json"),
                                         "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20})
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--deadline", type=float, required=True)
    arguments = parser.parse_args()
    main(arguments.model, arguments.deadline)
