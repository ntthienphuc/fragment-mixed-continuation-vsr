"""Compare native and bounded searches on prespecified source-development pixels."""

import argparse
import json
from pathlib import Path
import sys
import time

from common import HERE, INPUTS, PHASE, ROOT, check_lock, read, save, sha, utc


def main(model, output, deadline):
    lock = check_lock()
    assert model in lock["models"]
    assert time.time() < deadline and not output.exists()
    from runtime_bridge import create_runtime
    sys.path.insert(0, str(PHASE))
    from strict_inputs import load_pixels
    import torch

    runtime = create_runtime(model)
    for name, digest in runtime.identity["source_files"].items():
        assert lock["sha256"][name] == digest, name
    population = [json.loads(line)
                  for line in (INPUTS / "Development.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["clip_id"]: row for row in population}
    torch.cuda.reset_peak_memory_stats(runtime.device)
    checks = []
    for selected in lock["probes"]:
        row = by_id[selected["clip_id"]]
        assert row["frames"] == selected["frames"] and row["split"] == "development"
        assert time.time() < deadline
        pixels = load_pixels(row)
        encoded = runtime.encode(pixels)
        assert encoded["logp"].shape == (row["frames"], 5049)
        comparisons = []
        for weight in lock["ctc_weights"]:
            assert time.time() < deadline
            native = runtime.decode_native(encoded["states"], weight, beam_size=40, penalty=0)
            bounded = runtime.decode(encoded["states"], weight, cap=row["frames"], beam_size=40, penalty=0)
            assert native["tokens"] == bounded["tokens"]
            assert native["text"] == bounded["text"]
            error = abs(native["score"] - bounded["score"])
            assert error <= 1e-5
            comparisons.append({"ctc_weight": weight, "tokens": native["tokens"], "score_error": error,
                                "native_seconds": native["seconds"], "bounded_seconds": bounded["seconds"]})
            print({"utc": utc(), "model": model, "frames": row["frames"],
                   "ctc_weight": weight, "score_error": error}, flush=True)
        checks.append({"clip_id": row["clip_id"], "frames": row["frames"],
                       "encoding": encoded["metadata"], "comparisons": comparisons})
        del encoded
    save(output, {"utc": utc(), "status": "passed", "model": model, "identity": runtime.identity,
                  "checks": checks, "lock_sha256": sha(HERE / "Lock.json"),
                  "peak_gpu_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
                  "reference_used_for_selection_or_decoding": False, "test_clips_used": False,
                  "accuracy_claimed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.deadline)
