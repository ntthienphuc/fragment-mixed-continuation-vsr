"""Replay all source decoder candidates using cached and freshly encoded pixels."""

import argparse
from pathlib import Path
import time

from admission import checked
from english_common import HERE, READY, lines, read, require, save, sha, utc


def main(model, output, admission_path):
    plan, admission, _, locations = checked("preflight", model, admission_path)
    require(not output.exists(), "Preserve the preceding preflight")
    from runtime_bridge import DevelopmentCache, create_runtime, decode, pixels_for
    import numpy as np

    runtime = create_runtime(model)
    rows = lines(HERE / "Development.jsonl")
    by_id = {row["clip_id"]: number for number, row in enumerate(rows)}
    probes = read(READY / "Runtime_Validation/Lock.json")["probes"]
    cache = DevelopmentCache(model, locations, runtime)
    comparisons = []
    try:
        for probe in probes:
            require(time.time() < admission["deadline_unix"], "Preflight deadline reached")
            position = by_id[probe["clip_id"]]
            row = rows[position]
            require(row["frames"] == probe["frames"], "Probe population changed")
            cached = cache.encoded(position, row)
            fresh = runtime.encode(pixels_for(row, locations))
            state_error = float((cached["states"] - fresh["states"]).abs().max().item())
            logp_error = float(np.max(np.abs(cached["logp"] - fresh["logp"])))
            require(np.isfinite(state_error) and np.isfinite(logp_error), "Nonfinite replay difference")
            for candidate in plan["candidate_order"]:
                require(time.time() < admission["deadline_unix"], "Preflight search deadline reached")
                restored = decode(runtime, cached, candidate, plan)
                native = decode(runtime, fresh, candidate, plan)
                require(restored["tokens"] == native["tokens"] and restored["text"] == native["text"],
                        "Cache-to-native decoding differs")
                error = abs(restored["score"] - native["score"]) if "score" in restored else 0
                require(error <= 1e-4, "Cache-to-native score differs")
                comparisons.append({"clip_id": row["clip_id"], "frames": row["frames"], "candidate": candidate,
                                    "state_max_abs_error": state_error, "logp_max_abs_error": logp_error,
                                    "score_error": error, "cached_seconds": restored["seconds"],
                                    "fresh_seconds": native["seconds"]})
            del cached, fresh
    finally:
        cache.close()
    require(len(comparisons) == 33, "Missing preflight comparison")
    save(output, {"utc": utc(), "status": "native_source_cache_preflight_passed", "model": model,
                  "prediction_comparisons": len(comparisons), "checks": comparisons, "identity": runtime.identity,
                  "test_clips_used": False, "accuracy_claimed": False,
                  "execution_lock_sha256": sha(HERE / "Execution_Lock.json"), "plan_sha256": sha(HERE / "Plan.json")})
    print({"model": model, "status": "passed", "checks": len(comparisons)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.admission.resolve())
