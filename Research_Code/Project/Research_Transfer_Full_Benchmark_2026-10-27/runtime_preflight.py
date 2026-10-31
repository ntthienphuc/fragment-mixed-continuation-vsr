"""Native/readout parity on duration-selected development clips, without reference scoring."""

import argparse
from pathlib import Path
import time

from admission import checked, require
from common import HERE, lines, read, save, sha, utc


def main(model, output, device, threads):
    plan, admission, _ = checked("preflight", model)
    require(model in {"VLR_B0", "VC_NATIVE"}, "Use one registered model per graph family")
    require(not output.exists(), "Preserve previous preflight attempts")
    from runtime_bridge import create_runtime, decode, family_for, pixels_for

    runtime = create_runtime(model, device=device, threads=threads)
    rows = sorted(lines(HERE / "Development.jsonl"), key=lambda row: (row["frames"], row["clip_id"]))
    pixels_path = HERE / "Pixels_01/Index.jsonl"
    require(sha(pixels_path) == read(HERE / "Pixels_01/Verification.json")["index_sha256"], "Pixel proof changed")
    pixel_index = {row["clip_id"]: row for row in lines(pixels_path)}
    checks = []
    for index in (0, len(rows) // 2, len(rows) - 1):
        require(time.time() < admission["deadline_unix"], "Preflight deadline reached")
        row = rows[index]
        encoded = runtime.encode(pixels_for(row, model, pixel_index[row["clip_id"]]))
        comparisons = []
        for weight in plan["beam_weights"]:
            native = decode(runtime, encoded, "beam_c" + str(weight), plan)
            bounded = runtime.decode(encoded["states"], weight, cap=row["frames"], beam_size=40, penalty=0)
            error = abs(native["score"] - bounded["score"])
            require(native["tokens"] == bounded["tokens"] and native["text"] == bounded["text"],
                    "Native versus bounded search differs")
            require(error <= 1e-5, "Native versus bounded score differs")
            comparisons.append({"weight": weight, "tokens": native["tokens"], "score_error": error,
                                "native_seconds": native["seconds"], "bounded_seconds": bounded["seconds"]})
        checks.append({"clip_id": row["clip_id"], "frames": row["frames"],
                       "encoding": encoded["metadata"], "comparisons": comparisons})
    save(output, {"status": "passed", "utc": utc(), "model": model, "family": family_for(model),
                  "identity": runtime.identity, "duration_selected_development_checks": checks,
                  "reference_used": False, "accuracy_claimed": False, "test_clips_used": False,
                  "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    print({"model": model, "native_replays": 12, "status": "passed"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.device, arguments.threads)
