"""A bounded real-input resource probe, never a replacement for full evaluation."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from retention_common import ENGINE, HERE, ROOT, checked_plan, decode, read, require, runtime
from retention_common import save, sha, source, utc


def verify_prediction(instance, encoded, predicted, candidate, choices, reference):
    sys.path.insert(0, str(ENGINE))
    from confirmation_common import edit_counts

    sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
    from verify_probe import counts

    require(counts(reference, predicted["text"]) == edit_counts(reference, predicted["text"]),
            "Independent edit-count disagreement")
    repeated = decode(instance, encoded, candidate, choices)
    require(repeated["tokens"] == predicted["tokens"], "Repeated decoder tokens differ")
    require(repeated["text"] == predicted["text"], "Repeated decoder text differs")
    if candidate.startswith("beam_"):
        require(math.isfinite(predicted["score"]), "Nonfinite score")
        require(abs(repeated["score"] - predicted["score"]) <= 1e-4, "Repeated beam score differs")
        require(predicted["cap"] == len(encoded["states"]), "Truncated search horizon")
        trace = predicted["trace"]
        require(trace["implementation"] == "upstream_BatchBeamSearch", "Wrong beam implementation")
        require(trace["beam_size"] == 40 and trace["penalty"] == 0, "Wrong beam policy")
    return repeated


def main(arguments):
    plan = checked_plan()
    require(arguments.model in plan["models"], "Unregistered model")
    require(arguments.attempt.isidentifier(), "Invalid attempt name")
    require(60 <= arguments.maximum_seconds <= 1800, "Probe must have a finite small budget")
    output = HERE / "Runs" / arguments.attempt
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + arguments.maximum_seconds
    model = arguments.model
    language = "vlr" if model.startswith("VLR_") else "lrs2"
    positions = plan["cohorts"][language]["probe_positions"]
    choices = plan["models"][model]["execution_candidates"]
    save(output / "Start.json", {
        "utc": utc(), "pid": os.getpid(), "purpose": "resource_probe_only",
        "model": model, "corpus": language, "device": arguments.device,
        "positions": positions, "candidates": choices, "plan_sha256": sha(HERE / "Plan.json"),
        "deadline_unix": deadline, "maximum_seconds": arguments.maximum_seconds,
    })
    try:
        import torch
        sys.path.insert(0, str(ENGINE))
        from confirmation_common import edit_counts

        instance = runtime(model, arguments.device)
        save(output / "Identity.json", instance.identity)
        if instance.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(instance.device)
        rows = []
        with source(language) as dataset:
            with (output / "Predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for position in positions:
                    require(time.time() < deadline, "Probe deadline expired")
                    began = time.monotonic()
                    metadata, frames = dataset.load(position)
                    encoded = instance.encode(frames)
                    require(encoded["metadata"]["pixel_sha256"] == dataset.pixels[metadata["clip_id"]]["pixel_sha256"],
                            "Encoder pixels changed")
                    predictions = {}
                    for candidate in choices:
                        require(time.time() < deadline, "Probe decoder deadline expired")
                        prediction = decode(instance, encoded, candidate, choices)
                        prediction["counts"] = edit_counts(metadata["text"], prediction["text"])
                        predictions[candidate] = prediction
                    inference_seconds = time.monotonic() - began
                    del encoded
                    encoded = instance.encode(frames)
                    for candidate, prediction in predictions.items():
                        require(time.time() < deadline, "Probe replay deadline expired")
                        verify_prediction(instance, encoded, prediction, candidate, choices, metadata["text"])
                    del encoded
                    row = {
                        "position": position, "clip_id": metadata["clip_id"], "frames": len(frames),
                        "pixel_sha256": dataset.pixels[metadata["clip_id"]]["pixel_sha256"],
                        "model": model, "predictions": predictions, "inference_seconds": inference_seconds,
                        "fresh_pixel_replay": True, "utc": utc(),
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    print({"utc": utc(), "model": model, "clips": len(rows), "total": len(positions),
                           "frames": len(frames), "inference_seconds": inference_seconds}, flush=True)
        peak = None
        if instance.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(instance.device) / 2**20
        save(output / "Complete.json", {
            "utc": utc(), "status": "resource_probe_replayed_not_population_evaluation",
            "model": model, "corpus": language, "clips": len(rows), "peak_gpu_mib": peak,
            "seconds": time.time() - started, "predictions_sha256": sha(output / "Predictions.jsonl"),
            "identity_sha256": sha(output / "Identity.json"),
            "population_wer_computed": False, "decoder_tuned": False,
        })
    except BaseException:
        save(output / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--maximum-seconds", type=int, default=1800)
    main(parser.parse_args())
