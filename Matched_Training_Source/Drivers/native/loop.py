"""Training lifecycle around the unchanged native update and checkpoint functions."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


@contextmanager
def run_lock(output):
    import fcntl

    if os.name != "posix":
        raise RuntimeError("This isolated driver requires Linux")
    path = Path(output) / "Run.lock"
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_validation(records, metrics, expected):
    if [row["clip_id"] for row in records] != expected["clip_ids"]:
        raise ValueError("Source diagnostic clip identities or order changed")
    for field in ("requested_rows", "scored_rows", "window_ineligible"):
        if metrics[field] != expected[field]:
            raise ValueError("Source diagnostic coverage changed: " + field)
    if metrics["inference_failed"] or any(row["status"] != "ok" for row in records):
        raise ValueError("Source diagnostic contains inference failures")
    if metrics["metrics"]["word"]["units"] != expected["reference_units"]:
        raise ValueError("Source diagnostic reference denominator changed")
    if (metrics["view"], metrics["context"], metrics["decoder"], metrics["subset_limit"]) != (
            "window", "both", "greedy", 512):
        raise ValueError("Source diagnostic procedure changed")


def execute(model, optimizer, scaler, objective, training, validation, identity,
            output, attempt, first_step, stop_after, deadline):
    from engine import update, save_checkpoint
    from phase5.evaluation import evaluate
    from phase5.runtime import samples

    began = time.monotonic()
    step = first_step
    latest_saved = first_step
    stream = samples(training, first_step * 8, 48000, 17, workers=identity["workers"])
    log = output / "Attempts" / (attempt + ".steps.jsonl")
    try:
        for next_step in range(first_step + 1, 6001):
            if time.time() >= deadline:
                break
            result = update(model, optimizer, scaler, objective, stream, next_step,
                            identity["precision"], identity["device"], deadline)
            step = next_step
            result["seconds_this_attempt"] = time.monotonic() - began
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({key: result[key] for key in ("step", "loss", "lr", "gradient_norm",
                                                         "seconds_this_attempt")}), flush=True)
            if step % 500 == 0:
                records, metrics = evaluate(model, validation, context="both", decoder="greedy",
                                            view="window", limit=512)
                check_validation(records, metrics, identity["source_diagnostic"])
                prefix = output / "Attempts" / (attempt + f".validation_{step:06}")
                with Path(str(prefix) + ".predictions.jsonl").open("x", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                save(Path(str(prefix) + ".metrics.json"), metrics)
            if step % 500 == 0 or step == stop_after:
                save_checkpoint(output / "last.pt", model, optimizer, scaler, step, identity)
                latest_saved = step
            if step >= stop_after:
                break
        if step > latest_saved:
            save_checkpoint(output / "last.pt", model, optimizer, scaler, step, identity)
            latest_saved = step
    finally:
        stream.close()
    if step <= first_step or latest_saved != step:
        raise RuntimeError("No new complete checkpoint was produced")
    return {"updates": step, "presentations": step * 8, "seconds_this_attempt": time.monotonic() - began,
            "fixed_final_endpoint_reached": step == 6000, "checkpoint_step": latest_saved}
