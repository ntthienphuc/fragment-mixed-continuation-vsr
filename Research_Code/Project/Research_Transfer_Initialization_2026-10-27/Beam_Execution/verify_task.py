"""Verify every count and fixed, outcome-independent native beam/score replays."""

import argparse
import sys
import time

import numpy as np
import torch

from common import HERE, ROOT, SOURCE, STORE, configure, lines, plan, read, save, sha, utc
from independent import independent_score
from portable_runtime import DecoderRuntime, clear_attention
from worker import check_execution, destination

sys.path.insert(0, str(SOURCE))
from verify_probe import counts, render


def main(arguments):
    configure()
    frozen = plan()
    lock = check_execution()
    task = lock["tasks"][arguments.task]
    folder = destination(arguments.attempt, arguments.task)
    completed = read(folder / "Complete.json")
    assert completed["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json")
    assert completed["predictions_sha256"] == sha(folder / "Predictions.jsonl")
    rows = lines(folder / "Predictions.jsonl")
    assert [row["index"] for row in rows] == task["positions"]
    cohort = lines(SOURCE / "Cohort.jsonl")
    cache = STORE / "Cache" / task["model"]
    index = lines(cache / "Index.jsonl")
    runtime = DecoderRuntime(task["model"])
    count_checks = replays = score_checks = 0
    checks = []
    for row in rows:
        assert time.time() < arguments.deadline
        position = row["index"]
        source = cohort[position]
        item = index[position]
        assert row["clip_id"] == source["clip_id"] == item["clip_id"]
        assert row["source_id"] == source["source_id"]
        assert row["cache_sha256"] == item["sha256"]
        assert row["pixels_sha256"] == item["pixels_sha256"]
        assert row["maximum_ctc_error"] <= frozen["ctc_logp_tolerance"]
        keys = [f"beam_c{weight:.1f}" for weight in frozen["beam_weights"]]
        assert list(row["predictions"]) == keys
        if position in task["score_positions"]:
            path = cache / item["file"]
            assert sha(path) == item["sha256"]
            with np.load(path, allow_pickle=False) as values:
                states = torch.from_numpy(values["full_states"].copy()).to(runtime.device)
                logp = values["full_logp"].copy()
        for weight in frozen["beam_weights"]:
            prediction = row["predictions"][f"beam_c{weight:.1f}"]
            assert counts(source["reference"], prediction["text"]) == prediction["counts"]
            assert render(prediction["tokens"], runtime.vocabulary) == prediction["text"]
            trace = prediction["trace"]
            assert trace["beam_size"] == 40 and trace["penalty"] == 0
            assert trace["ctc_weight"] == weight and trace["step_cap"] == source["frames"]
            assert trace["input_states"] == source["frames"]
            assert trace["implementation"] == "SafeHorizonBeamSearch"
            count_checks += 1
            if position in task["score_positions"]:
                score = independent_score(runtime, states, logp, prediction["tokens"], weight)
                error = abs(score["merged"] - prediction["score"])
                assert error <= frozen["score_tolerance"], (position, weight, error)
                score_checks += 1
                checks.append({"index": position, "weight": weight, "score_error": error})
            if position in task["replay_positions"]:
                replay = runtime.decode(states, weight, cap=source["frames"], beam_size=40, penalty=0)
                assert replay["tokens"] == prediction["tokens"] and replay["text"] == prediction["text"]
                assert abs(replay["score"] - prediction["score"]) <= frozen["score_tolerance"]
                clear_attention(runtime.model)
                replays += 1
    receipt = {"status": "passed", "utc": utc(), "task": arguments.task, "clips": len(rows),
               "independent_count_checks": count_checks, "exact_beam_replays": replays,
               "independent_score_checks": score_checks, "score_checks": checks,
               "predictions_sha256": sha(folder / "Predictions.jsonl"),
               "execution_lock_sha256": sha(HERE / "Execution_Lock.json")}
    save(folder / "Verification.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args())
