"""Reference-free fixed probes and independent likelihood checks before task admission."""

import argparse
import gc
import time

import numpy as np
import torch

from common import HERE, PHASE, SOURCE, STORE, configure, lines, plan, read, save, sha, utc
from independent import independent_score, verify_ctc
from portable_runtime import DecoderRuntime, clear_attention


def main(deadline):
    configure()
    frozen = plan()
    assert time.time() < deadline <= frozen["admission_deadline_unix"]
    folder = HERE / "Preflight_01"
    folder.mkdir(exist_ok=False)
    source = lines(SOURCE / "Cohort.jsonl")
    positions = read(PHASE / "Plan.json")["probe_positions"]
    began = time.monotonic()
    save(folder / "Start.json", {"utc": utc(), "deadline": deadline})
    checks = []
    replays = 0
    try:
        brute = verify_ctc()
        for model in frozen["models"]:
            runtime = DecoderRuntime(model)
            index = lines(STORE / "Cache" / model / "Index.jsonl")
            for position in positions:
                assert time.time() < deadline
                record = index[position]
                path = STORE / "Cache" / model / record["file"]
                assert sha(path) == record["sha256"]
                with np.load(path, allow_pickle=False) as archive:
                    states = torch.from_numpy(archive["full_states"].copy()).to(runtime.device)
                    logp = archive["full_logp"].copy()
                with torch.inference_mode():
                    actual = runtime.model.ctc.log_softmax(states.unsqueeze(0))[0].cpu().numpy()
                assert np.abs(actual - logp).max() <= frozen["ctc_logp_tolerance"]
                outputs = {}
                for weight in frozen["beam_weights"]:
                    assert time.time() < deadline
                    predicted = runtime.decode(states, weight, cap=source[position]["frames"], beam_size=40, penalty=0)
                    score = independent_score(runtime, states, actual, predicted["tokens"], weight)
                    error = abs(score["merged"] - predicted["score"])
                    assert error <= frozen["score_tolerance"]
                    clear_attention(runtime.model)
                    if position == positions[0]:
                        repeated = runtime.decode(states, weight, cap=len(states), beam_size=40, penalty=0)
                        assert repeated["tokens"] == predicted["tokens"]
                        assert repeated["text"] == predicted["text"]
                        assert abs(repeated["score"] - predicted["score"]) <= frozen["score_tolerance"]
                        clear_attention(runtime.model)
                        replays += 1
                    outputs[f"beam_c{weight:.1f}"] = predicted
                    checks.append({"model": model, "position": position, "weight": weight, "score_error": error})
                save(folder / f"{model}_{position:04d}.json", {
                    "clip_id": record["clip_id"], "cache_sha256": record["sha256"], "predictions": outputs})
                print({"utc": utc(), "model": model, "position": position, "score_checks": len(checks)}, flush=True)
            del runtime, states
            gc.collect()
            torch.cuda.empty_cache()
        assert len(checks) == 24 and replays == 8
        save(folder / "Verification.json", {
            "status": "passed", "utc": utc(), "independent_score_checks": checks,
            "exact_beam_replays": replays, "ctc_bruteforce_checks": brute,
            "seconds": time.monotonic() - began, "plan_sha256": sha(HERE / "Plan.json")})
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args().deadline)
