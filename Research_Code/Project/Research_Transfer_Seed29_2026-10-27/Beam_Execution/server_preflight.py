"""Require exact cross-device replay of all fixed probes before server work."""

import argparse
import numpy as np
import torch

from common import HERE, STORE, configure, lines, plan, read, save, sha, utc
from independent import independent_score
from portable_runtime import DecoderRuntime, clear_attention
from worker import check_execution


def main(model):
    configure()
    frozen = plan()
    check_execution()
    assert model in frozen["models"]
    runtime = DecoderRuntime(model)
    folder = HERE / "Server_Preflight" / model
    folder.mkdir(parents=True, exist_ok=False)
    cache = STORE / "Cache" / model
    index = {row["clip_id"]: row for row in lines(cache / "Index.jsonl")}
    checks = []
    for path in sorted((HERE / "Preflight_01").glob(model + "_*.json")):
        reference = read(path)
        item = index[reference["clip_id"]]
        data_path = cache / item["file"]
        assert sha(data_path) == item["sha256"] == reference["cache_sha256"]
        with np.load(data_path, allow_pickle=False) as arrays:
            states = torch.from_numpy(arrays["full_states"].copy()).to(runtime.device)
            old_logp = arrays["full_logp"].copy()
        with torch.inference_mode():
            logp = runtime.model.ctc.log_softmax(states.unsqueeze(0))[0].cpu().numpy()
        assert np.abs(logp - old_logp).max() <= frozen["ctc_logp_tolerance"]
        for weight in frozen["beam_weights"]:
            output = runtime.decode(states, weight, cap=len(states), beam_size=40, penalty=0)
            expected = reference["predictions"][f"beam_c{weight:.1f}"]
            assert output["tokens"] == expected["tokens"] and output["text"] == expected["text"]
            assert abs(output["score"] - expected["score"]) <= frozen["score_tolerance"]
            independent = independent_score(runtime, states, logp, output["tokens"], weight)
            assert abs(output["score"] - independent["merged"]) <= frozen["score_tolerance"]
            checks.append({"clip_id": item["clip_id"], "weight": weight,
                           "score_error": abs(output["score"] - independent["merged"])})
            clear_attention(runtime.model)
    assert len(checks) == 12
    save(folder / "Verification.json", {"status": "passed", "utc": utc(), "checks": checks,
                                       "runtime": runtime.identity,
                                       "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    print({"model": model, "exact_cross_device_replays": len(checks)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    main(parser.parse_args().model)
