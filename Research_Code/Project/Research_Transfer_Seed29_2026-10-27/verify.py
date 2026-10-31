"""Every-cache CTC reconstruction, every greedy count, and fresh fixed GPU replay."""

import argparse
import gc
import hashlib
import sys
import time

from common import COHORT, ENGINE, HERE, ROOT, STORE, append, checked_plan, lines, read, save, sha, utc

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
import numpy as np
import torch
from confirmation_common import edit_counts, normalize
from data_io import load_pixels
from verify_probe import counts
from runtime_core import Runtime


def collapsed(labels):
    return [int(token) for index, token in enumerate(labels)
            if token not in (0, 2001) and (index == 0 or token != labels[index - 1])]


def independent_tokens(values, bias):
    labels = []
    for frame in values:
        lexical = int(np.argmax(frame[1:])) + 1
        labels.append(0 if float(frame[0]) + bias >= float(frame[lexical]) else lexical)
    output = []
    previous = None
    for label in labels:
        if label != previous and label not in (0, 2001):
            output.append(label)
        previous = label
    return output


def main(model, deadline):
    frozen = checked_plan()
    assert model in frozen["models"] and time.time() < deadline <= frozen["admission_deadline_unix"]
    folder = STORE / "Cache" / model
    complete = read(folder / "Complete.json")
    assert complete["index_sha256"] == sha(folder / "Index.jsonl")
    assert complete["plan_sha256"] == sha(HERE / "Plan.json")
    entry = read(HERE / "model_registry.json")["models"][model]
    checkpoint = ROOT / entry["checkpoint"]
    assert sha(checkpoint) == entry["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    exported = entry["state_field"] == "model"
    state = payload["model"] if exported else payload
    if exported:
        assert payload["step"] == 6000 and payload["identity"] == entry["training_identity"]
    prefix = "native." if exported else ""
    torch.set_num_threads(2)
    weight = state[prefix + "ctc.ctc_lo.weight"].float()
    bias = state[prefix + "ctc.ctc_lo.bias"].float()
    readout = torch.load(STORE / "Models" / (model + ".pt"), map_location="cpu", weights_only=True, mmap=True)
    tensor_checks = 0
    for group in ("decoder", "ctc"):
        for key, value in readout[group].items():
            assert torch.equal(state[prefix + group + "." + key], value)
            tensor_checks += 1
    assert tensor_checks == 163
    vocabulary = read(folder / "Vocabulary.json")
    assert vocabulary == readout["vocabulary"] and len(vocabulary) == 2002
    rows = lines(COHORT)
    index = lines(folder / "Index.jsonl")
    assert len(index) == len(rows) == complete["clips"] == 531
    output = HERE / "Greedy_01" / model
    output.mkdir(parents=True, exist_ok=False)
    maximum = 0.0
    checks = 0
    for row, record in zip(rows, index):
        assert time.time() < deadline
        assert record["clip_id"] == row["clip_id"] and record["source_id"] == row["source_id"]
        assert record["frames"] == row["frames"] and record["media_sha256"] == row["input"]["sha256"]
        path = folder / record["file"]
        assert path.resolve().parent == folder.resolve() and sha(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            states = arrays["full_states"].copy()
            logp = arrays["full_logp"].copy()
        assert states.shape == (row["frames"], 768) and logp.shape == (row["frames"], 2002)
        assert states.dtype == np.float32 and logp.dtype == np.float32
        assert np.isfinite(states).all() and np.isfinite(logp).all()
        with torch.inference_mode():
            logits = torch.from_numpy(states) @ weight.T + bias
            independent = (logits - torch.logsumexp(logits, dim=-1, keepdim=True)).numpy()
        error = float(np.abs(independent - logp).max())
        maximum = max(maximum, error)
        assert error <= 2e-4, (row["clip_id"], error)
        predictions = {}
        for blank_bias in frozen["blank_biases"]:
            altered = logp.astype(np.float64)
            altered[:, 0] += blank_bias
            tokens = collapsed(altered.argmax(axis=1))
            assert tokens == independent_tokens(logp, blank_bias)
            text = normalize("".join(vocabulary[token] for token in tokens)
                             .replace("\u2581", " ").replace("<space>", " "))
            result = edit_counts(row["reference"], text)
            assert result == counts(row["reference"], text)
            predictions[f"greedy_b{blank_bias:.1f}"] = {"text": text, "tokens": tokens, "counts": result}
            checks += 1
        append(output / "Predictions.jsonl", {"clip_id": row["clip_id"], "source_id": row["source_id"],
                                             "fold": frozen["source_folds"][row["source_id"]],
                                             "cache_sha256": record["sha256"], "predictions": predictions})
    del payload, state, readout, weight, bias, logits, independent
    gc.collect()
    runtime = Runtime(model, device="cuda", threads=2)
    replay_errors = []
    for position in frozen["probe_positions"]:
        assert time.time() < deadline
        row = rows[position]
        gray = {**row, "input": {**row["input"], "color_space": "gray"}}
        gray["metadata"] = {key: value for key, value in row["metadata"].items() if key != "pixels_sha256"}
        pixels = load_pixels(gray)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == index[position]["pixels_sha256"]
        value = runtime.encode(pixels)
        with np.load(folder / index[position]["file"], allow_pickle=False) as expected:
            state_error = float(np.abs(value["states"].cpu().numpy() - expected["full_states"]).max())
            logp_error = float(np.abs(value["logp"] - expected["full_logp"]).max())
        assert state_error <= 1e-5 and logp_error <= 1e-5
        replay_errors.append({"clip_id": row["clip_id"], "state_error": state_error, "logp_error": logp_error})
    receipt = {"status": "passed", "utc": utc(), "model": model, "clips": len(rows),
               "ctc_reconstructions": len(rows), "maximum_cpu_ctc_error": maximum,
               "independent_greedy_count_and_path_checks": checks, "tensor_export_checks": tensor_checks,
               "gpu_replays": replay_errors, "index_sha256": sha(folder / "Index.jsonl"),
               "predictions_sha256": sha(output / "Predictions.jsonl"),
               "plan_sha256": sha(HERE / "Plan.json"), "full_decoder_matrix_complete": False}
    save(folder / "Verification.json", receipt)
    save(output / "Verification.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--deadline", type=float, required=True)
    arguments = parser.parse_args()
    main(arguments.model, arguments.deadline)
