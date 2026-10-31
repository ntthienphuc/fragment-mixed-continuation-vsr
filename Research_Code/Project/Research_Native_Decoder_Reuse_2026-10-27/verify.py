"""Independently reconstruct eval-mode decoder math from checkpoint tensors."""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as functional

from common import BANK, DATA, HERE, ROOT, check_plan, configure, read, save, sha, utc


def linear(value, weights, prefix):
    return functional.linear(value, weights[prefix + ".weight"], weights[prefix + ".bias"])


def norm(value, weights, prefix):
    return functional.layer_norm(value, (768,), weights[prefix + ".weight"],
                                  weights[prefix + ".bias"], eps=1e-12)


def attention(query, memory, weights, prefix, allowed):
    batch = len(query)
    query_heads = linear(query, weights, prefix + ".linear_q").view(batch, -1, 12, 64).transpose(1, 2)
    key_heads = linear(memory, weights, prefix + ".linear_k").view(batch, -1, 12, 64).transpose(1, 2)
    value_heads = linear(memory, weights, prefix + ".linear_v").view(batch, -1, 12, 64).transpose(1, 2)
    energy = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / 8.0
    if allowed is not None:
        forbidden = ~allowed.unsqueeze(1)
        energy = energy.masked_fill(forbidden, torch.finfo(energy.dtype).min)
        probability = energy.softmax(-1).masked_fill(forbidden, 0.0)
    else:
        probability = energy.softmax(-1)
    context = torch.matmul(probability, value_heads).transpose(1, 2).contiguous().view(batch, -1, 768)
    return linear(context, weights, prefix + ".linear_out")


def manual_decoder(tokens, memory, weights, causal=True, memory_mask=None):
    # Recreate positional values on CPU, as in the frozen native constructor.
    positions = torch.arange(tokens.shape[1], dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, 768, 2, dtype=torch.float32) * (-math.log(10000.0) / 768))
    encoding = torch.zeros(tokens.shape[1], 768)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    value = functional.embedding(tokens, weights["embed.0.weight"]) * math.sqrt(768)
    value = value + encoding.to(value.device).unsqueeze(0)
    mask = torch.ones(1, tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=value.device)
    if causal:
        mask = mask.tril()
    for layer in range(6):
        prefix = f"decoders.{layer}"
        normalized = norm(value, weights, prefix + ".norm1")
        value = value + attention(normalized, normalized, weights, prefix + ".self_attn", mask)
        normalized = norm(value, weights, prefix + ".norm2")
        value = value + attention(normalized, memory, weights, prefix + ".src_attn", memory_mask)
        normalized = norm(value, weights, prefix + ".norm3")
        hidden = linear(normalized, weights, prefix + ".feed_forward.w_1").relu()
        value = value + linear(hidden, weights, prefix + ".feed_forward.w_2")
    return linear(norm(value, weights, "after_norm"), weights, "output_layer")


def compare(actual, expected, tolerance):
    assert actual.shape == expected.shape
    error = float((actual - expected).abs().max())
    assert error <= tolerance, error
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    output = HERE / args.attempt
    assert output.resolve().is_relative_to(HERE) and output.parent == HERE
    began = time.perf_counter()
    configure()
    plan = check_plan()
    completed = read(output / "Complete.json")
    assert completed["status"] == "completed_pending_independent_verification"
    assert completed["plan_sha256"] == sha(HERE / "Plan.json")
    assert completed["decoder_sha256"] == sha(output / "Decoder.pt")
    exported = torch.load(output / "Decoder.pt", map_location="cpu", weights_only=True)
    assert exported["plan_sha256"] == sha(HERE / "Plan.json")
    assert len(exported["vocabulary"]) == 2002
    assert exported["vocabulary"][0] == "<blank>" and exported["vocabulary"][-1] == "<eos>"
    native = torch.load(ROOT / plan["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    native_weights = {name[len("native.decoder."):]: value for name, value in native["model"].items()
                      if name.startswith("native.decoder.")}
    assert set(exported["state"]) == set(native_weights)
    for name, value in exported["state"].items():
        assert torch.equal(value, native_weights[name]), name
    assert sum(value.numel() for value in exported["state"].values()) == completed["decoder_parameters"]
    del native, native_weights
    weights = {key: value.to("cuda") for key, value in exported["state"].items()}
    original_predictions = {}
    with (BANK / "Collection_01/Predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            original_predictions[item["clip_id"]] = item["hybrid"]
    assert len(completed["probes"]) == len(plan["rows"]) == 8
    checks = []
    with torch.inference_mode():
        for row, result in zip(plan["rows"], completed["probes"]):
            assert row["clip_id"] == result["clip_id"]
            path = output / result["file"]
            assert sha(path) == result["sha256"]
            with np.load(DATA / row["file"], allow_pickle=False) as content:
                memory = torch.from_numpy(content["states"].copy()).to("cuda").unsqueeze(0)
            with np.load(path, allow_pickle=False) as content:
                tensors = {key: torch.from_numpy(content[key].copy()).to("cuda") for key in content.files}
            tokens = tensors["tokens"]
            assert tokens.tolist() == [[2001] + row["baseline_tokens"]]
            fresh_logits = manual_decoder(tokens, memory, weights)
            logp = fresh_logits.log_softmax(-1)
            errors = dict(manual_original=compare(fresh_logits, tensors["original_logits"], plan["tolerance"]),
                          manual_extracted=compare(fresh_logits, tensors["extracted_logits"], plan["tolerance"]),
                          incremental=compare(logp, tensors["incremental"], plan["tolerance"]))
            assert torch.equal(tensors["original_logits"], tensors["extracted_logits"])
            changed_logits = manual_decoder(tensors["changed_tokens"], memory, weights).log_softmax(-1)
            errors["future_replay"] = compare(changed_logits, tensors["changed_scores"], plan["tolerance"])
            errors["causal_future"] = compare(logp[:, :-1], changed_logits[:, :-1], plan["tolerance"])
            padded_memory = torch.cat((memory, tensors["padding"]), dim=1)
            mask = torch.arange(padded_memory.shape[1], device="cuda").lt(memory.shape[1]).view(1, 1, -1)
            padded = manual_decoder(tokens, padded_memory, weights, memory_mask=mask).log_softmax(-1)
            errors["padding_replay"] = compare(padded, tensors["padded_scores"], plan["tolerance"])
            errors["padding_invariance"] = compare(logp, padded, plan["tolerance"])
            bidirectional = manual_decoder(tokens, memory, weights, causal=False).log_softmax(-1)
            errors["bidirectional_replay"] = compare(bidirectional, tensors["bidirectional"], plan["tolerance"])
            changed_positions = int((logp.argmax(-1) != bidirectional.argmax(-1)).sum())
            assert changed_positions == result["changed_bidirectional_top1_positions"]
            archived = original_predictions[row["clip_id"]]
            for mode in ("before", "after"):
                assert result[mode]["tokens"] == archived["tokens"]
                assert result[mode]["text"] == archived["text"]
                assert result[mode]["trace"]["ctc_weight"] == .1
                assert result[mode]["trace"]["beam_size"] == 40
                assert result[mode]["cap"] == row["frames"]
            checks.append(dict(clip_id=row["clip_id"], errors=errors,
                               positions=tokens.shape[1], changed_bidirectional_positions=changed_positions))
    receipt = dict(status="passed", utc=utc(), checks=checks, checkpoint_tensors=len(weights),
                   decoder_parameters=completed["decoder_parameters"], manual_decoder_replays=8,
                   original_and_extracted_hypotheses_checked=16, beam_search_reexecuted_by_verifier=False,
                   complete_sha256=sha(output / "Complete.json"), plan_sha256=sha(HERE / "Plan.json"),
                   verifier_sha256=sha(HERE / "verify.py"), seconds=time.perf_counter() - began,
                   recognition_improvement_claim=False)
    save(output / "Verification.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "checks"}), flush=True)


if __name__ == "__main__":
    main()
