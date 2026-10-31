"""Independent count/path checks and fresh GPU replay of fixed timing probes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Research_Server_Confirmation_2026-10-19"))
from confirmation_common import read_json, save_json, sha, utc


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def counts(reference, hypothesis):
    ref = normalized(reference).split()
    hyp = normalized(hypothesis).split()
    cost = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int64)
    cost[:, 0] = np.arange(len(ref) + 1)
    cost[0, :] = np.arange(len(hyp) + 1)
    for row in range(1, len(ref) + 1):
        for column in range(1, len(hyp) + 1):
            cost[row, column] = min(cost[row - 1, column - 1] + (ref[row - 1] != hyp[column - 1]),
                                    cost[row - 1, column] + 1, cost[row, column - 1] + 1)
    row = len(ref)
    column = len(hyp)
    substitutions = deletions = insertions = 0
    while row or column:
        mismatch = int(ref[row - 1] != hyp[column - 1]) if row and column else 0
        if row and column and cost[row, column] == cost[row - 1, column - 1] + mismatch:
            substitutions += mismatch
            row -= 1
            column -= 1
        elif row and cost[row, column] == cost[row - 1, column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return {"errors": int(cost[-1, -1]), "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "units": len(ref), "metric": "wer"}


def collapsed(logp):
    labels = logp.argmax(axis=1).tolist()
    runs = []
    for frame, label in enumerate(labels):
        if frame == 0 or label != labels[frame - 1]:
            runs.append([label, frame, frame + 1])
        else:
            runs[-1][2] = frame + 1
    lexical = [run for run in runs if run[0] not in (0, 2001)]
    return [run[0] for run in lexical], [[run[1], run[2]] for run in lexical]


def alignment(logp, tokens):
    if not tokens:
        return {"spans": [], "score": float(np.asarray(logp[:, 0], dtype=np.float64).sum())}
    labels = [0]
    for token in tokens:
        labels.extend((token, 0))
    values = np.full((len(logp), len(labels)), -np.inf, dtype=np.float64)
    parent = np.full(values.shape, -1, dtype=np.int64)
    values[0, :2] = logp[0, labels[:2]]
    for frame in range(1, len(logp)):
        for state, label in enumerate(labels):
            possible = [state]
            if state:
                possible.append(state - 1)
            if state >= 2 and label and label != labels[state - 2]:
                possible.append(state - 2)
            previous = max(possible, key=lambda index: values[frame - 1, index])
            values[frame, state] = values[frame - 1, previous] + float(logp[frame, label])
            parent[frame, state] = previous
    state = len(labels) - 1 if values[-1, -1] >= values[-1, -2] else len(labels) - 2
    score = float(values[-1, state])
    if not np.isfinite(score):
        return None
    spans = [[len(logp), -1] for token in tokens]
    for frame in range(len(logp) - 1, -1, -1):
        if state % 2:
            position = (state - 1) // 2
            spans[position][0] = min(spans[position][0], frame)
            spans[position][1] = max(spans[position][1], frame + 1)
        state = int(parent[frame, state])
    return {"spans": spans, "score": score}


def render(tokens, vocabulary):
    return normalized("".join(vocabulary[token] for token in tokens).replace("\u2581", " ")
                      .replace("<space>", " "))


def project(tokens, spans, vocabulary, rule):
    words = []
    text = ""
    positions = []
    for index, token in enumerate(tokens):
        piece = vocabulary[token].replace("\u2581", " ").replace("<space>", " ")
        for letter in piece:
            if letter.isspace():
                if text:
                    words.append((text, positions))
                    text = ""
                    positions = []
            else:
                text += letter
                positions.append(index)
    if text:
        words.append((text, positions))
    selected = []
    for text, positions in words:
        start = min(spans[index][0] for index in positions)
        end = max(spans[index][1] for index in positions)
        include = 16 <= start + (end - start) / 2 < 66 if rule == "midpoint" else start < 66 and end > 16
        if include:
            selected.append(text)
    return normalized(" ".join(selected))


def check(model_id):
    from runtime_probe import checked_plan
    plan = checked_plan()
    folder = HERE / "Probe" / model_id
    complete = read_json(folder / "Complete.json")
    assert sha(folder / "Predictions.jsonl") == complete["predictions_sha256"]
    records = [json.loads(line) for line in (folder / "Predictions.jsonl").read_text().splitlines()]
    assert [record["clip_id"] for record in records] == plan["probe_ids"]
    vocabulary = read_json(folder / "Vocabulary.json")
    score_checks = 0
    for record in records:
        logits = folder / f"Case_{record['index']:02d}.npz"
        assert sha(logits) == record["logits_sha256"]
        predictions = record["predictions"]
        with np.load(logits, allow_pickle=False) as cache:
            for name, logp in (("full_greedy", cache["full"]), ("crop_greedy", cache["crop"]),
                               ("direct_greedy", cache["context"][16:66])):
                tokens, spans = collapsed(logp)
                assert predictions[name]["tokens"] == tokens
                assert predictions[name]["spans"] == spans
                assert predictions[name]["text"] == render(tokens, vocabulary)
            tokens, spans = collapsed(cache["context"])
            for rule in ("midpoint", "overlap"):
                assert predictions["projection_greedy_" + rule]["text"] == project(tokens, spans, vocabulary, rule)
            tokens = predictions["context_hybrid"]["tokens"]
            aligned = alignment(cache["context"], tokens)
            for rule in ("midpoint", "overlap"):
                output = predictions["projection_hybrid_" + rule]
                assert output["fallback"] == (aligned is None)
                if aligned is not None:
                    assert aligned["spans"] == output["alignment"]["spans"]
                    assert abs(aligned["score"] - output["alignment"]["score"]) < 1e-8
                    assert output["text"] == project(tokens, aligned["spans"], vocabulary, rule)
                else:
                    assert output["text"] == predictions["direct_greedy"]["text"]
        for name, prediction in predictions.items():
            for rule, expected in prediction.get("counts", {}).items():
                reference = record["full_reference"] if rule == "full" else record["references"][rule]
                assert counts(reference, prediction["text"]) == expected, (model_id, name, rule)
                score_checks += 1
    value = {"status": "passed", "utc": utc(), "clips": len(records), "score_checks": score_checks,
             "independent_alignment_and_allocation": True, "complete_sha256": sha(folder / "Complete.json"),
             "verifier_sha256": sha(Path(__file__)), "scope": "Timing probes, not transfer efficacy"}
    save_json(folder / "Verification.json", value)
    print(json.dumps(value), flush=True)


def replay(model_id):
    from runtime_probe import checked_plan, free_memory
    plan = checked_plan()
    assert free_memory() >= 3.0
    from data_io import load_pixels
    from native_runtime import Runtime
    folder = HERE / "Probe" / model_id
    previous = json.loads((folder / "Predictions.jsonl").read_text().splitlines()[0])
    rows = [json.loads(line) for line in (HERE / "Cohort.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(row for row in rows if row["clip_id"] == previous["clip_id"])
    row["input"]["color_space"] = "gray"
    row["metadata"].pop("pixels_sha256", None)
    pixels = load_pixels(row)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == previous["pixels_sha256"]
    bounds = row["target"]
    runtime = Runtime(model_id, device="cuda", threads=2)
    views = {"full": pixels, "context": pixels[bounds["context_start"]:bounds["context_end"]],
             "crop": pixels[bounds["start"]:bounds["end"]]}
    encoded = {name: runtime.encode(view) for name, view in views.items()}
    with np.load(folder / "Case_00.npz", allow_pickle=False) as saved:
        differences = {name: float(np.max(np.abs(saved[name] - encoded[name]["logp"]))) for name in views}
    assert max(differences.values()) <= 1e-5, differences
    weight = plan["source_ctc_weights"][model_id]
    cases = (("full_hybrid", encoded["full"]["states"], 0.1, row["frames"]),
             ("crop_hybrid", encoded["crop"]["states"], weight, 82),
             ("direct_hybrid", encoded["context"]["states"][16:66], weight, 82),
             ("context_hybrid", encoded["context"]["states"], weight, 82))
    for name, states, ctc_weight, cap in cases:
        fresh = runtime.decode(states, ctc_weight, cap=cap, beam_size=40, penalty=0)
        old = previous["predictions"][name]
        assert fresh["tokens"] == old["tokens"] and fresh["text"] == old["text"]
        assert abs(fresh["score"] - old["score"]) <= 1e-4
    value = {"status": "passed", "utc": utc(), "clip_id": row["clip_id"], "decoder_replays": len(cases),
             "maximum_logit_difference_by_view": differences, "verifier_sha256": sha(Path(__file__))}
    save_json(folder / "Replay.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("VLR_T0", "VLR_T1"))
    parser.add_argument("--replay", action="store_true")
    arguments = parser.parse_args()
    replay(arguments.model) if arguments.replay else check(arguments.model)
