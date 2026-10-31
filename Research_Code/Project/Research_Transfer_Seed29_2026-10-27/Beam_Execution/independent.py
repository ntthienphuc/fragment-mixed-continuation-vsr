"""Scalar CTC probability and an independently implemented native decoder."""

import ast
import math
from pathlib import Path

import numpy as np

from common import ROOT


def ctc_total(logp, tokens):
    values = np.asarray(logp, dtype=np.float64)
    if not tokens:
        return float(values[:, 0].sum())
    labels = [0]
    for token in tokens:
        assert token > 0
        labels.extend((token, 0))
    previous = np.full(len(labels), -np.inf)
    previous[0] = values[0, 0]
    previous[1] = values[0, labels[1]]
    for frame in range(1, len(values)):
        current = np.full(len(labels), -np.inf)
        for state, label in enumerate(labels):
            total = previous[state]
            if state:
                total = np.logaddexp(total, previous[state - 1])
            if state >= 2 and label != 0 and label != labels[state - 2]:
                total = np.logaddexp(total, previous[state - 2])
            current[state] = total + values[frame, label]
        previous = current
    return float(np.logaddexp(previous[-1], previous[-2]))


def independent_decoder():
    import torch
    from torch.nn import functional
    path = ROOT / "Research_Native_Decoder_Reuse_2026-10-27/verify.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = ("linear", "norm", "attention", "manual_decoder")
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {"torch": torch, "functional": functional, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["manual_decoder"]


def independent_score(runtime, states, logp, tokens, ctc_weight):
    import torch
    ctc = ctc_total(logp, tokens)
    attention = 0.0
    if ctc_weight < 1:
        prefix = torch.tensor([[2001] + tokens], device=states.device, dtype=torch.long)
        target = torch.tensor(tokens + [2001], device=states.device, dtype=torch.long)
        with torch.inference_mode():
            values = independent_decoder()(prefix, states.unsqueeze(0), runtime.model.decoder.state_dict())
            probabilities = values[0].log_softmax(-1)
            attention = float(probabilities[torch.arange(len(target), device=states.device), target].sum())
    return {"ctc": ctc, "attention": attention, "merged": ctc_weight * ctc + (1 - ctc_weight) * attention}


def verify_ctc():
    import itertools
    rng = np.random.default_rng(17)
    checked = 0
    for frames in range(1, 6):
        probability = rng.uniform(0.1, 1.0, (frames, 3))
        probability /= probability.sum(axis=1, keepdims=True)
        expected = {}
        for path in itertools.product(range(3), repeat=frames):
            tokens = tuple(label for index, label in enumerate(path)
                           if label and (index == 0 or path[index - 1] != label))
            mass = math.prod(probability[index, label] for index, label in enumerate(path))
            expected[tokens] = expected.get(tokens, 0.0) + mass
        for tokens, mass in expected.items():
            assert abs(math.exp(ctc_total(np.log(probability), tokens)) - mass) < 1e-12
            checked += 1
    return checked
