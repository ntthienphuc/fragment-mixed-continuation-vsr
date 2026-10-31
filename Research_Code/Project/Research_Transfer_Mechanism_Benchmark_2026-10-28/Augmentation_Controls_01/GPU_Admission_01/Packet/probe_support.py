"""Portable packet integrity, real input streams and exact state fingerprints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def verify_packet(packet):
    manifest = read(packet / "Manifest.json")
    for name, expected in manifest.items():
        path = (packet / name).resolve()
        if not path.is_relative_to(packet.resolve()) or sha(path) != expected:
            raise ValueError("Packet member changed: " + name)
    plan = read(packet / "Plan.json")
    if plan["assessment_read"] or plan["training_admitted"]:
        raise ValueError("Resource check must not claim assessment or training admission")
    return plan


class Inputs:
    def __init__(self, packet, plan):
        self.packet = packet
        self.cases = {item["sample"]: item for item in plan["cases"]}
        if len(self.cases) != len(plan["cases"]) or not set(range(16)).issubset(self.cases):
            raise ValueError("Initial training updates are incomplete")

    def sample(self, position):
        case = self.cases[position]
        path = self.packet / case["file"]
        if sha(path) != case["file_sha256"] or case["row"]["split"] != "train":
            raise ValueError("Input identity or split changed")
        with np.load(path, allow_pickle=False) as archive:
            pixels = archive["frames"]
        if hashlib.sha256(pixels.tobytes()).hexdigest() != case["pixels_sha256"]:
            raise ValueError("Training pixels changed")
        return case["row"], pixels

    def stream(self, first, stop):
        for position in range(first, stop):
            yield self.sample(position)


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("Nonfinite saved parameter or optimizer state")
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, ensure_ascii=True, allow_nan=False).encode() + b";")

    visit(value)
    return digest.hexdigest()


def complete_state(model, optimizer, scaler):
    from phase5.runtime import rng_state

    return fingerprint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng_state()})
