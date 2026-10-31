"""Paired word-error comparisons on the same complete released clip population."""

import numpy as np

from execution_common import require


FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def total_counts(values):
    require(bool(values), "Empty count population")
    for item in values:
        require(all(type(item[field]) is int and item[field] >= 0 for field in FIELDS), "Invalid count")
        require(item["units"] > 0, "Empty reference")
        require(item["errors"] == item["substitutions"] + item["deletions"] + item["insertions"], "Wrong edits")
        require(item["substitutions"] + item["deletions"] <= item["units"], "Impossible reference allocation")
    result = {field: sum(value[field] for value in values) for field in FIELDS}
    result["wer_percent"] = 100 * result["errors"] / result["units"]
    return result


def bootstrap(pairs, words, draws, seed):
    require(type(draws) is int and 0 < draws <= 100000, "Invalid bootstrap budget")
    denominator = np.asarray(words, dtype=np.int64)
    require(denominator.ndim == 1 and len(denominator) > 0 and np.all(denominator > 0), "Invalid denominators")
    differences = {}
    for name, (left, right) in pairs.items():
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        require(left.shape == right.shape == denominator.shape, "Unpaired population")
        require(np.all(left >= 0) and np.all(right >= 0), "Negative errors")
        differences[name] = left - right
    require(bool(differences), "Empty comparisons")
    samples = {name: [] for name in differences}
    generator = np.random.default_rng(seed)
    remaining = draws
    while remaining:
        count = min(256, remaining)
        indices = generator.integers(0, len(denominator), size=(count, len(denominator)))
        denominators = denominator[indices].sum(axis=1)
        for name, values in differences.items():
            samples[name].extend((100 * values[indices].sum(axis=1) / denominators).tolist())
        remaining -= count
    return {name: np.asarray(value, dtype=np.float64) for name, value in samples.items()}
