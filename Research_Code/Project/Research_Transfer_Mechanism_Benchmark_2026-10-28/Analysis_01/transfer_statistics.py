"""Paired whole-population counts and the unchanged parent bootstrap schemes."""

import numpy as np


FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_arrays(records):
    require(bool(records), "No models")
    arrays = {name: np.asarray(rows) for name, rows in records.items()}
    first = next(iter(arrays.values()))
    require(first.ndim == 2 and first.shape[1] == 5 and len(first) > 0, "Invalid count shape")
    for name, value in arrays.items():
        require(value.shape == first.shape, "Incomplete model population: " + name)
        require(np.issubdtype(value.dtype, np.integer), "Counts must be integers")
        require(np.all(value >= 0), "Negative count")
        require(np.all(value[:, 0] == value[:, 1:4].sum(axis=1)), "S/D/I decomposition differs")
        require(np.all(value[:, 4] > 0), "Empty reference")
        require(np.array_equal(value[:, 4], first[:, 4]), "Unpaired reference counts")
    return arrays


def aggregate_sources(arrays, source_ids):
    names = sorted(set(source_ids))
    require(len(source_ids) == len(next(iter(arrays.values()))), "Missing source identity")
    require(all(isinstance(name, str) and bool(name) for name in names), "Invalid source identity")
    result = {model: np.zeros((len(names), 5), dtype=np.int64) for model in arrays}
    positions = {name: index for index, name in enumerate(names)}
    for position, source in enumerate(source_ids):
        for model, values in arrays.items():
            result[model][positions[source]] += values[position]
    return names, result


def bootstrap(arrays, pairs, unit, draws, seed):
    require(unit in {"video_source", "released_clip"}, "Unknown resampling unit")
    require(type(draws) is int and draws > 0 and type(seed) is int, "Invalid bootstrap parameters")
    require(bool(pairs) and len(set(pairs)) == len(pairs), "Missing or duplicated comparisons")
    words = next(iter(arrays.values()))[:, 4]
    differences = {left + "_minus_" + right: arrays[left][:, 0] - arrays[right][:, 0]
                   for left, right in pairs}
    output = {name: [] for name in differences}
    generator = np.random.default_rng(seed)
    remaining = draws
    checked = 0
    while remaining:
        size = min(256, remaining)
        if unit == "video_source":
            weights = generator.multinomial(len(words), np.full(len(words), 1 / len(words)), size)
            denominators = weights @ words
            batch = {name: 100 * (weights @ values) / denominators for name, values in differences.items()}
        else:
            indices = generator.integers(0, len(words), size=(size, len(words)))
            denominators = words[indices].sum(axis=1)
            batch = {name: 100 * values[indices].sum(axis=1) / denominators
                     for name, values in differences.items()}
        for name, values in differences.items():
            if checked == 0:
                for index in range(min(17, size)):
                    if unit == "video_source":
                        numerator = sum(int(weight) * int(error) for weight, error in zip(weights[index], values))
                        denominator = sum(int(weight) * int(word) for weight, word in zip(weights[index], words))
                    else:
                        numerator = sum(int(values[item]) for item in indices[index])
                        denominator = sum(int(words[item]) for item in indices[index])
                    require(abs(batch[name][index] - 100 * numerator / denominator) < 1e-12,
                            "Independent scalar bootstrap differs")
            output[name].extend(batch[name].tolist())
        checked += min(17, size) * len(differences) if checked == 0 else 0
        remaining -= size
    return {name: np.asarray(values) for name, values in output.items()}, checked


def describe(records, pairs, source_ids, unit, draws, seed):
    arrays = checked_arrays(records)
    if unit == "video_source":
        groups, resampled = aggregate_sources(arrays, source_ids)
    else:
        require(source_ids is None, "Do not present unavailable English source groups as reconstructed")
        groups = None
        resampled = arrays
    samples, scalar_checks = bootstrap(resampled, pairs, unit, draws, seed)
    models = {}
    for model, values in arrays.items():
        total = values.sum(axis=0)
        models[model] = {"clips": len(values), "counts": dict(zip(FIELDS, total.tolist())),
                         "wer_percent": float(100 * total[0] / total[4]),
                         "hypothesis_units": int(total[4] - total[2] + total[3])}
    comparisons = {}
    for left, right in pairs:
        name = left + "_minus_" + right
        result = {"gain_pp": models[left]["wer_percent"] - models[right]["wer_percent"],
                  "conditional_ci95_pp": np.quantile(samples[name], [.025, .975]).tolist(),
                  "positive_difference_favors": right,
                  "error_reductions": {field: models[left]["counts"][field] - models[right]["counts"][field]
                                       for field in FIELDS[:4]}}
        if groups is not None:
            differences = resampled[left][:, 0] - resampled[right][:, 0]
            result["sources_better_worse_equal"] = [int((differences > 0).sum()), int((differences < 0).sum()),
                                                    int((differences == 0).sum())]
        comparisons[name] = result
    return {"models": models, "comparisons": comparisons, "resampling_unit": unit,
            "source_groups": groups, "bootstrap_draws": draws, "bootstrap_seed": seed,
            "scalar_bootstrap_checks": scalar_checks, "pointwise_intervals": True,
            "multiple_comparison_adjustment": False, "training_seed_population_inference": False}
