"""Whole-corpus edit totals and paired source-group uncertainty."""

from collections import defaultdict
import hashlib

import numpy as np

FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def totals(records, decoder):
    assert records and len({row["clip_id"] for row in records}) == len(records)
    result = {field: sum(row["predictions"][decoder]["counts"][field] for row in records) for field in FIELDS}
    assert result["units"] > 0 and result["errors"] == sum(result[field] for field in FIELDS[1:4])
    result["wer_percent"] = 100 * result["errors"] / result["units"]
    return result


def paired(left, right, left_decoder, right_decoder, label, draws=10000):
    assert draws == 10000 and left and len(left) == len(right)
    assert len({row["clip_id"] for row in left}) == len(left)
    lookup = {row["clip_id"]: row for row in right}
    assert len(lookup) == len(right) and set(lookup) == {row["clip_id"] for row in left}
    grouped = defaultdict(lambda: [0, 0])
    clip_outcomes = [0, 0, 0]
    for baseline in left:
        comparison = lookup[baseline["clip_id"]]
        assert baseline["source_id"] == comparison["source_id"]
        first = baseline["predictions"][left_decoder]["counts"]
        second = comparison["predictions"][right_decoder]["counts"]
        assert first["units"] == second["units"] and first["units"] > 0
        difference = first["errors"] - second["errors"]
        grouped[baseline["source_id"]][0] += difference
        grouped[baseline["source_id"]][1] += first["units"]
        clip_outcomes[0 if difference > 0 else 1 if difference < 0 else 2] += 1
    sources = sorted(grouped)
    values = np.asarray([grouped[source] for source in sources], dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(("20261030|" + label).encode()).digest()[:8], "big")
    random = np.random.default_rng(seed)
    samples = []
    scalar_checks = 0
    for start in range(0, draws, 200):
        indices = random.integers(0, len(values), size=(min(200, draws - start), len(values)))
        summed = values[indices].sum(axis=1)
        batch = 100 * summed[:, 0] / summed[:, 1]
        if start == 0:
            for sampled, estimate in zip(indices[:101], batch[:101]):
                numerator = sum(int(values[index, 0]) for index in sampled)
                denominator = sum(int(values[index, 1]) for index in sampled)
                assert abs(float(estimate) - 100 * numerator / denominator) < 1e-12
                scalar_checks += 1
        samples.extend(batch.tolist())
    bounds = np.quantile(samples, [.025, .975], method="linear").tolist()
    return {"reference_minus_comparison_pp": 100 * int(values[:, 0].sum()) / int(values[:, 1].sum()),
            "ci95_low_pp": bounds[0], "ci95_high_pp": bounds[1], "source_groups": len(sources),
            "source_order": sources, "source_error_difference_and_units": values.tolist(),
            "clip_better_worse_equal": clip_outcomes, "bootstrap_unit": "archived_source_id",
            "bootstrap_resamples": draws, "bootstrap_seed": seed, "independent_scalar_checks": scalar_checks,
            "conditional_on_fitted_checkpoints": True, "multiplicity_adjusted": False}
