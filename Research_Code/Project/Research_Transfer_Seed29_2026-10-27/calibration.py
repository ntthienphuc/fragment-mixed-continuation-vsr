"""Source-disjoint decoder selection repeated inside each cluster bootstrap draw."""

import numpy as np


def evaluate(errors, units, folds, multiplicities):
    """errors [model, source, candidate]; weights [draw, source]. No label leakage across folds."""
    errors = np.asarray(errors, dtype=np.int64)
    weights = np.asarray(multiplicities, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    assert errors.shape[1] == len(units) == len(folds) == weights.shape[1]
    assert np.all(weights >= 0)
    total = np.zeros((errors.shape[0], len(weights)), dtype=np.int64)
    choices = []
    for fold in range(5):
        calibration = folds != fold
        assessment = folds == fold
        assert np.all(weights[:, calibration].sum(axis=1) > 0)
        calibration_errors = np.einsum("ds,msc->mdc", weights[:, calibration], errors[:, calibration, :])
        selected = calibration_errors.argmin(axis=-1)
        assessment_errors = np.einsum("ds,msc->mdc", weights[:, assessment], errors[:, assessment, :])
        total += np.take_along_axis(assessment_errors, selected[..., None], axis=-1)[..., 0]
        choices.append(selected)
    denominators = weights @ np.asarray(units, dtype=np.int64)
    assert np.all(denominators > 0)
    return {"errors": total, "wer": 100.0 * total / denominators[None, :],
            "choices": np.stack(choices, axis=2), "units": denominators}


def resampling(sources, draws, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sources, size=(draws, sources))
    return np.stack([np.bincount(row, minlength=sources) for row in indices])


def scalar_evaluate(errors, units, folds, weights):
    """Separate nested-loop implementation for exact finite checks."""
    model_totals = []
    for model in range(len(errors)):
        draw_totals = []
        for draw in weights:
            total = 0
            for fold in range(5):
                candidate_scores = []
                for candidate in range(errors.shape[2]):
                    score = 0
                    for source in range(len(units)):
                        if folds[source] != fold:
                            score += int(draw[source]) * int(errors[model, source, candidate])
                    candidate_scores.append(score)
                chosen = min(range(len(candidate_scores)), key=lambda index: candidate_scores[index])
                for source in range(len(units)):
                    if folds[source] == fold:
                        total += int(draw[source]) * int(errors[model, source, chosen])
            draw_totals.append(total)
        model_totals.append(draw_totals)
    return np.asarray(model_totals)


def selfcheck():
    rng = np.random.default_rng(127)
    errors = rng.integers(0, 400, (2, 40, 11))
    units = rng.integers(400, 600, 40)
    folds = np.tile(np.arange(5), 8)
    weights = resampling(40, 101, 20261027)
    vector = evaluate(errors, units, folds, weights)
    scalar = scalar_evaluate(errors, units, folds, weights)
    assert np.array_equal(vector["errors"], scalar)
    for fold in range(5):
        changed = errors.copy()
        changed[:, folds == fold, :] += rng.integers(1000, 2000, changed[:, folds == fold, :].shape)
        altered = evaluate(changed, units, folds, weights)
        assert np.array_equal(vector["choices"][:, :, fold], altered["choices"][:, :, fold])
    tied = evaluate(np.ones_like(errors), units, folds, weights)
    assert np.all(tied["choices"] == 0)
    return {"exact_scalar_model_draw_checks": 202, "heldout_label_exclusion_checks": 1010,
            "tie_order_checks": 1010}


if __name__ == "__main__":
    print(selfcheck())
