"""Check weighted paired uncertainty against a scalar calculation before target inference."""

import numpy as np

from full_analysis_core import bootstrap, total_counts
from mechanism_common import HERE, read, save, sha, utc


def rejected(action):
    try:
        action()
    except (ValueError, AssertionError):
        return
    raise AssertionError("Invalid statistical input was accepted")


def main():
    values = [{"errors": 2, "substitutions": 1, "deletions": 1, "insertions": 0, "units": 10},
              {"errors": 1, "substitutions": 0, "deletions": 0, "insertions": 1, "units": 2}]
    assert total_counts(values)["wer_percent"] == 25
    words = [3, 8, 11, 2, 1, 6, 4]
    left = [2, 5, 10, 3, 0, 4, 2]
    right = [1, 7, 6, 1, 1, 3, 2]
    draws = 521
    result = bootstrap({"contrast": (left, right), "identity": (left, left)}, words, draws, 17)
    generator = np.random.default_rng(17)
    expected = []
    for repeat in range(draws):
        selected = generator.integers(0, len(words), size=len(words)).tolist()
        numerator = sum(left[index] - right[index] for index in selected)
        denominator = sum(words[index] for index in selected)
        expected.append(100 * numerator / denominator)
    np.testing.assert_allclose(result["contrast"], expected, atol=1e-12, rtol=0)
    assert np.all(result["identity"] == 0)
    reversed_result = bootstrap({"contrast": (right, left)}, words, draws, 17)
    np.testing.assert_allclose(reversed_result["contrast"], -result["contrast"], atol=0, rtol=0)
    rejected(lambda: total_counts([]))
    for field, value in (("errors", -1), ("errors", 9), ("units", 0), ("substitutions", 99)):
        rejected(lambda field=field, value=value: total_counts([{**values[0], field: value}]))
    rejected(lambda: bootstrap({"x": (left, right[:-1])}, words, 10, 17))
    rejected(lambda: bootstrap({"x": (left, right)}, words[:-1], 10, 17))
    rejected(lambda: bootstrap({"x": (left, right)}, [0] + words[1:], 10, 17))
    rejected(lambda: bootstrap({"x": (left, [-1] + right[1:])}, words, 10, 17))
    rejected(lambda: bootstrap({}, words, 10, 17))
    rejected(lambda: bootstrap({"x": (left, right)}, words, 0, 17))
    receipt = {"utc": utc(), "status": "passed", "scalar_bootstrap_comparisons": draws,
               "weighted_WER_verified": True, "identity_and_sign_reversal_verified": True,
               "malformed_cases_rejected": 11, "assessment_predictions_read": False,
               "scripts": {name: sha(HERE / name) for name in ("full_analysis_core.py", "test_full_analysis.py")}}
    save(HERE / "Analysis_Verification_01.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
