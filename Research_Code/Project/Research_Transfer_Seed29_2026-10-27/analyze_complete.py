"""Require the complete second-seed matrix and recount every scored prediction."""

import sys
import numpy as np

from common import COHORT, HERE, ROOT, checked_plan, lines, read, save, sha, utc
import calibration

sys.path.insert(0, str(COHORT.parent))
from verify_probe import counts

FIELDS = ("errors", "substitutions", "deletions", "insertions", "units")


def collect(frozen):
    folder = HERE / "Beam_Execution"
    lock_path = folder / "Execution_Lock.json"
    lock = read(lock_path)
    assert lock["plan_sha256"] == sha(folder / "Plan.json")
    for name, expected in lock["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert len(lock["tasks"]) == 16
    records = {model: {} for model in frozen["models"]}
    evidence = {lock_path.relative_to(ROOT).as_posix(): sha(lock_path)}
    for task_name, task in lock["tasks"].items():
        matches = list((folder / "Runs").glob("*/" + task_name + "/Verification.json"))
        assert len(matches) == 1, (task_name, "exactly one verified complete attempt is required")
        proof = read(matches[0])
        path = matches[0].parent / "Predictions.jsonl"
        assert proof["status"] == "passed" and proof["execution_lock_sha256"] == sha(lock_path)
        assert proof["predictions_sha256"] == sha(path)
        assert proof["independent_count_checks"] == 4 * len(task["positions"])
        assert proof["exact_beam_replays"] == 8 and proof["independent_score_checks"] == 16
        rows = lines(path)
        assert [row["index"] for row in rows] == task["positions"]
        target = records[task["model"]]
        for row in rows:
            assert row["clip_id"] not in target
            target[row["clip_id"]] = row
        for item in (path, matches[0]):
            evidence[item.relative_to(ROOT).as_posix()] = sha(item)
    cohort = lines(COHORT)
    sources = sorted(frozen["source_folds"])
    candidates = frozen["candidate_order"]
    metrics = np.zeros((2, len(sources), 11, len(FIELDS)), dtype=np.int64)
    checked = 0
    for model_index, model in enumerate(frozen["models"]):
        path = HERE / "Greedy_01" / model / "Predictions.jsonl"
        proof_path = path.parent / "Verification.json"
        proof = read(proof_path)
        assert proof["status"] == "passed" and proof["predictions_sha256"] == sha(path)
        assert proof["plan_sha256"] == sha(HERE / "Plan.json")
        blank = {row["clip_id"]: row for row in lines(path)}
        assert set(blank) == set(records[model]) == {row["clip_id"] for row in cohort}
        assert len(blank) == 531
        for item in (path, proof_path):
            evidence[item.relative_to(ROOT).as_posix()] = sha(item)
        for row in cohort:
            beam = records[model][row["clip_id"]]
            greedy = blank[row["clip_id"]]
            assert beam["source_id"] == greedy["source_id"] == row["source_id"]
            assert beam["frames"] == row["frames"]
            assert beam["cache_sha256"] == greedy["cache_sha256"]
            outputs = {**beam["predictions"], **greedy["predictions"]}
            assert set(outputs) == set(candidates)
            source = sources.index(row["source_id"])
            for candidate_index, candidate in enumerate(candidates):
                output = outputs[candidate]
                actual = counts(row["reference"], output["text"])
                assert actual == output["counts"]
                metrics[model_index, source, candidate_index] += [actual[field] for field in FIELDS]
                checked += 1
    assert checked == frozen["expected_final_prediction_count"] == 11682
    assert metrics[0, :, 0, 4].sum() == 13887
    return sources, metrics, evidence, checked


def describe(metrics, sources, frozen, candidates, weights):
    folds = np.array([frozen["source_folds"][source] for source in sources])
    units = metrics[0, :, 0, 4]
    assert np.all(metrics[..., 4] == units[None, :, None])
    errors = metrics[:, :, candidates, 0]
    center = calibration.evaluate(errors, units, folds, np.ones((1, len(sources)), dtype=np.int64))
    draws = calibration.evaluate(errors, units, folds, weights)
    scalar = calibration.scalar_evaluate(errors, units, folds, weights[:101])
    assert np.array_equal(draws["errors"][:, :101], scalar)
    chosen = center["choices"][:, 0]
    selected = np.zeros((2, len(sources), len(FIELDS)), dtype=np.int64)
    for model in range(2):
        for source, fold in enumerate(folds):
            selected[model, source] = metrics[model, source, candidates[chosen[model, fold]]]
    assert np.array_equal(selected[..., 0].sum(axis=1), center["errors"][:, 0])
    gain = draws["wer"][0] - draws["wer"][1]
    folds_output = []
    for fold in range(5):
        total = selected[:, folds == fold].sum(axis=1)
        wers = 100.0 * total[:, 0] / total[:, 4]
        folds_output.append({
            "fold": fold, "wer": wers.tolist(), "gain_pp": float(wers[0] - wers[1]),
            "selected": [frozen["candidate_order"][candidates[chosen[model, fold]]] for model in range(2)]})
    return {
        "wer": center["wer"][:, 0].tolist(),
        "gain_pp": float(center["wer"][0, 0] - center["wer"][1, 0]),
        "source_ci95_pp": np.quantile(gain, [.025, .975]).tolist(), "folds": folds_output,
        "counts": [dict(zip(FIELDS, row.tolist())) for row in selected.sum(axis=1)],
        "source_counts": {source: selected[:, index].tolist() for index, source in enumerate(sources)}}


def main():
    frozen = checked_plan()
    design = read(HERE / "Execution_Design_Lock.json")
    for name, expected in design["files"].items():
        assert sha(ROOT / name) == expected, name
    synthetic = calibration.selfcheck()
    sources, metrics, evidence, checked = collect(frozen)
    weights = calibration.resampling(len(sources), 10000, 20261027)
    primary = describe(metrics, sources, frozen, list(range(11)), weights)
    greedy = describe(metrics, sources, frozen, list(range(4, 11)), weights)
    beam = describe(metrics, sources, frozen, list(range(4)), weights)
    fixed = {}
    units = metrics[0, :, 0, 4]
    for index, candidate in enumerate(frozen["candidate_order"]):
        totals = metrics[:, :, index].sum(axis=1)
        wers = 100.0 * totals[:, 0] / totals[:, 4]
        gain = 100.0 * (weights @ (metrics[0, :, index, 0] - metrics[1, :, index, 0])) / (weights @ units)
        fixed[candidate] = {
            "wer": wers.tolist(), "gain_pp": float(wers[0] - wers[1]),
            "source_ci95_pp": np.quantile(gain, [.025, .975]).tolist(),
            "counts": [dict(zip(FIELDS, row.tolist())) for row in totals]}
    result = {
        "status": "complete_verified_seed29_final_pair_decoder_matrix", "utc": utc(),
        "models": frozen["models"], "primary": primary, "greedy_only": greedy, "beam_only": beam, "fixed": fixed,
        "independent_counts_checked": checked, "real_scalar_model_draw_checks": 606,
        "synthetic_verification": synthetic, "input_sha256": evidence,
        "plan_sha256": sha(HERE / "Plan.json"), "design_lock_sha256": sha(HERE / "Execution_Design_Lock.json"),
        "new_independent_test": False, "novelty_established": False, "paper_changed": False}
    save(HERE / "Analysis_01/Summary.json", result)
    for name, expected in evidence.items():
        assert sha(ROOT / name) == expected
    save(HERE / "Completion_Receipt.json", {
        "status": "seed29_full_matrix_analysis_verified", "utc": utc(),
        "summary_sha256": sha(HERE / "Analysis_01/Summary.json"),
        "verified_tasks": 16, "independent_counts_checked": checked})
    print({"status": result["status"], "primary": primary}, flush=True)


if __name__ == "__main__":
    main()
