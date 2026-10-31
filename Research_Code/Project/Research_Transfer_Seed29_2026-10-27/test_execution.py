"""Replay published-in-project calibration and test complete cohort coverage without new inference."""

import os
import subprocess
import sys

import numpy as np

from common import COHORT, HERE, ROOT, STORE, lines, read, save, sha, utc
from analyze_complete import describe
import calibration
from seal_execution import partition


def main():
    output = HERE / "Execution_Test_01"
    output.mkdir(exist_ok=False)
    origin = ROOT / "Research_Transfer_Decoder_Controls_2026-10-27"
    script = """
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from analyze import inputs
from common import save
frozen, sources, metrics, receipts, checked = inputs()
destination = Path(sys.argv[2])
np.save(destination / 'Previous_Metrics.npy', metrics, allow_pickle=False)
save(destination / 'Previous_Metadata.json', dict(frozen=frozen, sources=sources, checked=checked))
print(dict(status='previous_complete_matrix_recounted', checked=checked))
"""
    process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", script, str(origin), str(output)],
                             cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=240)
    (output / "Previous_Recount.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    meta = read(output / "Previous_Metadata.json")
    metrics = np.load(output / "Previous_Metrics.npy", allow_pickle=False)
    frozen = {**meta["frozen"], "candidate_order": meta["frozen"]["candidate_tie_order"]}
    weights = calibration.resampling(len(meta["sources"]), 10000, 20261027)
    expected = read(origin / "Analysis_01/Summary.json")
    comparisons = 0
    for name, candidates in (("primary", list(range(11))), ("greedy_only", list(range(4, 11))),
                             ("beam_only", list(range(4)))):
        actual = describe(metrics, meta["sources"], frozen, candidates, weights)
        assert actual == expected[name], name
        comparisons += 1
    models = read(HERE / "Plan.json")["models"]
    cohort = lines(COHORT)
    for owner in ("local", "server"):
        tasks = partition(cohort, models, owner)
        assert len(tasks) == 16
        for model in models:
            selected = [task for task in tasks.values() if task["model"] == model]
            assert len(selected) == 8
            assert sorted(position for task in selected for position in task["positions"]) == list(range(531))
        for task in tasks.values():
            assert len(task["positions"]) in (66, 67)
            assert set(task["replay_positions"]) <= set(task["score_positions"]) <= set(task["positions"])
    assert not (HERE / "Admission.json").exists()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    refused = subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "Beam_Execution/worker.py"),
                              "--task", "VLR_T0_29_LAST_00", "--attempt", "No_Admission",
                              "--deadline", "9999999999"], cwd=ROOT, env=environment,
                             text=True, encoding="utf-8", capture_output=True, timeout=90)
    (output / "Expected_Worker_Refusal.log").write_text(refused.stdout + refused.stderr, encoding="utf-8")
    assert refused.returncode != 0
    assert "Seed-29 inference is not admitted" in refused.stderr
    assert not (HERE / "Beam_Execution/Runs").exists() and not (STORE / "Cache").exists()
    save(output / "Verification.json", {
        "status": "complete_prior_calibration_replayed_and_execution_guards_passed", "utc": utc(),
        "prior_independent_counts": meta["checked"], "exact_complete_calibration_outputs": comparisons,
        "scalar_model_draw_checks": 606, "disjoint_allocation_configurations": 2,
        "real_worker_refused_before_output_creation": True, "new_seed29_predictions": False,
        "previous_summary_sha256": sha(origin / "Analysis_01/Summary.json"), "test_sha256": sha(__file__)})
    print(read(output / "Verification.json"), flush=True)


if __name__ == "__main__":
    main()
