"""Read back completed exposure results and bind them to the unchanged training code."""

import sys

from common import HERE, PHASE5, ROOT, read, save, sha, utc

sys.path.insert(0, str(PHASE5))
from phase5.common import code_identity


def main():
    folder = HERE / "Replay_03"
    summary = read(folder / "Summary.json")
    assert summary["status"] == "complete_deterministic_schedule_reconstruction"
    assert summary["native_objective_control_flow_checks"] == 192000
    assert summary["neural_forward_calls"] == 0
    assert not summary["pixels_read"] and not summary["gradient_or_optimizer_replay"]
    assert summary["training_receipt"]["code_sha256"] == code_identity()
    assert summary["code_sha256"] == sha(HERE / "reconstruct.py")
    assert summary["helpers_sha256"] == sha(HERE / "common.py")
    assert summary["protocol_sha256"] == sha(HERE / "Protocol.md")
    for path, expected in summary["training_code_sha256"].items():
        assert sha(ROOT / path) == expected
    assert set(summary["seeds"]) == {"17", "29"}
    assertions = 0
    for seed, item in summary["seeds"].items():
        counts = read(folder / ("Counts_" + seed + ".json"))
        assert counts["source_samples"]["T0"] == counts["source_samples"]["T1"]
        assert len(counts["source_samples"]["T0"]) == item["sources_seen"]
        for method in ("T0", "T1"):
            total = item["final_exposure"][method]
            assert sum(counts["source_samples"][method].values()) == total["samples"] == 48000
            assert sum(counts["source_frames"][method].values()) == total["input_frames"]
            assert sum(counts["target_token_histograms"][method].values()) == total["target_tokens"]
            assert total["attention_positions"] == total["target_tokens"] + total["samples"]
            assert total["full_samples"] + total["crop_samples"] == total["samples"]
            selected = item["source_selected_exposure"][method]
            assert selected["exposure"]["samples"] == selected["step"] * 8
            assert all(selected["exposure"][field] <= total[field] for field in total)
            assertions += 8
        for field, actual in item["T1_over_T0"].items():
            expected = item["final_exposure"]["T1"][field] / item["final_exposure"]["T0"][field]
            assert abs(actual - expected) < 1e-12
            assertions += 1
    loss = read(HERE / "Loss_Check_01/Verification.json")
    assert loss["status"] == "actual_criterion_and_independent_kl_passed"
    assert loss["native_normalize_length"] is False
    assert loss["code_sha256"] == sha(HERE / "verify_loss_reduction.py")
    assert loss["criterion_sha256"] == sha(ROOT / loss["criterion_path"])
    assert abs(loss["results"]["False"]["repeated_over_original"] - 2) < 1e-12
    assert abs(loss["results"]["True"]["repeated_over_original"] - 1) < 1e-12
    paths = [folder / name for name in ("Summary.json", "Counts_17.json", "Counts_29.json")]
    paths.append(HERE / "Loss_Check_01/Verification.json")
    save(HERE / "Completion_Receipt.json", {
        "status": "deterministic_exposure_and_loss_semantics_verified", "utc": utc(),
        "native_scheduling_checks": 192000, "aggregate_assertions": assertions,
        "original_training_code_identity": code_identity(),
        "evidence_sha256": {path.relative_to(HERE).as_posix(): sha(path) for path in paths},
        "verification_code_sha256": sha(HERE / "verify_summary.py"),
        "recorded_loss_reduction_checked_with_independent_kl": True,
        "gradient_equivalence_established": False, "recognition_or_efficiency_gain_established": False,
        "failed_attempts_preserved": ["Replay_01", "Replay_02"],
    })
    print({"status": "passed", "native_checks": 192000, "aggregate_assertions": assertions}, flush=True)


if __name__ == "__main__":
    main()
