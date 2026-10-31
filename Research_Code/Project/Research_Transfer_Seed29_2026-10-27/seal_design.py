"""Freeze tested code and design before admitting any second-seed destination output."""

from common import HERE, ROOT, STORE, read, save, sha, utc


def main():
    assert not (HERE / "Admission.json").exists() and not (STORE / "Cache").exists()
    test = read(HERE / "Execution_Test_01/Verification.json")
    assert test["status"] == "complete_prior_calibration_replayed_and_execution_guards_passed"
    assert test["prior_independent_counts"] == 11682 and test["exact_complete_calibration_outputs"] == 3
    lifecycle = read(HERE / "Lifecycle_Test_01/Verification.json")
    assert lifecycle["status"] == "owned_parent_and_child_termination_verified"
    assert lifecycle["queue_code_sha256"] == sha(HERE / "run_local.py")
    files = list(HERE.glob("*.py")) + list((HERE / "Beam_Execution").glob("*.py"))
    files += [HERE / "Execution_Design.md", HERE / "Plan.json", HERE / "model_registry.json"]
    proof = {"status": "code_prepared_and_checked_before_admission", "utc": utc(),
             "files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(files))},
             "test_receipt_sha256": sha(HERE / "Execution_Test_01/Verification.json"),
             "lifecycle_receipt_sha256": sha(HERE / "Lifecycle_Test_01/Verification.json"),
             "native_seed29_forward_verified": False, "new_destination_predictions": False}
    save(HERE / "Execution_Design_Lock.json", proof)
    print({"status": proof["status"], "files": len(proof["files"]), "new_inference": False}, flush=True)


if __name__ == "__main__":
    main()
