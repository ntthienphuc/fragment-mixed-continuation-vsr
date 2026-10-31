"""Seal the tested control worker; no allocation or assessment is admitted here."""

from mechanism_common import HERE, ROOT, read, save, sha, utc, validate_inputs


def main():
    plan = validate_inputs()
    proof = read(HERE / "Execution_Verification_01.json")
    assert proof["status"] == "passed" and proof["actual_source_prediction_checks"] == 132
    assert proof["complete_partitions_checked"] == 8 and len(proof["rejected_cases"]) == 114
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    names = list(proof["scripts"]) + ["experiment_core.py", "mechanism_common.py", "runtime_bridge.py",
                                     "source_grid_probe.py", "freeze_execution.py"]
    prerequisites = ["Preparation_Verification_02.json", "Core_Verification_01.json",
                     "Source_Grid_CPU_01/Complete.json", "Execution_Verification_01.json"]
    source_proof = read(HERE / "Source_Grid_CPU_01/Complete.json")
    assert source_proof["independent_grid_comparisons"] == 132
    code = {(HERE / name).relative_to(ROOT).as_posix(): sha(HERE / name) for name in names}
    evidence = {(HERE / name).relative_to(ROOT).as_posix(): sha(HERE / name) for name in prerequisites}
    save(HERE / "Execution_Lock.json", {"utc": utc(), "status": "mechanism_execution_frozen",
         "plan_sha256": sha(HERE / "Plan.json"), "tasks": plan["tasks"], "code_sha256": code,
         "prerequisite_sha256": evidence, "new_recognition_admitted": False,
         "remaining": ["actual worker/verifier integration", "execution-device preflight",
                       "finite resource admission", "complete comparison analysis"]})
    print({"status": "mechanism_execution_frozen", "tasks": len(plan["tasks"]), "GPU_work_admitted": False})


if __name__ == "__main__":
    main()
