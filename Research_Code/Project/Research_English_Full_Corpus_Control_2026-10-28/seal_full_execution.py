"""Freeze the tested full-population worker without admitting any benchmark run."""

from mechanism_common import HERE, ROOT, read, save, sha, utc, validate_inputs


def main():
    plan = validate_inputs()
    tests = read(HERE / "Execution_Verification_01.json")
    assert tests["status"] == "passed" and tests["malformed_cases_rejected"] == 51
    assert tests["plan_sha256"] == sha(HERE / "Plan.json")
    for name, digest in tests["scripts"].items():
        assert sha(HERE / name) == digest
    proof_path = HERE / "Source_Grid_CPU_01/Proofs/EN_F0_LAST/Complete.json"
    proof = read(proof_path)
    assert proof["status"] == "three_source_inputs_full_grid_verified"
    assert len(proof["comparisons"]) == 33 and proof["assessment_used"] is False
    assert proof["plan_sha256"] == sha(HERE / "Plan.json")
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    names = set(tests["scripts"]) | set(proof["scripts"])
    names.update({"seal_full_execution.py", "run_cpu_integration.py"})
    prerequisites = [HERE / "Execution_Verification_01.json", HERE / "Core_Verification_01.json",
                     HERE / "Source_Grid_CPU_01/Complete.json", proof_path]
    value = {"utc": utc(), "status": "F0_execution_frozen", "plan_sha256": sha(HERE / "Plan.json"),
             "tasks": plan["tasks"], "validation_task": "source_integration_EN_F0_LAST",
             "code_sha256": {(HERE / name).relative_to(ROOT).as_posix(): sha(HERE / name) for name in sorted(names)},
             "prerequisite_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in prerequisites},
             "benchmark_admitted": False, "source_integration_is_not_assessment": True}
    save(HERE / "Execution_Lock.json", value)
    print({"status": value["status"], "tasks": len(value["tasks"]), "code_files": len(names),
           "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})


if __name__ == "__main__":
    main()
