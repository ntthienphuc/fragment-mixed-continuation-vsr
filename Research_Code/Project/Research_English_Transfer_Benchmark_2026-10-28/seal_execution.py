"""Bind the full runtime and task implementation after CPU contract checks."""

import argparse
from pathlib import Path

from english_common import HERE, READY, ROOT, read, require, save, sha, utc
from task_design import build_tasks


def main(verification_path):
    require(not (HERE / "Execution_Lock.json").exists(), "Execution already sealed")
    verification = read(verification_path)
    require(verification["status"] == "cpu_execution_checks_passed", "CPU checks incomplete")
    require(verification["source_prediction_counts_checked"] == 47608, "Selection check incomplete")
    require(verification["plan_sha256"] == sha(HERE / "Plan.json"), "Tests use a different plan")
    current = {path.name: sha(path) for path in sorted(HERE.glob("*.py"))}
    require(current == verification["code_sha256"], "Code changed after CPU checks")
    previous = read(READY / "Runtime_Validation/Lock.json")
    source = {}
    for name, digest in previous["sha256"].items():
        if name.endswith(".py"):
            require(sha(ROOT / name) == digest, "Existing runtime source changed")
            source[name] = digest
    for path in sorted(HERE.glob("*.py")):
        source[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ("Research_Interval_Transfer_2026-10-27/verify_probe.py",
                 "Research_Server_Confirmation_2026-10-19/confirmation_common.py"):
        source[name] = sha(ROOT / name)
    plan = read(HERE / "Plan.json")
    tasks = build_tasks(plan)
    require(len(tasks) == 64, "Task scope changed")
    save(HERE / "Execution_Lock.json", {"utc": utc(), "status": "code_and_all_tasks_frozen_not_admitted",
                                       "plan_sha256": sha(HERE / "Plan.json"), "tasks": tasks,
                                       "cpu_verification_path": verification_path.relative_to(ROOT).as_posix(),
                                       "cpu_verification_sha256": sha(verification_path), "code_sha256": source})
    print({"status": "sealed", "tasks": 64, "source_files": len(source),
           "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("verification", type=Path)
    main(parser.parse_args().verification.resolve())
