"""Freeze the tested inference design without admitting or launching recognition."""

from admission import require, verify_plan
from common import ENGINE, HERE, PARENT, ROOT, SEED29, read, save, sha, utc
from task_design import build_tasks


def main():
    plan = verify_plan()
    require(not (HERE / "Admission.json").exists(), "Execution must freeze before admission")
    reports = (HERE / "Execution_Tests_01/Verification.json", HERE / "Protocol_Verification.json",
               HERE / "Selection_Verification.json")
    for path in reports:
        require(read(path)["status"] == "passed", "Preparation tests failed")
    require(read(reports[0])["historical_prediction_counts_checked"] == 17523, "Reuse checks incomplete")
    require(read(reports[1])["real_record_count_checks"] == 5841, "Verifier checks incomplete")
    files = {}
    for phase in (PARENT, SEED29):
        prior = read(phase / "Beam_Execution/Execution_Lock.json")
        for name, expected in prior["code_sha256"].items():
            require(sha(ROOT / name) == expected, "Prior numerical code changed: " + name)
            files[name] = expected
    native = read(HERE / "Native_Reference_Registry.json")
    for name, expected in native["support_sources_sha256"].items():
        require(sha(ROOT / name) == expected, "Native support source changed")
        files[name] = expected
    entry = native["models"]["VC_NATIVE"]
    source = ROOT / entry["source"]
    for path in source.rglob("*.py"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in HERE.glob("*.py"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in plan["frozen_sha256"].items():
        if name.endswith(".py"):
            files[name] = expected
    for path in (ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py",
                 ENGINE / "confirmation_common.py"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    tasks = build_tasks(plan)
    save(HERE / "Execution_Lock.json", {
        "status": "prepared_and_frozen_not_admitted", "utc": utc(), "plan_sha256": sha(HERE / "Plan.json"),
        "code_sha256": files, "task_chunks_per_model_stage": 8, "tasks": tasks,
        "maximum_queue_concurrency": 2, "runtime_preflights_required": ["standard", "vicocktail"],
        "native_beam_step_cap": "complete encoded input length, matching historical native search",
        "fresh_pixel_replay_rule": "first and last row in each length-strided task, all admitted decoders",
        "preparation_proofs": {path.relative_to(ROOT).as_posix(): sha(path) for path in reports},
        "new_gpu_work_admitted": False, "test_decoder_selection_completed": False,
    })
    require(all(sha(ROOT / name) == expected for name, expected in files.items()), "Code changed while locking")
    print({"status": "frozen_not_admitted", "tasks": len(tasks), "source_files": len(files)}, flush=True)


if __name__ == "__main__":
    main()
