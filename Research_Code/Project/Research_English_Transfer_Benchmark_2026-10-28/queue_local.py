"""Finite local source-only queue; verified tasks survive later server continuation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from english_common import HERE, ROOT, read, require, save, sha, utc
from local_resources import resources, stop_owned


def wait_resources(deadline):
    while time.time() < deadline:
        state = resources()
        if state["available_ram_gib"] >= 3 and state["free_gpu_mib"] >= 3000:
            return state
        print({"utc": utc(), "status": "waiting_for_memory", **state}, flush=True)
        time.sleep(min(15, max(.1, deadline - time.time())))
    raise TimeoutError("Local resource window expired")


def execute(command, folder, label, deadline):
    state = wait_resources(deadline)
    with (folder / (label + ".stdout.log")).open("x", encoding="utf-8") as stdout:
        with (folder / (label + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=HERE, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            save(folder / (label + ".Start.json"), {"utc": utc(), "pid": process.pid, "resources": state})
            try:
                while process.poll() is None:
                    if time.time() >= deadline:
                        raise TimeoutError("Local hard deadline reached")
                    time.sleep(min(2, max(.1, deadline - time.time())))
                require(process.returncode == 0, label + " failed; inspect its preserved logs")
            finally:
                stop_owned(process)


def backup(label):
    destination = Path("C:/Users/thien/Lip_Research_Backups/English_Transfer_20261027")
    path = destination / ("Benchmark_" + label + "_" + str(time.time_ns()) + ".zip")
    expected = {}
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for source in sorted(HERE.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".tmp":
                continue
            if "Runs" in source.parts and not (source.parent / "Verification.json").exists():
                continue
            require(source.stat().st_size < 64 * 2**20, "Unexpected large metadata file")
            payload = source.read_bytes()
            name = source.relative_to(ROOT).as_posix()
            expected[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == set(expected), "Backup members differ")
        for name, digest in expected.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Backup readback differs")
    return {"utc": utc(), "status": "sha256_readback_passed", "path": str(path),
            "sha256": sha(path), "members": len(expected), "files": expected}


def main(mode, attempt, seconds, tasks, preflight_attempt):
    require(os.name == "nt" and attempt.isidentifier(), "Local Windows attempt required")
    require(type(seconds) is int and 0 < seconds <= 7200, "Invalid local bound")
    plan = read(HERE / "Plan.json")
    execution = read(HERE / "Execution_Lock.json")
    require(execution["plan_sha256"] == sha(HERE / "Plan.json"), "Execution plan differs")
    for name, digest in execution["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Execution changed: " + name)
    folder = HERE / "Local_Queues" / attempt
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + seconds
    proofs = {}
    if mode == "development":
        require(bool(tasks) and len(set(tasks)) == len(tasks), "Explicit distinct source tasks required")
        for name in tasks:
            require(name in execution["tasks"] and execution["tasks"][name]["stage"] == "development",
                    "Only source-development tasks may run locally")
        for model in plan["models"]:
            path = HERE / "Runtime_Preflights" / preflight_attempt / (model + ".json")
            proof = read(path)
            require(proof["status"] == "native_source_cache_preflight_passed", "Missing runtime preflight")
            require(proof["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json"), "Preflight code changed")
            proofs[model] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    else:
        require(mode == "preflight" and not tasks, "Preflight cannot launch benchmark tasks")
        (HERE / "Runtime_Preflights" / attempt).mkdir(parents=True, exist_ok=False)
    admission_path = HERE / "Admissions" / (attempt + ".json")
    admission = {"utc": utc(), "status": "bounded_english_execution_admitted", "execution_location": "local",
                 "admitted_unix": started, "deadline_unix": deadline, "maximum_seconds": seconds,
                 "paid_allocation": False, "stages": ["preflight" if mode == "preflight" else "development"],
                 "models": plan["models"], "tasks": tasks, "runtime_preflights": proofs,
                 "plan_sha256": sha(HERE / "Plan.json"), "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
                 "locations_path": (HERE / "Local_Locations.json").relative_to(ROOT).as_posix(),
                 "locations_sha256": sha(HERE / "Local_Locations.json"),
                 "reason": "Native source-only preparation after replicated Vietnamese development benefit",
                 "assessment_admitted": False, "maximum_concurrent_processes": 1}
    save(admission_path, admission)
    save(folder / "Start.json", {"utc": utc(), "pid": os.getpid(), "mode": mode,
                                  "admission_sha256": sha(admission_path), "deadline_unix": deadline})
    completed = {}
    try:
        if mode == "preflight":
            for model in plan["models"]:
                target = HERE / "Runtime_Preflights" / attempt / (model + ".json")
                command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "preflight.py"), model,
                           str(target), "--admission", str(admission_path)]
                execute(command, folder, model, deadline)
                proof = read(target)
                require(proof["status"] == "native_source_cache_preflight_passed", "Failed preflight")
                completed[model] = sha(target)
                print({"utc": utc(), "model": model, "status": "preflight_passed"}, flush=True)
        else:
            prior_seconds = 0
            for task in tasks:
                if completed and deadline - time.time() < max(300, 1.5 * prior_seconds):
                    save(folder / "Stopped_Between_Tasks.json", {"utc": utc(), "completed": completed,
                                                               "reason": "Reserve insufficient for another complete task"})
                    save(folder / "Backup.json", backup(attempt))
                    return
                task_start = time.time()
                command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "run_task.py"), task,
                           "--attempt", attempt, "--admission", str(admission_path)]
                execute(command, folder, task + "_worker", deadline)
                target = HERE / "Runs" / attempt / task
                command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "verify_task.py"), task,
                           str(target), "--admission", str(admission_path)]
                execute(command, folder, task + "_verify", deadline)
                proof = read(target / "Verification.json")
                require(proof["status"] == "passed" and proof["task"] == task, "Task verification failed")
                completed[task] = sha(target / "Verification.json")
                prior_seconds = time.time() - task_start
                save(folder / (task + "_Backup.json"), backup(task))
                print({"utc": utc(), "task": task, "status": "verified", "seconds": prior_seconds}, flush=True)
        save(folder / "Complete.json", {"utc": utc(), "status": "bounded_local_queue_complete", "mode": mode,
                                        "completed": completed, "seconds": time.time() - started,
                                        "full_benchmark_complete": False, "assessment_clips": 0})
        save(folder / "Backup.json", backup(attempt))
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error), "completed": completed})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "development"])
    parser.add_argument("attempt")
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--preflight-attempt", default="Preflight_Local_01")
    arguments = parser.parse_args()
    main(arguments.mode, arguments.attempt, arguments.seconds, arguments.tasks, arguments.preflight_attempt)
