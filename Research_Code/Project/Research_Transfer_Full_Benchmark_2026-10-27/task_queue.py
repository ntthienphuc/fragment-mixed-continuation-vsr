"""Finite sequential queue; run separate admitted lanes only within the resource allocation."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from admission import checked, require
from common import HERE, read, save, sha, utc


def terminate_owned(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                       capture_output=True, timeout=20)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def execute(command, prefix, deadline):
    kwargs = {"cwd": HERE}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout:
        with prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **kwargs)
            try:
                while process.poll() is None:
                    if time.time() >= deadline:
                        raise TimeoutError("Allocated queue time expired")
                    time.sleep(min(2, max(0.01, deadline - time.time())))
                require(process.returncode == 0, "Child failed; inspect its preserved stderr")
            finally:
                terminate_owned(process)


def main(stage, attempt, lane, lanes, device, threads):
    lock_path = HERE / "Execution_Lock.json"
    require(lock_path.exists(), "Execution is not locked")
    lock = read(lock_path)
    tasks = [name for name, task in lock["tasks"].items() if task["stage"] == stage]
    require(tasks and 0 <= lane < lanes <= 2, "Invalid finite queue assignment")
    require(attempt.isidentifier(), "Attempt must be a simple identifier")
    _, admission, _ = checked(stage, lock["tasks"][tasks[0]]["model"])
    require(lanes <= admission["maximum_concurrent_workers"], "Concurrency exceeds admission")
    folder = HERE / "Queues" / attempt / f"{stage}_{lane}"
    folder.mkdir(parents=True, exist_ok=False)
    assigned = tasks[lane::lanes]
    save(folder / "Start.json", {"utc": utc(), "tasks": assigned, "pid": os.getpid(),
                                 "execution_lock_sha256": sha(lock_path),
                                 "admission_sha256": sha(HERE / "Admission.json")})
    try:
        for name in assigned:
            require(time.time() < admission["deadline_unix"], "Queue deadline reached")
            common = ["--device", device, "--threads", str(threads)]
            worker = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "run_task.py"),
                      name, "--attempt", attempt, *common]
            execute(worker, folder / (name + "_worker"), admission["deadline_unix"])
            result = HERE / "Runs" / attempt / name
            verifier = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "verify_task.py"),
                        name, str(result), *common]
            execute(verifier, folder / (name + "_verify"), admission["deadline_unix"])
            proof = read(result / "Verification.json")
            require(proof["status"] == "passed", "Task verification failed")
        save(folder / "Complete.json", {"utc": utc(), "status": "all_assigned_tasks_verified", "tasks": assigned})
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "assessment"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    main(args.stage, args.attempt, args.lane, args.lanes, args.device, args.threads)
