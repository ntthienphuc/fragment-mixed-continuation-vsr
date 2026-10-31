"""Finite disjoint local/server queue; no outcome-dependent retries or reselection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import subprocess
import sys
import time

from common import HERE, ROOT, plan, read, save, sha, utc


def execute(task, args, folder):
    for script in ("worker.py", "verify_task.py"):
        label = task + "_" + script.removesuffix(".py")
        command = [sys.executable, "-B", "-X", "utf8", str(HERE / script), "--task", task,
                   "--attempt", args.attempt, "--deadline", str(args.deadline)]
        with (folder / (label + ".log")).open("x", encoding="utf-8") as log:
            worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=os.name != "nt")
            save(folder / (label + ".Start.json"), {"utc": utc(), "pid": worker.pid, "command": command})
            try:
                code = worker.wait(timeout=max(1, args.deadline - time.time()))
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    worker.terminate()
                else:
                    os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                code = 124
            save(folder / (label + ".Exit.json"), {"utc": utc(), "exit_code": code})
            assert code == 0, f"{label}: exit {code}; preserve this attempt"
    return task


def main(args):
    lock = read(HERE / "Execution_Lock.json")
    tasks = [name for name, task in lock["tasks"].items() if task["owner"] == args.owner]
    assert args.parallel in (1, 2) and (args.owner != "local" or args.parallel == 1)
    frozen = plan()
    assert time.time() < args.deadline <= frozen["admission_deadline_unix"]
    folder = HERE / "Queues" / args.attempt
    assert folder.resolve().parent == (HERE / "Queues").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "Start.json", {"utc": utc(), "pid": os.getpid(), "tasks": tasks,
                               "deadline": args.deadline, "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    try:
        for start in range(0, len(tasks), args.parallel):
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = [pool.submit(execute, name, args, folder) for name in tasks[start:start + args.parallel]]
                for future in futures:
                    future.result()
        save(folder / "Complete.json", {"status": "all_assigned_tasks_verified", "utc": utc(), "tasks": tasks,
                                      "execution_lock_sha256": sha(HERE / "Execution_Lock.json")})
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", choices=("local", "server"), required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args())
