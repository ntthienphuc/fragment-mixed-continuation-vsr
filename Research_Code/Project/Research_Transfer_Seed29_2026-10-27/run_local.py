"""Finite local queue: verified inputs first, then the complete frozen decoder matrix."""

import argparse
import subprocess
import sys
import time

import psutil

from common import HERE, ROOT, checked_plan, read, save, sha, utc


def stop_owned_tree(worker):
    if worker.poll() is not None:
        return
    parent = psutil.Process(worker.pid)
    processes = parent.children(recursive=True)
    processes.append(parent)
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(processes, timeout=10)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    worker.wait(timeout=15)


def execute(script, arguments, folder, deadline, label):
    while psutil.virtual_memory().available < 3 * 2**30:
        assert time.time() < deadline, "Memory admission deadline; no child launched"
        print({"utc": utc(), "state": "waiting_for_3_GiB_RAM", "label": label}, flush=True)
        time.sleep(min(30, max(1, deadline - time.time())))
    command = [sys.executable, "-B", "-X", "utf8", str(HERE / script), *arguments]
    with (folder / (label + ".log")).open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        save(folder / (label + ".Start.json"), {"utc": utc(), "pid": worker.pid, "command": command})
        try:
            code = worker.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            stop_owned_tree(worker)
            code = 124
        save(folder / (label + ".Exit.json"), {"utc": utc(), "exit_code": code})
        assert code == 0, (label, code, "inspect preserved log")


def main(stage, deadline):
    frozen = checked_plan()
    assert time.time() < deadline <= frozen["admission_deadline_unix"]
    assert read(HERE / "Admission.json")["execution_location"] == "local"
    lock = read(HERE / "Execution_Design_Lock.json")
    for name, expected in lock["files"].items():
        assert sha(ROOT / name) == expected, name
    folder = HERE / ("Inputs_Local_01" if stage == "inputs" else "Matrix_Local_01")
    folder.mkdir(exist_ok=False)
    save(folder / "Start.json", {"utc": utc(), "stage": stage, "deadline": deadline,
                                 "design_lock_sha256": sha(HERE / "Execution_Design_Lock.json")})
    try:
        if stage == "inputs":
            for model in frozen["models"]:
                for script in ("encode.py", "verify.py"):
                    execute(script, [model, "--deadline", str(deadline)], folder, deadline,
                            model + "_" + script.removesuffix(".py"))
            execute("prepare_execution.py", [], folder, deadline, "prepare_readout")
            execute("Beam_Execution/probe_runtime.py", ["--deadline", str(deadline)],
                    folder, deadline, "readout_preflight")
            proof = read(HERE / "Beam_Execution/Preflight_01/Verification.json")
            assert proof["status"] == "passed" and proof["exact_beam_replays"] == 8
            status = "encoder_and_readout_preflight_verified"
        else:
            proof = read(HERE / "Inputs_Local_01/Complete.json")
            assert proof["status"] == "encoder_and_readout_preflight_verified"
            execute("seal_execution.py", ["--owner", "local"], folder, deadline, "seal_tasks")
            execute("Beam_Execution/task_queue.py", ["--owner", "local", "--attempt", "Local_01",
                    "--parallel", "1", "--deadline", str(deadline)], folder, deadline, "complete_beams")
            execute("analyze_complete.py", [], folder, deadline, "analysis")
            proof = read(HERE / "Completion_Receipt.json")
            assert proof["status"] == "seed29_full_matrix_analysis_verified"
            status = "complete_matrix_verified"
        save(folder / "Complete.json", {"utc": utc(), "status": status})
        print({"utc": utc(), "status": status}, flush=True)
    except BaseException as error:
        save(folder / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inputs", "beams"), required=True)
    parser.add_argument("--deadline", type=float, required=True)
    arguments = parser.parse_args()
    main(arguments.stage, arguments.deadline)
