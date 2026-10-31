"""One bounded, sequential local GPU preflight; preserve all partial attempts."""

import argparse
import ctypes
import os
import subprocess
import sys
import time

from common import HERE, check_lock, read, save, sha, utc


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]
    _fields_ += [(name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
                 "avail_virtual", "avail_extended")]


def resource_state():
    assert os.name == "nt", "This admission is local Windows only"
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                     text=True, timeout=10)
    values = output.strip().splitlines()
    assert len(values) == 1
    return {"available_ram_gib": status.avail_phys / 2**30, "free_gpu_mib": float(values[0])}


def terminate_owned(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=20)
        process.wait(timeout=20)


def main(attempt):
    assert attempt.isidentifier()
    lock = check_lock()
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    started = time.time()
    deadline = started + lock["maximum_seconds"]
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "started_unix": started,
                                  "deadline_unix": deadline, "lock_sha256": sha(HERE / "Lock.json")})
    completed = {}
    try:
        for model in lock["models"]:
            while True:
                assert time.time() < deadline, "Local preflight allocation expired"
                state = resource_state()
                if (state["available_ram_gib"] >= lock["minimum_available_ram_gib"]
                        and state["free_gpu_mib"] >= lock["minimum_free_gpu_mib"]):
                    break
                print({"utc": utc(), "status": "waiting_for_memory", **state}, flush=True)
                time.sleep(min(30, deadline - time.time()))
            proof_path = output / (model + ".json")
            command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "preflight.py"), model,
                       str(proof_path), "--deadline", str(deadline)]
            with (output / (model + ".stdout.log")).open("x", encoding="utf-8") as stdout:
                with (output / (model + ".stderr.log")).open("x", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=HERE, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW
                                               | subprocess.CREATE_NEW_PROCESS_GROUP)
                    save(output / (model + ".Start.json"), {"utc": utc(), "pid": process.pid, "resources": state})
                    try:
                        while process.poll() is None:
                            if time.time() >= deadline:
                                raise TimeoutError("Registered local runtime deadline reached")
                            time.sleep(min(2, deadline - time.time()))
                        assert process.returncode == 0, model + " failed; inspect its preserved stderr"
                    finally:
                        terminate_owned(process)
            proof = read(proof_path)
            assert proof["status"] == "passed" and proof["model"] == model
            assert proof["lock_sha256"] == sha(HERE / "Lock.json")
            assert len(proof["checks"]) == 3
            assert sum(len(row["comparisons"]) for row in proof["checks"]) == 12
            completed[model] = sha(proof_path)
            print({"utc": utc(), "model": model, "status": "passed"}, flush=True)
        save(output / "Complete.json", {"utc": utc(), "status": "all_four_english_runtime_checks_passed",
                                        "models": completed, "native_bounded_pairs": 48,
                                        "test_clips_used": False, "accuracy_claimed": False,
                                        "lock_sha256": sha(HERE / "Lock.json")})
    except BaseException as error:
        save(output / "Failure.json", {"utc": utc(), "error": repr(error), "completed_models": completed})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt")
    main(parser.parse_args().attempt)
