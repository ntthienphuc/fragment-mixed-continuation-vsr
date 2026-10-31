"""Run four source encoders sequentially with memory gates and a two-hour ceiling."""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

from encoding_common import HERE, lock, read, save, sha, utc


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]
    _fields_ += [(name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
                 "avail_virtual", "avail_extended")]


def resources():
    assert os.name == "nt", "Only the local Windows GPU is admitted"
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                     text=True, timeout=10)
    values = output.strip().splitlines()
    assert len(values) == 1
    return {"available_ram_gib": status.avail_phys / 2**30, "free_gpu_mib": float(values[0])}


def stop_owned(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=20)
        process.wait(timeout=20)


def main(attempt):
    assert attempt.isidentifier()
    registered = lock()
    output = HERE / attempt
    output.mkdir(exist_ok=False)
    started = time.time()
    deadline = started + registered["maximum_seconds"]
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "started_unix": started,
                                  "deadline_unix": deadline, "lock_sha256": sha(HERE / "Lock.json")})
    completed = {}
    try:
        for model in registered["models"]:
            while True:
                assert time.time() < deadline, "Source-encoding allocation expired"
                state = resources()
                if (state["available_ram_gib"] >= registered["minimum_available_ram_gib"]
                        and state["free_gpu_mib"] >= registered["minimum_free_gpu_mib"]):
                    break
                print({"utc": utc(), "status": "waiting_for_memory", **state}, flush=True)
                time.sleep(min(30, deadline - time.time()))
            command = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "encode_model.py"), model,
                       str(output / model), "--deadline", str(deadline)]
            with (output / (model + ".stdout.log")).open("x", encoding="utf-8") as stdout:
                with (output / (model + ".stderr.log")).open("x", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=HERE, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW
                                               | subprocess.CREATE_NEW_PROCESS_GROUP)
                    save(output / (model + ".Start.json"), {"utc": utc(), "pid": process.pid, "resources": state})
                    try:
                        while process.poll() is None:
                            if time.time() >= deadline:
                                raise TimeoutError("Registered source-encoding deadline reached")
                            time.sleep(min(2, deadline - time.time()))
                        assert process.returncode == 0, model + " failed; read its preserved stderr"
                    finally:
                        stop_owned(process)
            proof_path = output / model / "Complete.json"
            proof = read(proof_path)
            assert proof["status"] == "all_source_encoder_states_verified" and proof["model"] == model
            assert proof["clips"] == 1082 and proof["lock_sha256"] == sha(HERE / "Lock.json")
            assert len(proof["fresh_pixel_replays"]) == 3 and proof["test_clips_used"] is False
            assert sha(Path(proof["archive"])) == proof["archive_sha256"]
            assert sha(output / model / "Index.jsonl") == proof["index_sha256"]
            completed[model] = sha(proof_path)
            print({"utc": utc(), "model": model, "status": "encoding_verified"}, flush=True)
        save(output / "Complete.json", {"utc": utc(), "status": "all_four_source_encoder_caches_verified",
                                        "models": completed, "source_clips": 1082, "model_clip_encodings": 4328,
                                        "test_clips_used": False, "accuracy_claimed": False,
                                        "seconds": time.time() - started,
                                        "lock_sha256": sha(HERE / "Lock.json")})
    except BaseException as error:
        save(output / "Failure.json", {"utc": utc(), "error": repr(error), "completed_models": completed})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt")
    main(parser.parse_args().attempt)
