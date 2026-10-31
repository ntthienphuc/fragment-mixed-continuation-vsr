"""Two resource probes, then one frozen 24-shard matrix with four bounded workers."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

from common import HERE, ROOT, read, save, sha, utc


def atomic(name, value):
    target = HERE / name
    temporary = target.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(target)


def run(command, log, deadline, gpu, maximum):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        save(log.with_suffix(".process.json"), {"pid": process.pid, "gpu": gpu, "command": command})
        try:
            code = process.wait(timeout=min(maximum, deadline - time.time()))
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            raise
    if code:
        raise RuntimeError("Worker failed: " + str(log))


def job(task, mode, gpu, deadline):
    began = time.time()
    binding = HERE / "Binding"
    stage = task.rsplit("_", 1)[0]
    ticket = HERE / "Admissions" / (task + ".json")
    admission = {"utc": utc(), "status": "bounded_final_model_resource_probe_admitted" if mode == "probe"
                 else "matched_population_assessment_admitted", "mode": mode, "model": "MATCHED_SO17",
                 "device": "cuda", "cuda_visible_device": str(gpu), "readout": "common", "tasks": [task],
                 "assessment_plan_sha256": sha(HERE / "Plan.json"), "code_lock_sha256": sha(HERE / "Code_Lock.json"),
                 "registry_sha256": sha(binding / "model_registry.json"), "deadline_unix": deadline,
                 "maximum_task_seconds": 1500 if mode == "probe" else 3300,
                 "resource_probe": str(HERE / "Runs" / (stage + "_probe"))}
    save(ticket, admission)
    output = HERE / "Runs" / task
    args = [task, "--model", "MATCHED_SO17", "--binding", str(binding), "--admission", str(ticket),
            "--output", str(output), "--device", "cuda", "--mode", mode, "--readout", "common",
            "--destination-root", str(ROOT / "Destination_Stage_01/Payload")]
    for script, maximum in (("run_task.py", 3400), ("verify_task.py", 600)):
        run([sys.executable, "-B", "-u", str(HERE / script), *args],
            HERE / "Logs" / (task + "_" + script + ".log"), deadline, gpu, maximum)
    proof = read(output / "Verification.json")
    assert proof["status"] == "all_records_recounted_and_fresh_replay_passed"
    assert sha(output / "Predictions.jsonl") == proof["predictions_sha256"]
    archive = HERE / "Packages" / (task + ".zip")
    manifest = {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        files = [(path, "Run/" + path.name) for path in output.iterdir() if path.is_file()]
        files.append((ticket, "Admission.json"))
        for path, name in files:
            content = path.read_bytes()
            bundle.writestr(name, content)
            manifest[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        bundle.writestr("Manifest.json", json.dumps(manifest))
    result = {"task": task, "mode": mode, "gpu": gpu, "clips": proof["clips"],
              "seconds": time.time() - began, "peak_gpu_mib": proof["peak_gpu_mib"],
              "path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size}
    save(archive.with_suffix(".json"), result)
    return result


def resources():
    available = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    free_ram = int(available.split()[1]) / 1024**2
    value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    devices = [tuple(map(int, line.split(","))) for line in value.splitlines()]
    return free_ram, devices


def main():
    budget = read(HERE.parent / "Budget.json")
    deadline = budget["stop_unix"] - 15 * 60
    assert deadline - time.time() > 105 * 60
    guard = read(HERE.parent / "Guard_Status.json")
    assert guard["reason"] is None and time.time() - guard["unix"] < 90
    ram, devices = resources()
    assert ram > 24 and len(devices) == 2 and all(used < 1000 for _, used in devices)
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    for name in ("Admissions", "Runs", "Logs", "Packages"):
        (HERE / name).mkdir(exist_ok=False)
    save(HERE / "Start.json", {"utc": utc(), "pid": os.getpid(), "deadline_unix": deadline,
                               "workers": 4, "source_clips": 11307, "destination_clips": 1167})
    with ThreadPoolExecutor(max_workers=2) as executor:
        probes = [executor.submit(job, "source_probe", "probe", 0, deadline),
                  executor.submit(job, "assessment_probe", "probe", 1, deadline)]
        probe_results = [future.result() for future in probes]
    peak = max(item["peak_gpu_mib"] for item in probe_results)
    assert all(2 * peak + 4096 < total for total, _ in devices)
    save(HERE / "Resource_Verification.json", {"probes": probe_results, "two_workers_per_gpu_admitted": True})
    pending = [f"source_{i:02}" for i in range(16)] + [f"assessment_{i:02}" for i in range(8)]
    active = {}
    finished = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        while pending or active:
            guard = read(HERE.parent / "Guard_Status.json")
            ram, _ = resources()
            guard_ok = guard["reason"] is None and time.time() - guard["unix"] < 120
            for gpu in (0, 1):
                count = sum(item["gpu"] == gpu for item in active.values())
                while pending and count < 2 and not failures and guard_ok and ram > 24:
                    if deadline - time.time() < 35 * 60:
                        break
                    task = pending.pop(0)
                    future = executor.submit(job, task, "population", gpu, deadline)
                    active[future] = {"task": task, "gpu": gpu}
                    count += 1
            atomic("Progress.json", {"utc": utc(), "active": list(active.values()), "pending": pending,
                                     "finished": finished, "failures": failures, "deadline_unix": deadline})
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                try:
                    finished.append(future.result())
                except Exception:
                    failures.append({**item, "error": traceback.format_exc()})
    atomic("Progress.json", {"utc": utc(), "active": [], "pending": pending,
                             "finished": finished, "failures": failures, "deadline_unix": deadline})
    assert not pending and not failures and len(finished) == 24, "Retain incomplete matrix; no auto retry"
    assert sum(item["clips"] for item in finished if item["task"].startswith("source_")) == 11307
    assert sum(item["clips"] for item in finished if item["task"].startswith("assessment_")) == 1167
    save(HERE / "Terminal.json", {"utc": utc(), "status": "all_24_shards_verified", "finished": finished,
                                 "source_clips": 11307, "destination_clips": 1167})


if __name__ == "__main__":
    main()
