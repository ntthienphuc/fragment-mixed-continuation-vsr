"""Coherent snapshots and live observations of this pod's assigned tasks only."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

BASE = Path("/root/lip_transfer_seed29_20261027")
HERE = BASE / "project/Research_Transfer_Seed29_2026-10-27/Beam_Execution"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def state():
    value = {"unix": time.time(), "tasks": {}, "preflight": {}}
    for name in ("Queues/Server_01/Complete.json", "Queues/Server_01/Failure.json", "Server_Failure.json"):
        value[name] = read(HERE / name)
    for model in read(HERE / "Plan.json")["models"]:
        value["preflight"][model] = read(HERE / "Server_Preflight" / model / "Verification.json")
    for folder in sorted((HERE / "Runs/Server_01").glob("*")):
        if not folder.is_dir():
            continue
        item = {name: read(folder / name) for name in ("Complete.json", "Verification.json", "Failure.json")}
        predictions = folder / "Predictions.jsonl"
        if predictions.exists():
            item["completed_clips"] = sum(1 for line in predictions.open() if line.endswith("\n"))
        value["tasks"][folder.name] = item
    value["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                   "--format=csv,noheader"], capture_output=True, text=True).stdout
    value["tmux"] = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True).stdout
    for name in ("guard.log", "queue.log", "queue_environment_retry.log"):
        path = BASE / "logs" / name
        value[name] = path.read_text(errors="replace")[-2000:] if path.exists() else None
    return value


def snapshot():
    folder = BASE / "snapshots"
    folder.mkdir(exist_ok=True)
    target = folder / (str(time.time_ns()) + ".zip")
    files = {}
    skipped = []
    candidates = list(HERE.rglob("*")) + list((BASE / "logs").rglob("*"))
    candidates += [BASE / name for name in ("Launched.json", "bootstrap.log", "remote_bootstrap.py",
                                            "retry_environment.py", "remote_monitor.py", "Package.json", "Overlay.json")]
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(set(candidates)):
            if not path.is_file() or path.suffix in (".tmp", ".pyc"):
                continue
            name = path.relative_to(BASE).as_posix()
            expected = sha(path)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected or sha(path) != expected:
                skipped.append(name)
                continue
            archive.writestr(name, data)
            files[name] = expected
        archive.writestr("Snapshot_Manifest.json", json.dumps({"files": files, "skipped_active": skipped}))
    return {"path": str(target), "sha256": sha(target), "bytes": target.stat().st_size,
            "files": len(files), "skipped_active": skipped}


if __name__ == "__main__":
    print(json.dumps(snapshot() if "--snapshot" in sys.argv else state()), flush=True)
