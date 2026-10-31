"""Owned-pod cost guard with persistent result mirroring and backup-before-stop gating."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

from common import HERE, read, sha, utc

SAFE = Path("/workspace/lip_transfer_seed29_20261027")


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def provider(config):
    values = dict(item.split(b"=", 1) for item in Path("/proc/1/environ").read_bytes().split(b"\0") if b"=" in item)
    assert values[b"RUNPOD_POD_ID"].decode() == config["pod_id"]
    endpoint = "https://api.runpod.io/v2/pods/" + config["pod_id"]
    headers = {"Authorization": "Bearer " + values[b"RUNPOD_API_KEY"].decode(), "Content-Type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(endpoint, headers=headers), timeout=25) as response:
        pod = json.load(response)
    assert pod["id"] == config["pod_id"] and pod["name"] == config["pod_name"]
    assert pod["status"] == "RUNNING" and "stop" in pod["actions"]
    assert 0 < pod["cost"] < 1
    return endpoint, headers


def mirror():
    target = SAFE / "project" / HERE.name
    target.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.suffix == ".tmp" or "__pycache__" in path.parts:
            continue
        destination = target / path.relative_to(HERE)
        expected = sha(path)
        if destination.exists() and sha(destination) == expected:
            records.append(dict(name=path.relative_to(HERE).as_posix(), sha256=expected))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(path, temporary)
        if sha(temporary) != expected or sha(path) != expected:
            temporary.unlink()
            skipped.append(path.relative_to(HERE).as_posix())
            continue
        temporary.replace(destination)
        records.append(dict(name=path.relative_to(HERE).as_posix(), sha256=expected))
    atomic(SAFE / "Mirror.json", dict(utc=utc(), files=records, active_files_skipped=skipped))
    return len(records)


def stop_workers():
    for path in (HERE / "Queues/Server_01").glob("*.Start.json"):
        value = read(path)
        pid = value["pid"]
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and str(HERE).encode() in command.read_bytes():
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    config = read(HERE / "Server_Config.json")
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", "/workspace"], text=True))
    assert mount["filesystems"][0]["target"] == "/workspace", "Require the configured persistent volume"
    provider(config)
    if args.probe:
        print(json.dumps(dict(status="passed", pod_id=config["pod_id"], persistent_mount="/workspace",
                              injected_credential_available=True, credential_exported=False)))
        return
    SAFE.mkdir(exist_ok=True)
    atomic(SAFE / "Guard_Start.json", dict(utc=utc(), pid=os.getpid(), config=config))
    while True:
        mirrored = mirror()
        completed = HERE / "Queues/Server_01/Complete.json"
        failed = HERE / "Server_Failure.json"
        ack = HERE / "Local_Backup_Ack.json"
        reason = None
        if completed.exists() and ack.exists():
            proof = read(ack)
            assert proof["queue_complete_sha256"] == sha(completed)
            assert proof["status"] == "sha256_readback_passed"
            reason = "COMPLETE_AND_LOCAL_BACKUP_VERIFIED"
        elif failed.exists() and time.time() - failed.stat().st_mtime > 180:
            reason = "FAILED_ATTEMPT_PRESERVED_ON_VOLUME"
        elif time.time() >= config["stop_deadline_unix"]:
            reason = "REGISTERED_THREE_HOUR_COST_LIMIT"
        if reason:
            stop_workers()
            time.sleep(5)
            mirrored = mirror()
            endpoint, headers = provider(config)
            atomic(SAFE / "Stop_Request.json", dict(utc=utc(), reason=reason, pod_id=config["pod_id"],
                                                      persistent_files=mirrored, container_outputs_preserved=True))
            request = urllib.request.Request(endpoint + "/action", headers=headers, method="POST",
                                             data=json.dumps({"action": "stop"}).encode())
            with urllib.request.urlopen(request, timeout=25) as response:
                response.read()
                atomic(SAFE / "Stop_Response.json", dict(utc=utc(), status=response.status, accepted=True))
            return
        atomic(SAFE / "Guard_Heartbeat.json", dict(utc=utc(), files=mirrored,
                                                    seconds_to_stop=config["stop_deadline_unix"] - time.time()))
        time.sleep(30)


if __name__ == "__main__":
    main()
