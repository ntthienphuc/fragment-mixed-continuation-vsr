"""Back up remote milestones, verify every archive member and acknowledge completion."""

import hashlib
import json
from pathlib import Path
import time
import zipfile

from cloud import REMOTE, download, remote, upload
from common import HERE, ROOT, STORE, read, sha, utc


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def backup():
    value = json.loads(remote(["python", REMOTE + "/remote_monitor.py", "--snapshot"], timeout=120))
    folder = STORE / "Server_Backups"
    folder.mkdir(exist_ok=True)
    target = folder / Path(value["path"]).name
    download(value["path"], target, timeout=180)
    assert sha(target) == value["sha256"]
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("Snapshot_Manifest.json"))
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        prefix = "project/" + HERE.relative_to(ROOT).as_posix() + "/"
        assignment = read(HERE / "Dispatch/Server_01.json")
        completed = prefix + "Queues/Server_01/Complete.json"
        terminal = completed in manifest["files"]
        for task in assignment["tasks"]:
            base = prefix + "Runs/Server_01/" + task + "/"
            proof = base + "Verification.json"
            if proof not in manifest["files"]:
                terminal = False
                continue
            verification = json.loads(archive.read(proof))
            assert verification["status"] == "passed" and verification["task"] == task
            assert verification["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json")
            assert verification["predictions_sha256"] == manifest["files"][base + "Predictions.jsonl"]
            for name in manifest["files"]:
                if not name.startswith(base):
                    continue
                destination = HERE / name.removeprefix(prefix)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    assert sha(destination) == manifest["files"][name], name
                else:
                    destination.write_bytes(archive.read(name))
                    assert sha(destination) == manifest["files"][name], name
        if terminal:
            complete = json.loads(archive.read(completed))
            assert complete["status"] == "all_assigned_tasks_verified"
            assert complete["tasks"] == assignment["tasks"]
            assert complete["execution_lock_sha256"] == sha(HERE / "Execution_Lock.json")
            acknowledgement = {"status": "sha256_readback_passed", "utc": utc(),
                               "archive_sha256": value["sha256"],
                               "queue_complete_sha256": manifest["files"][completed]}
            atomic(HERE / "Local_Backup_Ack.json", acknowledgement)
            upload(HERE / "Local_Backup_Ack.json", REMOTE + "/" + prefix + "Local_Backup_Ack.json")
    receipt = {**value, "local": str(target), "utc": utc(), "status": "sha256_readback_passed",
               "all_remote_tasks_verified": terminal}
    atomic(HERE / "Latest_Server_Backup.json", receipt)
    return receipt


def main():
    config = read(HERE / "Server_Config.json")
    last_backup = 0
    last_key = None
    failures = 0
    while time.time() < config["stop_deadline_unix"] + 180:
        try:
            value = json.loads(remote(["python", REMOTE + "/remote_monitor.py"]))
            atomic(HERE / "Server_Live_Status.json", value)
            failures = 0
            key = json.dumps([(task, item.get("Verification.json"))
                              for task, item in value["tasks"].items()], sort_keys=True)
            if key != last_key or time.time() - last_backup > 300:
                receipt = backup()
                print(json.dumps({"event": "backup", **receipt}), flush=True)
                last_key = key
                last_backup = time.time()
                if receipt["all_remote_tasks_verified"]:
                    return
            progress = {task: item.get("completed_clips", 0) for task, item in value["tasks"].items()}
            print(json.dumps({"utc": utc(), "gpu": value["gpu"], "progress": progress}), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({"utc": utc(), "error": str(error), "failures": failures}), flush=True)
            if failures >= 5:
                return
        time.sleep(45)


if __name__ == "__main__":
    main()
