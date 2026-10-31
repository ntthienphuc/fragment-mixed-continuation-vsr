"""Verify the uploaded workload, start a cost guard, then launch the sealed queue."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile

BASE = Path("/root/lip_transfer_seed29_20261027")
PHASE = "Research_Transfer_Seed29_2026-10-27/Beam_Execution"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(package, receipt, replace=False):
    assert sha(package) == receipt["sha256"]
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == set(receipt["files"])
        for name, expected in receipt["files"].items():
            target = (BASE / name).resolve()
            assert target.is_relative_to(BASE) and name.startswith(("project/", "store/"))
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
            if target.exists() and not replace:
                assert sha(target) == expected, name
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            assert sha(target) == expected, name


def tmux(name, arguments, output, environment):
    keys = ("LIP_SEED29_STORE", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")
    explicit = ["env", *(key + "=" + environment[key] for key in keys), *arguments]
    command = " ".join(shlex.quote(str(argument)) for argument in explicit)
    command += " > " + shlex.quote(str(output)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True, env=environment)


def main():
    receipt = json.loads((BASE / "Package.json").read_text())
    overlay = json.loads((BASE / "Overlay.json").read_text())
    unpack(BASE / "Server_Workload.zip", receipt)
    unpack(BASE / "Server_Overlay.zip", overlay, replace=True)
    phase = BASE / "project" / PHASE
    config = json.loads((phase / "Server_Config.json").read_text())
    assert time.time() < config["science_deadline_unix"]
    assert not (phase / "Server_Preflight").exists()
    assert not (phase / "Queues/Server_01").exists()
    logs = BASE / "logs"
    logs.mkdir(exist_ok=True)
    environment = {**os.environ, "LIP_SEED29_STORE": str(BASE / "store"),
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-B", str(phase / "server_guard.py"), "--probe"],
                   check=True, env=environment)
    guard = subprocess.run(["tmux", "has-session", "-t", "lip-seed29-guard"], capture_output=True)
    if guard.returncode == 0:
        started = json.loads(Path("/workspace/lip_transfer_seed29_20261027/Guard_Start.json").read_text())
        assert started["config"]["pod_id"] == config["pod_id"]
        assert started["config"]["stop_deadline_unix"] == config["stop_deadline_unix"]
    else:
        tmux("lip-seed29-guard", [sys.executable, "-B", phase / "server_guard.py"],
             logs / "guard.log", environment)
    try:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(BASE / "venv")], check=True)
        python = BASE / "venv/bin/python"
        subprocess.run([str(python), "-m", "pip", "install", "numpy==1.26.4"], check=True, env=environment)
        identity = subprocess.check_output([str(python), "-c",
            "import torch,numpy; print(torch.__version__,numpy.__version__,torch.cuda.get_device_name())"],
            text=True, env=environment)
        tmux("lip-seed29-beams", [python, "-B", phase / "server_launch.py", "--deadline", str(config["science_deadline_unix"])],
             logs / "queue.log", environment)
        (BASE / "Launched.json").write_text(json.dumps({"unix": time.time(), "runtime": identity,
            "assignment": config["assignment"], "stop_deadline_unix": config["stop_deadline_unix"],
            "package_sha256": receipt["sha256"], "overlay_sha256": overlay["sha256"]}, indent=2))
        print(identity, flush=True)
    except BaseException as error:
        (phase / "Server_Failure.json").write_text(json.dumps({"unix": time.time(), "error": repr(error)}))
        raise


if __name__ == "__main__":
    main()
