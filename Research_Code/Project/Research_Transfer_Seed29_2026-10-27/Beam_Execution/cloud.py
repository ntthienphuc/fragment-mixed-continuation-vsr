"""SSH transport for the explicitly configured decoder-control pod."""

import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = "/root/lip_transfer_seed29_20261027"


def connection():
    value = json.loads((HERE / "Server_Connection.json").read_text(encoding="utf-8"))
    config = json.loads((HERE / "Server_Config.json").read_text(encoding="utf-8"))
    assert value["pod_id"] == config["pod_id"]
    options = ["-i", value["key_path"], "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=accept-new"]
    return value, options


def remote(arguments, timeout=55):
    value, options = connection()
    command = " ".join(shlex.quote(str(argument)) for argument in arguments)
    result = subprocess.run(["ssh", *options, "-p", str(value["port"]), value["destination"], command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:] + result.stdout[-2500:])
    return result.stdout


def upload(path, destination, timeout=120):
    value, options = connection()
    subprocess.run(["scp", *options, "-P", str(value["port"]), str(path),
                    value["destination"] + ":" + destination], check=True, timeout=timeout)


def download(source, destination, timeout=120):
    value, options = connection()
    subprocess.run(["scp", *options, "-P", str(value["port"]), value["destination"] + ":" + source,
                    str(destination)], check=True, timeout=timeout)
