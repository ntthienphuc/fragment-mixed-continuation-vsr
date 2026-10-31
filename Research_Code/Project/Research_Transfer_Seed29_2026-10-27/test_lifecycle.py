"""Check termination of only a deliberately created local child process tree."""

import subprocess
import sys
import time

import psutil

from common import HERE, save, sha, utc
from run_local import stop_owned_tree


def main():
    output = HERE / "Lifecycle_Test_01"
    output.mkdir(exist_ok=False)
    script = (
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "print(child.pid,flush=True)\n"
        "time.sleep(60)\n"
    )
    worker = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    child = int(worker.stdout.readline().strip())
    assert child in [process.pid for process in psutil.Process(worker.pid).children(recursive=True)]
    began = time.monotonic()
    stop_owned_tree(worker)
    assert worker.poll() is not None and not psutil.pid_exists(child)
    save(output / "Verification.json", {
        "status": "owned_parent_and_child_termination_verified", "utc": utc(),
        "parent_pid": worker.pid, "child_pid": child, "seconds": time.monotonic() - began,
        "worker_exit": worker.returncode, "queue_code_sha256": sha(HERE / "run_local.py")})
    print({"status": "owned_tree_stopped", "parent": worker.pid, "child": child}, flush=True)


if __name__ == "__main__":
    main()
