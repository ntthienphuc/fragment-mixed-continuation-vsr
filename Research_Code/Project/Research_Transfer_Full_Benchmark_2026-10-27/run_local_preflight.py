"""Run two graph checks sequentially within a finite local-only allocation."""

import json
from pathlib import Path
import subprocess
import sys
import time

from admission import checked
from common import HERE, read, save, utc


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    _, admission, _ = checked("preflight", "VLR_B0")
    assert admission["execution_location"] == "local"
    folder = HERE / "Local_Preflight_01"
    folder.mkdir(exist_ok=False)
    save(folder / "Start.json", {"utc": utc(), "deadline_unix": admission["deadline_unix"]})
    for model, family in (("VLR_B0", "standard"), ("VC_NATIVE", "vicocktail")):
        remaining = admission["deadline_unix"] - time.time()
        assert remaining > 30, "Insufficient remaining local allocation"
        output = folder / (family + ".json")
        atomic(folder / "Status.json", {"utc": utc(), "active_model": model, "status": "running"})
        arguments = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "runtime_preflight.py"),
                     model, str(output), "--device", "cuda", "--threads", "2"]
        with (folder / (family + ".stdout.log")).open("x", encoding="utf-8") as stdout:
            with (folder / (family + ".stderr.log")).open("x", encoding="utf-8") as stderr:
                try:
                    process = subprocess.run(arguments, cwd=HERE, stdout=stdout, stderr=stderr,
                                             timeout=max(1, remaining - 10))
                except subprocess.TimeoutExpired:
                    save(folder / "Failure.json", {"utc": utc(), "model": model, "reason": "deadline"})
                    raise
        save(folder / (family + ".Exit.json"), {"utc": utc(), "exit_code": process.returncode})
        if process.returncode != 0:
            save(folder / "Failure.json", {"utc": utc(), "model": model,
                                           "reason": "runtime_exit", "exit_code": process.returncode})
            raise RuntimeError("Preflight failed; preserve the logs and investigate before retrying")
        proof = read(output)
        assert proof["status"] == "passed" and proof["family"] == family
        assert proof["reference_used"] is False and proof["test_clips_used"] is False
    atomic(folder / "Status.json", {"utc": utc(), "status": "both_graphs_passed"})
    save(folder / "Complete.json", {"utc": utc(), "status": "both_graphs_passed",
                                    "assessment_started": False, "paid_allocation_created": False})


if __name__ == "__main__":
    main()
