"""Verify release bytes and recompute primary and exploratory paper results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(command, output, name):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=600)
    (output / (name + ".stdout.log")).write_bytes(result.stdout)
    (output / (name + ".stderr.log")).write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-5000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-python", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "verification/Payload_Manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or path.stat().st_size != item["bytes"]:
            raise ValueError(item["path"])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Hash mismatch: " + item["path"])
    command = [sys.executable, "-B", str(ROOT / "Research_Code/verify_release.py"),
               "--data", str(ROOT / "Data"), "--output", str(output / "Primary")]
    if args.test_python:
        command.extend(["--test-python", str(args.test_python.resolve())])
    run(command, output, "primary")
    run([sys.executable, "-B", str(ROOT / "Greedy_Diagnostic/Code/verify_supplement.py"),
         "--package", str(ROOT / "Greedy_Diagnostic"), "--output", str(output / "Greedy")], output, "greedy")
    result = {"utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "payload_files": len(manifest["files"]), "primary_aggregates": 60,
              "primary_paired_intervals": 49, "diagnostic_clip_model_records": 37422,
              "diagnostic_aggregates": 6, "diagnostic_paired_intervals": 4,
              "full_neural_training_or_inference_rerun": False}
    (output / "Verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
