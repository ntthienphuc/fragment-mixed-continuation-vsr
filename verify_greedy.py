"""Reproduce frozen greedy results, allowing only CRLF/LF export differences."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = ROOT / "Greedy_Diagnostic"
    manifest = json.loads((package / "Manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (package / name).resolve()
        assert path.is_relative_to(package), name
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    comparisons = []
    stages = {}
    for stage in ["source", "assessment"]:
        command = [sys.executable, "-B", str(package / "Code/analyze.py"),
                   "--inputs", str(package / "Data/Inputs"),
                   "--collected", str(package / "Data/Collected"),
                   "--stage", stage, "--output", str(args.output / stage)]
        result = subprocess.run(command, cwd=package, env=environment, capture_output=True, timeout=900)
        (args.output / (stage + ".stdout.log")).write_bytes(result.stdout)
        (args.output / (stage + ".stderr.log")).write_bytes(result.stderr)
        assert result.returncode == 0, result.stderr.decode(errors="replace")[-3000:]
        for name in ["Aggregate.csv", "Contrasts.json", "Per_Clip_Counts.csv", "Greedy_Minus_Common.csv"]:
            actual = (args.output / stage / name).read_bytes()
            expected = (package / "Results" / stage / name).read_bytes()
            exact = actual == expected
            # Keep every digit, space, key, row and character. Only platform line endings may differ.
            same = actual.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")
            assert same, "Scientific output mismatch: " + stage + "/" + name
            comparisons.append({"file": stage + "/" + name, "byte_identical": exact,
                                "identical_after_crlf_to_lf": same})
        stages[stage] = json.loads((args.output / stage / "Verification.json").read_text(encoding="utf-8"))
    proof = {"utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
             "comparison": "Exact bytes after CRLF-to-LF normalization; no numeric tolerance",
             "frozen_input_hashes_verified": len(manifest["files"]), "comparisons": comparisons,
             "clip_model_records": 37422, "aggregate_rows": 6, "paired_intervals": 4,
             "full_neural_inference_repeated": False, "stages": stages}
    (args.output / "Verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({key: proof[key] for key in ["status", "comparison", "comparisons"]}))


if __name__ == "__main__":
    main()
