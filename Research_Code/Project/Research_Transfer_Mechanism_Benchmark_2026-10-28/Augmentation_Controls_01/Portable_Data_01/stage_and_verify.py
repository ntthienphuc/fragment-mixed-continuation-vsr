"""Deploy the source-data provider into a separate owned scope and run its CPU verification."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Research_Source_Retention_2026-10-29/Server_01"))
from transport import REMOTE, powershell, upload, download


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main(attempt):
    if not attempt.startswith("Attempt_") or not attempt[8:].isdigit():
        raise ValueError("Use an explicit numbered verification attempt")
    local_attempt = HERE / "Attempts" / attempt
    local_attempt.mkdir(parents=True, exist_ok=False)
    scope = REMOTE + "/Portable_Data_01/" + attempt
    runtime = HERE.parent / "Training_Inputs_01/Runtime_Inputs.json"
    spec = ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03/Inputs.json"
    assert sha(spec) == "ab7df96dcd97d658b3147b708cc2ae6c749e461720426de470eb0f78bb271fa1"
    powershell("if (Test-Path -LiteralPath '" + scope + "') { throw 'Preserve existing provider attempt' };\n"
               "New-Item -ItemType Directory -Path '" + scope + "' | Out-Null")
    files = {name: HERE / name for name in ("portable_data.py", "verify_runtime.py", "Protocol.md")}
    files["Native_Inputs.json"] = spec
    verified = {}
    for name, path in files.items():
        remote = scope + "/" + name
        upload(path, remote)
        actual = powershell("(Get-FileHash -LiteralPath '" + remote + "' -Algorithm SHA256).Hash.ToLower()")
        assert actual == sha(path)
        verified[name] = actual
    remote_runtime = REMOTE + "/Matched_Training_Inputs_01/Runtime_Inputs.json"
    actual = powershell("(Get-FileHash -LiteralPath '" + remote_runtime +
                        "' -Algorithm SHA256).Hash.ToLower()")
    assert actual == sha(runtime)
    save(local_attempt / "Deployment.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                     "remote_scope": scope, "files": verified,
                                     "runtime_sha256": actual, "gpu_work": False})
    arguments = [REMOTE + "/venv/Scripts/python.exe", "-X", "utf8", "-B", "-u",
                 scope + "/verify_runtime.py", "--packet", REMOTE + "/GPU_Admission_01/Packet",
                 "--runtime", remote_runtime, "--runtime-sha256", actual,
                 "--spec", scope + "/Native_Inputs.json", "--spec-sha256", sha(spec),
                 "--output", scope + "/Preflight_01"]
    command = "& " + " ".join("'" + value.replace("'", "''") + "'" for value in arguments)
    command += "\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    print({"status": "staged_provider_cpu_verification_running", "scope": scope}, flush=True)
    try:
        output = powershell(command, timeout=1800)
    except BaseException:
        (local_attempt / "Verification_Error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    (local_attempt / "Runtime.log").write_text(output, encoding="utf-8")
    received = local_attempt / "Received_01"
    received.mkdir(exist_ok=False)
    for name in ("Complete.json", "Validation_Pixels.json"):
        remote = scope + "/Preflight_01/" + name
        download(remote, received / name)
        actual = powershell("(Get-FileHash -LiteralPath '" + remote + "' -Algorithm SHA256).Hash.ToLower()")
        assert actual == sha(received / name)
    proof = json.loads((received / "Complete.json").read_text())
    assert proof["status"] == "portable_source_inputs_native_order_pixels_and_coverage_verified"
    assert proof["provider_sha256"] == verified["portable_data.py"]
    assert proof["verifier_sha256"] == verified["verify_runtime.py"]
    assert not proof["cuda_initialized"] and not proof["training_admitted"]
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    main(parser.parse_args().attempt)
