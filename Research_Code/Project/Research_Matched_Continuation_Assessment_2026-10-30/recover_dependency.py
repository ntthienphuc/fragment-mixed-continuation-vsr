"""Restore the one absent locked helper without replacing any existing file."""

import importlib.util
import json
from pathlib import Path

from common import HERE, ROOT, read, save, sha, utc


def load_transport(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cloud = load_transport("cloud_transport", "Research_Vast_Source_2026-10-29/transport.py")
    private = load_transport("private_transport", "Research_Source_Retention_2026-10-29/Server_01/transport.py")
    recovery = HERE / "Dependency_Recovery_01"
    recovery.mkdir(exist_ok=False)
    remote_phase = private.REMOTE + "/project/" + HERE.name
    for name in ("Failure.json", "Start.json", "Admission.json", "model_registry.json", "run_task.py.log"):
        private.download(remote_phase + "/Worker_Integration_01/" + name, recovery / name, 60)
    private.download(remote_phase + "/integration.py", recovery / "integration_attempt01.py", 60)
    name = "Research_Server_Confirmation_2026-10-19/data_io.py"
    digest = read(HERE / "Code_Lock.json")["external_code_sha256"][name]
    assert sha(ROOT / name) == digest
    target = cloud.BASE + "/" + name
    staged = cloud.BASE + "/Destination_Stage_01/Payload/" + name
    script = "\n".join([
        "import hashlib,json,shutil", "from pathlib import Path",
        "target=Path(" + repr(target) + ")", "staged=Path(" + repr(staged) + ")",
        "assert not target.exists()", "assert hashlib.sha256(staged.read_bytes()).hexdigest()==" + repr(digest),
        "with target.open('xb') as output:", "    output.write(staged.read_bytes())",
        "assert hashlib.sha256(target.read_bytes()).hexdigest()==" + repr(digest),
        "print(json.dumps({'target':str(target),'sha256':" + repr(digest) + ",'copied_from_verified_payload':True}))",
    ])
    cloud_result = json.loads(cloud.remote(["python3", "-c", script], 60))
    private_target = private.REMOTE + "/project/" + name
    part = private_target + ".restore.part"
    preflight = private.powershell(
        "if((Test-Path -LiteralPath '" + private_target + "') -or (Test-Path -LiteralPath '" + part
        + "')) { throw 'Restore target already exists' }; 'absent'", 60)
    assert preflight == "absent"
    private.upload(ROOT / name, part, 60)
    verified = private.powershell(
        "$path='" + part + "'\n"
        "if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne '" + digest
        + "') { throw 'Restore hash mismatch' }\n"
        "if(Test-Path -LiteralPath '" + private_target + "') { throw 'Target appeared' }\n"
        "[IO.File]::Move($path,'" + private_target + "')\n"
        "(Get-FileHash -LiteralPath '" + private_target + "' -Algorithm SHA256).Hash.ToLower()", 60)
    assert verified == digest
    save(recovery / "Complete.json", {
        "utc": utc(), "status": "absent_locked_dependency_restored_on_both_hosts",
        "dependency": name, "sha256": digest, "vast": cloud_result,
        "private": private_target, "failed_attempt_preserved": True,
        "scientific_core_or_training_changed": False,
        "failure_files_sha256": {path.name: sha(path) for path in recovery.iterdir() if path.is_file()},
    })
    print(json.dumps(read(recovery / "Complete.json")), flush=True)


if __name__ == "__main__":
    main()
