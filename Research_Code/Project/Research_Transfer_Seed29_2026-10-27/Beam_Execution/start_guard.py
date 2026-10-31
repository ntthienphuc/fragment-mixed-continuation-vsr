"""Start the owned-pod limit before the large workload transfer begins."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

BASE = Path("/root/lip_transfer_seed29_20261027")
PHASE = BASE / "project/Research_Transfer_Seed29_2026-10-27/Beam_Execution"


def main():
    with zipfile.ZipFile(BASE / "Guard_Setup.zip") as archive:
        manifest = json.loads(archive.read("Guard_Manifest.json"))
        assert set(archive.namelist()) == set(manifest) | {"Guard_Manifest.json"}
        for name, expected in manifest.items():
            target = (BASE / name).resolve()
            assert target.is_relative_to(BASE) and name.startswith("project/")
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == expected
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
    environment = {**os.environ, "LIP_SEED29_STORE": str(BASE / "store")}
    proof = subprocess.check_output([sys.executable, "-B", str(PHASE / "server_guard.py"), "--probe"],
                                    env=environment, text=True)
    (BASE / "Guard_Probe.json").write_text(proof)
    logs = BASE / "logs"
    logs.mkdir(exist_ok=True)
    command = ["env", "LIP_SEED29_STORE=" + str(BASE / "store"),
               sys.executable, "-B", str(PHASE / "server_guard.py")]
    shell = " ".join(shlex.quote(item) for item in command)
    shell += " > " + shlex.quote(str(logs / "guard.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", "lip-seed29-guard", shell], check=True)
    print(proof, flush=True)


if __name__ == "__main__":
    main()
