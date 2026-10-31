"""Validate existing transport parts and assemble only the exact locked archive."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(base, mode):
    base = Path(base).resolve()
    plan = json.loads((base / "Transport_Parts.json").read_text(encoding="utf-8"))
    folder = base / "parts"
    verified = []
    for part in plan["parts"]:
        path = (folder / part["name"]).resolve()
        assert path.parent == folder.resolve(), "Unsafe part path"
        if not path.exists():
            continue
        assert path.stat().st_size == part["bytes"], f"Wrong part size: {part['name']}"
        assert digest(path) == part["sha256"], f"Wrong part hash: {part['name']}"
        verified.append(part["name"])
    if mode == "status":
        return {"verified": verified, "total_parts": len(plan["parts"])}
    assert mode == "assemble"
    assert len(verified) == len(plan["parts"]), "Missing verified parts"
    target = base / "Server_Workload.zip"
    temporary = base / "Server_Workload.assembling"
    assert not target.exists() and not temporary.exists(), "Preserve existing assembly"
    with temporary.open("xb") as destination:
        for part in plan["parts"]:
            with (folder / part["name"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=2**20)
    assert temporary.stat().st_size == plan["bytes"], "Wrong assembled size"
    assert digest(temporary) == plan["sha256"], "Wrong assembled hash"
    temporary.rename(target)
    return {"status": "assembled_sha256_passed", "bytes": target.stat().st_size,
            "sha256": plan["sha256"], "verified_parts": len(verified)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("status", "assemble"))
    parser.add_argument("--base", default="/root/lip_transfer_seed29_20261027")
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.base, arguments.mode)), flush=True)
