"""Prepare a content-checked inference payload without allocating a server."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from common import ENGINE, HERE, PARENT, ROOT, SEED29, lines, read, save, sha, utc

STORE = Path("C:/Users/thien/Lip_Research_Backups/Transfer_Full_Benchmark_20261027")


def normalized(path):
    value = Path(str(path).replace("\\", "/"))
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def inventory():
    files = {}

    def add(path, expected=None):
        path = normalized(path)
        assert path.is_relative_to(ROOT.resolve()), path
        assert path.is_file(), path
        name = path.relative_to(ROOT).as_posix()
        digest = sha(path)
        if expected:
            assert digest == expected, name
        if name in files:
            assert files[name]["sha256"] == digest, name
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}

    def locked_mapping(path, field):
        add(path)
        for name, expected in read(path)[field].items():
            add(name, expected)

    locked_mapping(HERE / "Plan.json", "frozen_sha256")
    locked_mapping(HERE / "Execution_Lock.json", "code_sha256")
    for phase in (PARENT, SEED29):
        locked_mapping(phase / "Analysis_01/Summary.json", "input_sha256")
        for path in phase.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".tmp":
                assert path.stat().st_size < 64 * 2**20, (path, "Unexpected bulk file in evidence folder")
                add(path)
    for path in HERE.rglob("*"):
        relative = path.relative_to(HERE)
        if not path.is_file() or "__pycache__" in relative.parts or path.suffix == ".tmp":
            continue
        if relative.parts[0] in {"Local_Preflight_01", "Payload_01"}:
            continue
        if path.name.startswith(("Admission", "Local_Preflight", "Payload_")):
            continue
        assert path.stat().st_size < 64 * 2**20, (path, "Unexpected bulk file in preparation folder")
        add(path)
    for path in ENGINE.glob("*.py"):
        add(path)
    add(ENGINE / "VC_Model_Registry.json")
    add(ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py")

    registries = (read(HERE / "model_registry.json"), read(HERE / "Native_Reference_Registry.json"))
    for registry in registries:
        for model, entry in registry["models"].items():
            for field in ("checkpoint", "base_checkpoint", "tokenizer", "units", "config", "native_config"):
                if field in entry:
                    add(entry[field], entry.get(field + "_sha256"))
            for path in normalized(entry["source"]).rglob("*.py"):
                add(path)
        for field in ("factory_source", "safe_search_source", "horizon_source", "alignment_source"):
            add(registry[field])
        for name in registry.get("required_provenance_assets", []):
            add(name)
    for path in (ROOT / "Research_Phase_05_2026-10-13/phase5").rglob("*.py"):
        add(path)
    for stage in ("Development", "Assessment"):
        for row in lines(HERE / (stage + ".jsonl")):
            assert row["input"]["kind"] == "video"
            add(row["input"]["path"], row["input"]["sha256"])
    return dict(sorted(files.items()))


def main(pack):
    folder = HERE / "Payload_01"
    folder.mkdir(exist_ok=False)
    files = inventory()
    value = {"utc": utc(), "status": "payload_inventory_checked_no_server_allocation",
             "files": files, "bytes": sum(item["bytes"] for item in files.values()),
             "local_only_merge": True,
             "excluded_mutable_admission_and_runtime_proofs": True,
             "runtime_admission_must_be_added_for_the_actual_server": True}
    save(folder / "Inventory.json", value)
    print({"files": len(files), "bytes": value["bytes"]}, flush=True)
    if not pack:
        return
    STORE.mkdir(parents=True, exist_ok=True)
    target = STORE / "Full_Benchmark_Payload_01.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, item in files.items():
            path = ROOT / name
            archive.write(path, "project/" + name)
            assert sha(path) == item["sha256"], (name, "Changed while packaging")
        archive.writestr("Payload_Manifest.json", json.dumps(value, indent=2))
    with zipfile.ZipFile(target) as archive:
        for name, item in files.items():
            with archive.open("project/" + name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            assert actual == item["sha256"], name
    save(folder / "Verification.json", {
        "utc": utc(), "status": "all_payload_members_sha256_readback_passed",
        "path": str(target), "sha256": sha(target), "bytes": target.stat().st_size,
        "files": len(files), "inventory_sha256": sha(folder / "Inventory.json"),
        "server_allocated": False, "new_recognition_result": False,
    })
    print({"status": "payload_verified", "path": str(target)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="store_true")
    main(parser.parse_args().pack)
