"""Package terminal integration evidence and identify only disposable fixture exports."""

import hashlib
import json
from pathlib import Path
import zipfile

from common import HERE, ROOT, read, save, sha, utc


def main():
    attempt = HERE / "Worker_Integration_02"
    complete = read(attempt / "Complete.json")
    assert complete["status"] == "real_cpu_worker_and_independent_replay_passed"
    assert not complete["cuda_initialized"] and not complete["final_assessment_admitted"]
    assert not (attempt / "Failure.json").exists()
    verification = read(attempt / "Run/Verification.json")
    assert verification == complete["verification"]
    assert verification["mode"] == "fixture" and verification["clips"] == 1
    assert not verification["complete_population_receipt"]
    exports = []
    paths = []
    for name in ("Worker_Integration_01", "Worker_Integration_02"):
        folder = HERE / name
        registry = read(folder / "model_registry.json")
        entry = registry["models"]["WW_RECOVERY_RUNTIME_CHECK"]
        model = folder / "model.pt"
        assert (ROOT / entry["checkpoint"]).resolve() == model.resolve()
        assert sha(model) == entry["checkpoint_sha256"]
        exports.append({"path": str(model), "bytes": model.stat().st_size, "sha256": sha(model)})
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                assert path.stat().st_size < 2 * 2**20
                paths.append((path, path.relative_to(HERE).as_posix()))
    for name in [*read(HERE / "Code_Lock.json")["code_sha256"],
                 "Plan.json", "Code_Lock.json", "integration.py", "Launch_Integration_02.ps1"]:
        path = HERE / name
        paths.append((path, name))
    for name in ("finalize.py", "evidence.py"):
        paths.append((ROOT.parent / "WholeWord_Finalization_01" / name, "export_helper/" + name))
    assert len({name for _, name in paths}) == len(paths)
    package = HERE / "Worker_Integration_Receipts_02.zip"
    manifest = {}
    with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as archive:
        for path, name in paths:
            assert path.resolve().is_relative_to(ROOT.parent.resolve())
            content = path.read_bytes()
            archive.writestr(name, content)
            manifest[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        archive.writestr("Manifest.json", json.dumps(manifest, indent=2))
    proof = {"utc": utc(), "status": "terminal_cpu_worker_and_replay_evidence_packaged",
             "package": str(package), "sha256": sha(package), "bytes": package.stat().st_size,
             "files": len(manifest), "disposable_exports": exports, "actual_assessment_result": False}
    save(HERE / "Worker_Integration_Package_02.json", proof)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
