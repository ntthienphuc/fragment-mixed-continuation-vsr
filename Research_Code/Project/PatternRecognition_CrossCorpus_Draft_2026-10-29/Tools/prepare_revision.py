"""Preserve both retired drafts before creating the replacement manuscript."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OLD_NAMES = [
    "PatternRecognition_FullUtterance_Draft_2026-10-29",
    "PatternRecognition_Vietnamese_Transfer_Draft_2026-10-29",
]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    evidence = HERE / "QA"
    evidence.mkdir(exist_ok=True)
    receipt_path = evidence / "Previous_Drafts_Backup.json"
    assert not receipt_path.exists(), "Do not overwrite an existing backup receipt"
    backup_root = Path("C:/Users/thien/Lip_Research_Backups/Paper_Replacement_20261029")
    backup_root.mkdir(parents=True, exist_ok=True)
    archive_path = backup_root / ("Previous_Drafts_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")
    manifest = {}
    for name in OLD_NAMES:
        folder = (ROOT / name).resolve()
        assert folder.parent == ROOT.resolve() and folder.is_dir()
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                assert not path.is_symlink()
                manifest[path.relative_to(ROOT).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(ROOT / name, name)
        archive.writestr("File_Manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            content = archive.read(name)
            assert len(content) == expected["bytes"]
            assert hashlib.sha256(content).hexdigest() == expected["sha256"], name
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_archives_read_back_and_sha256_verified",
        "folders": OLD_NAMES,
        "archive": str(archive_path),
        "archive_sha256": sha(archive_path),
        "files": manifest,
        "deletion_performed": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    source = ROOT / OLD_NAMES[1]
    for name in ("Source", "Data"):
        shutil.copytree(source / name, HERE / name)
    for name in ("build.py", "verify_results.py", "verify_package.py", "completed_results.py"):
        shutil.copy2(source / "Tools" / name, HERE / "Tools" / name)
    (HERE / "Build").mkdir()
    notes = HERE / "Research_Notes"
    notes.mkdir()
    shutil.copy2(source / "Review_02/External_Review_Response.md", notes / "Previous_Review_Response.md")
    shutil.copy2(source / "Review_01/Independent_Review.md", notes / "Previous_Independent_Review.md")
    analysis = ROOT / "Research_Source_Retention_2026-10-29/Analysis_01"
    inputs = analysis / "Inputs_01"
    inputs.mkdir()
    table = ROOT / OLD_NAMES[0] / "Data/Results.csv"
    shutil.copy2(table, inputs / "Destination_Results.csv")
    assert sha(table) == sha(inputs / "Destination_Results.csv") == sha(HERE / "Data/Results.csv")
    archived = notes / "Before_Path_Migration"
    archived.mkdir()
    for name in ("summarize.py", "watch_analysis.py", "verify_analysis.py"):
        shutil.copy2(analysis / name, archived / name)
    (inputs / "Provenance.json").write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(),
        "source": table.relative_to(ROOT).as_posix(),
        "sha256": sha(table),
        "purpose": "Stable destination result table independent of manuscript folders",
        "backup_archive": str(archive_path),
        "data_values_changed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "archive": str(archive_path),
                      "files": len(manifest), "bytes": archive_path.stat().st_size}))


if __name__ == "__main__":
    main()
