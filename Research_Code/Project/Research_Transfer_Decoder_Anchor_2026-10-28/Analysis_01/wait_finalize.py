"""Wait finitely for every closed shard, then analyze and archive the complete matrix."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from collect_anchor import FULL, PHASE, ROOT, manifest, read, sha
from anchor_core import save, utc

HERE = Path(__file__).resolve().parent


def update(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def backup():
    output = HERE / "Results_01"
    summary = read(output / "Summary.json")
    receipt = read(PHASE / "Completion_Receipt.json")
    assert receipt["status"] == "complete_vi_anchor_verified"
    assert receipt["summary_sha256"] == sha(output / "Summary.json")
    assert receipt["per_clip_sha256"] == sha(output / "Per_Clip_Counts.csv")
    files = {ROOT / name for name in summary["input_sha256"]}
    files.update(path for path in HERE.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".csv"})
    files.update({PHASE / "Plan.json", PHASE / "Completion_Receipt.json", FULL / "Completion_Receipt.json"})
    directory = Path("C:/Users/thien/Lip_Research_Backups/VI_Anchor_20261028")
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / ("Complete_" + str(time.time_ns()) + ".zip")
    index = {}
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            assert path.resolve().is_relative_to(ROOT.resolve())
            payload = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            index[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            archive.writestr(name, payload)
        archive.writestr("Manifest.json", json.dumps(index, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(index) | {"Manifest.json"}
        for name, record in index.items():
            payload = archive.read(name)
            assert len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == record["sha256"]
    proof = {"utc": utc(), "status": "complete_analysis_archive_readback_passed",
             "path": str(archive_path), "sha256": sha(archive_path), "files": len(index)}
    save(HERE / "Final_Backup.json", proof)
    lines = ["", "[" + utc() + "] VI COMMON DECODER: ĐỦ MA TRẬN VÀ BACKUP",
             "Đã kiểm chứng 6 model x 1.167 clip; cùng decoder beam_c0.1, 20 nguồn video."]
    for pair, result in summary["pairs"].items():
        lines.append(pair + ": giảm WER " + str(round(result["gain_pp"], 4)) +
                     " điểm phần trăm; source CI95 " + str(result["source_ci95_pp"]) + ".")
    lines.extend(["Đây là đánh giá phụ cùng decoder trên tập đã dùng trước; không phải test ẩn mới.",
                  "Kết luận chỉ điều kiện trên các seed đã train; chưa tự đưa vào paper.",
                  "Backup: " + str(archive_path) + "; SHA-256 readback passed.", ""])
    with (ROOT / "result.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return proof


def main():
    assert read(HERE / "Analysis_Lock.json")["status"] == "vi_anchor_analysis_fixed_before_complete_matrix"
    deadline = time.time() + 21600
    save(HERE / "Watcher_Start.json", {"utc": utc(), "pid": os.getpid(), "deadline_unix": deadline})
    while time.time() < deadline:
        try:
            manifest()
        except ValueError as error:
            if not str(error).startswith("Incomplete task coverage:"):
                raise
            update("Watcher_Status.json", {"utc": utc(), "pid": os.getpid(),
                    "status": "waiting_for_complete_matrix", "detail": str(error), "deadline_unix": deadline})
            time.sleep(120)
            continue
        update("Watcher_Status.json", {"utc": utc(), "pid": os.getpid(), "status": "analyzing_complete_matrix"})
        with (HERE / "analysis.stdout.log").open("x") as stdout, (HERE / "analysis.stderr.log").open("x") as stderr:
            subprocess.run([sys.executable, "-X", "utf8", str(HERE / "analyze_anchor.py")],
                           cwd=ROOT, stdout=stdout, stderr=stderr, check=True, timeout=900)
        result = backup()
        save(HERE / "Watcher_Complete.json", {"utc": utc(), "status": "analysis_and_backup_complete", **result})
        return
    save(HERE / "Watcher_Complete.json", {"utc": utc(), "status": "finite_wait_expired_without_analysis"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save(HERE / "Finalization_Failure.json", {"utc": utc(), "error": repr(error)})
        raise
