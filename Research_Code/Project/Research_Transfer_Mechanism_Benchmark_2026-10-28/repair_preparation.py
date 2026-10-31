"""Preserve the unexecuted first plan and repair its incomplete Vietnamese calibration queue."""

from copy import deepcopy
import hashlib
import json
import zipfile

from mechanism_common import HERE, ROOT, lines, read, save, sha, utc


def main():
    original = read(HERE / "Plan.json")
    assert "preparation_revision" not in original
    assert original["new_gpu_work_admitted"] is False
    assert not (HERE / "Runtime_Validation_01").exists()
    previous_sha = sha(HERE / "Plan.json")
    folder = HERE / "Preparation_01"
    folder.mkdir(exist_ok=False)
    files = ["mechanism_common.py", "model_registry.json", "Plan.json", "prepare.py", "Protocol.md",
             "runtime_bridge.py", "validate_runtime.py"]
    hashes = {name: sha(HERE / name) for name in files}
    snapshot = folder / "Rejected_Snapshot.zip"
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(HERE / name, name)
        archive.writestr("Manifest.json", json.dumps(hashes, sort_keys=True))
    with zipfile.ZipFile(snapshot) as archive:
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    before = {}
    corrected = deepcopy(original)
    rows = lines(ROOT / corrected["cohorts"]["vi"]["development"]["path"])
    ordered = sorted(range(len(rows)), key=lambda index: (rows[index]["frames"], rows[index]["clip_id"]))
    for model in ("VLR_T2_LAST", "VLR_T4_LAST"):
        tasks = [task for task in original["tasks"].values()
                 if task["model"] == model and task["stage"] == "development"]
        positions = [position for task in tasks for position in task["positions"]]
        assert len(positions) == len(set(positions)) == 237
        before[model] = len(positions)
        for shard in range(8):
            name = "development_" + model + "_" + str(shard).zfill(2)
            corrected["tasks"][name]["positions"] = ordered[shard::8]
            corrected["tasks"][name]["parent_task"] = None
            corrected["tasks"][name]["partition_basis"] = "complete_manifest_sorted_frames_stride8"
    save(folder / "Rejection.json", {"utc": utc(), "status": "rejected_before_recognition",
         "reason": "Parent T1 queue encoded only 237 new development clips and reused 531 prior predictions. "
                   "No corresponding T2/T4 predictions exist; each new model must process all 768 clips.",
         "coverage_before": before, "required_clips_per_model": 768,
         "recognition_started": False, "parent_benchmarks_changed": False,
         "previous_plan_sha256": previous_sha, "snapshot_sha256": sha(snapshot),
         "snapshot_readback_passed": True})
    protocol = (HERE / "Protocol.md").read_text(encoding="utf-8")
    old = "shards. Task memberships inherit each parent's frozen T1 partition. At most"
    new = ("shards. Each new state covers the complete parent cohort exactly once.\n"
           "English and assessment partitions retain the parent memberships; Vietnamese\n"
           "development uses all 768 positions sorted by length and distributed over eight\n"
           "shards. Parent T1's 237-clip incremental queue cannot be reused for a new state. At most")
    assert protocol.count(old) == 1
    (HERE / "Protocol.md").write_text(protocol.replace(old, new), encoding="utf-8")
    corrected["utc"] = utc()
    corrected["preparation_revision"] = 2
    corrected["superseded_plan_sha256"] = previous_sha
    corrected["preparation_correction"] = "full_calibration_coverage_before_any_new_recognition"
    for name in ("Protocol.md", "repair_preparation.py", "verify_preparation.py",
                 "Preparation_01/Rejection.json", "Preparation_01/Rejected_Snapshot.zip"):
        corrected["input_sha256"][(HERE / name).relative_to(ROOT).as_posix()] = sha(HERE / name)
    assert sha(HERE / "Plan.json") == previous_sha
    (HERE / "Plan.json").write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"revision": 2, "previous_plan_sha256": previous_sha, "plan_sha256": sha(HERE / "Plan.json")})


if __name__ == "__main__":
    main()
