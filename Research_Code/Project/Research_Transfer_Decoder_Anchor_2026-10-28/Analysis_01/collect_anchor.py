"""Require the complete common-decoder matrix before reading its target predictions."""

from pathlib import Path
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "Vi_Execution"))
from anchor_core import FULL, ROOT, admitted, design, lines, pixel_index, read, sha
from verify_anchor import records_checked


FILES = ("Start.json", "Complete.json", "Identity.json", "Predictions.jsonl", "Verification.json")


def require_complete(found, expected):
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError("Incomplete task coverage: " + str(len(missing)) + " missing; " + str(extra))


def manifest():
    plan, tasks, population, search = design()
    expected = {name for name, task in tasks.items() if task["model"] != "VC_NATIVE"}
    found = {}
    for name in sorted(expected):
        paths = list((PHASE / "Runs").glob("*/" + name + "/Verification.json"))
        if len(paths) > 1:
            raise ValueError("Multiple verified attempts require an explicit reviewed manifest: " + name)
        if not paths:
            continue
        folder = paths[0].parent
        assert not (folder / "Failure.json").exists()
        assert all((folder / item).is_file() for item in FILES)
        proof = read(paths[0])
        assert proof["status"] == "passed" and proof["task"] == name
        assert proof["plan_sha256"] == sha(PHASE / "Plan.json")
        assert proof["execution_lock_sha256"] == sha(PHASE / "Vi_Execution/Execution_Lock.json")
        found[name] = {"path": folder.relative_to(ROOT).as_posix(),
                       "files": {item: sha(folder / item) for item in FILES}}
    require_complete(found, expected)
    assert len(found) == 40
    reuse_path = PHASE / "Vi_Execution/Preparation_Complete_02/Primary_Reuse.json"
    reuse = read(reuse_path)
    assert reuse["status"] == "native_reference_reuse_independently_verified"
    assert reuse["plan_sha256"] == sha(PHASE / "Plan.json")
    assert reuse["primary_completion_sha256"] == sha(FULL / "Completion_Receipt.json")
    assert reuse["decoder_selection_sha256"] == sha(FULL / "Decoder_Selection.json")
    assert reuse["execution_lock_sha256"] == sha(PHASE / "Vi_Execution/Execution_Lock.json")
    assert len(reuse["tasks"]) == 8 and reuse["independent_count_checks"] == 1167
    for entry in reuse["original_files"].values():
        for name, digest in entry["files"].items():
            assert sha(ROOT / entry["path"] / name) == digest
    return plan, tasks, population, found, reuse, reuse_path


def collect():
    plan, tasks, population, found, reuse, reuse_path = manifest()
    models = plan["model_order"]["vi"]
    records = {model: {} for model in models}
    evidence = {reuse_path.relative_to(ROOT).as_posix(): sha(reuse_path)}
    registry = read(FULL / "model_registry.json")["models"]
    registry.update(read(FULL / "Native_Reference_Registry.json")["models"])
    pixels = pixel_index()
    checks = 0
    for name, entry in found.items():
        folder = ROOT / entry["path"]
        start = read(folder / "Start.json")
        proof = read(folder / "Verification.json")
        complete = read(folder / "Complete.json")
        admission_path = ROOT / start["admission_path"]
        assert sha(admission_path) == start["admission_sha256"] == proof["admission_sha256"]
        _, task, rows, _ = admitted(name, admission_path, now=start["started_unix"])
        assert proof["checkpoint_sha256"] == registry[task["model"]]["checkpoint_sha256"]
        assert proof["model"] == start["model"] == task["model"]
        assert proof["clips"] == complete["clips"] == len(rows)
        assert complete["start_sha256"] == entry["files"]["Start.json"]
        assert complete["identity_sha256"] == entry["files"]["Identity.json"]
        assert proof["predictions_sha256"] == complete["predictions_sha256"] == entry["files"]["Predictions.jsonl"]
        replay = proof["fresh_pixel_replays"]
        assert len(replay) == 2
        assert [item["clip_id"] for item in replay] == [rows[0][1]["clip_id"], rows[-1][1]["clip_id"]]
        assert all(0 <= item["score_error"] <= 1e-4 for item in replay)
        values = lines(folder / "Predictions.jsonl")
        count = records_checked(rows, values, task["model"], pixels)
        assert count == proof["independent_count_checks"]
        checks += count
        for value in values:
            assert value["index"] not in records[task["model"]]
            records[task["model"]][value["index"]] = value
        for item, digest in entry["files"].items():
            evidence[(folder / item).relative_to(ROOT).as_posix()] = digest
        evidence[admission_path.relative_to(ROOT).as_posix()] = sha(admission_path)
    original = {}
    for name, entry in reuse["original_files"].items():
        folder = ROOT / entry["path"]
        for value in lines(folder / "Predictions.jsonl"):
            assert value["index"] not in original
            original[value["index"]] = value
        for item, digest in entry["files"].items():
            evidence[(folder / item).relative_to(ROOT).as_posix()] = digest
    assert set(original) == set(range(1167))
    for name, entry in reuse["tasks"].items():
        task = tasks[name]
        assert entry["positions"] == task["positions"]
        rows = [(index, population[index]) for index in task["positions"]]
        values = [original[index] for index, _ in rows]
        checks += records_checked(rows, values, "VC_NATIVE", pixels)
        for value in values:
            assert value["index"] not in records["VC_NATIVE"]
            records["VC_NATIVE"][value["index"]] = value
    assert checks == 7002
    assert all(set(values) == set(range(1167)) for values in records.values())
    return plan, population, records, evidence, {"new_tasks": found, "native_reuse_path": str(reuse_path)}
