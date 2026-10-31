"""Preserve revision 1 and reseal the corrected calibration provenance before execution."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
sys.path.insert(0, str(PHASE))

from mechanism_common import read, save, sha, utc


def function_tree(source, name):
    module = ast.parse(source)
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def main():
    intent = read(HERE / "Intent.json")
    archive = HERE / "Previous_Files.zip"
    assert sha(archive) == intent["archive_sha256"]
    assert not list((PHASE / "Runs").glob("*"))
    assert not (PHASE / "Decoder_Selection.json").exists()
    assert sha(PHASE / "Execution_Lock.json") == intent["previous_files"]["Execution_Lock.json"]
    proof_path = PHASE / "Execution_Verification_02.json"
    proof = read(proof_path)
    assert proof["status"] == "passed"
    assert proof["actual_source_prediction_checks"] == 132
    assert proof["complete_partitions_checked"] == 8
    assert len(proof["rejected_cases"]) == 117
    assert proof["calibration_provenance_checked"] == {
        "calibration_regime": {"vi": "target_corpus_development", "en": "source_corpus_development"},
        "destination_calibration_labels_used_by_language": {"vi": True, "en": False},
        "destination_calibration_labels_used": True,
    }
    for name, digest in proof["scripts"].items():
        assert sha(PHASE / name) == digest
    with zipfile.ZipFile(archive) as previous:
        original = previous.read("select_full_decoder.py").decode("utf-8")
        current = (PHASE / "select_full_decoder.py").read_text(encoding="utf-8")
        assert function_tree(original, "choose") == function_tree(current, "choose")
        old_lock = json.loads(previous.read("Execution_Lock.json"))
    allowed = {
        (PHASE / name).relative_to(ROOT).as_posix()
        for name in ("select_full_decoder.py", "test_execution.py")
    }
    for name, digest in old_lock["code_sha256"].items():
        if name not in allowed:
            assert sha(ROOT / name) == digest
    for name, digest in old_lock["prerequisite_sha256"].items():
        assert sha(ROOT / name) == digest
    updated = deepcopy(old_lock)
    updated["utc"] = utc()
    updated["revision"] = 2
    updated["supersedes_lock_sha256"] = intent["previous_files"]["Execution_Lock.json"]
    updated["revision_reason"] = intent["reason"]
    for name in allowed:
        updated["code_sha256"][name] = sha(ROOT / name)
    updated["code_sha256"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    old_proof_name = (PHASE / "Execution_Verification_01.json").relative_to(ROOT).as_posix()
    del updated["prerequisite_sha256"][old_proof_name]
    updated["prerequisite_sha256"][proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
    updated["prerequisite_sha256"][(HERE / "Intent.json").relative_to(ROOT).as_posix()] = sha(HERE / "Intent.json")
    lock_path = PHASE / "Execution_Lock.json"
    lock_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    from execution_common import checked_execution

    plan, checked_lock = checked_execution()
    assert checked_lock == updated and plan["tasks"] == old_lock["tasks"]
    result = {
        "utc": utc(),
        "status": "calibration_provenance_corrected_and_execution_resealed",
        "previous_archive_sha256": sha(archive),
        "previous_lock_sha256": intent["previous_files"]["Execution_Lock.json"],
        "execution_lock_sha256": sha(lock_path),
        "verification_sha256": sha(proof_path),
        "selection_algorithm_ast_unchanged": True,
        "plan_and_cohorts_unchanged": True,
        "new_recognition_admitted": False,
        "existing_active_benchmarks_changed": False,
    }
    save(HERE / "Complete.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
