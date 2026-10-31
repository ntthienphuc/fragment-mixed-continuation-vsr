"""Preserve the failed metadata contract and reseal it before full-cohort inference."""

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


def main():
    intent = read(HERE / "Intent.json")
    archive = HERE / "Previous_Files.zip"
    assert sha(archive) == intent["archive_sha256"]
    assert not list((PHASE / "Runs").glob("*/assessment_*/Predictions.jsonl"))
    assert not (PHASE / "Decoder_Selection.json").exists()
    proof_path = PHASE / "Execution_Verification_03.json"
    proof = read(proof_path)
    assert proof["status"] == "passed" and proof["actual_source_prediction_checks"] == 132
    assert len(proof["rejected_cases"]) == 121 and len(proof["observed_gpu_device_records"]) == 2
    for name, digest in proof["scripts"].items():
        assert sha(PHASE / name) == digest
    with zipfile.ZipFile(archive) as previous:
        old_lock = json.loads(previous.read("Execution_Lock.json"))
        old_analysis = json.loads(previous.read("Analysis_01/Lock.json"))
    allowed = {(PHASE / name).relative_to(ROOT).as_posix() for name in ("execution_common.py", "test_execution.py")}
    for name, digest in old_lock["code_sha256"].items():
        if name not in allowed:
            assert sha(ROOT / name) == digest, name
    for name, digest in old_lock["prerequisite_sha256"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(PHASE / "Execution_Lock.json") == intent["previous_files"]["Execution_Lock.json"]
    updated = deepcopy(old_lock)
    updated["utc"] = utc()
    updated["revision"] = 3
    updated["supersedes_lock_sha256"] = intent["previous_files"]["Execution_Lock.json"]
    updated["revision_reason"] = intent["reason"]
    for name in allowed:
        updated["code_sha256"][name] = sha(ROOT / name)
    updated["code_sha256"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    old_proof = (PHASE / "Execution_Verification_02.json").relative_to(ROOT).as_posix()
    updated["prerequisite_sha256"].pop(old_proof)
    updated["prerequisite_sha256"][proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
    updated["prerequisite_sha256"][(HERE / "Intent.json").relative_to(ROOT).as_posix()] = sha(HERE / "Intent.json")
    (PHASE / "Execution_Lock.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    analysis = deepcopy(old_analysis)
    analysis["utc"] = utc()
    analysis["execution_lock_sha256"] = sha(PHASE / "Execution_Lock.json")
    analysis["supersedes_analysis_lock_sha256"] = intent["previous_files"]["Analysis_01/Lock.json"]
    analysis["execution_metadata_revision_only"] = True
    (PHASE / "Analysis_01/Lock.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    from execution_common import checked_execution

    plan, checked = checked_execution()
    assert checked == updated and plan["tasks"] == old_lock["tasks"]
    result = {"utc": utc(), "status": "cuda_device_identity_corrected_before_full_evaluation",
              "execution_lock_sha256": sha(PHASE / "Execution_Lock.json"),
              "analysis_lock_sha256": sha(PHASE / "Analysis_01/Lock.json"),
              "previous_archive_sha256": sha(archive), "verification_sha256": sha(proof_path),
              "plan_decoding_checkpoint_and_analysis_algorithms_unchanged": True,
              "assessment_started": False}
    save(HERE / "Complete.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
