"""Freeze the complete statistical comparison before new target predictions exist."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent
ROOT = SCIENCE.parent
sys.path.insert(0, str(SCIENCE))

from execution_common import checked_execution
from mechanism_common import read, save, sha, utc


def main():
    plan, execution = checked_execution()
    assert not list((SCIENCE / "Runs").glob("*/assessment_*/Predictions.jsonl"))
    proof = read(HERE / "Verification_01.json")
    assert proof["status"] == "passed"
    assert proof["completed_parent_csv_rows_checked"] == 42544
    assert len(proof["parent_WER_and_bootstrap_reproductions"]) == 4
    assert len(proof["rejected_cases"]) == 13
    assert proof["zero_gain_control_exact"] is True and proof["new_T2_T4_predictions_read"] is False
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    scripts = {str((HERE / name).relative_to(ROOT).as_posix()): digest
               for name, digest in proof["scripts"].items()}
    for name in ("freeze.py", "Protocol.md"):
        scripts[(HERE / name).relative_to(ROOT).as_posix()] = sha(HERE / name)
    inputs = dict(proof["input_sha256"])
    inputs[(HERE / "Verification_01.json").relative_to(ROOT).as_posix()] = sha(HERE / "Verification_01.json")
    models = {
        "vi": ["VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST", "VLR_T2_LAST", "VLR_T4_LAST", "VC_NATIVE"],
        "en": ["EN_B0", "EN_T0_LAST", "EN_T1_LAST", "EN_T2_LAST", "EN_T4_LAST", "EN_F0_LAST", "EN_B1"],
    }
    pairs = {}
    for language, prefix in (("vi", "VLR"), ("en", "EN")):
        pairs[language] = [[prefix + "_" + left + "_LAST", prefix + "_" + right + "_LAST"]
                           for left, right in (("T1", "T2"), ("T1", "T4"), ("T0", "T1"),
                                               ("T0", "T2"), ("T0", "T4"))]
    result = {
        "utc": utc(), "status": "mechanism_analysis_frozen_before_new_assessment",
        "plan_sha256": sha(SCIENCE / "Plan.json"),
        "execution_lock_sha256": sha(SCIENCE / "Execution_Lock.json"),
        "model_order": models, "pairs": pairs,
        "primary_new_contrasts": {language: values[:2] for language, values in pairs.items()},
        "bootstrap": {"vi": {"unit": "video_source", "draws": 10000, "seed": 20261027},
                      "en": {"unit": "released_clip", "draws": 10000, "seed": 20261028}},
        "script_sha256": scripts, "prerequisite_sha256": inputs,
        "new_target_predictions_exist": False, "parent_results_already_known": True,
        "paid_execution_admitted": False,
    }
    save(HERE / "Lock.json", result)
    print({"status": result["status"], "new_models": len(plan["models"]), "readouts": 2}, flush=True)


if __name__ == "__main__":
    main()
