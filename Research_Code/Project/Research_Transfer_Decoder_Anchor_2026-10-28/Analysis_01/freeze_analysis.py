"""Fix the inherited source bootstrap and explicit secondary decoder before consolidation."""

from pathlib import Path

from collect_anchor import FULL, PHASE, read, sha
from anchor_core import save, utc


HERE = Path(__file__).resolve().parent


def main():
    proof_path = HERE / "Verification_03.json"
    proof = read(proof_path)
    assert proof["status"] == "passed" and proof["synthetic_rows_checked"] == 7002
    assert proof["scalar_bootstrap_checks"] == 202
    assert proof["target_prediction_counts_parsed_by_this_test"] is False
    assert proof["closed_prediction_file_bytes_read_for_hash_checks"] is True
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    result = {"utc": utc(), "status": "vi_anchor_analysis_fixed_before_complete_matrix",
              "primary_contrasts": ["VLR_T0_LAST_minus_VLR_T1_LAST", "VLR_T0_29_LAST_minus_VLR_T1_29_LAST"],
              "decoder_selection_on_target": False, "new_hidden_test": False,
              "bootstrap": read(FULL / "Plan.json")["bootstrap"],
              "plan_sha256": sha(PHASE / "Plan.json"),
              "primary_completion_sha256": sha(FULL / "Completion_Receipt.json"),
              "parent_analysis_sha256": sha(FULL / "analyze_assessment.py"),
              "scripts": proof["scripts"], "tests_path": proof_path.name, "tests_sha256": sha(proof_path)}
    save(HERE / "Analysis_Lock.json", result)
    print({"status": result["status"], "bootstrap": result["bootstrap"]})


if __name__ == "__main__":
    main()
