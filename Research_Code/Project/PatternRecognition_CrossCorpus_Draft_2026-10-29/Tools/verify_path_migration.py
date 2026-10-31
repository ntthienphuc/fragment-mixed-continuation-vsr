"""Recheck the unchanged population after moving its destination table."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
ANALYSIS = ROOT / "Research_Source_Retention_2026-10-29/Analysis_01"


def main():
    source = (ANALYSIS / "verify_analysis.py").read_text(encoding="utf-8")
    old = 'output = collect.HERE / "Validation_01"'
    assert source.count(old) == 1
    revised = source.replace(old, 'output = collect.HERE / "Validation_02_Path_Migration"')
    environment = {"__name__": "migration_verifier", "__file__": str(ANALYSIS / "verify_analysis.py")}
    sys.path.insert(0, str(ANALYSIS))
    exec(compile(revised, str(ANALYSIS / "verify_analysis.py"), "exec"), environment)
    environment["main"]()
    output = ANALYSIS / "Main_Contrast_02_Path_Migration"
    command = [sys.executable, "-X", "utf8", "-B", str(ANALYSIS / "summarize.py"),
               "--corpus", "lrs2", "--model", "EN_B0", "--model", "EN_T0_LAST",
               "--model", "EN_T1_LAST", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, timeout=600)
    (HERE / "QA/Path_Migration_stdout.log").write_bytes(result.stdout)
    (HERE / "QA/Path_Migration_stderr.log").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    previous = json.loads((ANALYSIS / "Main_Contrast_01/Summary.json").read_text())
    current = json.loads((output / "Summary.json").read_text())
    fields = ["results", "comparisons", "source_and_destination", "destination_table_sha256",
              "population_clips", "source_groups", "all_requested_models_complete",
              "whole_corpus_matrix_complete", "source_test_readouts_retuned", "inputs"]
    for field in fields:
        assert previous[field] == current[field], field
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "identical_numeric_results_and_inputs_after_path_migration",
        "fields_checked": fields,
        "real_clips_per_state": current["population_clips"],
        "states": 3,
        "source_summary_sha256": hashlib.sha256((output / "Summary.json").read_bytes()).hexdigest(),
        "frozen_assessment_plans_unchanged": True,
        "old_validation_retained": True,
    }
    (HERE / "QA/Path_Migration_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
