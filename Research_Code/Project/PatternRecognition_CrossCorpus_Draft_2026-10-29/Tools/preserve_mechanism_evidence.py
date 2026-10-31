"""Move paper admission inputs out of the two draft directories being retired."""

import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent


def main():
    source = ROOT / "PatternRecognition_FullUtterance_Draft_2026-10-29/Mechanism_Update_01"
    target = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Paper_Evidence_01"
    target.mkdir(exist_ok=False)
    files = [source / "stage_completed.py", source / "merge_counts.py"]
    files.extend(path for path in (source / "Staged_01").rglob("*") if path.is_file())
    manifest = {}
    for path in files:
        relative = path.relative_to(source)
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == original
        manifest[relative.as_posix()] = original
    (target / "Relocation.json").write_text(json.dumps({
        "source": source.relative_to(ROOT).as_posix(),
        "status": "all_admission_inputs_copied_without_content_changes",
        "files": manifest,
        "purpose": "Stable inputs for the current manuscript numerical verifier",
        "stage_completed_main": "Historical staging entrypoint; only check_finalized is used here",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "admission_inputs_preserved", "files": len(files)}))


if __name__ == "__main__":
    main()
