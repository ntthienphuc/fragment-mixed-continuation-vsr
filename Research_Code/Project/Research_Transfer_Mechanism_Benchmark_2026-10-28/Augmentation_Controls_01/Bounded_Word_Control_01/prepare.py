"""Build a separately named bounded control; parent code and schedules remain immutable."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

from word_span import MINIMUM, MAXIMUM, choose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "Readiness_02"
SPEC = importlib.util.spec_from_file_location("bounded_parent_preparation", PARENT / "prepare.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def main():
    assert not (HERE / "Summary.json").exists()
    parent_summary = BUILD.read(PARENT / "Summary.json")
    parent_proof = BUILD.read(PARENT / "Verification.json")
    assert parent_proof["status"] == "complete_training_schedule_verified"
    assert parent_proof["summary_sha256"] == BUILD.sha(PARENT / "Summary.json")
    source_hashes = {name: BUILD.sha(PARENT / name) for name in ["prepare.py", "word_span.py"]}
    assert source_hashes == parent_summary["scripts"]
    sys.path.insert(0, str(ROOT / "Research_Phase_05_2026-10-13"))
    from phase5.views import CONTEXT

    for dataset in ["vlr", "lrs2"]:
        config = BUILD.read(ROOT / "Research_Phase_05_2026-10-13/configs" / (dataset + ".json"))
        assert MINIMUM == config["core_frames"] and MAXIMUM == config["core_frames"] + 2 * CONTEXT
    BUILD.HERE = HERE
    BUILD.choose = choose
    replay = BUILD.read(BUILD.REPLAY / "Summary.json")
    proof = BUILD.read(BUILD.REPLAY / "Verification.json")
    assert proof["status"] == "passed" and proof["replay_sha256"] == BUILD.sha(BUILD.REPLAY / "Summary.json")
    results = []
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        for item in replay["datasets"]:
            result = BUILD.build(item["dataset"], item, archive)
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {"utc": datetime.now(timezone.utc).isoformat(),
               "status": "bounded_schedule_pending_independent_verification", "datasets": results,
               "duration_band_frames": [MINIMUM, MAXIMUM], "parent_implementation_sha256": source_hashes,
               "geometry_sha256": BUILD.sha(ROOT / "Research_Phase_05_2026-10-13/phase5/views.py"),
               "scripts": {name: BUILD.sha(HERE / name) for name in ["prepare.py", "word_span.py"]},
               "protocol_sha256": BUILD.sha(HERE / "Protocol.md"), "training_admitted": False,
               "assessment_predictions_used": False}
    with (HERE / "Summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
