"""Exercise the real selector on explicitly synthetic complete six-model fixtures."""

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from common import HERE, read, save, sha, utc


def main(attempt, expect_portable):
    assert attempt.isidentifier()
    folder = HERE / attempt
    folder.mkdir(exist_ok=False)
    phase = folder / "Synthetic_Project" / "Synthetic_Phase"
    phase.mkdir(parents=True)
    models = ["VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST", "VLR_T0_29_LAST", "VLR_T1_29_LAST", "VC_NATIVE"]
    candidates = read(HERE / "Plan.json")["candidate_order"]
    save(phase / "Plan.json", {"frozen_sha256": {}, "matched_models": models[:-1],
                               "practical_reference": models[-1], "candidate_order": candidates})
    entries = {model: {"checkpoint_sha256": "synthetic_checkpoint_not_a_model"} for model in models}
    save(phase / "model_registry.json", {"models": {model: entries[model] for model in models[:-1]}})
    save(phase / "Native_Reference_Registry.json", {"models": {models[-1]: entries[models[-1]]}})
    cohort = [{"clip_id": str(index), "source_id": "synthetic_source", "split": "development",
               "reference": "one"} for index in range(768)]
    with (phase / "Development.jsonl").open("x", encoding="utf-8") as stream:
        for row in cohort:
            stream.write(json.dumps(row) + "\n")
    (phase / "Assessment.jsonl").write_text("Synthetic marker only; never used for inference\n", encoding="utf-8")
    bundle = phase / "Bundle"
    bundle.mkdir()
    manifest = {"status": "complete_verified_development_bundle", "models": {},
                "plan_sha256": sha(phase / "Plan.json"), "synthetic_test_only": True}
    for model in models:
        prediction_path = bundle / (model + ".jsonl")
        verification_path = bundle / (model + "_Verification.json")
        with prediction_path.open("x", encoding="utf-8") as stream:
            for row in cohort:
                predictions = {}
                for index, candidate in enumerate(candidates):
                    error = int(index != 0)
                    predictions[candidate] = {"text": "two" if error else "one",
                                              "counts": {"errors": error, "substitutions": error,
                                                         "deletions": 0, "insertions": 0,
                                                         "units": 1, "metric": "wer"}}
                stream.write(json.dumps({**row, "model": model, "predictions": predictions}) + "\n")
        save(verification_path, {"status": "passed", "model": model, "split": "development", "clips": 768,
                                 "plan_sha256": sha(phase / "Plan.json"),
                                 "predictions_sha256": sha(prediction_path),
                                 "checkpoint_sha256": entries[model]["checkpoint_sha256"],
                                 "candidate_order": candidates, "synthetic_test_only": True})
        manifest["models"][model] = {
            "predictions_path": prediction_path.relative_to(phase.parent).as_posix(),
            "predictions_sha256": sha(prediction_path),
            "verification_path": verification_path.relative_to(phase.parent).as_posix(),
            "verification_sha256": sha(verification_path),
        }
    manifest_path = bundle / "Manifest.json"
    save(manifest_path, manifest)
    source = HERE / "select_decoder.py"
    spec = importlib.util.spec_from_file_location("portable_selector_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = phase
    module.ROOT = phase.parent
    module.main(manifest_path)
    selected = read(phase / "Decoder_Selection.json")
    assert all(value["decoder"] == candidates[0] for value in selected["models"].values())
    checked = sum(value["independent_prediction_counts_checked"] for value in selected["models"].values())
    assert checked == 50688
    invalid = [name for name in selected["input_sha256"]
               if PureWindowsPath(name).is_absolute() or PurePosixPath(name).is_absolute() or "\\" in name]
    assert bool(invalid) is not expect_portable, (invalid, expect_portable)
    if expect_portable:
        for name, digest in selected["input_sha256"].items():
            assert sha(phase.parent / name) == digest
            assert str(PurePosixPath("/root/project") / name).startswith("/root/project/")
    save(folder / "Verification.json", {
        "utc": utc(), "status": "passed", "synthetic_only": True,
        "selector_sha256": sha(source), "candidate_counts_checked": checked,
        "selected_decoders": {model: value["decoder"] for model, value in selected["models"].items()},
        "portable": not invalid, "nonportable_keys": invalid,
        "expected_original_failure_reproduced": bool(invalid) and not expect_portable,
        "new_recognition_result": False,
    })
    print({"portable": not invalid, "candidate_counts_checked": checked}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt")
    parser.add_argument("--expect-portable", action="store_true")
    arguments = parser.parse_args()
    main(arguments.attempt, arguments.expect_portable)
