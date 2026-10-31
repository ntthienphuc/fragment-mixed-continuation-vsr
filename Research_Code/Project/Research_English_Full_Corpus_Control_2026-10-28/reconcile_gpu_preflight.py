"""Reconcile a device-label assertion after a successful native GPU source probe.

All original evidence remains unchanged. This creates an explicitly derived
receipt for the CLI device alias and does not execute the recognition model.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import zipfile

from experiment_core import check_prediction, cohort
from mechanism_common import HERE, ROOT, read, save, sha, utc


ATTEMPT = HERE / "Shared_Server_Preflight_03"
RAW = ATTEMPT / "Remote/Proof"
CPU = HERE / "Source_Grid_CPU_01/Proofs/EN_F0_LAST"
OUTPUT = ATTEMPT / "Reconciliation_01"


def check_alias(value):
    assert value["status"] == "single_device_alias_execution_verified"
    assert type(value["visible_cuda_devices"]) is int and value["visible_cuda_devices"] == 1
    assert type(value["current_cuda_index"]) is int and value["current_cuda_index"] == 0
    assert value["implicit_device"] == value["explicit_device"] == "cuda:0"
    assert value["gpu"] == "NVIDIA L40"
    expected = "/root/lip_english_transfer_20261028_r2/venv/bin/python"
    assert value["python"] == expected


def check_probe_files(folder, proof):
    assert sha(folder / "Identity.json") == proof["identity_sha256"]
    for name, digest in proof["probe_sha256"].items():
        assert sha(folder / name) == digest
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    assert proof["plan_sha256"] == sha(HERE / "Plan.json")
    assert proof["assessment_used"] is False and proof["reference_used_for_inference"] is False


def main():
    assert not OUTPUT.exists(), "Preserve existing reconciliation; do not overwrite"
    alias_path = ATTEMPT / "Device_Alias_Runtime_Verification.json"
    alias = read(alias_path)
    check_alias(alias)
    rejected = 0
    mutations = [("visible_cuda_devices", 2), ("current_cuda_index", 1), ("implicit_device", "cpu"),
                 ("explicit_device", "cuda:1"), ("gpu", "Unknown"), ("python", "/usr/bin/python")]
    for field, value in mutations:
        invalid = {**alias, field: value}
        try:
            check_alias(invalid)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError("Accepted wrong device/environment: " + field)
    assert rejected == len(mutations)

    archive_receipt = read(ATTEMPT / "Backup.json")
    assert archive_receipt["status"] == "sha256_readback_passed"
    assert sha(archive_receipt["path"]) == archive_receipt["sha256"]
    with zipfile.ZipFile(archive_receipt["path"]) as archive:
        manifest = json.loads(archive.read("Manifest.json"))
        assert set(archive.namelist()) == set(manifest) | {"Manifest.json"}
        for name, digest in manifest.items():
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == digest
            path = (ROOT / name).resolve()
            path.relative_to((ATTEMPT / "Remote").resolve())
            assert sha(path) == digest

    original_exit = read(ATTEMPT / "Remote/Exit.json")
    assert original_exit["returncode"] == 0 and original_exit["success"] is False
    failure = read(ATTEMPT / "Remote/Failure.json")["error"]
    assert 'proof["device"] == "cuda"' in failure and "AssertionError" in failure
    assert original_exit["seconds"] < 300
    raw = read(RAW / "Complete.json")
    cpu = read(CPU / "Complete.json")
    assert raw["status"] == cpu["status"] == "three_source_inputs_full_grid_verified"
    assert raw["device"] == "cuda:0" and cpu["device"] == "cpu"
    assert raw["model"] == cpu["model"] == "EN_F0_LAST"
    assert raw["positions"] == cpu["positions"] == [518, 176, 1012]
    assert raw["frames"] == cpu["frames"] == [14, 46, 153]
    check_probe_files(RAW, raw)
    check_probe_files(CPU, cpu)
    identity = read(RAW / "Identity.json")
    assert identity["strict_state_load"] is True and identity["selected_training_step"] == 6000
    assert identity["precision"] == "float32" and identity["tf32"] is False
    assert identity["classes"] == 5049 and identity["blank_id"] == 0 and identity["eos_id"] == 5048
    registry = read(HERE / "model_registry.json")["models"]["EN_F0_LAST"]
    assert raw["checkpoint_sha256"] == cpu["checkpoint_sha256"] == registry["checkpoint_sha256"]
    assert sha(ROOT / registry["checkpoint"]) == registry["checkpoint_sha256"]

    plan = read(HERE / "Plan.json")
    rows = cohort(plan, "en", "development")
    expected = {(position, candidate) for position in raw["positions"] for candidate in plan["candidate_order"]}
    for proof in (raw, cpu):
        assert proof["independent_count_checks"] == len(proof["comparisons"]) == 33
        assert {(item["position"], item["candidate"]) for item in proof["comparisons"]} == expected
        for item in proof["comparisons"]:
            assert item["tokens_exact"] is True and item["text_exact"] is True
            assert math.isfinite(item["score_error"]) and 0 <= item["score_error"] <= 1e-4
    comparisons = []
    for position in raw["positions"]:
        name = "Probe_" + str(position) + ".json"
        left = read(CPU / name)
        right = read(RAW / name)
        row = rows[position]
        assert left["clip_id"] == right["clip_id"] == row["clip_id"]
        assert left["frames"] == right["frames"] == row["frames"]
        assert left["encoding"]["pixel_sha256"] == right["encoding"]["pixel_sha256"]
        assert set(left["predictions"]) == set(right["predictions"]) == set(plan["candidate_order"])
        for candidate in plan["candidate_order"]:
            first = left["predictions"][candidate]
            second = right["predictions"][candidate]
            check_prediction(row, first, candidate, 5049)
            check_prediction(row, second, candidate, 5049)
            assert first["tokens"] == second["tokens"] and first["text"] == second["text"]
            difference = abs(first["score"] - second["score"]) if "score" in first else 0.0
            assert math.isfinite(difference)
            comparisons.append({"position": position, "candidate": candidate, "tokens_and_text_exact": True,
                                "cross_backend_score_difference_diagnostic": difference})
    assert len(comparisons) == 33

    inputs = [RAW / "Complete.json", CPU / "Complete.json", alias_path, ATTEMPT / "Backup.json",
              ATTEMPT / "Remote/Exit.json", ATTEMPT / "Remote/Failure.json", HERE / "Execution_Lock.json"]
    original_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}
    canonical = copy.deepcopy(raw)
    canonical["device"] = "cuda"
    canonical["device_label_normalization"] = {
        "native_reported_device": "cuda:0", "cli_device_alias": "cuda", "visible_cuda_devices": 1,
        "raw_receipt_path": (RAW / "Complete.json").relative_to(ROOT).as_posix(),
        "raw_receipt_sha256": sha(RAW / "Complete.json"), "runtime_alias_sha256": sha(alias_path),
        "reconciled_utc": utc(), "recognition_outputs_modified": False, "inference_repeated": False,
    }
    OUTPUT.mkdir()
    save(OUTPUT / "Canonical_Runtime_Proof.json", canonical)
    restored = read(OUTPUT / "Canonical_Runtime_Proof.json")
    del restored["device_label_normalization"]
    restored["device"] = "cuda:0"
    assert restored == raw, "Only the explicit device-label metadata may change"
    assert all(sha(ROOT / name) == digest for name, digest in original_hashes.items())
    save(OUTPUT / "Complete.json", {
        "utc": utc(), "status": "F0_L40_preflight_reconciled_without_inference",
        "raw_child_returncode": 0, "original_wrapper_failure_preserved": True,
        "device_alias_negative_cases_rejected": rejected, "native_comparisons": 33,
        "independent_counts_rechecked": 66, "cpu_gpu_comparisons": comparisons,
        "canonical_receipt_sha256": sha(OUTPUT / "Canonical_Runtime_Proof.json"),
        "input_sha256": original_hashes, "script_sha256": sha(Path(__file__)),
        "new_inference": False, "assessment_used": False, "full_evaluation_admitted": False,
    })
    print({"status": "reconciled", "native_comparisons": 33, "independent_counts": 66,
           "cpu_gpu_exact_tokens": 33, "rejected_device_cases": rejected,
           "maximum_cross_backend_score_difference_diagnostic": max(
               item["cross_backend_score_difference_diagnostic"] for item in comparisons)})


if __name__ == "__main__":
    main()
