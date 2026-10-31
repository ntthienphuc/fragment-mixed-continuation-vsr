"""Exercise admission rejection and the real encoder CLI without GPU permission."""

from copy import deepcopy
import os
import subprocess
import sys
import time

from common import HERE, ROOT, STORE, read, save, sha, utc, validate_admission


def validate(admission, receipt, result):
    validate_admission(admission, receipt, result, "plan", "receipt", "summary")


def main():
    folder = HERE / "Preparation_Test_01"
    folder.mkdir(exist_ok=False)
    admission = {"status": "admitted_after_parent_review", "plan_sha256": "plan",
                 "parent_receipt_sha256": "receipt", "parent_summary_sha256": "summary",
                 "complete_decoder_grid_required": True, "models": ["VLR_T0_29_LAST", "VLR_T1_29_LAST"],
                 "scientific_decision": "Synthetic unit fixture only; no real admission file is created.",
                 "maximum_execution_seconds": 3600, "execution_location": "local",
                 "admitted_unix": 1000, "deadline_unix": 4600}
    receipt = {"status": "complete_matrix_analysis_verified", "summary_sha256": "summary",
               "verified_new_tasks": 24, "independent_counts_checked": 29205}
    result = {"status": "complete_verified_five_model_decoder_matrix", "independent_counts_checked": 29205}
    validate(admission, receipt, result)
    mutations = [
        ("admission", "status", "pending"), ("admission", "plan_sha256", "changed"),
        ("admission", "parent_summary_sha256", "changed"),
        ("admission", "complete_decoder_grid_required", False),
        ("admission", "models", ["VLR_T1_29_LAST"]),
        ("admission", "scientific_decision", ""), ("admission", "maximum_execution_seconds", float("inf")),
        ("admission", "deadline_unix", 5000), ("receipt", "verified_new_tasks", 23),
        ("receipt", "independent_counts_checked", 100), ("result", "status", "partial"),
        ("result", "independent_counts_checked", 100),
    ]
    rejected = 0
    for category, field, value in mutations:
        fixture = {"admission": deepcopy(admission), "receipt": deepcopy(receipt), "result": deepcopy(result)}
        fixture[category][field] = value
        try:
            validate(fixture["admission"], fixture["receipt"], fixture["result"])
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError((category, field, "unsafe admission accepted"))
    server = {**admission, "execution_location": "server", "usd_per_hour": 0.6,
              "maximum_cost_usd": 0.6, "reserved_balance_usd": 1, "observed_balance_usd": 1.6}
    validate(server, receipt, result)
    for field, value in (("usd_per_hour", 1), ("observed_balance_usd", 1.2), ("maximum_cost_usd", 0.1)):
        altered = {**server, field: value}
        try:
            validate(altered, receipt, result)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError((field, "unsafe server budget accepted"))
    assert rejected == 15
    assert not (HERE / "Admission.json").exists()
    assert not (STORE / "Cache").exists()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-B", "-X", "utf8", str(HERE / "encode.py"), "VLR_T0_29_LAST",
               "--deadline", str(time.time() + 60)]
    observed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=40)
    with (folder / "Expected_Encoder_Refusal.log").open("x", encoding="utf-8") as stream:
        stream.write(observed.stdout + observed.stderr)
    assert observed.returncode != 0
    assert "Seed-29 inference is not admitted" in observed.stderr
    assert not (STORE / "Cache").exists() and not (HERE / "Greedy_01").exists()
    plan = read(HERE / "Plan.json")
    assert plan["expected_final_prediction_count"] == 2 * 531 * 11
    assert len(plan["candidate_order"]) == 11
    save(folder / "Verification.json", {
        "status": "admission_controls_and_real_cli_refusal_passed", "utc": utc(),
        "unsafe_admissions_rejected": rejected, "valid_local_and_server_fixtures_accepted": 2,
        "real_encoder_refused_without_admission": True, "encoder_exit_code": observed.returncode,
        "cache_created": False, "new_gpu_inference": False, "new_destination_predictions": False,
        "plan_sha256": sha(HERE / "Plan.json"), "code_sha256": sha(HERE / "test_admission.py"),
        "log_sha256": sha(folder / "Expected_Encoder_Refusal.log"),
        "scope": "Input preparation and execution admission only; numerical seed-29 inference remains untested.",
    })
    print({"status": "passed", "rejected_unsafe_fixtures": rejected,
           "actual_encoder_refusal": observed.returncode, "gpu_inference": False}, flush=True)


if __name__ == "__main__":
    main()
