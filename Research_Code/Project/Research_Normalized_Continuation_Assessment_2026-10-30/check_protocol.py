"""Validate copied numerical protocol and normalized checkpoint gates before any final model exists."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from common import ROOT, HERE, read, sha
from bind_normalized import validate
from data_tasks import Destination, positions


def main(destination_root=None):
    native = ROOT / "Research_Matched_Continuation_Assessment_2026-10-30"
    plan = read(HERE / "Plan.json")
    old = read(native / "Plan.json")
    same = ("source", "destination", "primary_decoder", "secondary_candidate_order", "secondary_selection",
            "secondary_source_decoder", "beam_size", "penalty", "precision", "external_lm", "horizon",
            "runtime_source", "runtime_code_sha256", "training_endpoint", "bootstrap_resamples")
    for name in same:
        assert plan[name] == old[name], name
    assert plan["parent_native_assessment_plan_sha256"] == sha(native / "Plan.json")
    unchanged = ("common.py", "runtime_bridge.py", "data_tasks.py", "execution.py", "run_task.py",
                 "verify_task.py", "select_development.py")
    for name in unchanged:
        assert sha(HERE / name) == sha(native / name)
    lock = read(HERE / "Code_Lock.json")
    assert lock["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    for name, digest in lock["code_sha256"].items():
        assert sha(HERE / name) == digest
    for name, digest in lock["external_code_sha256"].items():
        assert sha(ROOT / name) == digest
    cohorts = {}
    for stage in ("development", "assessment"):
        data = Destination(stage, destination_root or ROOT, ROOT)
        assigned = [position for index in range(8)
                    for position in positions(data, f"{stage}_{index:02}", "population")]
        assert sorted(assigned) == list(range(len(data)))
        cohorts[stage] = {"clips": len(data), "unique_positions": len(set(assigned))}
    source = ROOT / "Research_Vast_Source_2026-10-29/Normalized_Production_01/Training_Verification_01/Lock.json"
    if not source.exists():
        source = ROOT / "Normalized_Production_01/Training_Verification_01/Lock.json"
    assert plan["normalized_training_plan_sha256"] == sha(source.parent.parent / "Plan.json")
    training = read(source)
    failures = []
    # Explicit synthetic receipt fixtures exercise validation; they are not final-model evidence.
    for arm, alias in plan["models"].items():
        assert alias in ("full", "fixed_time")
        identity = training["identities"][alias]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        assert digest == training["identity_sha256"][alias]
        checkpoint_hash = "synthetic-fixture-not-a-checkpoint"
        verification = {"identity_sha256": digest, "arm": alias,
                        "status": "normalized_training_records_checkpoint_and_strict_load_verified",
                        "native_strict_load": True, "batchnorm_buffers_unchanged": True,
                        "source_checkpoint_stable": True, "checkpoint_sha256": checkpoint_hash,
                        "records": {"training": {"updates": 6000, "exposure": {"presentations": 48000}}}}
        files = {name: {"sha256": checkpoint_hash, "bytes": 10} for name in ("model", "checkpoint", "receipts")}
        backup = {"verified_members": 1, "local_files": deepcopy(files),
                  "proof": {"status": "terminal_training_files_packaged", "arm": alias,
                            "identity_sha256": digest, "updates": 6000, "training_endpoint_reached": True,
                            "independent_training_audit_passed": True,
                            "independent_training_audit_required": False, "files": files}}
        assert validate(alias, identity, 6000, verification, backup, training) == digest

        def reject(label, value_identity=identity, step=6000, value_verify=verification, value_backup=backup):
            try:
                validate(alias, value_identity, step, value_verify, value_backup, training)
            except (AssertionError, ValueError, KeyError):
                failures.append(arm + ":" + label)
            else:
                raise AssertionError("Invalid binding accepted: " + label)

        reject("incomplete_step", step=500)
        reject("native_receipt", value_verify={**verification,
               "status": "native_training_records_checkpoint_and_strict_load_verified"})
        reject("wrong_arm", value_verify={**verification, "arm": "whole_word"})
        reject("unverified_checkpoint", value_verify={**verification, "native_strict_load": False})
        reject("missing_backup_members", value_backup={**backup, "verified_members": 0})
        altered = deepcopy(identity)
        altered["objective"] = "matched_" + alias
        reject("native_objective_identity", value_identity=altered)
        altered_backup = deepcopy(backup)
        altered_backup["local_files"]["checkpoint"]["sha256"] = "different"
        reject("backup_mismatch", value_backup=altered_backup)
    value = {"utc": datetime.now(timezone.utc).isoformat(),
             "status": "normalized_plan_same_numerical_protocol_and_binding_cases_verified",
             "unchanged_runtime_modules": list(unchanged), "identical_plan_fields": list(same),
             "actual_manifest_cohorts": cohorts, "synthetic_bad_binding_cases_rejected": failures,
             "actual_final_normalized_binding_tested": False, "gpu_assessment_started": False,
             "checkpoint_or_reference_substitution": False}
    (HERE / "Protocol_Check.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination-root", type=Path)
    arguments = parser.parse_args()
    main(arguments.destination_root)
