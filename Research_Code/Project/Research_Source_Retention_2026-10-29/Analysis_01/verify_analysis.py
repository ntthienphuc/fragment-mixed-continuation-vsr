"""Exercise the analysis on real completed evidence and deliberately invalid copies."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import zipfile

import collect
from summarize import paired, save


def edited_archive(original, destination, edit):
    with zipfile.ZipFile(original) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if name != "Manifest.json"}
    prefix = next(name.rsplit("/", 1)[0] for name in files if name.endswith("/Predictions.jsonl"))
    edit(files, prefix)
    for name in ("Complete.json", "Verification.json"):
        key = prefix + "/" + name
        if key not in files:
            continue
        receipt = json.loads(files[key])
        receipt["predictions_sha256"] = collect.digest(files[prefix + "/Predictions.jsonl"])
        receipt["identity_sha256"] = collect.digest(files[prefix + "/Identity.json"])
        files[key] = json.dumps(receipt).encode()
    manifest = {name: {"bytes": len(data), "sha256": collect.digest(data)} for name, data in files.items()}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("Manifest.json", json.dumps(manifest))


def alter_record(files, prefix, action):
    path = prefix + "/Predictions.jsonl"
    records = [json.loads(line) for line in files[path].splitlines()]
    action(records)
    files[path] = ("\n".join(json.dumps(row) for row in records) + "\n").encode()


def wrong_counts(files, prefix):
    def change(records):
        records[0]["predictions"]["beam_c0.5"]["counts"]["errors"] += 1
    alter_record(files, prefix, change)


def wrong_position(files, prefix):
    def change(records):
        records[1] = deepcopy(records[0])
    alter_record(files, prefix, change)


def wrong_checkpoint(files, prefix):
    path = prefix + "/Identity.json"
    identity = json.loads(files[path])
    identity["registry_entry"]["checkpoint_sha256"] = "0" * 64
    files[path] = json.dumps(identity).encode()


def toy(clip, group, errors, units):
    return {"clip_id": clip, "source_id": group,
            "predictions": {"frozen": {"counts": {"errors": errors, "units": units}}}}


def main():
    output = collect.HERE / "Validation_01"
    output.mkdir(exist_ok=False)
    parent, assessment = collect.plans()
    metadata, pixels = collect.metadata("lrs2")
    task_name = "EN_B0_00"
    task = assessment["tasks"][task_name]
    original_resolver = collect.archive_for
    original = original_resolver(task_name)
    original_sha = collect.digest(original.read_bytes())
    records, provenance = collect.shard(task_name, task, parent, metadata, pixels)
    assert len(records) == 622
    rejected = []
    cases = [
        ("missing_independent_replay", lambda files, prefix: files.pop(prefix + "/Verification.json")),
        ("wrong_edit_counts_despite_resealed_hashes", wrong_counts),
        ("duplicate_position_despite_resealed_hashes", wrong_position),
        ("different_checkpoint_despite_resealed_hashes", wrong_checkpoint),
    ]
    try:
        for name, edit in cases:
            fixture = output / (name + ".zip")
            edited_archive(original, fixture, edit)
            collect.archive_for = lambda task_name, path=fixture: path
            try:
                collect.shard(task_name, task, parent, metadata, pixels)
            except (ValueError, KeyError) as error:
                rejected.append({"case": name, "reason": str(error)})
            else:
                raise AssertionError("Invalid evidence was accepted: " + name)
    finally:
        collect.archive_for = original_resolver
    assert collect.digest(original.read_bytes()) == original_sha
    left = [toy("a1", "A", 2, 1), toy("a2", "A", 0, 9), toy("b1", "B", 1, 1)]
    right = [toy("a1", "A", 0, 1), toy("a2", "A", 1, 9), toy("b1", "B", 0, 1)]
    estimated = paired(left, right, "frozen", "frozen", "known_two_source_example")
    assert math.isclose(estimated["reference_minus_comparison_pp"], 200 / 11)
    assert estimated["groups"] == 2
    assert estimated["ci95_low_pp"] == 10 and estimated["ci95_high_pp"] == 100
    reverse = paired(right, left, "frozen", "frozen", "known_two_source_example")
    assert math.isclose(reverse["reference_minus_comparison_pp"], -200 / 11)
    assert reverse["ci95_low_pp"] == -100 and reverse["ci95_high_pp"] == -10
    save(output / "Verification.json", {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "real_shard_recount_and_adversarial_checks_passed",
        "real_clips_recounted": len(records), "prediction_variants_recounted": 2 * len(records),
        "source_groups_in_complete_lrs2": len({row["source_id"] for row in metadata}),
        "rejected": rejected, "known_source_cluster_case": estimated,
        "original_archive_unchanged": True, "original": provenance,
        "whole_matrix_evaluated": False,
        "code_sha256": {path.name: collect.digest(path.read_bytes()) for path in collect.HERE.glob("*.py")},
    })
    print(json.dumps({"status": "analysis_verified", "real_clips": len(records), "negative_cases": rejected}))


if __name__ == "__main__":
    main()
