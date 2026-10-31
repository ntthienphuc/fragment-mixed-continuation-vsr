"""Restore only complete, locally verified short-only evidence for final analysis."""

import json
import hashlib
from pathlib import Path
import sys
import zipfile

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE))
from collect import sha, verify_zip


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_file(item):
    path = Path(item["local"])
    assert path.stat().st_size == item["bytes"]
    assert sha(path) == item["sha256"], "Backup content changed: " + str(path)
    return path


def binding_metadata(terminal, training_metadata):
    if terminal.get("execution_host") == "verified_split_hosts":
        return split_binding_metadata(terminal, training_metadata)
    if terminal.get("execution_host") != "private_windows":
        with zipfile.ZipFile(training_metadata) as archive:
            return {name: archive.read("Assessment/Binding/" + name)
                    for name in ("Binding.json", "model_registry.json")}
    receipt_path = PHASE / "Private_Metadata_Backup.json"
    assert sha(receipt_path) == terminal["private_metadata_receipt_sha256"]
    receipt = read(receipt_path)
    private_metadata = checked_file(receipt)
    assert verify_zip(private_metadata) == receipt["verified_members"]
    with zipfile.ZipFile(private_metadata) as archive, zipfile.ZipFile(training_metadata) as training:
        for name in ("Plan.json", "Code_Lock.json"):
            assert archive.read("Assessment/" + name) == (PHASE / "Assessment" / name).read_bytes()
        assert archive.read("Private_Assessment/Training_Evidence/Training_Verification.json") == training.read(
            "Training_Verification.json")
        private_terminal = json.loads(archive.read("Private_Assessment/Terminal.json"))
        assert private_terminal["status"] == "all_24_shards_verified"
        assert private_terminal["finished"] == terminal["finished"]
        return {name: archive.read("Assessment/Binding_Private/" + name)
                for name in ("Binding.json", "model_registry.json")}


def split_binding_metadata(terminal, training_metadata):
    receipt_path = PHASE / "Split_Metadata_Backup.json"
    assert sha(receipt_path) == terminal["split_metadata_receipt_sha256"]
    receipt = read(receipt_path)
    metadata = checked_file(receipt)
    assert verify_zip(metadata) == receipt["verified_members"]
    with zipfile.ZipFile(metadata) as archive, zipfile.ZipFile(training_metadata) as training:
        for name in ("Plan.json", "Code_Lock.json"):
            assert archive.read("Assessment/" + name) == (PHASE / "Assessment" / name).read_bytes()
        assert archive.read("Training_Verification.json") == training.read("Training_Verification.json")
        matrix = json.loads(archive.read("Split/Completed_Matrix.json"))
        assert matrix["status"] == "all_24_shards_verified_across_disjoint_hosts"
        assert matrix["finished"] == terminal["finished"]
        assert matrix["task_execution_hosts"] == terminal["task_execution_hosts"]
        assert matrix["overlap_shard_in_aggregation"] is False
        protocol = json.loads(archive.read("Split/Comparability_Protocol.json"))
        private = json.loads(archive.read("Split/Private_Subset_Completion.json"))
        paid = json.loads(archive.read("Split/Paid_Backup_Complete.json"))
        assert protocol["private_tasks"] == [f"source_{index:02}" for index in range(6)]
        assert protocol["paid_tasks"] == [f"source_{index:02}" for index in range(6, 16)] + [
            f"assessment_{index:02}" for index in range(8)]
        assert protocol["score_absolute_tolerance"] == 1e-4
        assert paid["status"] == "paid_subset_full_backup_and_cross_host_parity_verified"
        assert private["status"] == "six_private_shards_verified_queue_closed_at_handoff"
        assert private["interrupted_inference_tasks"] == 0
        joined = private["finished"] + paid["finished"]
        assert len(joined) == len({item["task"] for item in joined}) == 24
        assert {item["task"]: item for item in joined} == {
            item["task"]: item for item in terminal["finished"]}
        for name, cohort, host in (("private_tasks", private, "WORKSTATION-TT"),
                                    ("paid_tasks", paid, "Vast58265651")):
            assert {item["task"] for item in cohort["finished"]} == set(protocol[name])
            assert all(terminal["task_execution_hosts"][task] == host for task in protocol[name])
        comparisons = [*paid["probes"], paid["extended"]]
        assert sorted(item["clips"] for item in comparisons) == [6, 6, 703]
        for comparison in comparisons:
            assert comparison["status"] == "cross_host_predictions_and_protocol_match"
            assert comparison["tokens_text_counts_pixels_trace_exact"]
            assert comparison["score_atol"] == 1e-4 and comparison["max_score_error"] <= 1e-4
        for name, member, expected in (
            ("Private_Metadata_Backup.json", "Private_Assessment/Private_Subset_Completion_01.json", private),
            ("Metadata_Backup.json", "Parallel_Assessment_01/Population_Terminal.json", None)):
            source_receipt = json.loads(archive.read("Split/" + name))
            source_metadata = checked_file(source_receipt)
            assert verify_zip(source_metadata) == source_receipt["verified_members"]
            with zipfile.ZipFile(source_metadata) as source:
                actual = json.loads(source.read(member).decode("utf-8-sig"))
                if expected is not None:
                    assert actual == expected
                else:
                    assert actual["status"] == "exclusive_paid_subset_and_overlap_verified"
                    assert actual["finished"] == paid["finished"]
        return {name: archive.read("Assessment/Binding_Private/" + name)
                for name in ("Binding.json", "model_registry.json")}


def restore():
    terminal = read(PHASE / "Assessment_Backup_Complete.json")
    assert terminal["status"] == "training_and_24_shards_backed_up"
    assert terminal["source_clips"] == 11307 and terminal["destination_clips"] == 1167
    expected = {f"source_{index:02}" for index in range(16)}
    expected |= {f"assessment_{index:02}" for index in range(8)}
    assert len(terminal["finished"]) == 24
    assert {item["task"] for item in terminal["finished"]} == expected
    for stage, count in (('source', 11307), ('assessment', 1167)):
        assert sum(item['clips'] for item in terminal['finished']
                   if item['task'].startswith(stage + '_')) == count
    training = read(PHASE / "Training_Backup_Verified.json")
    assert training["status"] == "all_training_files_locally_hash_verified"
    assert training["updates"] == 6000
    for item in training["files"].values():
        checked_file(item)
    receipts = Path(training["files"]["receipts"]["local"])
    assert verify_zip(receipts) == training["verified_members"]
    metadata_receipt = read(PHASE / "Final_Metadata_Backup.json")
    metadata = checked_file(metadata_receipt)
    assert verify_zip(metadata) == metadata_receipt["verified_members"]
    archives = []
    for item in terminal["finished"]:
        receipt = read(PHASE / "Shard_Backups" / (item["task"] + ".json"))
        for field in ("sha256", "bytes", "clips", "task"):
            assert receipt[field] == item[field]
        archive = checked_file(receipt)
        assert verify_zip(archive) == receipt["verified_members"]
        with zipfile.ZipFile(archive) as bundle:
            verification = json.loads(bundle.read('Run/Verification.json'))
            content = bundle.read('Run/Predictions.jsonl')
            assert content.endswith(b'\n')
            assert hashlib.sha256(content).hexdigest() == verification['predictions_sha256']
            records = [json.loads(line) for line in content.splitlines()]
            assert len(records) == verification['clips'] == item['clips']
            assert len({row['clip_id'] for row in records}) == len(records)
            assert verification['status'] == 'all_records_recounted_and_fresh_replay_passed'
            assert verification['model'] == 'MATCHED_SO17' and verification['readout'] == 'common'
            assert verification['task'] == item['task'] and verification['mode'] == 'population'
            assert verification['code_lock_sha256'] == sha(PHASE / 'Assessment/Code_Lock.json')
            assert verification['assessment_plan_sha256'] == sha(PHASE / 'Assessment/Plan.json')
        archives.append((item["task"], archive))
    binding_files = binding_metadata(terminal, metadata)
    registry_sha = hashlib.sha256(binding_files['model_registry.json']).hexdigest()
    for task, path in archives:
        with zipfile.ZipFile(path) as bundle:
            proof = json.loads(bundle.read('Run/Verification.json'))
            assert proof['registry_sha256'] == registry_sha, task
    with zipfile.ZipFile(metadata) as archive:
        for name in ("Plan.json", "Code_Lock.json"):
            assert archive.read("Assessment/" + name) == (PHASE / "Assessment" / name).read_bytes()
        proof = json.loads(archive.read("Training_Verification.json"))
        assert proof["status"] == "native_training_records_checkpoint_and_strict_load_verified"
        assert proof["updates"] == 6000 and proof["exposure"]["presentations"] == 48000
        assert proof["native_strict_load"] and proof["batchnorm_buffers_unchanged"]
        binding = json.loads(binding_files["Binding.json"])
        assert binding["checkpoint_sha256"] == training["files"]["model"]["sha256"]
        assert binding["verification_sha256"] == __import__("hashlib").sha256(
            archive.read("Training_Verification.json")).hexdigest()
        output = PHASE / "Analysis_Inputs_01"
        output.mkdir(exist_ok=False)
        folder = output / "Binding"
        folder.mkdir()
        for name in ("Binding.json", "model_registry.json"):
            (folder / name).write_bytes(binding_files[name])
        (output / "Training_Verification.json").write_bytes(archive.read("Training_Verification.json"))
    for task, path in archives:
        folder = output / "Runs" / task
        folder.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("Run/"):
                    leaf = name.removeprefix("Run/")
                    assert "/" not in leaf and "\\" not in leaf and leaf
                    (folder / leaf).write_bytes(archive.read(name))
    provenance = {
        "status": "all_24_backups_rehashed_and_restored",
        "training_receipt_sha256": sha(PHASE / "Training_Backup_Verified.json"),
        "assessment_receipt_sha256": sha(PHASE / "Assessment_Backup_Complete.json"),
        "metadata_sha256": metadata_receipt["sha256"],
        "archives": {task: sha(path) for task, path in archives},
        "execution_host": terminal.get("execution_host", "paid_linux"),
        "private_metadata_receipt_sha256": terminal.get("private_metadata_receipt_sha256"),
        "split_metadata_receipt_sha256": terminal.get("split_metadata_receipt_sha256"),
    }
    (output / "Complete.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance))


if __name__ == "__main__":
    restore()
