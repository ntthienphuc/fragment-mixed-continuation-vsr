"""Collect complete matched populations without loading a model or selecting on test data."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent / "Assessment"
sys.path.insert(0, str(PHASE))
from common import ROOT, load_module, read, rows, sha
from data_tasks import Destination, candidates, positions
from verify_task import prediction_rows, verify_records


class Metadata:
    stage = "source"

    def __init__(self):
        original = load_module("matched_source_metadata", ROOT /
                               "Research_Source_Retention_2026-10-29/Analysis_01/collect.py")
        self.rows, self.pixels = original.metadata("vlr")

    def __len__(self):
        return len(self.rows)

    def metadata(self, position):
        row = self.rows[position]
        return {"clip_id": row["clip_id"], "source_id": row["source_id"], "frames": row["frames"],
                "reference": row["text"], "pixel_sha256": self.pixels[row["clip_id"]]["pixel_sha256"]}


def check_lock():
    lock = read(PHASE / "Code_Lock.json")
    assert lock["assessment_plan_sha256"] == sha(PHASE / "Plan.json")
    for relative, expected in lock["code_sha256"].items():
        assert sha(PHASE / relative) == expected
    for relative, expected in lock["external_code_sha256"].items():
        assert sha(ROOT / relative) == expected
    return read(PHASE / "Plan.json")


def population_metadata(stage, destination_root=ROOT):
    plan = check_lock()
    dataset = Metadata() if stage == "source" else Destination(stage, destination_root, ROOT)
    spec = plan["source"] if stage == "source" else plan["destination"][stage]
    assert len(dataset) == spec["clips"]
    cohort = [dataset.metadata(index) for index in range(len(dataset))]
    assert len({row["clip_id"] for row in cohort}) == len(cohort)
    assert sum(len(row["reference"].split()) for row in cohort) == spec["reference_units"]
    assert len({row["source_id"] for row in cohort}) == spec["source_groups"]
    return dataset, spec


def check_shard(folder, task, model, stage, readout, choices, binding, selection, dataset):
    folder = Path(folder)
    assert not (folder / "Failure.json").exists(), "A failed attempt cannot enter aggregation"
    proof = read(folder / "Verification.json")
    start = read(folder / "Start.json")
    complete = read(folder / "Complete.json")
    identity = read(folder / "Identity.json")
    registry = read(Path(binding) / "model_registry.json")
    entry = registry["models"][model]
    assert proof["status"] == "all_records_recounted_and_fresh_replay_passed"
    assert complete["status"] == "complete_pending_independent_verification"
    expected = {"task": task, "model": model, "mode": "population", "stage": stage,
                "readout": readout, "candidates": choices, "registry_sha256": sha(Path(binding) /
                "model_registry.json"), "assessment_plan_sha256": sha(PHASE / "Plan.json"),
                "code_lock_sha256": sha(PHASE / "Code_Lock.json")}
    for field, value in expected.items():
        assert start[field] == proof[field] == value, (task, field)
    for field in ("task", "model", "mode"):
        assert complete[field] == expected[field]
    assert proof["device"] == start["device"] == identity["device"].split(":")[0]
    assert start["selection_sha256"] == (sha(selection) if selection else None)
    assert identity["model_id"] == model and identity["registry_entry"] == entry
    assert identity["registry_sha256"] == expected["registry_sha256"]
    for filename, field in (("Predictions.jsonl", "predictions_sha256"), ("Identity.json", "identity_sha256")):
        assert sha(folder / filename) == proof[field] == complete[field]
    assert sha(folder / "Start.json") == complete["start_sha256"]
    selected = positions(dataset, task, "population")
    records = prediction_rows(folder / "Predictions.jsonl")
    assert len(selected) == complete["clips"] == proof["clips"] == len(records)
    checks = verify_records(dataset, selected, records, choices, model, "population")
    assert proof["independent_count_checks"] == checks
    offsets = sorted({0, len(records) // 3, 2 * len(records) // 3, len(records) - 1})
    replay_pairs = [(row["position"], row["candidate"]) for row in proof["replays"]]
    assert replay_pairs == [(records[index]["position"], choice) for index in offsets for choice in choices]
    assert all(0 <= row["score_error"] <= 1e-4 for row in proof["replays"])
    stable_fields = ("registry_entry", "asset_sha256", "source_files", "torch", "numpy", "device",
                     "matched_bridge_sha256", "native_graph_binding")
    signature = json.dumps({field: identity[field] for field in stable_fields}, sort_keys=True)
    provenance = {"task": task, "folder": str(folder.resolve()), "clips": len(records),
                  "files": {name: sha(folder / name) for name in
                            ("Verification.json", "Start.json", "Identity.json", "Complete.json", "Predictions.jsonl")}}
    return records, provenance, signature


def validate_selection(model, binding, selection, development_shards, destination_root):
    from select_development import choose

    assert development_shards is not None, "Selected readout requires complete development evidence"
    predictions, development = collect("development", model, "grid", binding, development_shards,
                                      destination_root=destination_root)
    selected = read(selection)
    spec = read(PHASE / "Plan.json")["destination"]["development"]
    source = Path(destination_root) / spec["manifest"]
    cohort = [{**row, "stage": row["split"]} for row in rows(source)]
    result = choose(cohort, predictions, read(PHASE / "Plan.json")["secondary_candidate_order"])
    for field, value in result.items():
        assert selected[field] == value, "Development selection differs: " + field
    assert selected["development_manifest_sha256"] == sha(source)
    assert selected["selection_code_sha256"] == sha(PHASE / "select_development.py")
    assert selected["code_lock_sha256"] == sha(PHASE / "Code_Lock.json")
    assert selected["assessment_predictions_read"] is False
    for item in development["shards"]:
        saved = selected["bound_shards"][item["task"]]
        assert saved["predictions_sha256"] == item["files"]["Predictions.jsonl"]
        assert saved["verification_sha256"] == item["files"]["Verification.json"]
        assert saved["clips"] == item["clips"]
    return development


def collect(stage, model, readout, binding, shards, selection=None, destination_root=ROOT,
            development_shards=None):
    plan = check_lock()
    assert stage in ("source", "assessment", "development") and model in plan["models"]
    binding = Path(binding)
    proof = read(binding / "Binding.json")
    registry = read(binding / "model_registry.json")
    assert proof["model"] == model and proof["assessment_plan_sha256"] == sha(PHASE / "Plan.json")
    assert proof["registry_sha256"] == sha(binding / "model_registry.json")
    entry = registry["models"][model]
    assert entry["selected_training_step"] == 6000
    assert entry["checkpoint_sha256"] == proof["checkpoint_sha256"]
    assert entry["training_identity"]["steps"] == 6000
    assert entry["training_identity"]["selection"] == "fixed_final_step"
    choices = candidates(stage, readout, model, binding, selection)
    development = None
    if readout == "selected":
        development = validate_selection(model, binding, selection, development_shards, destination_root)
    dataset, spec = population_metadata(stage, destination_root)
    records = []
    evidence = []
    signatures = set()
    for index in range(16 if stage == "source" else 8):
        task = f"{stage}_{index:02}"
        chunk, provenance, signature = check_shard(Path(shards) / task, task, model, stage, readout,
                                                   choices, binding, selection, dataset)
        records.extend(chunk)
        evidence.append(provenance)
        signatures.add(signature)
    assert len(signatures) == 1, "Mixed runtimes require a separately verified comparability protocol"
    assert sorted(row["position"] for row in records) == list(range(len(dataset)))
    assert len({row["clip_id"] for row in records}) == len(dataset)
    ordered = sorted(records, key=lambda row: row["position"])
    return ordered, {"model": model, "stage": stage, "readout": readout, "candidates": choices,
                     "clips": len(ordered), "reference_units": spec["reference_units"],
                     "source_groups": spec["source_groups"], "shards": evidence,
                     "binding_sha256": sha(binding / "Binding.json"),
                     "registry_sha256": sha(binding / "model_registry.json"),
                     "selection_sha256": sha(selection) if selection else None,
                     "development_reverified": development}
