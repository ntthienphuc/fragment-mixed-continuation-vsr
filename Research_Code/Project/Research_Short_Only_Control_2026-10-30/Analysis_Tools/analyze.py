"""Join the completed short-only populations to locked, verified matched comparators."""

import csv
import json
from pathlib import Path

from collect_population import PHASE, ROOT, collect
from common import read, save, sha, utc
from paired_statistics import FIELDS, paired, totals
from export_tables import export

CONTROL = PHASE.parent
DECODER = "beam_c0.1"
NEW_MODEL = "MATCHED_SO17"
OLD_MODELS = ("MATCHED_FC17", "MATCHED_FT17", "MATCHED_WW17")
ANALYSIS_FILES = ('analyze.py', 'collect_population.py', 'paired_statistics.py',
                  'export_tables.py', 'restore_inputs.py')


def load_old(stage, lock, current):
    summary_path = Path(lock["summary"])
    assert sha(summary_path) == lock["summary_sha256"]
    summary = read(summary_path)
    assert summary["status"] == "complete_requested_matched_populations_recounted"
    assert summary["stage"] == stage and summary["readout"] == "common"
    csv_path = summary_path.parent / "Per_Clip_Counts.csv"
    assert sha(csv_path) == lock["per_clip_counts_sha256"]
    assert summary["output_sha256"][csv_path.name] == lock["per_clip_counts_sha256"]
    lookup = {row["clip_id"]: row for row in current}
    assert len(lookup) == len(current)
    models = {model: [] for model in OLD_MODELS}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            model = item["model"]
            assert model in models and item["stage"] == stage
            assert item["decoder"] == DECODER and item["readout"] == "common"
            identity = lookup[item["clip_id"]]
            assert item["source_id"] == identity["source_id"]
            assert int(item["position"]) == identity["position"]
            assert int(item["frames"]) == identity["frames"]
            counts = {field: int(item[field]) for field in FIELDS}
            assert counts["units"] == identity["predictions"][DECODER]["counts"]["units"]
            assert min(counts.values()) >= 0 and counts["units"] > 0
            assert counts["errors"] == sum(counts[field] for field in FIELDS[1:4])
            models[model].append({
                "clip_id": item["clip_id"], "source_id": item["source_id"],
                "predictions": {DECODER: {"counts": counts}},
            })
    for model, rows in models.items():
        assert len(rows) == len(current)
        assert {row["clip_id"] for row in rows} == set(lookup)
        result = totals(rows, DECODER)
        expected = next(item for item in summary["results"] if item["model"] == model)
        for field, value in result.items():
            assert abs(value - expected[field]) < 1e-10, (model, field)
    return models, summary


def runtime_comparison(summary, current_provenance):
    new_shard = current_provenance["shards"][0]
    current = read(Path(new_shard["folder"]) / "Identity.json")
    matches = {}
    shared = ("source_files", "torch", "numpy", "device", "precision", "tf32",
              "native_graph_binding", "reference_used_for_inference")
    previous_bridge = ROOT / "Research_Matched_Continuation_Assessment_2026-10-30/runtime_bridge.py"
    current_bridge = PHASE / "runtime_bridge.py"
    assert previous_bridge.read_text(encoding="utf-8") == current_bridge.read_text(encoding="utf-8")
    training_fields = ("native_graph", "dataset", "scope", "seed", "steps", "accumulation", "lr",
                       "weight_decay", "gradient_clip", "eval_every", "selection", "initializer_sha256",
                       "precision", "fused_adam", "tf32", "deterministic", "threads", "workers")
    for model in OLD_MODELS:
        archived = summary["provenance"][model]["shards"][0]
        path = Path(archived["folder"]) / "Identity.json"
        assert sha(path) == archived["files"]["Identity.json"]
        previous = read(path)
        for field in shared:
            assert previous[field] == current[field], "Readout runtime differs: " + field
        assert previous["matched_bridge_sha256"] == sha(previous_bridge)
        assert current["matched_bridge_sha256"] in (sha(previous_bridge), sha(current_bridge))
        for field in training_fields:
            assert previous["registry_entry"]["training_identity"][field] == current[
                "registry_entry"]["training_identity"][field], "Training setting differs: " + field
        for field in ("tokenizer", "units", "config", "base_checkpoint"):
            assert previous["asset_sha256"][field] == current["asset_sha256"][field]
        for field in ("config", "source", "tokenizer", "units", "base_checkpoint_sha256", "native_graph"):
            assert previous["registry_entry"][field] == current["registry_entry"][field]
        matches[model] = {
            "previous_identity_sha256": sha(path), "shared_fields": list(shared),
            "training_settings_checked": list(training_fields),
            "bridge_text_identical_after_line_ending_normalization": True,
            "previous_bridge_sha256": sha(previous_bridge), "new_bridge_sha256": sha(current_bridge),
        }
    return matches


def main():
    design = read(CONTROL / "Comparison_Plan.json")
    assert design["status"] == "analysis_design_locked_before_short_only_training"
    assert design["readout"] == DECODER and design["bootstrap_resamples"] == 10000
    inputs = CONTROL / "Analysis_Inputs_01"
    assert read(inputs / "Complete.json")["status"] == "all_24_backups_rehashed_and_restored"
    stages = {}
    export_inputs = {}
    for stage in ("source", "assessment"):
        current, evidence = collect(stage, NEW_MODEL, "common", inputs / "Binding", inputs / "Runs")
        old, archived = load_old(stage, design["old_evidence"][stage], current)
        comparable = runtime_comparison(archived, evidence)
        all_records = {**old, NEW_MODEL: current}
        comparisons = []
        for role, pair in (("primary", design["primary_comparison"]),
                           ("reference", design["reference_comparison"])):
            left, right = pair
            label = "|".join([stage, "common", left, right])
            comparisons.append({"role": role, "reference": left, "comparison": right,
                                **paired(all_records[left], all_records[right], DECODER, DECODER, label)})
        assert set(all_records) == {*OLD_MODELS, NEW_MODEL}
        export_inputs[stage] = (all_records, current, comparisons)
        stages[stage] = {
            "clips": len(current), "readout": "common", "decoder": DECODER,
            "results": {model: totals(rows, DECODER) for model, rows in all_records.items()},
            "comparisons": comparisons, "new_provenance": evidence,
            "archived_summary_sha256": design["old_evidence"][stage]["summary_sha256"],
            "runtime_comparison": comparable,
        }
    output = CONTROL / "Analysis_01"
    output.mkdir(exist_ok=False)
    exports = {}
    for stage, (models, current, comparisons) in export_inputs.items():
        exports[stage] = export(stage, models, current, comparisons, output / stage)
    result = {
        "utc": utc(), "status": "both_locked_populations_recounted_and_compared",
        "stages": stages, "new_model": NEW_MODEL,
        "comparison_plan_sha256": sha(CONTROL / "Comparison_Plan.json"),
        "training_verification_sha256": sha(inputs / "Training_Verification.json"),
        "conditional_on_fitted_checkpoints": True, "multiplicity_adjusted": False,
        "new_blind_test": False, "equivalence_margin_declared": False,
        "manuscript_updated": False,
        "exports": exports,
        "output_sha256": {path.relative_to(output).as_posix(): sha(path)
                          for path in sorted(output.rglob('*.csv'))},
        "code_sha256": {name: sha(Path(__file__).parent / name) for name in ANALYSIS_FILES},
    }
    save(output / "Summary.json", result)
    print(json.dumps({"status": result["status"],
                      "results": {stage: value["results"] for stage, value in stages.items()}}))


if __name__ == "__main__":
    main()
