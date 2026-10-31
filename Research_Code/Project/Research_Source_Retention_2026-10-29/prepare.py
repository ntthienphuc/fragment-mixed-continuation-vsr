"""Freeze complete cohorts and deterministic resource probes before recognition."""

from retention_common import ENGINE, HERE, INPUTS, PARENT, PREPARED_SHA256, ROOT
from retention_common import logical, native_path, prepared_plan, read, require, save, sha, source, utc


def main():
    prepared = prepared_plan()
    completion = PARENT / "Completion_Receipt.json"
    require(read(completion)["status"] == "complete_mechanism_transfer_verified", "Target matrix incomplete")
    backup = ROOT / "Research_Tailscale_Server_2026-10-28/Matrix_Complete.json"
    require(read(backup)["status"] == "full_53960_row_matrix_recounted_and_backed_up", "Backup incomplete")
    files = set(HERE.glob("*.py"))
    files.update((INPUTS / "source_loader.py", ENGINE / "confirmation_common.py"))
    files.add(ROOT / "Research_English_Transfer_Benchmark_2026-10-28/runtime_bridge.py")
    files.add(ROOT / "Research_English_Transfer_Benchmark_2026-10-28/english_common.py")
    files.add(ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py")
    files.update(native_path(model) for model in prepared["models"])
    files.update((ROOT / "Research_Phase_05_2026-10-13/phase5").rglob("*.py"))
    files.update((ROOT / "Research_Phase_05_2026-10-13/vendor/auto_avsr").rglob("*.py"))
    for entry in prepared["models"].values():
        registry = read(ROOT / entry["registry"])
        for field in ("factory_source", "safe_search_source", "horizon_source"):
            files.add(ROOT / registry[field])
    cohorts = {}
    for language in ("lrs2", "vlr"):
        with source(language) as dataset:
            ordered = sorted(range(len(dataset)), key=lambda position: (dataset.rows[position]["frames"], position))
            quantiles = {str(percent): ordered[(len(ordered) - 1) * percent // 100]
                         for percent in (0, 25, 50, 75, 95, 100)}
            models = [model for model, item in prepared["models"].items()
                      if item["identity"]["corpus"].lower() == language]
            require(len(models) == 7, "Expected seven fixed states per corpus")
            cohorts[language] = {
                "clips": len(dataset), "models": models,
                "reference_units": sum(len(row["text"].split()) for row in dataset.rows),
                "length_quantile_positions": quantiles,
                "probe_positions": sorted(set(quantiles.values())),
                "total_frames": sum(row["frames"] for row in dataset.rows),
                "complete_population_required": True,
            }
    save(HERE / "Plan.json", {
        "utc": utc(), "status": "frozen_source_retention_resource_probe_plan",
        "prepared_plan_sha256": PREPARED_SHA256,
        "cohorts": cohorts,
        "models": prepared["models"],
        "code_sha256": {logical(path): sha(path) for path in sorted(files)},
        "prerequisite_sha256": {logical(path): sha(path) for path in (completion, backup)},
        "full_evaluation_admitted": False,
        "test_based_tuning_allowed": False,
        "probe_results_are_population_estimates": False,
        "precision": "float32", "beam_size": 40, "external_lm": False,
        "horizon": "full_encoded_length", "penalty": 0,
    })
    print({"status": "plan_frozen", "corpora": cohorts, "locked_code_files": len(files)}, flush=True)


if __name__ == "__main__":
    main()
