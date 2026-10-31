"""Freeze exhaustive shard coverage, selected readouts and resource limits."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from retention_common import ROOT, checked_plan, logical, save, sha, source, utc


def main():
    parent = checked_plan()
    tasks = {}
    for corpus in ("lrs2", "vlr"):
        with source(corpus) as dataset:
            count = 2 if corpus == "lrs2" else 16
            bins = [[] for _ in range(count)]
            loads = [0] * count
            ordered = sorted(range(len(dataset)), key=lambda index: (-dataset.rows[index]["frames"], index))
            for index in ordered:
                selected = min(range(count), key=lambda item: (loads[item], item))
                bins[selected].append(index)
                loads[selected] += dataset.rows[index]["frames"]
            assert sorted(index for group in bins for index in group) == list(range(len(dataset)))
            for model in parent["cohorts"][corpus]["models"]:
                for shard, positions in enumerate(bins):
                    task = model + "_" + str(shard).zfill(2)
                    tasks[task] = {"model": model, "corpus": corpus, "positions": sorted(positions),
                                   "frames": loads[shard], "candidates": parent["models"][model]["execution_candidates"]}
    files = sorted(HERE.glob("*.py"))
    save(HERE / "Plan.json", {
        "utc": utc(), "status": "complete_source_retention_matrix_locked",
        "parent_plan_sha256": sha(HERE.parent / "Plan.json"),
        "code_sha256": {logical(path): sha(path) for path in files},
        "tasks": tasks, "models": list(parent["models"]),
        "cohort_sizes": {key: value["clips"] for key, value in parent["cohorts"].items()},
        "assignment_uses_predictions": False, "source_test_tuning_allowed": False,
        "maximum_task_seconds": 14400, "requires_real_device_probe_before_launch": True,
        "publication_requires_complete_population_and_independent_verification": True,
    })
    print({"tasks": len(tasks), "model_states": len(parent["models"]), "complete_populations": True})


if __name__ == "__main__":
    main()
