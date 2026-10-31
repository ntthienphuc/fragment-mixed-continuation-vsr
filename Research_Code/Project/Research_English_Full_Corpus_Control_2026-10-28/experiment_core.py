"""Shared decoder contract for complete two-language T2/T4 experiments."""

import importlib.util
import math
from pathlib import Path
import sys

from mechanism_common import EN, ENGINE, HERE, ROOT, lines, read, sha


def cohort(plan, language, stage):
    entry = plan["cohorts"][language][stage]
    path = ROOT / entry["path"]
    assert sha(path) == entry["sha256"]
    rows = lines(path)
    assert len(rows) == entry["clips"]
    assert len({row["clip_id"] for row in rows}) == len(rows)
    assert all(row["language"] == language and row["split"] == stage for row in rows)
    return rows


def task_rows(plan, name):
    task = plan["tasks"][name]
    rows = cohort(plan, task["language"], task["stage"])
    positions = task["positions"]
    assert positions and len(set(positions)) == len(positions)
    assert all(type(position) is int and 0 <= position < len(rows) for position in positions)
    return task, [(position, rows[position]) for position in positions]


def validate_choices(plan, selection):
    assert selection["status"] == "all_mechanism_development_decoders_locked_before_assessment"
    assert selection["test_predictions_read"] is False
    assert set(selection["models"]) == set(plan["models"])
    registry = read(HERE / "model_registry.json")["models"]
    for model, result in selection["models"].items():
        expected = plan["cohorts"][registry[model]["language"]]["development"]
        assert result["clips"] == expected["clips"]
        assert result["reference_units"] == expected["reference_units"]
        assert result["independent_counts_checked"] == expected["clips"] * len(plan["candidate_order"])
        assert result["decoder"] in plan["candidate_order"]


def candidates(plan, task, selection=None):
    if task["stage"] == "development":
        assert selection is None
        return list(plan["candidate_order"])
    assert task["stage"] == "assessment"
    assert selection is not None
    validate_choices(plan, selection)
    chosen = selection["models"][task["model"]]["decoder"]
    return list(dict.fromkeys([chosen, plan["anchor"]]))


def parent_decoder():
    """Reuse the unchanged parent grid implementation for both native dictionaries."""
    name = "mechanism_parent_decoder"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, EN / "runtime_bridge.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        original_paths = list(sys.path)
        try:
            sys.path.insert(0, str(EN))
            specification.loader.exec_module(module)
        finally:
            sys.path[:] = original_paths
    return sys.modules[name].decode


def decode(runtime, encoded, candidate, plan):
    return parent_decoder()(runtime, encoded, candidate, plan)


def native_bridge():
    name = "mechanism_registered_runtime_bridge"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, HERE / "runtime_bridge.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


def pixels(row, locations=None):

    if row["language"] == "en" and row["split"] == "assessment" and locations is not None:
        original = Path(row["input"]["path"].replace("\\", "/")).name
        assert original == Path(row["clip_id"]).stem + ".npz"
        folder = Path(locations["wild_root"])
        folder = folder if folder.is_absolute() else ROOT / folder
        row = {**row, "input": {**row["input"], "path": str(folder / original)}}
    return native_bridge().pixels_for(row)


def independent_greedy(logp, bias, blank, eos):
    """Scalar argmax and run allocation, separate from the parent's NumPy collapse."""
    labels = []
    for frame in logp:
        winner = max(range(len(frame)), key=lambda token: float(frame[token]) + (bias if token == blank else 0))
        labels.append(winner)
    tokens = []
    for index, token in enumerate(labels):
        if index > 0 and token == labels[index - 1]:
            continue
        if token not in {blank, eos}:
            tokens.append(token)
    return tokens


def independent_counts(reference, hypothesis):
    sys.path.insert(0, str(ROOT / "Research_Interval_Transfer_2026-10-27"))
    from verify_probe import counts

    return counts(reference, hypothesis)


def check_prediction(row, result, candidate, classes):
    assert isinstance(result["tokens"], list)
    assert all(type(token) is int and 0 < token < classes - 1 for token in result["tokens"])
    assert independent_counts(row["reference"], result["text"]) == result["counts"]
    assert type(result["seconds"]) in {int, float} and math.isfinite(result["seconds"])
    assert result["seconds"] >= 0
    if candidate.startswith("beam_c"):
        assert result["cap"] == row["frames"]
        assert math.isfinite(result["score"])
        trace = result["trace"]
        assert trace["implementation"] == "upstream_BatchBeamSearch"
        assert trace["step_cap"] == row["frames"] and trace["beam_size"] == 40 and trace["penalty"] == 0
        assert trace["ctc_weight"] == float(candidate.removeprefix("beam_c"))
    else:
        assert candidate.startswith("greedy_b") and result["search_calls"] == 0
