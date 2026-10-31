"""Unchanged complete-clip populations with deterministic task assignments."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

from common import HERE, ROOT, load_module, read, rows, sha


class Source:
    stage = "source"

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def metadata(self, position):
        row = self.data.rows[position]
        return {"clip_id": row["clip_id"], "source_id": row["source_id"], "frames": row["frames"],
                "reference": row["text"], "pixel_sha256": self.data.pixels[row["clip_id"]]["pixel_sha256"]}

    def load(self, position):
        _, pixels = self.data.load(position)
        return self.metadata(position), pixels


class Destination:
    def __init__(self, stage, input_root, root):
        self.stage = stage
        self.root = Path(root)
        self.input_root = Path(input_root)
        plan = read(HERE / "Plan.json")["destination"]
        expected = plan[stage]
        path = self.input_root / expected["manifest"]
        assert sha(path) == expected["sha256"]
        self.rows = rows(path)
        pixel_path = self.input_root / plan["pixel_index"]
        assert sha(pixel_path) == plan["pixel_index_sha256"]
        pixels = [row for row in rows(pixel_path) if row["split"] == stage]
        self.pixels = {row["clip_id"]: row for row in pixels}
        assert len(self.rows) == len(self.pixels) == len(pixels) == expected["clips"]
        assert {row["clip_id"] for row in self.rows} == set(self.pixels)
        assert sum(len(row["reference"].split()) for row in self.rows) == expected["reference_units"]
        assert len({row["source_id"] for row in self.rows}) == expected["source_groups"]

    def __len__(self):
        return len(self.rows)

    def metadata(self, position):
        row = self.rows[position]
        return {"clip_id": row["clip_id"], "source_id": row["source_id"], "frames": row["frames"],
                "reference": row["reference"], "pixel_sha256": self.pixels[row["clip_id"]]["gray_sha256"]}

    def load(self, position):
        sys.path.insert(0, str(self.root / "Research_Server_Confirmation_2026-10-19"))
        from data_io import load_pixels

        row = self.rows[position]
        pixel = self.pixels[row["clip_id"]]
        assert pixel["media_sha256"] == row["input"]["sha256"]
        assert pixel["frames"] == row["frames"] and pixel["source_id"] == row["source_id"]
        metadata = {key: value for key, value in row.get("metadata", {}).items() if key != "pixels_sha256"}
        item = {**row, "input": {**row["input"], "color_space": "gray"}, "metadata": metadata}
        pixels = load_pixels(item, root=self.input_root)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == pixel["gray_sha256"]
        return self.metadata(position), pixels


@contextmanager
def open_data(stage, destination_root=None, root=ROOT):
    root = Path(root)
    assert stage in ("source", "development", "assessment")
    if stage == "source":
        relative = "Research_Transfer_Mechanism_Benchmark_2026-10-28/Source_Retention_Runtime_01/source_loader.py"
        loader = load_module("matched_complete_source_loader", root / relative)
        with loader.open_source("vlr") as data:
            assert len(data) == read(HERE / "Plan.json")["source"]["clips"]
            yield Source(data)
    else:
        yield Destination(stage, destination_root or root, root)


def positions(dataset, task, mode, root=ROOT):
    stage, suffix = task.rsplit("_", 1)
    assert stage == dataset.stage
    plan = read(HERE / "Plan.json")
    if mode == "fixture":
        assert stage == "source" and suffix == "fixture"
        return [7617]
    if mode == "probe":
        assert suffix == "probe"
        order = sorted(range(len(dataset)), key=lambda index: (dataset.metadata(index)["frames"], index))
        return sorted({order[round(fraction * (len(order) - 1))] for fraction in (0, .25, .5, .75, .95, 1)})
    assert mode == "population" and suffix.isdigit()
    shard = int(suffix)
    if stage == "source":
        assert 0 <= shard < 16
        path = Path(root) / plan["source"]["assignment_plan"]
        assert sha(path) == plan["upstream_lock_sha256"][plan["source"]["assignment_plan"]]
        return read(path)["tasks"][f"VLR_B0_{shard:02}"]["positions"]
    assert 0 <= shard < 8
    order = sorted(range(len(dataset)), key=lambda index: (
        dataset.metadata(index)["frames"], dataset.metadata(index)["clip_id"]))
    return order[shard::8]


def candidates(stage, readout, model, binding, selection_path=None):
    plan = read(HERE / "Plan.json")
    if stage == "development":
        assert readout == "grid" and selection_path is None
        return list(plan["secondary_candidate_order"])
    assert readout in ("common", "selected")
    if readout == "common":
        assert selection_path is None
        return [plan["primary_decoder"]]
    selection = read(selection_path)
    assert selection["status"] == "complete_development_selection"
    assert selection["model"] == model and selection["clips"] == 768 and selection["reference_units"] == 15919
    assert selection["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    assert selection["registry_sha256"] == sha(Path(binding) / "model_registry.json")
    assert selection["decoder"] in plan["secondary_candidate_order"]
    return [selection["decoder"]]
