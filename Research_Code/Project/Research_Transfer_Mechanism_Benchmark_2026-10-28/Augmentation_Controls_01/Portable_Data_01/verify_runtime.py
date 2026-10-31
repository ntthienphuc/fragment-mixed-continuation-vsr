"""Run real staged data through the native sampler and archived pixel expectations on CPU."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
HERE = Path(__file__).resolve().parent
from portable_data import InputPaths, TrainingData, ValidationData, relative_name, sha, digest


def main(args):
    began = time.monotonic()
    packet = Path(args.packet).resolve()
    sys.path.insert(0, str(packet))
    sys.path.insert(0, str(packet / "source"))
    sys.path.insert(0, str(packet / "support/Native_Matched_Adapter_01"))
    import numpy as np
    import torch
    from adapter import MatchedObjective
    from phase5.common import rank, code_identity
    from phase5.data import PreparedData, check_disjoint
    from phase5.runtime import samples
    from phase5.text import Tokenizer
    from probe_support import Inputs, verify_packet

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if sha(args.spec) != args.spec_sha256:
        raise ValueError("Original native input specification changed")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8-sig"))
    plan = verify_packet(packet)
    assert code_identity() == plan["native_code_identity"] == spec["identity"]["code_sha256"]
    paths = InputPaths(args.runtime, args.runtime_sha256)
    objective = MatchedObjective("vlr", "full")
    training = TrainingData(objective, spec, paths)
    validation = ValidationData(spec, paths)
    overlap = check_disjoint(training, validation)
    original = PreparedData.__new__(PreparedData)
    original.rows = training.rows
    for number, scheduled in enumerate(objective.schedule):
        epoch, position = divmod(number, len(original.rows))
        index = original.order(epoch, 17)[position]
        assert original.rows[index]["clip_id"] == scheduled["clip_id"]
    selected = validation.selection(512)
    independent_selection = sorted(validation.all_rows, key=lambda row: rank(row["clip_id"]))[:512]
    assert selected == independent_selection
    tokenizer = Tokenizer(packet / "source" / plan["config"]["tokenizer"],
                          packet / "source" / plan["config"]["units"], "vi")
    model = type("SelectionModel", (), {"method": "T0", "tokenizer": tokenizer})()
    expected_inputs = Inputs(packet, plan)
    comparisons = []
    views = 0
    for position in sorted(expected_inputs.cases):
        row, pixels = training.sample(position, 17)
        expected_row, expected_pixels = expected_inputs.sample(position)
        assert row == expected_row and np.array_equal(pixels, expected_pixels)
        comparisons.append({"sample": position, "clip_id": row["clip_id"],
                            "pixels_sha256": digest(pixels.tobytes()),
                            "original_pixel_digest": bool(row.get("pixels_sha256"))})
        for arm in plan["arms"]:
            objective.arm = arm
            actual = objective.select(model, pixels, row, position, 17)
            expected = objective.select(model, expected_pixels, expected_row, position, 17)
            assert np.array_equal(actual[0], expected[0]) and actual[1:] == expected[1:]
            views += 1
    sequential = [(row["clip_id"], digest(pixels.tobytes()))
                  for row, pixels in samples(training, 0, 16, 17, workers=0)]
    threaded = [(row["clip_id"], digest(pixels.tobytes()))
                for row, pixels in samples(training, 0, 16, 17, workers=2)]
    assert sequential == threaded
    validation_checked = []
    for index, row in enumerate(selected, 1):
        pixels = validation.frames(row["clip_id"])
        validation_checked.append({"clip_id": row["clip_id"], "frames": len(pixels),
                                   "pixels_sha256": digest(pixels.tobytes()),
                                   "window_eligible": row["window_eligible"]})
        if index % 64 == 0:
            print({"validation_caches_verified": index, "total": 512}, flush=True)
    assert len(comparisons) == 21 and views == 63 and len(validation_checked) == 512
    rejected = []

    def rejects(label, callback):
        try:
            callback()
        except (ValueError, KeyError):
            rejected.append(label)
        else:
            raise AssertionError("Invalid request accepted: " + label)

    rejects("wrong_runtime_sha256", lambda: InputPaths(args.runtime, "0" * 64))
    for limit in (0, -1, 513, True):
        rejects("validation_limit_" + str(limit), lambda limit=limit: validation.selection(limit))
    for position in (-1, 48000, True):
        rejects("sample_position_" + str(position), lambda position=position: training.sample(position, 17))
    rejects("wrong_seed", lambda: training.sample(0, 29))
    uncached = next(row["clip_id"] for row in validation.all_rows if row["clip_id"] not in validation.cached)
    rejects("uncached_validation", lambda: validation.frames(uncached))
    rejects("unscheduled_training", lambda: training.frames("unavailable/video"))
    for name in ("../outside", "/absolute", "C:/absolute", "a\\b"):
        rejects("unsafe_member_" + name, lambda name=name: relative_name(name))
    assert len(rejected) == 15 and not torch.cuda.is_initialized()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"utc": datetime.now(timezone.utc).isoformat(),
               "status": "portable_source_inputs_native_order_pixels_and_coverage_verified",
               "identity": paths.identity, "native_code_identity": code_identity(),
               "native_spec_sha256": args.spec_sha256, "packet_manifest_sha256": sha(packet / "Manifest.json"),
               "full_training_manifest_rows": len(training.all_rows), "training_eligible_rows": len(training.rows),
               "scheduled_presentations": 48000, "unique_scheduled_videos": len(training.scheduled),
               "training_real_pixel_replays": comparisons, "real_view_comparisons": views,
               "native_threaded_prefetch_compared": len(threaded),
               "full_validation_manifest_rows": len(validation.all_rows),
               "cached_validation_clips": len(validation_checked),
               "window_eligible_cached_clips": sum(row["window_eligible"] for row in selected),
               "window_ineligible_cached_clips": sum(not row["window_eligible"] for row in selected),
               "source_split_overlap": overlap, "rejected_requests": rejected,
               "cuda_initialized": False, "recognizer_instantiated": False, "training_admitted": False,
               "seconds": time.monotonic() - began, "python": sys.version, "torch": str(torch.__version__),
               "provider_sha256": sha(HERE / "portable_data.py"), "verifier_sha256": sha(Path(__file__))}
    for name, value in (("Validation_Pixels.json", validation_checked), ("Complete.json", receipt)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in (
        "status", "scheduled_presentations", "cached_validation_clips", "window_eligible_cached_clips",
        "window_ineligible_cached_clips", "cuda_initialized", "seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("packet", "runtime", "runtime-sha256", "spec", "spec-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
