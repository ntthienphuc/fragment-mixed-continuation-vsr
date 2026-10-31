"""Independent edit recount and fresh-pixel replay; a shard is never a population result."""

import json
import math
from pathlib import Path

from common import HERE, ROOT, read, save, sha, utc, load_module
from data_tasks import candidates, open_data, positions
from execution import checked
from run_task import parser


def prediction_rows(path):
    result = []
    with Path(path).open("rb") as stream:
        for line in stream:
            assert line.endswith(b"\n") and line.strip(), "Incomplete prediction stream"
            result.append(json.loads(line))
    return result


def verify_records(dataset, selected, records, choices, model, mode):
    assert len(selected) == len(records)
    independent = load_module("matched_independent_counts", ROOT / "Research_Interval_Transfer_2026-10-27/verify_probe.py")
    for position, record in zip(selected, records):
        original = dataset.metadata(position)
        assert record["position"] == position and record["model"] == model
        assert record["mode"] == mode and record["stage"] == dataset.stage
        for field in ("clip_id", "source_id", "frames"):
            assert record[field] == original[field]
        encoded = record["encoding"]
        assert encoded["frames"] == encoded["state_frames"] == original["frames"]
        assert encoded["model_id"] == model and encoded["precision"] == "float32"
        assert encoded["transform"] == "phase5_native_gray" and encoded["color_space"] == "gray"
        assert encoded["pixel_sha256"] == original["pixel_sha256"]
        assert list(record["predictions"]) == choices
        for candidate, result in record["predictions"].items():
            assert independent.counts(original["reference"], result["text"]) == result["counts"]
            assert all(type(token) is int and 0 < token < 2001 for token in result["tokens"])
            if candidate.startswith("beam_c"):
                assert result["cap"] == original["frames"] and math.isfinite(result["score"])
                trace = result["trace"]
                assert trace["beam_size"] == 40 and trace["penalty"] == 0
                assert trace["ctc_weight"] == float(candidate.removeprefix("beam_c"))
                assert trace["implementation"] == "upstream_BatchBeamSearch"
    return len(records) * len(choices)


def main(arguments):
    checked(arguments)
    folder = arguments.output.resolve()
    assert folder.is_relative_to(HERE) and folder != HERE
    start = read(folder / "Start.json")
    complete = read(folder / "Complete.json")
    assert complete["status"] == "complete_pending_independent_verification"
    assert sha(folder / "Predictions.jsonl") == complete["predictions_sha256"]
    assert sha(folder / "Identity.json") == complete["identity_sha256"]
    assert sha(folder / "Start.json") == complete["start_sha256"]
    for key in ("task", "model", "mode", "readout", "device"):
        assert start[key] == getattr(arguments, key)
    assert start["assessment_plan_sha256"] == sha(HERE / "Plan.json")
    assert start["code_lock_sha256"] == sha(HERE / "Code_Lock.json")
    assert start["admission_sha256"] == sha(arguments.admission)
    assert start["registry_sha256"] == sha(arguments.binding / "model_registry.json")
    assert start["selection_sha256"] == (sha(arguments.selection) if arguments.selection else None)
    assert start["destination_root"] == (str(arguments.destination_root) if arguments.destination_root else None)
    stage = arguments.task.rsplit("_", 1)[0]
    choices = candidates(stage, arguments.readout, arguments.model, arguments.binding, arguments.selection)
    assert start["candidates"] == choices
    records = prediction_rows(folder / "Predictions.jsonl")
    from runtime_bridge import create_runtime, decode

    with open_data(stage, arguments.destination_root) as dataset:
        selected = positions(dataset, arguments.task, arguments.mode)
        assert complete["clips"] == len(selected)
        checks = verify_records(dataset, selected, records, choices, arguments.model, arguments.mode)
        runtime = create_runtime(arguments.binding, arguments.model, device=arguments.device)
        expected_identity = read(folder / "Identity.json")
        for field in ("registry_entry", "asset_sha256", "source_files", "torch", "numpy", "device",
                      "matched_bridge_sha256", "native_graph_binding"):
            assert runtime.identity[field] == expected_identity[field], "Replay identity differs: " + field
        offsets = sorted({0, len(records) // 3, 2 * len(records) // 3, len(records) - 1})
        replay = []
        for offset in offsets:
            expected = records[offset]
            _, pixels = dataset.load(expected["position"])
            encoded = runtime.encode(pixels)
            assert encoded["metadata"]["pixel_sha256"] == expected["encoding"]["pixel_sha256"]
            for candidate in choices:
                actual = decode(runtime, encoded, candidate, choices)
                saved = expected["predictions"][candidate]
                assert actual["tokens"] == saved["tokens"] and actual["text"] == saved["text"]
                error = abs(actual["score"] - saved["score"]) if "score" in actual else 0
                assert error <= 1e-4
                replay.append({"position": expected["position"], "candidate": candidate, "score_error": error})
            del encoded
    save(folder / "Verification.json", {"utc": utc(), "status": "all_records_recounted_and_fresh_replay_passed",
         "task": arguments.task, "model": arguments.model, "mode": arguments.mode, "stage": stage,
         "readout": arguments.readout, "device": arguments.device, "candidates": choices, "clips": len(records),
         "independent_count_checks": checks, "replays": replay,
         "predictions_sha256": sha(folder / "Predictions.jsonl"), "identity_sha256": sha(folder / "Identity.json"),
         "code_lock_sha256": sha(HERE / "Code_Lock.json"), "registry_sha256": start["registry_sha256"],
         "assessment_plan_sha256": sha(HERE / "Plan.json"), "seconds": complete["seconds"],
         "peak_gpu_mib": complete["peak_gpu_mib"], "complete_population_receipt": False})
    print({"status": "verified", "task": arguments.task, "clips": len(records), "checks": checks}, flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
