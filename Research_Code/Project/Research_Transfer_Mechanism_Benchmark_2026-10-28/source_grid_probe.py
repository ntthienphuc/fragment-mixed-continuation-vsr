"""Actual fixed short/median/long development inputs through the full decoder grid."""

import argparse
import hashlib
import math
from pathlib import Path
import time

from experiment_core import check_prediction, cohort, decode, independent_greedy, native_bridge, pixels
from mechanism_common import HERE, ROOT, read, save, sha, utc, validate_inputs


def original_native(runtime, states, candidate, plan):
    import torch

    weight = float(candidate.removeprefix("beam_c"))
    with torch.inference_mode(), torch.autocast(runtime.device.type, enabled=False):
        search = runtime.native_factory(runtime.model, runtime.vocabulary, ctc_weight=weight,
                                        beam_size=plan["beam_size"], penalty=plan["penalty"])
        search.eval()
        hypotheses = search(states)
    assert hypotheses
    sequence = [int(token) for token in hypotheses[0].yseq]
    assert sequence[0] == sequence[-1] == runtime.eos_id
    tokens = [token for token in sequence[1:-1] if token not in {runtime.blank_id, runtime.eos_id}]
    score = float(hypotheses[0].score)
    assert math.isfinite(score)
    return tokens, score


def main(model, output, seconds, device):
    assert 0 < seconds <= 1800
    deadline = time.time() + seconds
    output.mkdir(parents=True, exist_ok=False)
    save(output / "Start.json", {"utc": utc(), "model": model, "device": device,
                                "deadline_unix": deadline, "assessment_used": False})
    try:
        plan = validate_inputs()
        registry = read(HERE / "model_registry.json")
        entry = registry["models"][model]
        rows = cohort(plan, entry["language"], "development")
        ordered = sorted(range(len(rows)), key=lambda position: (rows[position]["frames"], rows[position]["clip_id"]))
        probes = [ordered[index] for index in (0, len(rows) // 2, len(rows) - 1)]
        assert probes == plan["source_probe_positions"][entry["language"]]
        runtime = native_bridge().create_runtime(model, device=device, threads=2)
        from confirmation_common import edit_counts

        save(output / "Identity.json", runtime.identity)
        records = []
        comparisons = []
        for position in probes:
            assert time.time() < deadline
            row = rows[position]
            visual = pixels(row)
            encoded = runtime.encode(visual)
            assert encoded["metadata"]["pixel_sha256"] == hashlib.sha256(visual.tobytes()).hexdigest()
            assert len(encoded["states"]) == row["frames"]
            predictions = {}
            for candidate in plan["candidate_order"]:
                assert time.time() < deadline
                result = decode(runtime, encoded, candidate, plan)
                result["counts"] = edit_counts(row["reference"], result["text"])
                check_prediction(row, result, candidate, entry["classes"])
                assert runtime.text(result["tokens"]) == result["text"]
                if candidate.startswith("beam_c"):
                    expected, expected_score = original_native(runtime, encoded["states"], candidate, plan)
                    score_error = abs(result["score"] - expected_score)
                    assert math.isfinite(score_error) and score_error <= 1e-4
                else:
                    bias = float(candidate.removeprefix("greedy_b"))
                    expected = independent_greedy(encoded["logp"], bias, runtime.blank_id, runtime.eos_id)
                    score_error = 0.0
                assert expected == result["tokens"]
                comparisons.append({"position": position, "candidate": candidate, "score_error": score_error,
                                    "tokens_exact": True, "text_exact": runtime.text(expected) == result["text"]})
                predictions[candidate] = result
            record = {"position": position, "clip_id": row["clip_id"], "frames": row["frames"],
                      "encoding": encoded["metadata"], "predictions": predictions}
            records.append(record)
            save(output / ("Probe_" + str(position) + ".json"), record)
            print({"utc": utc(), "model": model, "position": position, "frames": row["frames"],
                   "grid": len(predictions), "independent_comparisons_passed": len(comparisons)}, flush=True)
            del encoded
            del visual
        assert len(records) == 3 and len(comparisons) == 33 and time.time() < deadline
        save(output / "Complete.json", {"utc": utc(), "status": "three_source_inputs_full_grid_verified",
             "model": model, "language": entry["language"], "device": str(runtime.device),
             "positions": probes, "frames": [row["frames"] for row in records],
             "comparisons": comparisons, "independent_count_checks": 33, "reference_used_for_inference": False,
             "assessment_used": False, "transfer_performance_claim": False,
             "plan_sha256": sha(HERE / "Plan.json"), "checkpoint_sha256": entry["checkpoint_sha256"],
             "scripts": {name: sha(HERE / name) for name in (
                 "source_grid_probe.py", "experiment_core.py", "runtime_bridge.py", "mechanism_common.py")},
             "identity_sha256": sha(output / "Identity.json"),
             "probe_sha256": {"Probe_" + str(position) + ".json": sha(output / ("Probe_" + str(position) + ".json"))
                              for position in probes}})
    except Exception as error:
        save(output / "Failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    arguments = parser.parse_args()
    main(arguments.model, arguments.output.resolve(), arguments.seconds, arguments.device)
