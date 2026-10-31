"""One actual development input per control on CPU; never a target-performance claim."""

import argparse
from pathlib import Path
import time

from mechanism_common import HERE, ROOT, lines, read, save, sha, utc, validate_inputs


def main(model, output, seconds):
    assert 0 < seconds <= 600
    deadline = time.time() + seconds
    assert not output.exists()
    plan = validate_inputs()
    entry = read(HERE / "model_registry.json")["models"][model]
    language = entry["language"]
    rows = lines(ROOT / plan["cohorts"][language]["development"]["path"])
    position = plan["source_probe_positions"][language][0]
    row = rows[position]
    assert position == min(range(len(rows)), key=lambda index: (rows[index]["frames"], rows[index]["clip_id"]))
    from runtime_bridge import create_runtime, pixels_for
    import torch

    runtime = create_runtime(model, device="cpu", threads=2)
    assert time.time() < deadline
    encoded = runtime.encode(pixels_for(row))
    assert len(encoded["states"]) == row["frames"]
    actual = runtime.decode_native(encoded["states"], .1, beam_size=40, penalty=0)
    assert time.time() < deadline
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        search = runtime.native_factory(runtime.model, runtime.vocabulary, ctc_weight=.1, beam_size=40, penalty=0)
        search.eval()
        hypotheses = search(encoded["states"])
    assert hypotheses
    sequence = [int(token) for token in hypotheses[0].yseq]
    assert sequence[0] == runtime.eos_id and sequence[-1] == runtime.eos_id
    expected_tokens = [token for token in sequence[1:-1] if token not in {runtime.blank_id, runtime.eos_id}]
    assert actual["tokens"] == expected_tokens
    expected_text = runtime.text(expected_tokens)
    assert actual["text"] == expected_text
    error = abs(actual["score"] - float(hypotheses[0].score))
    assert error <= 1e-4 and time.time() < deadline
    save(output, {"utc": utc(), "status": "cpu_shortest_development_native_replay_passed", "model": model,
                  "language": language, "clip_id": row["clip_id"], "frames": row["frames"],
                  "identity": runtime.identity, "encoding": encoded["metadata"],
                  "native_tokens": actual["tokens"], "native_text": actual["text"], "score_error": error,
                  "test_clips_used": False, "GPU_admission_granted": False,
                  "plan_sha256": sha(HERE / "Plan.json"), "script_sha256": sha(__file__)})
    print({"model": model, "frames": row["frames"], "status": "passed", "score_error": error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output.resolve(), arguments.seconds)
