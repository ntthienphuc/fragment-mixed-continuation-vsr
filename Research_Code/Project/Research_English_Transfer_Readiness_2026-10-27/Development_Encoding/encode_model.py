"""Encode every LRS2 development clip and verify lossless states plus native replay."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from encoding_common import HERE, INPUTS, PHASE, create_runtime, lock, save, sha, utc


def state_bytes(states):
    return hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()


def main(model, output, deadline):
    registered = lock()
    assert model in registered["models"] and not output.exists()
    assert time.time() < deadline
    source = [json.loads(line) for line in (INPUTS / "Development.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(source) == registered["clips_per_model"]
    output.mkdir(exist_ok=False)
    cache = Path(registered["cache_root"]) / model
    cache.mkdir(parents=True, exist_ok=False)
    runtime = create_runtime(model)
    for name, digest in runtime.identity["source_files"].items():
        assert registered["sha256"][name] == digest
    save(output / "Identity.json", runtime.identity)
    sys.path.insert(0, str(PHASE))
    from strict_inputs import load_pixels
    import torch

    torch.cuda.reset_peak_memory_stats(runtime.device)
    records = []
    archive_path = cache / "States.zip"
    began = time.monotonic()
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_STORED) as archive:
        with (output / "Index.jsonl").open("x", encoding="utf-8") as stream:
            for number, row in enumerate(source):
                assert time.time() < deadline, "Source-encoding deadline reached"
                assert row["split"] == "development" and row["language"] == "en"
                pixels = load_pixels(row)
                encoded = runtime.encode(pixels)
                states = np.ascontiguousarray(encoded["states"].detach().cpu().numpy())
                assert states.shape == (row["frames"], 768) and states.dtype == np.float32
                assert np.isfinite(states).all()
                buffer = io.BytesIO()
                np.save(buffer, states, allow_pickle=False)
                payload = buffer.getvalue()
                member = f"states/{number:04d}.npy"
                archive.writestr(member, payload)
                record = {"position": number, "clip_id": row["clip_id"], "source_id": row["source_id"],
                          "model": model, "frames": row["frames"], "source_payload_sha256": row["input"]["sha256"],
                          "member": member, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                          "state_sha256": state_bytes(states), "encoding": encoded["metadata"]}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                del encoded, states, pixels
                if (number + 1) % 25 == 0 or number + 1 == len(source):
                    elapsed = time.monotonic() - began
                    print({"utc": utc(), "model": model, "encoded": number + 1, "total": len(source),
                           "elapsed_seconds": elapsed,
                           "eta_seconds": elapsed * (len(source) - number - 1) / (number + 1)}, flush=True)
    replays = []
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {record["member"] for record in records}
        for record in records:
            payload = archive.read(record["member"])
            assert hashlib.sha256(payload).hexdigest() == record["payload_sha256"]
            states = np.load(io.BytesIO(payload), allow_pickle=False)
            assert states.shape == (record["frames"], 768) and states.dtype == np.float32
            assert np.isfinite(states).all() and state_bytes(states) == record["state_sha256"]
        indices = {row["clip_id"]: number for number, row in enumerate(source)}
        for probe in registered["probes"]:
            assert time.time() < deadline
            number = indices[probe["clip_id"]]
            states = np.load(io.BytesIO(archive.read(records[number]["member"])), allow_pickle=False)
            fresh = runtime.encode(load_pixels(source[number]))
            assert np.array_equal(states, fresh["states"].detach().cpu().numpy())
            assert records[number]["encoding"]["pixel_sha256"] == fresh["metadata"]["pixel_sha256"]
            restored = torch.from_numpy(states.copy()).to(runtime.device)
            cached_prediction = runtime.decode(restored, 0.1, cap=len(states), beam_size=40, penalty=0)
            native_prediction = runtime.decode_native(fresh["states"], 0.1, beam_size=40, penalty=0)
            assert cached_prediction["tokens"] == native_prediction["tokens"]
            assert cached_prediction["text"] == native_prediction["text"]
            error = abs(cached_prediction["score"] - native_prediction["score"])
            assert error <= 1e-5
            replays.append({"clip_id": probe["clip_id"], "frames": probe["frames"],
                            "states_bitwise_equal": True, "native_decoder_score_error": error})
            del fresh, restored
    save(output / "Complete.json", {"utc": utc(), "status": "all_source_encoder_states_verified",
                                    "model": model, "clips": len(records), "archive": str(archive_path),
                                    "archive_sha256": sha(archive_path), "index_sha256": sha(output / "Index.jsonl"),
                                    "lock_sha256": sha(HERE / "Lock.json"), "fresh_pixel_replays": replays,
                                    "seconds": time.monotonic() - began,
                                    "peak_gpu_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
                                    "reference_used_for_inference": False, "test_clips_used": False,
                                    "accuracy_claimed": False})
    print({"utc": utc(), "model": model, "status": "complete", "clips": len(records)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.deadline)
