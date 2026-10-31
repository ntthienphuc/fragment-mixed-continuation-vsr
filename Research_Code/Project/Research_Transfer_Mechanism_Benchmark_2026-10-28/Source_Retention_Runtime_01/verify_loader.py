"""Exercise the ZIP-backed provider on real shortest, longest and ordinary clips."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from source_loader import HERE, PLAN_SHA256, decode_frames, open_source


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    started = time.monotonic()
    save("Start.json", {"utc": datetime.now(timezone.utc).isoformat(),
                        "device": "cpu", "loads_recognizer": False})
    result = {}
    for language in ("vlr", "lrs2"):
        with open_source(language) as source:
            ordered = sorted(range(len(source)), key=lambda index: (source.rows[index]["frames"], index))
            positions = {0, len(source) - 1, ordered[0], ordered[len(ordered) // 2], ordered[-1]}
            positions.update(random.Random(17).sample(range(len(source)), 5))
            checks = []
            for position in sorted(positions):
                row, frames = source.load(position)
                pixel = source.pixels[row["clip_id"]]
                digest = hashlib.sha256(frames.tobytes()).hexdigest()
                assert digest == pixel["pixel_sha256"]
                assert frames.shape == (row["frames"], 96, 96)
                checks.append({"position": position, "frames": len(frames), "pixel_sha256": digest})
            for position in (-1, len(source), 0.0):
                try:
                    source.load(position)
                except IndexError:
                    pass
                else:
                    raise AssertionError("Out-of-cohort position accepted")
            row = source.rows[0]
            payload = source.archive.read("prepared/" + row["frame_path"])
            corrupted = bytes([payload[0] ^ 1]) + payload[1:]
            invalid_pixel = {**source.pixels[row["clip_id"]], "pixel_sha256": "0" * 64}
            for data, pixel in ((corrupted, source.pixels[row["clip_id"]]), (payload, invalid_pixel)):
                try:
                    decode_frames(data, row, pixel)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Corrupted frame identity accepted")
            result[language] = {
                "complete_cohort": len(source),
                "reference_units": sum(len(row["text"].split()) for row in source.rows),
                "minimum_frames": source.rows[ordered[0]]["frames"],
                "maximum_frames": source.rows[ordered[-1]]["frames"],
                "real_pixel_loads": checks, "out_of_cohort_rejections": 3, "corruption_rejections": 2,
                "archive_handle_open_during_reads": not source.archive.fp.closed,
            }
            archive = source.archive
        assert archive.fp is None
        result[language]["archive_handle_closed_after_context"] = True
    assert "torch" not in sys.modules, "No inference or model load belongs in this verification"
    save("Verification.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "real_source_loader_checks_passed",
        "seconds": time.monotonic() - started, "corpora": result,
        "prepared_plan_sha256": PLAN_SHA256, "source_sha256": sha(HERE / "source_loader.py"),
        "verifier_sha256": sha(Path(__file__)), "new_recognition_results": False,
        "files_extracted": 0, "inference_admitted": False,
    })
    print(json.dumps({"status": "real_source_loader_checks_passed",
                      "seconds": time.monotonic() - started,
                      "corpora": {name: {key: value for key, value in rows.items()
                                          if key != "real_pixel_loads"} for name, rows in result.items()}}, indent=2))


if __name__ == "__main__":
    main()
