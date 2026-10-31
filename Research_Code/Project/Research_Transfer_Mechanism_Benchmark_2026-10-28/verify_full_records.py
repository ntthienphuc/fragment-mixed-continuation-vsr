"""Check actual two-language tokens, populations, gray pixels and independent edit counts."""

import math

from execution_common import require
from experiment_core import check_prediction
from mechanism_common import VI, lines, read, sha


def gray_receipts():
    path = VI / "Pixels_01/Index.jsonl"
    require(sha(path) == read(VI / "Pixels_01/Verification.json")["index_sha256"], "Gray receipts changed")
    rows = lines(path)
    result = {row["clip_id"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate gray receipt")
    return result


def verify_records(rows, predictions, task, selected, classes):
    require(classes == (2002 if task["language"] == "vi" else 5049), "Wrong language vocabulary")
    require(len(rows) == len(predictions) and bool(rows), "Incomplete predictions")
    gray = gray_receipts() if task["language"] == "vi" else None
    checks = 0
    for (position, row), predicted in zip(rows, predictions):
        require(predicted["index"] == position, "Prediction order changed")
        require(predicted["model"] == task["model"] and predicted["split"] == task["stage"], "Wrong model/stage")
        for name in ("clip_id", "source_id", "frames"):
            require(predicted[name] == row[name], "Wrong input: " + name)
        require(predicted["media_sha256"] == row["input"]["sha256"], "Input packet changed")
        metadata = predicted["encoding"]
        require(metadata["model_id"] == task["model"], "Wrong encoder")
        require(metadata["frames"] == metadata["state_frames"] == row["frames"], "Frame count changed")
        require(metadata["color_space"] == "gray" and metadata["precision"] == "float32", "Input type changed")
        require(metadata["transform"] == "phase5_native_gray", "Input transform changed")
        digest = metadata["pixel_sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "Missing pixel digest")
        require(all(character in "0123456789abcdef" for character in digest), "Invalid pixel digest")
        if gray is not None:
            receipt = gray[row["clip_id"]]
            require(receipt["media_sha256"] == row["input"]["sha256"], "Wrong gray source")
            require(digest == receipt["gray_sha256"], "Vietnamese gray pixels changed")
        elif task["stage"] == "assessment":
            require(digest == row["input"]["pixels_sha256"], "Author-path pixels changed")
        require(set(predicted["predictions"]) == set(selected), "Wrong decoder grid")
        for candidate, result in predicted["predictions"].items():
            check_prediction(row, result, candidate, classes)
            checks += 1
    return checks


def same_backend_replay(expected, actual):
    require(expected["tokens"] == actual["tokens"] and expected["text"] == actual["text"], "Replay differs")
    require(("score" in expected) == ("score" in actual), "Score schema changed")
    difference = 0.0
    if "score" in expected:
        require(math.isfinite(expected["score"]) and math.isfinite(actual["score"]), "Nonfinite score")
        difference = abs(expected["score"] - actual["score"])
        require(difference <= 1e-4, "Same-backend score differs")
    return difference
