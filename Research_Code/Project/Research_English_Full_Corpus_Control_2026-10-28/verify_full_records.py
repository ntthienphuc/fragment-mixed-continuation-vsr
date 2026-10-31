"""Independent population, input, decoder, token and edit-count validation."""

import math

from execution_common import require
from experiment_core import check_prediction


def verify_records(rows, predictions, task, selected, classes=5049):
    require(len(rows) == len(predictions) and bool(rows), "Incomplete predictions")
    checks = 0
    for (position, row), predicted in zip(rows, predictions):
        require(predicted["index"] == position, "Prediction order changed")
        require(predicted["model"] == task["model"] and predicted["split"] == task["stage"], "Wrong model/stage")
        for name in ("clip_id", "source_id", "frames"):
            require(predicted[name] == row[name], "Wrong input identity: " + name)
        require(predicted["media_sha256"] == row["input"]["sha256"], "Input packet changed")
        metadata = predicted["encoding"]
        require(metadata["model_id"] == task["model"], "Wrong encoder model")
        require(metadata["frames"] == metadata["state_frames"] == row["frames"], "Frame count changed")
        require(metadata["color_space"] == "gray" and metadata["precision"] == "float32", "Input type changed")
        require(metadata["transform"] == "phase5_native_gray", "Input transform changed")
        digest = metadata["pixel_sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "Missing input digest")
        require(all(character in "0123456789abcdef" for character in digest), "Invalid pixel digest")
        if task["stage"] == "assessment":
            require(metadata["pixel_sha256"] == row["input"]["pixels_sha256"], "Author-path pixels changed")
        require(set(predicted["predictions"]) == set(selected), "Missing or extra decoder")
        for candidate, result in predicted["predictions"].items():
            check_prediction(row, result, candidate, classes)
            checks += 1
    return checks


def same_backend_replay(expected, actual):
    require(expected["tokens"] == actual["tokens"], "Fresh-pixel tokens differ")
    require(expected["text"] == actual["text"], "Fresh-pixel text differs")
    require(("score" in expected) == ("score" in actual), "Score schema changed")
    difference = 0.0
    if "score" in expected:
        require(math.isfinite(expected["score"]) and math.isfinite(actual["score"]), "Nonfinite score")
        difference = abs(expected["score"] - actual["score"])
        require(difference <= 1e-4, "Same-backend score replay differs")
    return difference
