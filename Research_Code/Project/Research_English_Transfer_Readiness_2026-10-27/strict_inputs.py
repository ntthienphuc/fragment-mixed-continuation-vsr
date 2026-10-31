"""Read verified English pixels without changing frozen shared input code."""

import hashlib
import io
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "Research_Server_Confirmation_2026-10-19"
sys.path.insert(0, str(ENGINE))

from confirmation_common import resolve
from data_io import checked_payload, load_pixels as shared_load_pixels
from data_io import payload_shape, validate_shape


def load_pixels(row, root=ROOT):
    """Return raw contiguous uint8 gray96 frames; never normalize or use text."""
    if row["language"] != "en" or abs(float(row["fps"]) - 25.0) > 0.001:
        raise ValueError("Expected a verified English 25fps input")
    source = row["input"]
    if source.get("color_space") != "gray":
        raise ValueError("English inputs must explicitly use grayscale")
    if source["kind"] == "zip_npz":
        return shared_load_pixels(row, root=root)
    if source["kind"] != "npz_frames":
        raise ValueError("Unsupported English input kind")
    path = resolve(source["path"], root)
    payload = checked_payload(path.read_bytes(), source["sha256"])
    shape, dtype = payload_shape(payload)
    validate_shape(shape, dtype, int(row["frames"]), "gray")
    with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
        pixels = np.ascontiguousarray(packet["frames"])
    validate_shape(pixels.shape, pixels.dtype, int(row["frames"]), "gray")
    expected = source.get("pixels_sha256")
    if not expected or hashlib.sha256(pixels.tobytes()).hexdigest() != expected:
        raise ValueError("Decoded pixels must match the author-path cache receipt")
    return pixels
