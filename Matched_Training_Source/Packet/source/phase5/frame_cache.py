"""Lossless uint8 frame storage; compression never changes model inputs."""
import io
import zipfile
from pathlib import Path
import numpy as np


def encode_frames(frames):
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (96, 96):
        raise ValueError('Expected uint8 T by 96 by 96 frames')
    payload = io.BytesIO()
    np.save(payload, frames, allow_pickle=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        info = zipfile.ZipInfo('frames.npy', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, payload.getvalue(), compresslevel=1)
    return output.getvalue()


def load_frames(path):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix != '.npz':
        raise ValueError('Unsupported frame cache format')
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ['frames']:
            raise ValueError('Unexpected compressed frame members')
        return archive['frames']


def save_frames(path, frames):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    if path.suffix == '.npz':
        temporary.write_bytes(encode_frames(frames))
    elif path.suffix == '.npy':
        with temporary.open('wb') as handle:
            np.save(handle, frames, allow_pickle=False)
    else:
        raise ValueError('Unsupported frame cache format')
    temporary.replace(path)
