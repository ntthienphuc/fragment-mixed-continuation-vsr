"""Storage changes must preserve the exact tensor and reject unrelated archives."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from phase5.frame_cache import encode_frames, load_frames, save_frames


class FrameCacheTests(unittest.TestCase):
    def test_exact_pixels_and_deterministic_bytes(self):
        rng = np.random.default_rng(81)
        frames = rng.integers(0, 256, (13, 96, 96), dtype=np.uint8)
        frames[0].fill(0)
        frames[1].fill(255)
        self.assertEqual(encode_frames(frames), encode_frames(frames.copy()))
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ['npy', 'npz']:
                path = Path(directory) / ('frames.' + suffix)
                save_frames(path, frames)
                restored = load_frames(path)
                self.assertTrue(np.array_equal(frames, restored))
                self.assertEqual(restored.dtype, np.uint8)
                self.assertFalse(path.with_suffix('.tmp').exists())

    def test_reject_unexpected_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.npz'
            np.savez(path, unrelated=np.zeros((2, 96, 96), dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_frames(path)
        with self.assertRaises(ValueError):
            encode_frames(np.zeros((2, 96, 96), dtype=np.float32))
