"""Hash-bound prepared data with common eligibility and deterministic ordering."""

import random
from functools import lru_cache
from pathlib import Path
import numpy as np
from .common import digest, read_json, read_rows, rank, code_identity
from .frame_cache import load_frames


class PreparedData:
    def __init__(self, directory, training=False, cohort='window'):
        if cohort not in ['window', 'full']:
            raise ValueError('Unknown training population')
        self.directory = Path(directory)
        self.manifest = self.directory / 'manifest.jsonl'
        self.receipt = read_json(self.directory / 'receipt.json')
        if self.receipt.get('code_sha256') != code_identity():
            raise ValueError('Prepared cache belongs to another code version; prepare a new cache')
        if digest(self.manifest) != self.receipt['manifest_sha256']:
            raise ValueError('Prepared manifest changed')
        failures = self.directory / 'failures.jsonl'
        if digest(failures) != self.receipt['failures_sha256']:
            raise ValueError('Preparation coverage changed')
        self.failures = read_rows(failures)
        self.all_rows = read_rows(self.manifest)
        key = 'train_eligible' if cohort == 'window' else 'full_eligible'
        self.rows = [r for r in self.all_rows if r[key]] if training else self.all_rows
        if not self.rows:
            raise ValueError('No eligible rows')
        if training and any(r['split'] != 'train' for r in self.rows):
            raise ValueError('Optimization can only consume train rows')

    @lru_cache(maxsize=8)
    def frames(self, clip_id):
        row = self.by_id[clip_id]
        if row['frame_path'] is None:
            raise ValueError('Ineligible training clip has metadata only; prepare a full cache to evaluate it')
        path = (self.directory / row['frame_path']).resolve()
        if self.directory.resolve() not in path.parents:
            raise ValueError('Cache path escapes prepared split')
        if digest(path) != row['frames_sha256']:
            raise ValueError('Frame cache drift')
        frames = load_frames(path)
        if frames.dtype != np.uint8 or frames.shape != (row['frames'], 96, 96):
            raise ValueError('Frame shape changed')
        return frames

    @property
    def by_id(self):
        if not hasattr(self, '_by_id'):
            self._by_id = {r['clip_id']: r for r in self.rows}
            if len(self._by_id) != len(self.rows):
                raise ValueError('Duplicate sample identities')
        return self._by_id

    @lru_cache(maxsize=2)
    def order(self, epoch, seed):
        indices = list(range(len(self.rows)))
        random.Random(seed + epoch).shuffle(indices)
        return indices

    def sample(self, number, seed):
        epoch, position = divmod(number, len(self.rows))
        row = self.rows[self.order(epoch, seed)[position]]
        return row, self.frames(row['clip_id'])

    def selection(self, limit=0):
        rows = sorted(self.rows, key=lambda r: rank(r['clip_id']))
        return rows[:limit] if limit else rows


def check_disjoint(train, validation):
    overlaps = {}
    for key in ['clip_id', 'source_id', 'video_sha256']:
        left = {r[key] for r in train.all_rows}
        right = {r[key] for r in validation.all_rows}
        overlaps[key] = len(left & right)
    if any(overlaps.values()):
        raise ValueError(f'Train/validation identity overlap: {overlaps}')
    return overlaps
