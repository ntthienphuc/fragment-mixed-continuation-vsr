"""Explicit precision, reproducible resume state and bounded CPU prefetch."""

import contextlib
import random
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import torch


def autocast(precision, device):
    if precision == 'fp32':
        return contextlib.nullcontext()
    if not str(device).startswith('cuda'):
        raise ValueError('Mixed precision is supported only on CUDA in this protocol')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16')
    return torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def samples(data, first, stop, seed, workers=0):
    if not workers:
        for number in range(first, stop):
            yield data.sample(number, seed)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        numbers = iter(range(first, stop))
        for _ in range(workers * 2):
            number = next(numbers, None)
            if number is not None:
                pending.append(pool.submit(data.sample, number, seed))
        while pending:
            yield pending.popleft().result()
            number = next(numbers, None)
            if number is not None:
                pending.append(pool.submit(data.sample, number, seed))
