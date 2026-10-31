"""Frame geometry and overlap targets; never use a reference to choose input availability."""

from .common import rank

CORE = 125
CONTEXT = 16


def target_text(words, start, end, fps=25):
    lower, upper = start / fps, end / fps
    selected = [w['text'] for w in words if
                (w['end'] > lower and w['start'] < upper and w['end'] > w['start'])
                or (w['start'] == w['end'] and lower <= w['start'] < upper)]
    return ' '.join(selected).strip()


def window_start(length, identity, sample_index=0, seed=17, core=CORE):
    if core < 1 or length < core + 2 * CONTEXT:
        raise ValueError('Clip too short for matched context windows')
    choices = length - core - 2 * CONTEXT + 1
    return CONTEXT + int(rank(f'{seed}|{identity}|{sample_index}')[:16], 16) % choices


def span(length, start, mode='base', core=CORE):
    if mode not in {'base', 'past', 'both'}:
        raise ValueError('Unknown context budget')
    if core < 1:
        raise ValueError('Core length must be positive')
    end = start + core
    left = start - CONTEXT if mode != 'base' else start
    right = end + CONTEXT if mode == 'both' else end
    if left < 0 or right > length:
        raise ValueError('Requested context is unavailable')
    return left, right, start - left, end - left


def context_for(method, budget):
    if budget not in {'past', 'both'}:
        raise ValueError('Context budget must be past or both')
    return budget if method in {'T4', 'T5', 'A5'} else 'base'
