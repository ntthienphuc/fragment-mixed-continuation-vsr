"""Restrict conventional spans to the already declared T1--T4 student input band."""

import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "Readiness_02/word_span.py"
SPEC = importlib.util.spec_from_file_location("original_word_span_for_bounded_control", SOURCE)
PARENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARENT)

MINIMUM = 50
MAXIMUM = 82


def choose(words, frame_count, core_start, core_stop, full_text, encode):
    assert core_stop - core_start == MINIMUM
    rejected = 0
    for priority, start, stop, first, last in PARENT.candidates(words, frame_count, core_start, core_stop):
        if not MINIMUM <= stop - start <= MAXIMUM:
            continue
        text = " ".join(word["text"] for word in words[first:last + 1]).strip()
        try:
            identifiers = encode(text)
        except ValueError:
            rejected += 1
            continue
        minimum_ctc = len(identifiers) + sum(a == b for a, b in zip(identifiers, identifiers[1:]))
        if minimum_ctc > stop - start:
            rejected += 1
            continue
        return {"view": "word_aligned", "start": start, "stop": stop, "first_word": first,
                "last_word": last, "text": text, "native_tokens": identifiers,
                "minimum_ctc_frames": minimum_ctc, "priority": priority,
                "earlier_ctc_or_empty_candidates": rejected}
    identifiers = encode(full_text)
    minimum_ctc = len(identifiers) + sum(a == b for a, b in zip(identifiers, identifiers[1:]))
    assert minimum_ctc <= frame_count
    return {"view": "shared_full_fallback", "start": 0, "stop": frame_count, "text": full_text,
            "native_tokens": identifiers, "minimum_ctc_frames": minimum_ctc,
            "earlier_ctc_or_empty_candidates": rejected}
