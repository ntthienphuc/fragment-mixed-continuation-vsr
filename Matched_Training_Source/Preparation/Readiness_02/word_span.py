"""Conventional nearest-boundary word spans, retaining coverage through explicit fallback."""

import math


def candidates(words, frame_count, core_start, core_stop):
    assert 0 <= core_start < core_stop <= frame_count
    if any(not math.isfinite(word[field]) for word in words for field in ("start", "end")):
        return []
    if any(left["start"] > right["start"] for left, right in zip(words, words[1:])):
        return []
    found = []
    for first in range(len(words)):
        start_time = words[first]["start"]
        stop_time = start_time
        for last in range(first, len(words)):
            word = words[last]
            if word["end"] <= word["start"]:
                break
            stop_time = max(stop_time, word["end"])
            if stop_time <= core_start / 25 or start_time >= core_stop / 25:
                continue
            start = math.floor(start_time * 25 + 1e-8)
            stop = math.ceil(stop_time * 25 - 1e-8)
            if not 0 <= start < stop <= frame_count:
                continue
            boundary_distance = abs(start - core_start) + abs(stop - core_stop)
            midpoint_distance = abs(start + stop - core_start - core_stop) / 2
            priority = (boundary_distance, abs(stop - start - 50), midpoint_distance, start, first, last)
            found.append((priority, start, stop, first, last))
    return sorted(found)


def choose(words, frame_count, core_start, core_stop, full_text, encode):
    rejected = 0
    for priority, start, stop, first, last in candidates(words, frame_count, core_start, core_stop):
        text = " ".join(word["text"] for word in words[first:last + 1]).strip()
        try:
            ids = encode(text)
        except ValueError:
            rejected += 1
            continue
        minimum = len(ids) + sum(left == right for left, right in zip(ids, ids[1:]))
        if minimum > stop - start:
            rejected += 1
            continue
        return {"view": "word_aligned", "start": start, "stop": stop, "first_word": first,
                "last_word": last, "text": text, "native_tokens": ids, "minimum_ctc_frames": minimum,
                "priority": priority, "earlier_ctc_or_empty_candidates": rejected}
    ids = encode(full_text)
    minimum = len(ids) + sum(left == right for left, right in zip(ids, ids[1:]))
    assert minimum <= frame_count, "Full fallback itself is CTC-infeasible"
    return {"view": "shared_full_fallback", "start": 0, "stop": frame_count, "text": full_text,
            "native_tokens": ids, "minimum_ctc_frames": minimum, "earlier_ctc_or_empty_candidates": rejected}
