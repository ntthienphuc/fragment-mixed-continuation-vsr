"""Counterexamples for timing gaps, local timing validity and CTC-safe fallback."""

from word_span import choose


def words(*items):
    return [{"text": text, "start": start, "end": end} for text, start, end in items]


def encode(text):
    if not text.strip():
        raise ValueError("empty")
    return list(range(1, len(text.split()) + 1))


def main():
    result = choose(words(("a", 0, 1), ("b", 1, 2), ("c", 2, 3)), 75, 25, 75, "a b c", encode)
    assert (result["start"], result["stop"], result["text"]) == (25, 75, "b c")
    result = choose(words(("z", 0, 0), ("a", 1, 2), ("b", 2, 3)), 75, 25, 75, "z a b", encode)
    assert result["text"] == "a b" and result["view"] == "word_aligned"
    result = choose(words(("a", 0, 1), ("z", 1, 1), ("b", 1, 2)), 50, 0, 50, "a z b", encode)
    assert result["text"] == "a" and result["stop"] == 25
    result = choose(words(("a", 0, .4), ("b", 8, 8.4)), 225, 0, 50, "a b", encode)
    assert result["text"] == "a" and result["stop"] == 10
    result = choose(words(("a", 0, 1.01)), 25, 0, 25, "a", encode)
    assert result["view"] == "shared_full_fallback" and result["stop"] == 25
    result = choose(words(("a", 0, 0)), 50, 0, 50, "a", encode)
    assert result["view"] == "shared_full_fallback"
    result = choose(words(("a a", 0, .08)), 6, 0, 2, "a a", lambda text: [1, 1])
    assert result["view"] == "shared_full_fallback" and result["minimum_ctc_frames"] == 3
    result = choose(words(("", 0, .5)), 50, 0, 50, "a", encode)
    assert result["view"] == "shared_full_fallback"
    result = choose(words(("a", 0, float("nan"))), 50, 0, 50, "a", encode)
    assert result["view"] == "shared_full_fallback"
    print({"passed": 9, "includes_observed_timing_gap_and_zero_duration_counterexamples": True})


if __name__ == "__main__":
    main()
