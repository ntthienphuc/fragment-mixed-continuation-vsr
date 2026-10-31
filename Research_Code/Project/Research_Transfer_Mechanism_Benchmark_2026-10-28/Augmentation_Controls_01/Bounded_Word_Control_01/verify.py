"""Independently enumerate bounded spans and bind every output slot to its archived trace."""

from datetime import datetime, timezone
import gzip
import importlib.util
import itertools
import json
import math
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "Readiness_02"
SPEC = importlib.util.spec_from_file_location("bounded_parent_verification", PARENT / "verify.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def exhaustive(words, frames, lower, upper, tokenizer):
    if any(not math.isfinite(word[key]) for word in words for key in ["start", "end"]):
        return []
    if any(words[index]["start"] > words[index + 1]["start"] for index in range(len(words) - 1)):
        return []
    valid = []
    for first in range(len(words)):
        for last in range(first, len(words)):
            segment = words[first:last + 1]
            if any(word["end"] <= word["start"] for word in segment):
                continue
            beginning = segment[0]["start"]
            ending = max(word["end"] for word in segment)
            if beginning >= upper / 25 or ending <= lower / 25:
                continue
            start = math.floor(beginning * 25 + 1e-8)
            stop = math.ceil(ending * 25 - 1e-8)
            if not 0 <= start < stop <= frames or not 50 <= stop - start <= 82:
                continue
            text = " ".join(word["text"] for word in segment).strip()
            try:
                _, minimum = CHECK.ids_and_minimum(tokenizer, text)
            except ValueError:
                continue
            if minimum > stop - start:
                continue
            valid.append((abs(start - lower) + abs(stop - upper), abs(stop - start - 50),
                          abs(start + stop - lower - upper) / 2, start, first, last))
    return valid


def verify_trace(item):
    dataset = item["dataset"]
    trace_path = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01"
    trace_path = trace_path / (dataset + "_trace.jsonl.gz")
    assert CHECK.sha(trace_path) == item["trace_sha256"]
    originals = (row for row in records(trace_path) if not row["full"])
    schedule = records(HERE / (dataset + "_short_slots.jsonl.gz"))
    fallback = []
    bounded = 0
    count = 0
    for count, (row, original) in enumerate(itertools.zip_longest(schedule, originals), 1):
        assert row is not None and original is not None
        assert row["sample"] == original["sample"] == 2 * count - 1
        assert row["clip_id"] == original["clip_id"]
        assert row["clip_frames"] == original["clip_frames"]
        assert row["original_start"] == original["core_start"]
        assert row["original_stop"] == original["core_stop"]
        assert row["original_target_sha256"] == original["target_sha256"]
        assert original["core_stop"] - original["core_start"] == 50
        if row["view"] == "shared_full_fallback":
            fallback.append(row["sample"])
            assert row["matched_T1"]["start"] == 0
            assert row["matched_T1"]["stop"] == row["clip_frames"]
        else:
            assert row["view"] == "word_aligned" and 50 <= row["stop"] - row["start"] <= 82
            bounded += 1
    assert count == 24000 and bounded + len(fallback) == count
    return {"dataset": dataset, "archived_slot_identities_checked": count,
            "bounded_short_slots": bounded, "shared_full_fallback_samples": fallback}


def main():
    summary = json.loads((HERE / "Summary.json").read_text())
    assert summary["status"] == "bounded_schedule_pending_independent_verification"
    assert summary["duration_band_frames"] == [50, 82]
    assert summary["protocol_sha256"] == CHECK.sha(HERE / "Protocol.md")
    assert summary["geometry_sha256"] == CHECK.sha(ROOT / "Research_Phase_05_2026-10-13/phase5/views.py")
    for name, digest in summary["scripts"].items():
        assert CHECK.sha(HERE / name) == digest
    for name, digest in summary["parent_implementation_sha256"].items():
        assert CHECK.sha(PARENT / name) == digest
    prior = json.loads((PARENT / "Verification.json").read_text())
    assert CHECK.sha(PARENT / "verify.py") == prior["script_sha256"]
    CHECK.HERE = HERE
    CHECK.exhaustive = exhaustive
    results = []
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        for item in summary["datasets"]:
            result = CHECK.check_dataset(item, archive)
            result["trace_identity"] = verify_trace(item)
            results.append(result)
    proof = {"utc": datetime.now(timezone.utc).isoformat(),
             "status": "all_bounded_slots_and_shared_fallbacks_verified", "datasets": results,
             "summary_sha256": CHECK.sha(HERE / "Summary.json"), "script_sha256": CHECK.sha(Path(__file__)),
             "parent_verifier_sha256": CHECK.sha(PARENT / "verify.py"),
             "training_admitted": False, "assessment_predictions_used": False}
    with (HERE / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
