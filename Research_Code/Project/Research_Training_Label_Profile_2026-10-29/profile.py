"""Count training annotation flags and their exact original FC/FM supervision exposure."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NATIVE = ROOT / "Research_Tailscale_Server_2026-10-28/Native_Loss_Probe_03/source"
sys.path.insert(0, str(NATIVE))
from phase5.text import Tokenizer, reference_text
from phase5.views import target_text

CUES = frozenset("the and of to is in that you for with are this your but it we our not be have".split())
FLAGS = ("ascii_english_cues", "repeated_fourgram", "majority_zero_duration_words")
EXAMPLES = ("part3/DONGC2/TRUCT28/TRUCT28_44", "part3/DONGC2/VIYEU33/VIYEU33_2",
            "part3/DONGC2/TRUCT28/TRUCT28_83")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features(text, words):
    normalized = reference_text(text, "vi")
    tokens = normalized.split()
    letters = [letter for letter in normalized if letter.isalpha()]
    ascii_only = bool(letters) and all(ord(letter) < 128 for letter in letters)
    cues = sorted(CUES.intersection(tokens))
    locations = defaultdict(list)
    for index in range(max(0, len(tokens) - 3)):
        locations[tuple(tokens[index:index + 4])].append(index)
    repetitions = 0
    for positions in locations.values():
        count = 0
        end = -1
        for position in positions:
            if position >= end:
                count += 1
                end = position + 4
        repetitions = max(repetitions, count)
    zero = sum(word["start"] == word["end"] for word in words)
    return {"ascii_english_cues": ascii_only and len(cues) >= 3,
            "repeated_fourgram": repetitions >= 3,
            "majority_zero_duration_words": bool(words) and zero * 2 >= len(words),
            "ascii_letters_only": ascii_only, "english_cue_words": cues,
            "max_nonoverlapping_fourgram_repeats": repetitions,
            "zero_duration_words": zero, "word_rows": len(words)}


def main():
    dataset = ROOT / "data/official_raw/vlr/extracted/cloud_dataset_signer_split"
    original_path = dataset / "manifests/train_strict.csv"
    with original_path.open(encoding="utf-8-sig", newline="") as stream:
        original = {row["clip_id"]: row for row in csv.DictReader(stream)}
    trace_path = ROOT / "Research_Transfer_Mechanism_Controls_2026-10-28/Replay_01/vlr_trace.jsonl.gz"
    assert sha(trace_path) == "e5e98cd1ff61682476c0aa195ef56cf7bc5cc05ebfa9f243411b4afaccfaf374"
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    schedule = {row["clip_id"]: row for row in trace}
    assert len(trace) == len(schedule) == 48000
    tokenizer = Tokenizer(NATIVE / "assets/tokenizers/vi.model", NATIVE / "assets/tokenizers/vi_units.txt", "vi")
    totals = Counter()
    strata = {flag: Counter() for flag in FLAGS}
    sources = defaultdict(Counter)
    examples = {flag: [] for flag in FLAGS}
    motivated = []
    output = HERE / "Results_01"
    output.mkdir(exist_ok=False)
    manifest_hash = hashlib.sha256()
    seen = set()
    archive_path = ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip"
    with (output / "Flagged_Training_Rows.jsonl").open("x", encoding="utf-8") as flagged_stream:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open("Phase05/prepared/vlr/train/manifest.jsonl") as stream:
                for line in stream:
                    manifest_hash.update(line)
                    row = json.loads(line)
                    identity = row["clip_id"]
                    assert row["split"] == "train" and identity not in seen
                    seen.add(identity)
                    assert reference_text(original[identity]["transcript"], "vi") == reference_text(row["text"], "vi")
                    detail = features(row["text"], row["words"])
                    totals["prepared_rows"] += 1
                    totals["eligible_rows"] += int(row["train_eligible"])
                    active = [flag for flag in FLAGS if detail[flag]]
                    if active:
                        record = {"clip_id": identity, "source_id": row["source_id"], "text": row["text"],
                                  "scheduled": identity in schedule, "flags": active, **detail}
                        flagged_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        for flag in active:
                            strata[flag]["prepared_rows"] += 1
                            strata[flag]["eligible_rows"] += int(row["train_eligible"])
                            sources[row["source_id"]][flag] += 1
                            examples[flag].append(record)
                    if identity in schedule:
                        presentation = schedule[identity]
                        text = row["text"] if presentation["full"] else target_text(
                            row["words"], presentation["core_start"], presentation["core_stop"])
                        assert digest(text.encode()) == presentation["target_sha256"]
                        full_tokens = len(tokenizer.encode(row["text"]))
                        fragment_tokens = len(tokenizer.encode(text))
                        exposure = {"presentations": 1, "short_slots": int(not presentation["full"]),
                                    "full_tokens": full_tokens, "fm_tokens": fragment_tokens,
                                    "full_frames": row["frames"],
                                    "fm_frames": presentation["views"]["T1"]["input_stop"] -
                                                 presentation["views"]["T1"]["input_start"]}
                        totals.update(exposure)
                        for flag in active:
                            strata[flag].update(exposure)
                    if identity in EXAMPLES:
                        label_path = dataset / original[identity]["label_path"]
                        with label_path.open(encoding="utf-8-sig", newline="") as labels:
                            labels = list(csv.DictReader(labels))
                        label_text = " ".join(word["Word"].strip() for word in labels if word["Word"].strip())
                        assert reference_text(label_text, "vi") == reference_text(row["text"], "vi")
                        motivated.append({"clip_id": identity, "prepared_text": row["text"],
                                          "word_csv_text": label_text, "label_sha256": sha(label_path),
                                          "video_path": str(dataset / original[identity]["video_path"]), **detail})
    assert manifest_hash.hexdigest() == "70ec83434982fd3e350ac43c51a20468f416d32b8bd48b254021d76b7f9e0dfb"
    assert totals["prepared_rows"] == 92632 and totals["eligible_rows"] == 90519
    assert totals["presentations"] == 48000 and totals["short_slots"] == 24000
    assert totals["full_tokens"] == 1144990 and totals["fm_tokens"] == 791474
    assert totals["full_frames"] == 7307270 and totals["fm_frames"] == 4851015
    assert len(motivated) == 3
    for flag, counts in strata.items():
        counts["scheduled_clip_percent"] = 100 * counts["presentations"] / totals["presentations"]
        counts["full_token_mass_percent"] = 100 * counts["full_tokens"] / totals["full_tokens"]
        counts["fm_token_mass_percent"] = 100 * counts["fm_tokens"] / totals["fm_tokens"]
        examples[flag] = sorted(examples[flag], key=lambda row: digest(row["clip_id"].encode()))[:8]
    receipt = {"utc": datetime.now(timezone.utc).isoformat(),
               "status": "training_profile_reconciled_with_original_fc_fm_exposure",
               "totals": dict(totals), "strata": strata, "source_counts": dict(sources),
               "examples_selected_by_id_hash": examples, "motivating_csv_checks": motivated,
               "flags_overlap": True, "wrong_language_labels_confirmed": False,
               "audio_review_performed": False, "train_rows_changed": False, "assessment_read": False,
               "manifest_sha256": manifest_hash.hexdigest(), "original_csv_sha256": sha(original_path),
               "original_trace_sha256": sha(trace_path), "tokenizer_sha256": sha(NATIVE / "assets/tokenizers/vi.model"),
               "code_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "Protocol.md"),
               "flagged_rows_sha256": sha(output / "Flagged_Training_Rows.jsonl")}
    with (output / "Summary.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": receipt["status"], "totals": receipt["totals"], "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
