"""Recount text flags from original CSV and inspect deterministic raw-label/video examples."""

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

import av

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data/official_raw/vlr/extracted/cloud_dataset_signer_split"
OUT = HERE / "Results_01"
ENGLISH = set("the and of to is in that you for with are this your but it we our not be have".split())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokens(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(char if unicodedata.category(char)[0] in "LNM" else " " for char in text)
    return text.split()


def repeated(values):
    for start in range(len(values) - 3):
        phrase = values[start:start + 4]
        occurrences = 0
        index = 0
        while index <= len(values) - 4:
            if values[index:index + 4] == phrase:
                occurrences += 1
                index += 4
            else:
                index += 1
        if occurrences >= 3:
            return True
    return False


def main():
    summary = json.loads((OUT / "Summary.json").read_text(encoding="utf-8"))
    assert sha(OUT / "Flagged_Training_Rows.jsonl") == summary["flagged_rows_sha256"]
    assert sha(DATA / "manifests/train_strict.csv") == summary["original_csv_sha256"]
    with (OUT / "Flagged_Training_Rows.jsonl").open(encoding="utf-8") as stream:
        flagged = {row["clip_id"]: row for row in map(json.loads, stream)}
    expected_english = {identity for identity, row in flagged.items() if row["ascii_english_cues"]}
    expected_repeated = {identity for identity, row in flagged.items() if row["repeated_fourgram"]}
    actual_english = set()
    actual_repeated = set()
    original = {}
    with (DATA / "manifests/train_strict.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assert row["split"] == "train" and row["clip_id"] not in original
            original[row["clip_id"]] = row
            values = tokens(row["transcript"])
            letters = [char for word in values for char in word if char.isalpha()]
            if letters and all(char.isascii() for char in letters) and len(ENGLISH.intersection(values)) >= 3:
                actual_english.add(row["clip_id"])
            if repeated(values):
                actual_repeated.add(row["clip_id"])
    assert actual_english == expected_english and actual_repeated == expected_repeated
    assert len(original) == summary["totals"]["prepared_rows"] == 92632
    assert len(actual_english) == summary["strata"]["ascii_english_cues"]["prepared_rows"]
    assert len(actual_repeated) == summary["strata"]["repeated_fourgram"]["prepared_rows"]
    sample = set()
    for examples in summary["examples_selected_by_id_hash"].values():
        sample.update(row["clip_id"] for row in examples)
    sample.update(row["clip_id"] for row in summary["motivating_csv_checks"])
    prepared = {}
    metadata_hash = hashlib.sha256()
    with zipfile.ZipFile(ROOT / "Server_Rescue_2026-10-15/Source_Results_Metadata.zip") as archive:
        with archive.open("Phase05/prepared/vlr/train/manifest.jsonl") as stream:
            for line in stream:
                metadata_hash.update(line)
                row = json.loads(line)
                if row["clip_id"] in sample:
                    prepared[row["clip_id"]] = row
    assert metadata_hash.hexdigest() == summary["manifest_sha256"] and set(prepared) == sample
    checks = []
    discrepancies = []
    for identity in sorted(sample):
        raw = original[identity]
        with (DATA / raw["label_path"]).open(encoding="utf-8-sig", newline="") as stream:
            words = [row for row in csv.DictReader(stream) if row["Word"].strip()]
        word_tokens = tokens(" ".join(row["Word"] for row in words))
        text_tokens = tokens(raw["transcript"])
        actual_words = [{"text": row["Word"].strip(), "start": float(row["Start"]), "end": float(row["End"])}
                        for row in words]
        assert actual_words == prepared[identity]["words"], "Word annotation provenance changed: " + identity
        if word_tokens != text_tokens:
            discrepancies.append({"clip_id": identity, "word_tokens": word_tokens,
                                  "manifest_tokens": text_tokens,
                                  "scheduled": flagged.get(identity, {}).get("scheduled", False)})
        zero = sum(float(row["Start"]) == float(row["End"]) for row in words)
        flag = bool(words) and zero * 2 >= len(words)
        assert flag == flagged.get(identity, {}).get("majority_zero_duration_words", False)
        checks.append({"clip_id": identity, "word_rows": len(words), "zero_duration_words": zero,
                       "majority_zero_duration_words": flag,
                       "label_sha256": sha(DATA / raw["label_path"])})
    audio = []
    for example in summary["motivating_csv_checks"]:
        path = Path(example["video_path"])
        with av.open(str(path)) as container:
            audio.append({"clip_id": example["clip_id"], "video_streams": len(container.streams.video),
                          "audio_streams": len(container.streams.audio), "video_sha256": sha(path)})
    for flag, counts in summary["strata"].items():
        scheduled = [row for row in flagged.values() if row[flag] and row["scheduled"]]
        assert len(scheduled) == counts["presentations"]
    proof = {"utc": datetime.now(timezone.utc).isoformat(),
             "status": "source_label_provenance_verified_with_text_discrepancies",
             "all_original_train_text_rows": len(original), "english_cue_rows": len(actual_english),
             "repeated_fourgram_rows": len(actual_repeated), "individual_word_csv_checks": checks,
             "sampled_full_text_vs_word_csv_discrepancies": discrepancies,
             "motivating_video_streams": audio, "audio_listened_to": False,
             "all_timing_flags_independently_recounted": False, "neural_inference": False,
             "wrong_language_annotations_proven": False, "summary_sha256": sha(OUT / "Summary.json"),
             "verifier_sha256": sha(Path(__file__))}
    with (OUT / "Verification.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps({"status": proof["status"], "all_original_train_text_rows": len(original),
                      "individual_word_csv_checks": len(checks), "audio_streams": audio,
                      "sampled_text_discrepancies": discrepancies}, ensure_ascii=False))


if __name__ == "__main__":
    main()
