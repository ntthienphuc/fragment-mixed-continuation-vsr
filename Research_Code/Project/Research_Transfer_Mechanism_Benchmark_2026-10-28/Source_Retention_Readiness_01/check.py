"""Reverify complete cached source tests and bind the current transfer model identities."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import traceback
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUP = ROOT / "Server_Rescue_2026-10-15"
META = BACKUP / "Source_Results_Metadata.zip"
CORPORA = {
    "vlr": ("Data/VLR/Prepared_Test.zip", "Data/VLR/vlr_test_inputs_Transfer_Status.json", 11307),
    "lrs2": ("Data/EN/LRS2_Prepared_Test.zip", "Data/EN/heldout_inputs_Transfer_Status.json", 1243),
}
PHASES = (
    "Research_Transfer_Full_Benchmark_2026-10-27",
    "Research_English_Transfer_Benchmark_2026-10-28",
    "Research_English_Full_Corpus_Control_2026-10-28",
    "Research_Transfer_Mechanism_Benchmark_2026-10-28",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sets(rows):
    return {field: {row[field] for row in rows} for field in ("clip_id", "source_id", "video_sha256")}


def check_archive(language, item, identities):
    relative, receipt_path, expected_clips = item
    path = BACKUP / relative
    acquisition = read(BACKUP / receipt_path)
    expected = next(row for row in acquisition["completed"] if row["filename"] == path.name)
    assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"]
    with zipfile.ZipFile(path) as archive:
        manifest_payload = archive.read("Manifest.json")
        manifest = json.loads(manifest_payload)
        members = manifest["files"]
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(members) | {"Manifest.json"}
        prepared_bytes = archive.read("prepared/manifest.jsonl")
        rows = [json.loads(line) for line in prepared_bytes.splitlines() if line.strip()]
        receipt = json.loads(archive.read("prepared/receipt.json"))
        assert len(rows) == receipt["planned"] == receipt["prepared"] == expected_clips
        assert receipt["failed"] == 0 and archive.read("prepared/failures.jsonl") == b""
        assert hashlib.sha256(prepared_bytes).hexdigest() == receipt["manifest_sha256"]
        assert receipt["config_sha256"] == identities[language]["config_sha256"]
        assert receipt["code_sha256"] == identities[language]["code_sha256"]
        assert all(row["split"] == "test" and row["text"].strip() for row in rows)
        assert len({row["clip_id"] for row in rows}) == expected_clips
        frame_members = {"prepared/" + row["frame_path"]: row for row in rows}
        assert len(frame_members) == expected_clips
        decoded = 0
        pixel_receipts = []
        for index, (name, expected_member) in enumerate(members.items(), 1):
            relative_name = PurePosixPath(name)
            assert not relative_name.is_absolute() and ".." not in relative_name.parts and ":" not in name
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            assert len(data) == expected_member["bytes"] and digest == expected_member["sha256"], name
            if name in frame_members:
                row = frame_members[name]
                assert digest == row["frames_sha256"]
                with np.load(io.BytesIO(data), allow_pickle=False) as cache:
                    assert cache.files == ["frames"], cache.files
                    frames = cache["frames"]
                assert frames.dtype == np.uint8 and frames.shape == (row["frames"], 96, 96)
                pixel_receipts.append({"clip_id": row["clip_id"], "frames": len(frames),
                                       "pixel_sha256": hashlib.sha256(frames.tobytes()).hexdigest()})
                decoded += 1
            if index % 5000 == 0:
                print(json.dumps({"corpus": language, "payloads": index, "frames_loaded": decoded}), flush=True)
        assert decoded == expected_clips
        historical = json.loads(archive.read("Preparation_Check.json"))
        assert len({row["source_id"] for row in rows}) == historical["source_groups"]
        cohort = sets(rows)
        overlaps = {}
        with zipfile.ZipFile(META) as metadata:
            for split in ("train", "val"):
                member = "Phase05/prepared/" + language + "/" + split + "/manifest.jsonl"
                with metadata.open(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                assert digest == identities[language][split + "_manifest_sha256"]
                source = {field: set() for field in cohort}
                with metadata.open(member) as stream:
                    for line in stream:
                        row = json.loads(line)
                        assert row["split"] == split
                        for field in source:
                            source[field].add(row[field])
                overlaps[split] = {field: len(cohort[field] & source[field]) for field in cohort}
                assert not any(overlaps[split].values())
        durations = np.asarray([row["frames"] / 25 for row in rows])
        result = {
            "archive": str(path.relative_to(ROOT)), "archive_sha256": expected["sha256"],
            "archive_bytes": path.stat().st_size, "payload_files_verified": len(members),
            "archive_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "prepared_manifest_sha256": receipt["manifest_sha256"], "clips": len(rows),
            "source_labels": len(cohort["source_id"]),
            "reference_whitespace_units": sum(len(row["text"].split()) for row in rows),
            "all_frame_caches_loaded": decoded, "total_frames": sum(row["frames"] for row in rows),
            "duration_seconds": dict(zip(("min", "median", "p95", "max"),
                                         [float(value) for value in np.quantile(durations, (0, .5, .95, 1))])),
            "mean_seconds": float(durations.mean()), "disjointness": overlaps,
            "speaker_independence_established": False, "human_label_review_added": False,
        }
        if language == "vlr":
            correction = archive.read("Dataset_Identity_Correction.md")
            result["directory_group_correction_sha256"] = hashlib.sha256(correction).hexdigest()
        with (HERE / (language + "_Pixel_Manifest.jsonl")).open("x", encoding="utf-8") as stream:
            for record in pixel_receipts:
                stream.write(json.dumps(record) + "\n")
        result["pixel_manifest_sha256"] = sha(HERE / (language + "_Pixel_Manifest.jsonl"))
        return result


def main():
    began = time.monotonic()
    save("Start.json", {"utc": datetime.now(timezone.utc).isoformat(), "gpu_inference": False})
    models = {}
    identities = {}
    prerequisites = {"Protocol.md": sha(HERE / "Protocol.md"), "check.py": sha(Path(__file__))}
    for phase in PHASES:
        folder = ROOT / phase
        registry = read(folder / "model_registry.json")["models"]
        selection = read(folder / "Decoder_Selection.json")
        for name in ("model_registry.json", "Decoder_Selection.json"):
            prerequisites[phase + "/" + name] = sha(folder / name)
        for name, entry in registry.items():
            if name == "VC_NATIVE":
                continue
            assert name not in models
            language = "vlr" if name.startswith("VLR_") else "lrs2"
            candidates = list(dict.fromkeys([selection["models"][name]["decoder"], "beam_c0.1"]))
            models[name] = {"registry": phase + "/model_registry.json", "identity": entry,
                            "frozen_transfer_decoder": candidates[0], "execution_candidates": candidates}
            if "training_identity" in entry:
                identity = entry["training_identity"]
                compare = {field: identity[field] for field in (
                    "config_sha256", "code_sha256", "train_manifest_sha256", "val_manifest_sha256")}
                assert identity["steps"] == 6000 and entry["selected_training_step"] == 6000
                if language in identities:
                    assert identities[language] == compare
                identities[language] = compare
    corpora = {language: check_archive(language, item, identities) for language, item in CORPORA.items()}
    physical = {}
    for name, model in models.items():
        entry = model["identity"]
        for field in ("checkpoint", "base_checkpoint", "tokenizer", "units", "config"):
            if field not in entry:
                continue
            path = (ROOT / entry[field]).resolve(strict=True)
            digest = entry[field + "_sha256"]
            if str(path) not in physical:
                assert sha(path) == digest, str(path)
                physical[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
            assert physical[str(path)]["sha256"] == digest
    save("Prepared_Plan.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_not_inference_admitted",
        "models": models, "corpora": corpora, "prerequisites": prerequisites,
        "all_checkpoints_and_inputs_hashed": physical, "historical_test_exposure": True,
        "beam_size": 40, "penalty": 0, "external_lm": False, "precision": "float32",
        "horizon": "full_encoded_length", "target_matrix_must_finish_first": True,
    })
    save("Complete.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "complete_source_tests_reverified",
        "corpora": corpora, "models": list(models), "physical_model_assets": len(physical),
        "prepared_plan_sha256": sha(HERE / "Prepared_Plan.json"),
        "seconds": time.monotonic() - began, "new_recognition_results": False,
    })
    print(json.dumps({"status": "complete_source_tests_reverified", "corpora": corpora}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save("Failure.json", {"utc": datetime.now(timezone.utc).isoformat(), "error": traceback.format_exc()})
        raise
