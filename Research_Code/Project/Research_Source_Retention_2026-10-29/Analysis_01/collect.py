"""Load only complete, independently verified source-population shard backups."""

import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
BACKUP = Path.home() / "Lip_Research_Backups/Source_Retention_20261029"
PARENT_SHA = "bf56e94190102756c38287e557ef6132ea86f8841b030a88e97134f4226aa1d4"
ASSESSMENT_SHA = "42a023aaa74808961aa3cb5e2fdefcb0c2009cfa12a94ee02ee086bb84140a94"
PREPARED_SHA = "57043031647da4c158d8b83b5680d4e2031dfbd8ec1d85af4d21c7521851e084"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locked(path, expected):
    payload = Path(path).read_bytes()
    require(digest(payload) == expected, "Changed locked input: " + str(path))
    return json.loads(payload)


def plans():
    parent = locked(PHASE / "Plan.json", PARENT_SHA)
    assessment = locked(PHASE / "Assessment_01/Plan.json", ASSESSMENT_SHA)
    for plan in (parent, assessment):
        for relative, expected in plan["code_sha256"].items():
            require(digest((ROOT / relative).read_bytes()) == expected, "Changed code: " + relative)
    require(assessment["parent_plan_sha256"] == PARENT_SHA, "Parent identity differs")
    return parent, assessment


def metadata(corpus):
    folder = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Source_Retention_Readiness_01"
    prepared = locked(folder / "Prepared_Plan.json", PREPARED_SHA)["corpora"][corpus]
    pixel_payload = (folder / (corpus + "_Pixel_Manifest.jsonl")).read_bytes()
    require(digest(pixel_payload) == prepared["pixel_manifest_sha256"], "Pixel manifest changed")
    pixels = {row["clip_id"]: row for row in map(json.loads, pixel_payload.splitlines())}
    with zipfile.ZipFile(ROOT / prepared["archive"]) as archive:
        payload = archive.read("prepared/manifest.jsonl")
    require(digest(payload) == prepared["prepared_manifest_sha256"], "Source text changed")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    require(len(rows) == len(pixels) == prepared["clips"], "Metadata coverage differs")
    require(len({row["clip_id"] for row in rows}) == len(rows), "Duplicate source clip")
    return rows, pixels


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def recount(reference, hypothesis):
    ref = normalized(reference).split()
    hyp = normalized(hypothesis).split()
    costs = [list(range(len(hyp) + 1))]
    for row, word in enumerate(ref, 1):
        current = [row]
        for column, predicted in enumerate(hyp, 1):
            current.append(min(costs[-1][column - 1] + (word != predicted),
                               costs[-1][column] + 1, current[-1] + 1))
        costs.append(current)
    row = len(ref)
    column = len(hyp)
    substitutions = deletions = insertions = 0
    while row or column:
        mismatch = int(ref[row - 1] != hyp[column - 1]) if row and column else 0
        if row and column and costs[row][column] == costs[row - 1][column - 1] + mismatch:
            substitutions += mismatch
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return {"errors": costs[-1][-1], "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "units": len(ref), "metric": "wer"}


def archive_for(task):
    paths = sorted(BACKUP.glob(task + "_*.zip"))
    local = PHASE / "Server_01/Local_EN_B0_01/Backups" / (task + ".zip")
    if local.is_file():
        paths.append(local)
    if not paths:
        return None
    hashes = {digest(path.read_bytes()) for path in paths}
    require(len(hashes) == 1, "Conflicting backups for " + task)
    return paths[0]


def shard(task_name, task, parent, rows, pixels):
    path = archive_for(task_name)
    if path is None:
        return None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("Manifest.json"))
        require(len(names) == len(set(names)) == len(manifest) + 1, "Invalid archive members")
        require(set(names) == set(manifest) | {"Manifest.json"}, "Archive manifest differs")
        data = {}
        for name, expected in manifest.items():
            payload = archive.read(name)
            require(len(payload) == expected["bytes"] and digest(payload) == expected["sha256"],
                    "Archive content changed: " + name)
            if "/Assessment_01/Runs/" in name:
                require("/" + task_name + "/" in name, "Unexpected scientific output")
                data[Path(name).name] = payload
    verification = json.loads(data["Verification.json"])
    require(verification["status"] == "all_records_recounted_and_native_replay_passed", "No replay proof")
    require(verification["plan_sha256"] == ASSESSMENT_SHA, "Replay used another plan")
    require(verification["task"] == task_name, "Replay used another task")
    complete = json.loads(data["Complete.json"])
    start = json.loads(data["Start.json"])
    identity = json.loads(data["Identity.json"])
    require(start["plan_sha256"] == ASSESSMENT_SHA and start["task"] == task_name, "Start differs")
    require(complete["task"] == task_name, "Completion differs")
    for name, key in (("Predictions.jsonl", "predictions_sha256"), ("Identity.json", "identity_sha256")):
        require(digest(data[name]) == complete[key] == verification[key], "Receipt hash differs")
    require(identity["model_id"] == task["model"], "Model identity differs")
    require(identity["registry_entry"] == parent["models"][task["model"]]["identity"], "Fitted state changed")
    require(identity["strict_state_load"] and identity["precision"] == "float32", "Runtime differs")
    require(data["Predictions.jsonl"].endswith(b"\n"), "Incomplete final record")
    records = [json.loads(line) for line in data["Predictions.jsonl"].splitlines()]
    require(len(records) == complete["clips"] == verification["clips"] == len(task["positions"]),
            "Shard coverage differs")
    for position, record in zip(task["positions"], records):
        row = rows[position]
        require(record["position"] == position, "Manifest order differs")
        require(record["model"] == task["model"] and record["corpus"] == task["corpus"], "Wrong cohort")
        for key in ("clip_id", "source_id", "frames"):
            require(record[key] == row[key], "Source identity changed: " + key)
        encoded = record["encoding"]
        require(encoded["frames"] == encoded["state_frames"] == row["frames"], "Truncated clip")
        require(encoded["pixel_sha256"] == pixels[row["clip_id"]]["pixel_sha256"], "Pixels differ")
        require(list(record["predictions"]) == task["candidates"], "Decoder set differs")
        for candidate, prediction in record["predictions"].items():
            require(recount(row["text"], prediction["text"]) == prediction["counts"], "Edit recount differs")
    provenance = {"path": str(path), "sha256": digest(path.read_bytes()), "clips": len(records),
                  "task": task_name, "all_archive_members_verified": True}
    return records, provenance


def model(corpus, name, parent, assessment, rows, pixels):
    tasks = {key: value for key, value in assessment["tasks"].items() if value["model"] == name}
    require(bool(tasks) and all(task["corpus"] == corpus for task in tasks.values()), "Wrong model/corpus")
    records = []
    provenance = []
    missing = []
    for task_name, task in tasks.items():
        loaded = shard(task_name, task, parent, rows, pixels)
        if loaded is None:
            missing.append(task_name)
        else:
            batch, receipt = loaded
            records.extend(batch)
            provenance.append(receipt)
    if missing:
        return {"model": name, "status": "incomplete", "missing": missing,
                "verified_clips": len(records), "required_clips": len(rows)}, None, provenance
    records.sort(key=lambda row: row["position"])
    require([record["position"] for record in records] == list(range(len(rows))), "Population differs")
    return {"model": name, "status": "complete_verified_population", "clips": len(records)}, records, provenance
