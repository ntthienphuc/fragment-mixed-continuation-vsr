"""Recover only complete, verified prior development predictions with identical inputs."""

from pathlib import Path

from admission import require, verify_plan
from common import HERE, ROOT, lines, read, sha
from select_decoder import choose


def prior_predictions(model):
    plan = verify_plan()
    reuse = plan["development_reuse"][model]
    phase = ROOT / reuse["source_phase"]
    receipt_path = phase / "Completion_Receipt.json"
    summary_path = phase / "Analysis_01/Summary.json"
    receipt = read(receipt_path)
    summary = read(summary_path)
    expected_status = ("seed29_full_matrix_analysis_verified" if "Seed29" in phase.name
                       else "complete_matrix_analysis_verified")
    require(receipt["status"] == expected_status, "Prior matrix is incomplete")
    require(receipt["summary_sha256"] == sha(summary_path), "Prior summary changed")
    expected = 11682 if "Seed29" in phase.name else 29205
    require(receipt["independent_counts_checked"] == expected, "Prior recount incomplete")
    evidence = {str(path.relative_to(ROOT).as_posix()): sha(path) for path in (receipt_path, summary_path)}
    for name, digest in summary["input_sha256"].items():
        require(sha(ROOT / name) == digest, "Prior evidence changed: " + name)
        evidence[name] = digest
    registry = read(phase / "model_registry.json")["models"][model]
    current = read(HERE / "model_registry.json")["models"][model]
    require(registry == current, "Checkpoint, vocabulary, graph, or transform registry differs")
    lock_path = phase / "Beam_Execution/Execution_Lock.json"
    lock = read(lock_path)
    records = {}
    for name, task in lock["tasks"].items():
        if task["model"] != model:
            continue
        matches = list((phase / "Beam_Execution/Runs").glob("*/" + name + "/Verification.json"))
        require(len(matches) == 1, "Expected one complete verified prior task: " + name)
        proof = read(matches[0])
        path = matches[0].parent / "Predictions.jsonl"
        require(proof["status"] == "passed", "Prior task failed verification")
        require(proof["execution_lock_sha256"] == sha(lock_path), "Prior task lock differs")
        require(proof["predictions_sha256"] == sha(path), "Prior predictions changed")
        rows = lines(path)
        require([row["index"] for row in rows] == task["positions"], "Prior task is incomplete")
        for row in rows:
            require(row["clip_id"] not in records, "Duplicate prior prediction")
            records[row["clip_id"]] = {**row, "model": model, "split": "development"}
        for item in (path, matches[0]):
            evidence[item.relative_to(ROOT).as_posix()] = sha(item)
    greedy_path = phase / "Greedy_01" / model / "Predictions.jsonl"
    greedy_proof_path = greedy_path.parent / "Verification.json"
    greedy_proof = read(greedy_proof_path)
    require(greedy_proof["status"] == "passed", "Prior greedy evidence is incomplete")
    require(greedy_proof["predictions_sha256"] == sha(greedy_path), "Prior greedy predictions changed")
    greedy = lines(greedy_path)
    require(len(greedy) == len(records) == 531, "Expected complete prior cohort")
    for row in greedy:
        output = records[row["clip_id"]]
        require(output["cache_sha256"] == row["cache_sha256"], "Greedy/beam inputs differ")
        require(output["source_id"] == row["source_id"], "Source identities differ")
        require(not set(output["predictions"]).intersection(row["predictions"]), "Candidate collision")
        output["predictions"].update(row["predictions"])
    old_cohort = lines(ROOT / "Research_Interval_Transfer_2026-10-27/Cohort.jsonl")
    current_rows = {row["clip_id"]: row for row in lines(HERE / "Development.jsonl")}
    cache = Path(reuse["cache_folder"])
    require(sha(cache / "Index.jsonl") == reuse["cache_index_sha256"], "Cache identity changed")
    cached = {row["clip_id"]: row for row in lines(cache / "Index.jsonl")}
    pixels = {row["clip_id"]: row for row in lines(HERE / "Pixels_01/Index.jsonl")}
    require(set(cached) == set(records), "Cache population differs")
    for row in old_cohort:
        current_row = current_rows[row["clip_id"]]
        for field in ("clip_id", "reference", "source_id", "frames", "input"):
            require(row[field] == current_row[field], "Prior/current cohort field differs: " + field)
        item = cached[row["clip_id"]]
        require(item["sha256"] == records[row["clip_id"]]["cache_sha256"], "Prior cache binding differs")
        require(item["pixels_sha256"] == pixels[row["clip_id"]]["gray_sha256"], "Pixel convention differs")
    check = choose(old_cohort, list(records.values()), plan["candidate_order"])
    require(check["independent_prediction_counts_checked"] == 5841, "Prior counts incomplete")
    for item in (lock_path, greedy_path, greedy_proof_path):
        evidence[item.relative_to(ROOT).as_posix()] = sha(item)
    return records, evidence
