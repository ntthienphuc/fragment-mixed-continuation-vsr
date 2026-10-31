"""Bind all required artifacts and disjoint tasks after actual beam preflight."""

import argparse

from common import COHORT, ENGINE, HERE, ROOT, STORE, checked_plan, lines, read, save, sha, utc


def partition(cohort, models, owner):
    assert owner in ("local", "server")
    order = sorted(range(len(cohort)), key=lambda index: (cohort[index]["frames"], cohort[index]["clip_id"]))
    tasks = {}
    for chunk in range(8):
        positions = order[chunk::8]
        samples = [positions[round(fraction * (len(positions) - 1))] for fraction in (0, 1/3, 2/3, 1)]
        for model in models:
            tasks[f"{model}_{chunk:02d}"] = {
                "model": model, "positions": positions, "owner": owner,
                "score_positions": samples, "replay_positions": [samples[0], samples[-1]]}
    for model in models:
        assigned = [position for task in tasks.values() if task["model"] == model for position in task["positions"]]
        assert sorted(assigned) == list(range(len(cohort)))
    return tasks


def main(owner):
    scientific = checked_plan()
    admission = read(HERE / "Admission.json")
    assert owner == admission["execution_location"]
    folder = HERE / "Beam_Execution"
    frozen = read(folder / "Plan.json")
    proof_path = folder / "Preflight_01/Verification.json"
    proof = read(proof_path)
    assert proof["status"] == "passed" and proof["exact_beam_replays"] == 8
    assert len(proof["independent_score_checks"]) == 24
    assert proof["plan_sha256"] == sha(folder / "Plan.json")
    design = read(HERE / "Execution_Design_Lock.json")
    for name, expected in design["files"].items():
        assert sha(ROOT / name) == expected, name
    vendor = ROOT / "Research_Phase_05_2026-10-13/vendor/auto_avsr"
    files = list(folder.glob("*.py")) + list(vendor.rglob("*.py"))
    files += [ENGINE / name for name in ("native_runtime.py", "confirmation_common.py", "model_registry.json")]
    registry = read(ENGINE / "model_registry.json")
    files += [ROOT / registry[key] for key in ("factory_source", "horizon_source", "safe_search_source")]
    files += [ROOT / "Research_Native_Decoder_Reuse_2026-10-27/verify.py", COHORT.parent / "verify_probe.py"]
    files += [ROOT / name for name in design["files"]]
    lock = {
        "status": "frozen", "utc": utc(), "plan_sha256": sha(folder / "Plan.json"),
        "tasks": partition(lines(COHORT), scientific["models"], owner),
        "runtime_receipt_sha256": sha(proof_path), "design_lock_sha256": sha(HERE / "Execution_Design_Lock.json"),
        "code_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(files))},
        "cache_index_sha256": {model: sha(STORE / "Cache" / model / "Index.jsonl") for model in frozen["models"]},
        "exports_sha256": {model: sha(STORE / "Models" / (model + ".pt")) for model in frozen["models"]}}
    save(folder / "Execution_Lock.json", lock)
    print({"status": "frozen", "tasks": len(lock["tasks"]), "owner": owner}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", choices=("local", "server"), required=True)
    main(parser.parse_args().owner)
