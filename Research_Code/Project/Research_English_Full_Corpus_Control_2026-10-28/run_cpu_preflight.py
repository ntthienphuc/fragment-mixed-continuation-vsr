"""Validate the existing F0 state on three source clips without using a GPU or test data."""

from pathlib import Path
import sys
import time
import traceback

from mechanism_common import HERE, ROOT, read, save, sha, utc


OPERATIONS = ROOT / "Research_Transfer_Decoder_Anchor_2026-10-28/Vi_Local_Operations"
sys.path.insert(0, str(OPERATIONS))
from queue_tools import child, free_memory_gib, snapshot


def validate_prerequisites():
    previous = ROOT / "Research_Transfer_Mechanism_Benchmark_2026-10-28/Source_Grid_CPU_01"
    complete = read(previous / "Complete.json")
    backup = read(previous / "Backup.json")
    assert complete["status"] == "four_models_three_source_grid_replays_passed"
    assert complete["independent_grid_comparisons"] == 132
    assert backup["status"] == "sha256_readback_passed"
    assert sha(backup["path"]) == backup["sha256"]
    proof = read(HERE / "Core_Verification_01.json")
    assert proof["status"] == "passed" and proof["plan_sha256"] == sha(HERE / "Plan.json")
    for name, digest in proof["scripts"].items():
        assert sha(HERE / name) == digest
    for name, entry in read(HERE / "Copied_Source.json").items():
        assert sha(HERE / name) == entry["sha256"]
    helper = read(OPERATIONS / "Operations_Lock.json")
    assert sha(OPERATIONS / "queue_tools.py") == helper["files"]["queue_tools.py"]
    assert sha(OPERATIONS / helper["verification_path"]) == helper["verification_sha256"]
    return {"previous_complete_sha256": sha(previous / "Complete.json"),
            "previous_backup_sha256": backup["sha256"],
            "core_proof_sha256": sha(HERE / "Core_Verification_01.json")}


def main():
    prerequisites = validate_prerequisites()
    folder = HERE / "Source_Grid_CPU_01"
    folder.mkdir(exist_ok=False)
    began = time.time()
    deadline = began + 1800
    model = "EN_F0_LAST"
    plan = read(HERE / "Plan.json")
    assert plan["models"] == [model]
    sources = [HERE / name for name in (
        "run_cpu_preflight.py", "source_grid_probe.py", "experiment_core.py",
        "runtime_bridge.py", "mechanism_common.py")]
    sources.append(OPERATIONS / "queue_tools.py")
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    lock = {"utc": utc(), "status": "bounded_cpu_source_grid_only", "model": model,
            "plan_sha256": sha(HERE / "Plan.json"), "scripts": hashes, "prerequisites": prerequisites,
            "device": "cpu", "threads": 2, "minimum_free_ram_gib": 5.5,
            "deadline_unix": deadline, "child_seconds": 900,
            "GPU_admission_granted": False, "assessment_used": False}
    save(folder / "Lock.json", lock)
    try:
        while free_memory_gib() < 5.5:
            if time.time() + 900 > deadline:
                raise TimeoutError("Insufficient RAM within the finite F0 CPU admission")
            print({"utc": utc(), "status": "waiting_for_memory",
                   "free_ram_gib": free_memory_gib()}, flush=True)
            time.sleep(15)
        assert time.time() + 900 <= deadline
        for name, digest in hashes.items():
            assert sha(ROOT / name) == digest
        python = ROOT / "Research_Phase_02_2026-10-08/environment/venv/Scripts/python.exe"
        output = folder / "Proofs" / model
        arguments = [str(python), "-B", "-X", "utf8", "-u", str(HERE / "source_grid_probe.py"),
                     model, str(output), "--seconds", "870", "--device", "cpu"]
        print({"utc": utc(), "status": "starting", "model": model,
               "free_ram_gib": free_memory_gib()}, flush=True)
        child(arguments, folder / "Process", time.time() + 900)
        receipt = read(output / "Complete.json")
        assert receipt["status"] == "three_source_inputs_full_grid_verified"
        assert receipt["model"] == model and receipt["plan_sha256"] == lock["plan_sha256"]
        assert receipt["device"] == "cpu" and receipt["assessment_used"] is False
        assert len(receipt["comparisons"]) == receipt["independent_count_checks"] == 33
        assert receipt["frames"] == [14, 46, 153]
        assert sha(output / "Identity.json") == receipt["identity_sha256"]
        for name, digest in receipt["probe_sha256"].items():
            assert sha(output / name) == digest
        save(folder / "Complete.json", {"utc": utc(), "status": "F0_three_source_grid_replays_passed",
             "model": model, "seconds": time.time() - began, "lock_sha256": sha(folder / "Lock.json"),
             "receipt_sha256": sha(output / "Complete.json"), "independent_grid_comparisons": 33,
             "performance_claim": False, "GPU_admission_granted": False})
        print({"utc": utc(), "status": "F0_preflight_passed", "comparisons": 33}, flush=True)
    except Exception:
        save(folder / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise
    finally:
        destination = Path("C:/Users/thien/Lip_Research_Backups/English_Full_Corpus_Control_20261028")
        destination = destination / ("Source_Grid_" + str(time.time_ns()) + ".zip")
        paths = [path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        backup = snapshot(ROOT, paths, destination)
        save(folder / "Backup.json", backup)
        print({"status": backup["status"], "backup": backup["path"], "sha256": backup["sha256"]}, flush=True)


if __name__ == "__main__":
    main()
