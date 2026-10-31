"""Finite serial CPU source-grid checks while the admitted GPU workloads continue."""

from pathlib import Path
import sys
import time
import traceback

from mechanism_common import HERE, ROOT, read, save, sha, utc


OPERATIONS = ROOT / "Research_Transfer_Decoder_Anchor_2026-10-28/Vi_Local_Operations"
sys.path.insert(0, str(OPERATIONS))
from queue_tools import child, free_memory_gib, snapshot


def main():
    folder = HERE / "Source_Grid_CPU_01"
    folder.mkdir(exist_ok=False)
    began = time.time()
    deadline = began + 3600
    plan = read(HERE / "Plan.json")
    core_proof = read(HERE / "Core_Verification_01.json")
    assert core_proof["status"] == "passed" and core_proof["plan_sha256"] == sha(HERE / "Plan.json")
    for name, digest in core_proof["scripts"].items():
        assert sha(HERE / name) == digest
    helper_lock = read(OPERATIONS / "Operations_Lock.json")
    assert sha(OPERATIONS / "queue_tools.py") == helper_lock["files"]["queue_tools.py"]
    assert sha(OPERATIONS / helper_lock["verification_path"]) == helper_lock["verification_sha256"]
    files = [HERE / name for name in (
        "source_grid_probe.py", "run_source_grid.py", "experiment_core.py", "runtime_bridge.py", "mechanism_common.py"
    )]
    files.append(OPERATIONS / "queue_tools.py")
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    lock = {"utc": utc(), "status": "bounded_cpu_source_grid_only", "script_sha256": hashes,
            "plan_sha256": sha(HERE / "Plan.json"), "core_verification_sha256": sha(HERE / "Core_Verification_01.json"),
            "models": plan["models"], "device": "cpu", "threads": 2, "minimum_free_ram_gib": 5.5,
            "deadline_unix": deadline, "per_model_seconds": 900,
            "GPU_admission_granted": False, "assessment_clips_used": False}
    save(folder / "Lock.json", lock)
    python = ROOT / "Research_Phase_02_2026-10-08/environment/venv/Scripts/python.exe"
    completed = []
    try:
        for model in plan["models"]:
            while free_memory_gib() < 5.5:
                if time.time() + 900 > deadline:
                    raise TimeoutError("Insufficient RAM within finite source-grid admission")
                print({"utc": utc(), "status": "waiting_for_memory", "model": model,
                       "free_ram_gib": free_memory_gib()}, flush=True)
                time.sleep(15)
            assert time.time() + 900 <= deadline
            for name, digest in hashes.items():
                assert sha(ROOT / name) == digest
            output = folder / "Proofs" / model
            print({"utc": utc(), "status": "starting", "model": model,
                   "free_ram_gib": free_memory_gib()}, flush=True)
            arguments = [str(python), "-B", "-X", "utf8", "-u", str(HERE / "source_grid_probe.py"),
                         model, str(output), "--seconds", "870", "--device", "cpu"]
            child(arguments, folder / "Processes" / model, min(deadline, time.time() + 900))
            receipt_path = output / "Complete.json"
            receipt = read(receipt_path)
            assert receipt["status"] == "three_source_inputs_full_grid_verified"
            assert receipt["model"] == model and receipt["plan_sha256"] == lock["plan_sha256"]
            assert receipt["assessment_used"] is False and receipt["device"] == "cpu"
            assert len(receipt["comparisons"]) == receipt["independent_count_checks"] == 33
            completed.append({"model": model, "receipt_sha256": sha(receipt_path), "frames": receipt["frames"]})
            print({"utc": utc(), "status": "model_passed", "model": model}, flush=True)
        save(folder / "Complete.json", {"utc": utc(), "status": "four_models_three_source_grid_replays_passed",
             "models": completed, "seconds": time.time() - began, "lock_sha256": sha(folder / "Lock.json"),
             "independent_grid_comparisons": 132, "performance_claim": False, "GPU_admission_granted": False})
    except Exception:
        save(folder / "Failure.json", {"utc": utc(), "error": traceback.format_exc(), "completed": completed})
        raise
    finally:
        destination = Path("C:/Users/thien/Lip_Research_Backups/Transfer_Mechanism_20261028")
        destination = destination / ("Source_Grid_" + str(time.time_ns()) + ".zip")
        paths = [path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        receipt = snapshot(ROOT, paths, destination)
        save(folder / "Backup.json", receipt)
        print({"status": receipt["status"], "backup": receipt["path"], "sha256": receipt["sha256"]}, flush=True)


if __name__ == "__main__":
    main()
