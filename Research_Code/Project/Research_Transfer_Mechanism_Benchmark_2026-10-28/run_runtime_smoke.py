"""Four serial CPU development replays, finite deadlines, and verified local backup."""

from pathlib import Path
import sys
import time
import traceback

from mechanism_common import HERE, ROOT, read, save, sha, utc


OPERATIONS = ROOT / "Research_Transfer_Decoder_Anchor_2026-10-28/Vi_Local_Operations"
sys.path.insert(0, str(OPERATIONS))
from queue_tools import child, free_memory_gib, snapshot


def main():
    folder = HERE / "Runtime_Validation_01"
    folder.mkdir(exist_ok=False)
    began = time.time()
    deadline = began + 1800
    plan = read(HERE / "Plan.json")
    preparation = read(HERE / "Preparation_Verification_02.json")
    assert preparation["status"] == "passed"
    assert preparation["plan_sha256"] == sha(HERE / "Plan.json")
    helper_lock = read(OPERATIONS / "Operations_Lock.json")
    helper_digest = sha(OPERATIONS / "queue_tools.py")
    assert helper_digest == helper_lock["files"]["queue_tools.py"]
    assert sha(OPERATIONS / helper_lock["verification_path"]) == helper_lock["verification_sha256"]
    scripts = [HERE / "runtime_bridge.py", HERE / "validate_runtime.py", HERE / "run_runtime_smoke.py",
               HERE / "mechanism_common.py", HERE / "verify_preparation.py", OPERATIONS / "queue_tools.py"]
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in scripts}
    lock = {"utc": utc(), "status": "bounded_cpu_development_only", "script_sha256": hashes,
            "plan_sha256": sha(HERE / "Plan.json"), "preparation_sha256": sha(HERE / "Preparation_Verification_02.json"),
            "models": plan["models"], "device": "cpu", "threads": 2, "minimum_free_ram_gib": 5.5,
            "deadline_unix": deadline, "per_model_seconds": 600,
            "GPU_admission_granted": False, "assessment_clips_used": False}
    save(folder / "Lock.json", lock)
    python = ROOT / "Research_Phase_02_2026-10-08/environment/venv/Scripts/python.exe"
    completed = []
    try:
        for model in plan["models"]:
            while free_memory_gib() < 5.5:
                if time.time() + 600 > deadline:
                    raise TimeoutError("Insufficient RAM within finite CPU admission")
                print({"utc": utc(), "status": "waiting_for_memory", "model": model,
                       "free_ram_gib": free_memory_gib()}, flush=True)
                time.sleep(15)
            assert time.time() + 600 <= deadline
            for name, digest in hashes.items():
                assert sha(ROOT / name) == digest
            output = folder / (model + ".json")
            print({"utc": utc(), "status": "starting", "model": model,
                   "free_ram_gib": free_memory_gib()}, flush=True)
            arguments = [str(python), "-B", "-X", "utf8", "-u", str(HERE / "validate_runtime.py"),
                         model, str(output), "--seconds", "570"]
            child(arguments, folder / model, min(deadline, time.time() + 600))
            receipt = read(output)
            assert receipt["status"] == "cpu_shortest_development_native_replay_passed"
            assert receipt["model"] == model and receipt["plan_sha256"] == lock["plan_sha256"]
            assert receipt["test_clips_used"] is False and receipt["GPU_admission_granted"] is False
            completed.append({"model": model, "receipt_sha256": sha(output), "frames": receipt["frames"]})
            print({"utc": utc(), "status": "model_passed", "model": model}, flush=True)
        save(folder / "Complete.json", {"utc": utc(), "status": "four_cpu_development_replays_passed",
             "models": completed, "seconds": time.time() - began, "lock_sha256": sha(folder / "Lock.json"),
             "performance_claim": False, "GPU_admission_granted": False})
    except Exception:
        save(folder / "Failure.json", {"utc": utc(), "error": traceback.format_exc(), "completed": completed})
        raise
    finally:
        destination = Path("C:/Users/thien/Lip_Research_Backups/Transfer_Mechanism_20261028")
        destination = destination / ("Runtime_Preparation_" + str(time.time_ns()) + ".zip")
        paths = [path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        receipt = snapshot(ROOT, paths, destination)
        save(folder / "Backup.json", receipt)
        print({"status": receipt["status"], "backup": receipt["path"], "sha256": receipt["sha256"]}, flush=True)


if __name__ == "__main__":
    main()
