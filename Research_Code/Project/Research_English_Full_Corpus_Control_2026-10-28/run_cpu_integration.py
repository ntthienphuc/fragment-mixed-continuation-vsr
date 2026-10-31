"""Execute the actual frozen worker and fresh-pixel verifier on source-only probes."""

from pathlib import Path
import sys
import time
import traceback

from execution_common import VALIDATION_TASK, checked_execution
from mechanism_common import HERE, ROOT, read, save, sha, utc


OPERATIONS = ROOT / "Research_Transfer_Decoder_Anchor_2026-10-28/Vi_Local_Operations"
sys.path.insert(0, str(OPERATIONS))
from queue_tools import child, free_memory_gib, snapshot


def main():
    plan, execution = checked_execution()
    helper = read(OPERATIONS / "Operations_Lock.json")
    assert sha(OPERATIONS / "queue_tools.py") == helper["files"]["queue_tools.py"]
    assert sha(OPERATIONS / helper["verification_path"]) == helper["verification_sha256"]
    folder = HERE / "Execution_Integration_01"
    folder.mkdir(exist_ok=False)
    began = time.time()
    deadline = began + 1800
    save(folder / "Start.json", {"utc": utc(), "deadline_unix": deadline,
         "execution_lock_sha256": sha(HERE / "Execution_Lock.json"), "GPU_used": False,
         "assessment_used": False, "minimum_free_ram_gib": 5.5})
    try:
        while free_memory_gib() < 5.5:
            if time.time() + 900 > deadline:
                raise TimeoutError("Insufficient RAM within the finite integration window")
            print({"utc": utc(), "status": "waiting_for_memory", "free_ram_gib": free_memory_gib()}, flush=True)
            time.sleep(15)
        locations = folder / "Locations.json"
        save(locations, {})
        proof_path = HERE / "Source_Grid_CPU_01/Proofs/EN_F0_LAST/Complete.json"
        admission_path = folder / "Admission.json"
        save(admission_path, {"utc": utc(), "status": "bounded_F0_execution_admitted",
             "purpose": "source_integration", "execution_location": "local", "paid_allocation": False,
             "admitted_unix": began, "deadline_unix": deadline, "maximum_seconds": 1800,
             "device": "cpu", "threads": 2, "tasks": [VALIDATION_TASK], "models": ["EN_F0_LAST"],
             "stages": ["development"], "plan_sha256": sha(HERE / "Plan.json"),
             "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
             "runtime_preflight": {"path": proof_path.relative_to(ROOT).as_posix(), "sha256": sha(proof_path)},
             "locations_path": locations.relative_to(ROOT).as_posix(), "locations_sha256": sha(locations)})
        python = ROOT / "Research_Phase_02_2026-10-08/environment/venv/Scripts/python.exe"
        common = [str(python), "-B", "-X", "utf8", "-u"]
        arguments = [str(HERE / "run_full_task.py"), VALIDATION_TASK, "--attempt", "CPUIntegration01",
                     "--admission", str(admission_path), "--device", "cpu"]
        print({"utc": utc(), "stage": "actual_frozen_worker", "free_ram_gib": free_memory_gib()}, flush=True)
        child(common + arguments, folder / "Worker", deadline)
        output = HERE / "Runs/CPUIntegration01" / VALIDATION_TASK
        print({"utc": utc(), "stage": "independent_fresh_pixel_verifier"}, flush=True)
        arguments = [str(HERE / "verify_full_task.py"), VALIDATION_TASK, str(output),
                     "--admission", str(admission_path), "--device", "cpu"]
        child(common + arguments, folder / "Verifier", deadline)
        proof = read(output / "Verification.json")
        assert proof["status"] == "passed" and proof["purpose"] == "source_integration"
        assert proof["clips"] == 3 and proof["independent_counts_checked"] == 33
        assert len(proof["fresh_pixel_replays"]) == 22
        save(folder / "Complete.json", {"utc": utc(), "status": "frozen_worker_and_verifier_integrated",
             "source_clips": 3, "independent_counts": 33, "fresh_pixel_replays": 22,
             "execution_lock_sha256": sha(HERE / "Execution_Lock.json"),
             "verification_sha256": sha(output / "Verification.json"), "seconds": time.time() - began,
             "benchmark_clips_completed": 0, "GPU_used": False, "assessment_used": False})
        print({"utc": utc(), "status": "F0_worker_integration_passed"}, flush=True)
    except BaseException:
        save(folder / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise
    finally:
        destination = Path("C:/Users/thien/Lip_Research_Backups/English_Full_Corpus_Control_20261028")
        destination = destination / ("Worker_Integration_" + str(time.time_ns()) + ".zip")
        paths = [path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        receipt = snapshot(ROOT, paths, destination)
        save(folder / "Backup.json", receipt)
        print({"status": receipt["status"], "backup": receipt["path"], "sha256": receipt["sha256"]}, flush=True)


if __name__ == "__main__":
    main()
