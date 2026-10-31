"""Exercise the actual worker and independent verifier on one real recovery-model clip."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import HERE, ROOT, read, save, sha, utc


def main(output):
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    save(output / "Start.json", {"utc": utc(), "pid": os.getpid(), "purpose": "cpu_worker_replay_integration"})
    private = ROOT.parent
    checkpoint = private / "Recovery_Snapshots/20261029T174340Z/last.pt"
    expected = "d758745a96cd04a5a45fadfe632a74238d1b36c6f10a4f3d399e8b5ba4741fc2"
    assert sha(checkpoint) == expected
    sys.path.insert(0, str(private / "WholeWord_Finalization_01"))
    from finalize import export_model
    import torch

    torch.set_num_threads(2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == 3000 and payload["identity"]["native_graph"] == "T0"
    exported = export_model(payload, expected, output / "model.pt")
    registry = read(ROOT / "Research_Transfer_Full_Benchmark_2026-10-27/model_registry.json")
    entry = dict(registry["models"]["VLR_T0_LAST"])
    entry.update({"checkpoint": (output / "model.pt").relative_to(ROOT).as_posix(),
                  "checkpoint_sha256": exported["sha256"], "native_graph": "T0",
                  "selected_training_step": 3000, "training_identity": payload["identity"]})
    registry["models"] = {"WW_RECOVERY_RUNTIME_CHECK": entry}
    registry["status"] = "recovery_worker_integration_only"
    save(output / "model_registry.json", registry)
    del payload
    gc.collect()
    admission = {"utc": utc(), "status": "cpu_worker_integration_admitted", "mode": "fixture",
                 "assessment_plan_sha256": sha(HERE / "Plan.json"),
                 "code_lock_sha256": sha(HERE / "Code_Lock.json"),
                 "model": "WW_RECOVERY_RUNTIME_CHECK", "device": "cpu", "readout": "common",
                 "tasks": ["source_fixture"], "registry_sha256": sha(output / "model_registry.json"),
                 "deadline_unix": time.time() + 900, "maximum_task_seconds": 600}
    save(output / "Admission.json", admission)
    arguments = ["source_fixture", "--model", "WW_RECOVERY_RUNTIME_CHECK", "--binding", str(output),
                 "--admission", str(output / "Admission.json"), "--output", str(output / "Run"),
                 "--device", "cpu", "--mode", "fixture", "--readout", "common"]
    for name in ("run_task.py", "verify_task.py"):
        with (output / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-X", "utf8", "-B", "-u", str(HERE / name), *arguments],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=650)
        assert process.returncode == 0, "Read exact subprocess failure: " + name
    proof = read(output / "Run/Verification.json")
    assert proof["mode"] == "fixture" and proof["clips"] == 1
    assert proof["status"] == "all_records_recounted_and_fresh_replay_passed"
    assert proof["candidates"] == ["beam_c0.1"] and not proof["complete_population_receipt"]
    assert not torch.cuda.is_initialized()
    save(output / "Complete.json", {"utc": utc(), "status": "real_cpu_worker_and_independent_replay_passed",
         "training_step": 3000, "model_export": exported, "verification": proof,
         "seconds": time.monotonic() - began, "cuda_initialized": False, "final_assessment_admitted": False})
    print(json.dumps(read(output / "Complete.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    arguments = parser.parse_args()
    assert 1 <= arguments.attempt <= 99
    output = HERE / f"Worker_Integration_{arguments.attempt:02}"
    assert not output.exists(), "Preserve prior integration attempts"
    try:
        main(output)
    except BaseException:
        save(output / "Failure.json", {"utc": utc(), "error": traceback.format_exc()})
        raise
