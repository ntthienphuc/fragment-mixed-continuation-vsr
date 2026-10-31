"""Guard first, upload four verified streams, and start the finite native beam queue."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import zipfile

from cloud import REMOTE, remote, upload
from common import HERE, PHASE, ROOT, STORE, read, save, sha, utc


def main():
    package = read(PHASE / "Beam_Package.json")
    assert sha(package["path"]) == package["sha256"]
    config = read(HERE / "Server_Config.json")
    assert time.time() < config["science_deadline_unix"] - 600
    assert read(HERE / "Transport_Resume_Verification.json")["byte_exact_readback"]
    setup = [HERE / "common.py", PHASE / "common.py", HERE / "server_guard.py", HERE / "Server_Config.json"]
    mapping = {"project/" + path.relative_to(ROOT).as_posix(): path for path in setup}
    hashes = {name: sha(path) for name, path in mapping.items()}
    target = STORE / "Guard_Setup.zip"
    with zipfile.ZipFile(target, "x", zipfile.ZIP_STORED) as archive:
        for name, path in mapping.items():
            archive.write(path, name)
        archive.writestr("Guard_Manifest.json", json.dumps(hashes))
    remote(["mkdir", "-p", REMOTE + "/parts"])
    upload(target, REMOTE + "/Guard_Setup.zip")
    upload(HERE / "start_guard.py", REMOTE + "/start_guard.py")
    guard = json.loads(remote(["python", "-B", REMOTE + "/start_guard.py"], timeout=90))
    assert guard["status"] == "passed" and guard["pod_id"] == config["pod_id"]
    save(HERE / "Guard_Verified_Local.json", {"utc": utc(), **guard})
    folder = STORE / "Transport_Parts"
    folder.mkdir(exist_ok=False)
    parts = []
    with open(package["path"], "rb") as stream:
        while payload := stream.read(128 * 2**20):
            name = f"part_{len(parts):03d}.bin"
            with (folder / name).open("xb") as output:
                output.write(payload)
            parts.append({"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    save(HERE / "Transport_Parts.json", {"parts": parts, "bytes": package["bytes"], "sha256": package["sha256"]})
    upload(HERE / "Transport_Parts.json", REMOTE + "/Transport_Parts.json")
    upload(HERE / "assemble_resume.py", REMOTE + "/assemble_resume.py")
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(upload, folder / part["name"], REMOTE + "/parts/" + part["name"], 360): part
                   for part in parts}
        for future in as_completed(futures):
            future.result()
            print({"utc": utc(), "uploaded_part": futures[future]["name"],
                   "seconds": time.monotonic() - began}, flush=True)
    verified = json.loads(remote(["python", REMOTE + "/assemble_resume.py", "assemble"], timeout=180))
    assert verified["sha256"] == package["sha256"]
    save(HERE / "Transport_Verification.json", {"utc": utc(), **verified})
    overlay_files = [HERE / "Server_Config.json", HERE / "Dispatch/Server_01.json"]
    overlay = {"project/" + path.relative_to(ROOT).as_posix(): path for path in overlay_files}
    archive_path = STORE / "Server_Overlay.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_STORED) as archive:
        for name, path in overlay.items():
            archive.write(path, name)
    save(HERE / "Overlay.json", {"sha256": sha(archive_path), "files": {
        name: sha(path) for name, path in overlay.items()}})
    upload(PHASE / "Beam_Package.json", REMOTE + "/Package.json")
    upload(archive_path, REMOTE + "/Server_Overlay.zip")
    upload(HERE / "Overlay.json", REMOTE + "/Overlay.json")
    for name in ("remote_bootstrap.py", "remote_monitor.py"):
        upload(HERE / name, REMOTE + "/" + name)
    response = remote(["python", "-B", REMOTE + "/remote_bootstrap.py"], timeout=300)
    print(response[-2500:], flush=True)
    save(HERE / "Deployment.json", {"status": "tmux_guard_and_queue_launched", "utc": utc(),
                                   "package_sha256": package["sha256"]})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(HERE / "Deployment_Failure.json", {"utc": utc(), "error": repr(error)})
        marker = REMOTE + "/project/Research_Transfer_Seed29_2026-10-27/Beam_Execution/Server_Failure.json"
        content = json.dumps({"utc": utc(), "deployment_failed": True})
        command = "from pathlib import Path; Path(" + repr(marker) + ").write_text(" + repr(content) + ")"
        try:
            remote(["python", "-c", command])
        except Exception:
            save(HERE / "Stop_Needed.json", {"utc": utc(), "reason": "remote_failure_marker_unreachable"})
        raise
