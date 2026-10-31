"""Archive phase code and evidence, with byte readback for every member."""

import hashlib
import json
import time
import zipfile

from common import HERE, STORE, save, sha, utc


def main():
    assert not (STORE / "Cache").exists() and not (HERE / "Greedy_01").exists()
    STORE.mkdir(parents=True, exist_ok=True)
    target = STORE / ("Preparation_" + str(time.time_ns()) + ".zip")
    entries = {}
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(HERE.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            name = path.relative_to(HERE).as_posix()
            content = path.read_bytes()
            expected = hashlib.sha256(content).hexdigest()
            assert sha(path) == expected
            archive.writestr(name, content)
            entries[name] = expected
        archive.writestr("Archive_Manifest.json", json.dumps(entries, indent=2))
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == set(entries) | {"Archive_Manifest.json"}
        for name, expected in entries.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    receipt = {"utc": utc(), "status": "all_preparation_members_sha256_readback_passed",
               "path": str(target), "sha256": sha(target), "bytes": target.stat().st_size,
               "files": len(entries), "new_destination_predictions": False}
    save(HERE / "Inference_Backups" / (target.stem + ".json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
