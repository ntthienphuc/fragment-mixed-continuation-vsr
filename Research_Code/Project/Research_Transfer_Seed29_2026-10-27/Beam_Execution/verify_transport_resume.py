"""Exercise real assembly, readback, and rejection of a corrupted part."""

import hashlib
import json
from pathlib import Path
import tempfile

from assemble_resume import execute
from common import HERE, save, utc


def fixture(base):
    pieces = [b"a" * 257, b"boundary" * 131, bytes(range(256))]
    (base / "parts").mkdir()
    parts = []
    for index, payload in enumerate(pieces):
        name = f"part_{index:03d}.bin"
        (base / "parts" / name).write_bytes(payload)
        parts.append({"name": name, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
    combined = b"".join(pieces)
    plan = {"parts": parts, "bytes": len(combined), "sha256": hashlib.sha256(combined).hexdigest()}
    (base / "Transport_Parts.json").write_text(json.dumps(plan), encoding="utf-8")
    return combined


def main():
    with tempfile.TemporaryDirectory(prefix="lip_transport_check_") as temporary:
        base = Path(temporary)
        combined = fixture(base)
        assert len(execute(base, "status")["verified"]) == 3
        proof = execute(base, "assemble")
        assert (base / "Server_Workload.zip").read_bytes() == combined
        assert proof["bytes"] == len(combined)
        try:
            execute(base, "assemble")
        except AssertionError as error:
            assert "Preserve existing" in str(error)
        else:
            raise AssertionError("Existing archive was not protected")
    with tempfile.TemporaryDirectory(prefix="lip_transport_reject_") as temporary:
        base = Path(temporary)
        fixture(base)
        path = base / "parts/part_001.bin"
        path.write_bytes(b"x" * path.stat().st_size)
        try:
            execute(base, "assemble")
        except AssertionError as error:
            assert "Wrong part hash" in str(error)
        else:
            raise AssertionError("Corruption was accepted")
        assert not (base / "Server_Workload.zip").exists()
    record = {"utc": utc(), "status": "real_assembly_and_corruption_rejection_passed",
              "verified_parts": 3, "byte_exact_readback": True, "existing_output_preserved": True}
    save(HERE / "Transport_Resume_Verification.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
