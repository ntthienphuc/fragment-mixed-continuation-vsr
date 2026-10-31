"""Compile the delivered ZIP from a separate directory without project dependencies."""

import hashlib
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / "Overleaf_Source.zip"
    target = ROOT / "Build" / ("Package_Check_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    target.mkdir()
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            resolved = (target / name).resolve()
            assert target.resolve() in resolved.parents, name
        package.extractall(target)
    texbin = Path(os.environ["APPDATA"]) / "TinyTeX" / "bin" / "windows"
    environment = dict(os.environ)
    environment["PATH"] = str(texbin) + os.pathsep + environment["PATH"]
    started = time.monotonic()
    result = subprocess.run(
        [str(texbin / "latexmk.exe"), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=target,
        env=environment,
        capture_output=True,
        timeout=120,
    )
    (target / "Compile_stdout.log").write_bytes(result.stdout)
    (target / "Compile_stderr.log").write_bytes(result.stderr)
    assert result.returncode == 0, result.stdout.decode(errors="replace")[-4000:]
    expected = pymupdf.open(ROOT / "Manuscript.pdf")
    actual = pymupdf.open(target / "main.pdf")
    assert len(expected) == len(actual)
    for old, new in zip(expected, actual):
        assert old.get_text() == new.get_text(), old.number
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_fresh_standalone_zip_compile_exact_page_text",
        "pages": len(actual),
        "compile_seconds": round(time.monotonic() - started, 2),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "check_directory": target.relative_to(ROOT).as_posix(),
        "overleaf_browser_tested": False,
    }
    (ROOT / "QA" / "Package_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
