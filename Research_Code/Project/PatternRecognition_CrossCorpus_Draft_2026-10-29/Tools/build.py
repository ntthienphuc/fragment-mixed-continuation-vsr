"""Build the journal draft, render every page, and package only LaTeX inputs."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Source"
BUILD = ROOT / "Build"
QA = ROOT / "QA"
TEXBIN = Path(os.environ["APPDATA"]) / "TinyTeX" / "bin" / "windows"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    environment = dict(os.environ)
    environment["PATH"] = str(TEXBIN) + os.pathsep + environment["PATH"]
    command = [str(TEXBIN / "latexmk.exe"), "-pdf", "-interaction=nonstopmode",
               "-halt-on-error", "-outdir=" + str(BUILD), "main.tex"]
    result = subprocess.run(command, cwd=SOURCE, env=environment, capture_output=True, timeout=180)
    (QA / "Build_stdout.log").write_bytes(result.stdout)
    (QA / "Build_stderr.log").write_bytes(result.stderr)
    if result.returncode:
        print(result.stdout.decode("utf-8", errors="replace")[-8000:])
        print(result.stderr.decode("utf-8", errors="replace")[-1500:])
        raise SystemExit(result.returncode)
    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    unresolved = re.findall(r".*(?:undefined|Overfull|Rerun to).*", log)
    if unresolved:
        print("\n".join(unresolved))
        raise RuntimeError("Resolve LaTeX references or layout warnings before packaging")
    final_pdf = ROOT / "Manuscript.pdf"
    shutil.copy2(BUILD / "main.pdf", final_pdf)
    document = pymupdf.open(final_pdf)
    text_pages = []
    thumbnails = []
    for index, page in enumerate(document):
        text_pages.append(page.get_text())
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)
        output = QA / f"page_{index + 1:02d}.png"
        pixmap.save(output)
        thumbnail = Image.open(output).convert("RGB")
        thumbnail.thumbnail((355, 475))
        tile = Image.new("RGB", (375, 505), "#eeeeee")
        tile.paste(thumbnail, ((375 - thumbnail.width) // 2, 22))
        ImageDraw.Draw(tile).text((12, 4), f"Page {index + 1}", fill="black")
        thumbnails.append(tile)
    for first in range(0, len(thumbnails), 6):
        sheet = Image.new("RGB", (1125, 1010), "#cccccc")
        for offset, tile in enumerate(thumbnails[first:first + 6]):
            sheet.paste(tile, ((offset % 3) * 375, (offset // 3) * 505))
        sheet.save(QA / f"contact_{first // 6 + 1:02d}.jpg", quality=90)
    text = "\n\n".join(text_pages)
    (QA / "Extracted_Text.txt").write_text(text, encoding="utf-8")
    assert "61.882" in text and "63.916" in text and "66.792" in text
    assert re.search(r"\bpending\b", text, flags=re.IGNORECASE) is None
    assert "T2" in text and "T4" in text
    assert all(value in text for value in ["62.002", "62.444", "66.749", "1,039"])
    assert "[?]" not in text
    archive = ROOT / "Overleaf_Source.zip"
    source_files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in source_files:
            package.write(path, path.relative_to(SOURCE).as_posix())
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        for path in source_files:
            assert package.read(path.relative_to(SOURCE).as_posix()) == path.read_bytes()
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "compiled_all_pages_rendered_package_readback_passed",
        "visual_review": "pending",
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "pages": len(document),
        "words_approximate": len(text.split()),
        "latex_warnings_requiring_correction": unresolved,
        "pdf_sha256": sha(final_pdf),
        "source_zip_sha256": sha(archive),
        "source_files": {path.relative_to(SOURCE).as_posix(): sha(path) for path in source_files},
        "zip_bytes": archive.stat().st_size,
        "no_T2_T4_partial_scores": True,
        "completed_T2_T4_scores_included": True,
    }
    (QA / "Build_Receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ["status", "pages", "elapsed_seconds", "zip_bytes"]}))


if __name__ == "__main__":
    main()
