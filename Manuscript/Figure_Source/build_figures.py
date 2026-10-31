"""Build two editorial figures with pdfLaTeX; leave all numerical plots unchanged."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
NAMES = ("training_design", "dataset_examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    if args.build_dir:
        build = args.build_dir.resolve()
        build.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="fragment-mixed-figures-")
        build = Path(temporary.name)
    try:
        for name in NAMES:
            for iteration in (1, 2):
                command = ["pdflatex", "-halt-on-error", "-interaction=nonstopmode",
                           "-output-directory=" + str(build), name + ".tex"]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=120)
                (build / f"{name}.pass{iteration}.log").write_bytes(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"{name}, pass {iteration}: " + result.stdout.decode(errors="replace")[-4000:])
            log = (build / f"{name}.log").read_text(errors="replace")
            for issue in ("Overfull", "Undefined control sequence", "LaTeX Error", "Rerun to get"):
                if issue in log:
                    raise RuntimeError(f"{name}: inspect {issue!r} in {build}")
            shutil.copy2(build / f"{name}.pdf", output / f"{name}.pdf")
            print(f"Built {output / (name + '.pdf')}")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
