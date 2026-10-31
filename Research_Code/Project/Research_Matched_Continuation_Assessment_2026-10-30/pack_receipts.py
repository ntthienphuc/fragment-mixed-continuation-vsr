"""Package completed runtime-check evidence, excluding the disposable model export."""

import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


complete = json.loads((HERE / "Runtime_Check_01/Complete.json").read_text())
assert complete["status"] == "cpu_real_recovery_runtime_equivalence_passed"
assert not (HERE / "Runtime_Check_01/Failure.json").exists()
model = HERE / "Runtime_Check_01/model.pt"
assert sha(model) == complete["model_export"]["sha256"]
files = [(path, path.relative_to(HERE).as_posix()) for path in HERE.glob("*.py")]
files += [(HERE / name, name) for name in ("Plan.json", "Preparation.json", "Check.log")]
files += [(path, path.relative_to(HERE).as_posix()) for path in (HERE / "Runtime_Check_01").glob("*.json")]
files += [(BASE / "WholeWord_Finalization_01" / name, "export_helper/" + name)
          for name in ("finalize.py", "evidence.py")]
manifest = {}
package = HERE / "Receipts.zip"
with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as archive:
    for path, name in files:
        assert path.resolve().is_relative_to(BASE.resolve())
        content = path.read_bytes()
        archive.writestr(name, content)
        manifest[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    archive.writestr("Manifest.json", json.dumps(manifest, indent=2))
value = {"bytes": package.stat().st_size, "sha256": sha(package), "files": len(manifest),
         "actual_recognition_result": False, "model_export_included": False}
with (HERE / "Package.json").open("x") as stream:
    json.dump(value, stream, indent=2)
print(json.dumps(value))
