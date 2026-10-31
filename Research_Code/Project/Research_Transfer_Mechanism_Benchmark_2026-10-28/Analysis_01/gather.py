"""Join complete new predictions with verified parent controls, retaining all planned outcomes."""

from datetime import datetime
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent
ROOT = SCIENCE.parent
sys.path.insert(0, str(SCIENCE))

from collect_full_results import collect
from execution_common import check_scope, checked_execution, checked_selection, validate_budget
from experiment_core import cohort
from mechanism_common import read, sha
from parent_tables import parents
from transfer_statistics import FIELDS, require


def check_admissions(manifest_path, plan, selection):
    manifest = read(manifest_path)
    inputs = {}
    selection_time = datetime.fromisoformat(selection["utc"]).timestamp()
    for name, item in manifest["tasks"].items():
        folder = (ROOT / item["path"]).resolve()
        folder.relative_to((SCIENCE / "Runs").resolve())
        task = plan["tasks"][name]
        start = read(folder / "Start.json")
        require(start["started_unix"] >= selection_time, "Assessment preceded complete decoder selection")
        admission_path = (ROOT / start["admission_path"]).resolve()
        admission_path.relative_to(ROOT)
        require(sha(admission_path) == start["admission_sha256"], "Admission changed")
        admission = read(admission_path)
        validate_budget(admission, start["started_unix"])
        check_scope(admission, name, task, start["device"], plan)
        require(admission["purpose"] == "full_evaluation", "Integration admission reused")
        require(admission["plan_sha256"] == sha(SCIENCE / "Plan.json"), "Wrong admitted plan")
        require(admission["execution_lock_sha256"] == sha(SCIENCE / "Execution_Lock.json"), "Wrong admitted code")
        require(admission["selection_sha256"] == start["selection_sha256"] == sha(SCIENCE / "Decoder_Selection.json"),
                "Wrong admitted decoder selection")
        location = ROOT / admission["locations_path"]
        require(sha(location) == admission["locations_sha256"], "Asset locations changed")
        probe = admission["runtime_preflights"][task["model"]]
        probe_path = ROOT / probe["path"]
        require(sha(probe_path) == probe["sha256"], "Runtime probe changed")
        for path in (admission_path, location, probe_path):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    return inputs


def gather(manifest_path, analysis_lock):
    plan, execution = checked_execution()
    selection = checked_selection(plan)
    require(selection["calibration_regime"] == plan["calibration_regime"], "Wrong calibration metadata")
    require(selection["destination_calibration_labels_used_by_language"] == {"vi": True, "en": False},
            "Calibration provenance differs")
    new, inputs = collect("assessment", manifest_path, plan, selection)
    inputs.update(check_admissions(manifest_path, plan, selection))
    populations = {language: cohort(plan, language, "assessment") for language in ("vi", "en")}
    previous, settings, evidence, _ = parents(populations)
    inputs.update(evidence)
    registry = read(SCIENCE / "model_registry.json")["models"]
    for model, predictions in new.items():
        language = registry[model]["language"]
        for view, decoder in (("primary", selection["models"][model]["decoder"]), ("anchor", plan["anchor"])):
            require((view, model) not in previous[language], "New model duplicates a parent")
            previous[language][(view, model)] = [row["predictions"][decoder]["counts"] for row in predictions]
            settings[language][(view, model)] = decoder
    matrices = {}
    export = []
    for language, models in analysis_lock["model_order"].items():
        matrices[language] = {}
        population = populations[language]
        for view in ("primary", "anchor"):
            matrices[language][view] = {}
            for model in models:
                values = previous[language][(view, model)]
                require(len(values) == len(population), "Incomplete comparison population")
                matrices[language][view][model] = [[row[field] for field in FIELDS] for row in values]
                for row, count in zip(population, values):
                    export.append({"language": language, "readout": view, "model": model,
                                   "clip_id": row["clip_id"], "source_id": row["source_id"],
                                   "frames": row["frames"], "decoder": settings[language][(view, model)], **count})
    for path in (SCIENCE / "Plan.json", SCIENCE / "Execution_Lock.json", SCIENCE / "Decoder_Selection.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    return populations, matrices, export, inputs
