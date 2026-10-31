"""Export the completed three-arm control without merging historical model identities."""

import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PHASE = ROOT / "Research_Matched_Continuation_Assessment_2026-10-30"
SOURCE = PHASE / "Analysis_Paper_Review05_01"
ORIGINAL = PHASE / "Analysis_Native_ThreeArm_01"
DESTINATION = HERE / "Data/Matched_Crops"
TRAINING = ROOT / "Research_Vast_Source_2026-10-29/Native_Production_01/Continuation_01"
ARMS = {"MATCHED_FC17": "full", "MATCHED_FT17": "fixed_time", "MATCHED_WW17": "whole_word"}
LABELS = {"full": "Full-only", "fixed_time": "Fixed-time mixed", "whole_word": "Whole-word mixed"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    summary = read(SOURCE / "Summary.json")
    assert summary["status"] == "complete_requested_matched_populations_recounted"
    assert summary["all_three_matched_models_complete"] is True
    assert summary["readout"] == "common" and summary["stage"] == "assessment"
    assert set(summary["models"]) == set(ARMS)
    for name, digest in summary["output_sha256"].items():
        assert sha(SOURCE / name) == digest == sha(ORIGINAL / name)
    totals = defaultdict(lambda: defaultdict(int))
    populations = defaultdict(set)
    groups = defaultdict(set)
    fields = ("errors", "substitutions", "deletions", "insertions", "units")
    for row in csv_rows(SOURCE / "Per_Clip_Counts.csv"):
        model = row["model"]
        assert row["clip_id"] not in populations[model]
        populations[model].add(row["clip_id"])
        groups[model].add(row["source_id"])
        for field in fields:
            totals[model][field] += int(row[field])
    results = csv_rows(SOURCE / "Results.csv")
    identities = {}
    exposure = {}
    for row in results:
        model = row["model"]
        arm = ARMS[model]
        assert len(populations[model]) == 1167 and len(groups[model]) == 20
        assert int(row["units"]) == 30091
        assert all(int(row[field]) == totals[model][field] for field in fields)
        assert totals[model]["errors"] == sum(totals[model][key] for key in fields[1:4])
        assert abs(float(row["wer_percent"]) - 100 * int(row["errors"]) / 30091) < 1e-10
        proof = read(TRAINING / "Training_Verification" / (arm + ".json"))
        assert proof["status"] == "native_training_records_checkpoint_and_strict_load_verified"
        training = proof["records"]["training"]
        assert training["updates"] == 6000 and training["exposure"]["presentations"] == 48000
        exposure[arm] = training
        identities[model] = {key: proof[key] for key in ("identity_sha256", "checkpoint_sha256")}
        identities[model]["binding_sha256"] = summary["provenance"][model]["binding_sha256"]
    assert len({frozenset(value) for value in populations.values()}) == 1
    assert exposure["fixed_time"]["full_slots"] == exposure["whole_word"]["full_slots"] == 24317
    DESTINATION.mkdir(exist_ok=True)
    for name in ("Results.csv", "Paired_Comparisons.csv", "Per_Clip_Counts.csv", "Summary.json"):
        shutil.copy2(SOURCE / name, DESTINATION / name)
        assert sha(SOURCE / name) == sha(DESTINATION / name)
    (DESTINATION / "Training_Exposure.json").write_text(json.dumps(exposure, indent=2) + "\n")
    (DESTINATION / "Model_Identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    source_profile = HERE / "Data/Matched_Source/Source_Destination_Profile.csv"
    source_rows = {row["model"]: row for row in csv_rows(source_profile)} if source_profile.exists() else {}
    if source_rows:
        assert set(source_rows) == set(ARMS)
        for model, row in source_rows.items():
            assert row["binding_sha256"] == identities[model]["binding_sha256"]
            assert int(row["source_clips"]) == 11307 and int(row["source_reference_units"]) == 225960
    lines = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{Matched fragment-labelling comparison on ViCocktail at seed 17.",
             r"All three newly fitted states use the same common decoder and 30,091 reference units.",
             r"Training frame and native-token counts are in millions.}",
             r"\label{tab:crop}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Training views & WER (\%) & S & D & I & Frames & Tokens\\", r"\midrule"]
    if source_rows:
        lines = [r"\begin{table}[tbp]", r"\centering\small",
                 r"\caption{Source and destination performance of the matched crop controls at seed 17.",
                 r"The same fitted states use a common decoder on 11,307 VLR and 1,167 ViCocktail clips.",
                 r"Training frame and native-token counts are in millions.}",
                 r"\label{tab:crop}", r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Training views & VLR WER (\%) & ViCocktail WER (\%) & Frames & Tokens\\", r"\midrule"]
    for row in results:
        arm = ARMS[row["model"]]
        counts = exposure[arm]["exposure"]
        values = [LABELS[arm], f"{float(row['wer_percent']):.3f}"]
        if source_rows:
            source = source_rows[row["model"]]
            assert abs(float(source["destination_wer_percent"]) - float(row["wer_percent"])) < 1e-10
            values = [LABELS[arm], f"{float(source['source_wer_percent']):.3f}", values[1]]
        else:
            values.extend(f"{int(row[field]):,}" for field in fields[1:4])
        values.extend(f"{counts[field] / 1e6:.3f}" for field in ("input_frames", "target_tokens"))
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
                  r"Each arm uses 48,000 source presentations and 6,000 updates.",
                  r"Both mixed arms share 317 full-clip fallbacks. This follow-up is distinct",
                  r"from the original FC/FM comparison; frame and token exposures are not equal.",
                  r"\end{minipage}", r"\end{table}"])
    (HERE / "Source/tables/crop_comparison.tex").write_text("\n".join(lines) + "\n")
    receipt = {"utc": datetime.now(timezone.utc).isoformat(),
               "status": "completed_three_arm_recount_export_verified", "records": 3501,
               "assessment_clips_each": 1167, "reference_units_each": 30091,
               "new_blind_confirmation": False, "historical_FM_unchanged": True,
               "new_native_source_profile_included": bool(source_rows),
               "source": str(SOURCE), "source_summary_sha256": sha(SOURCE / "Summary.json"),
               "export_sha256": {path.name: sha(path) for path in DESTINATION.iterdir() if path.is_file()},
               "table_sha256": sha(HERE / "Source/tables/crop_comparison.tex")}
    (HERE / "QA/Matched_Crop_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "records", "assessment_clips_each")}))


if __name__ == "__main__":
    main()
