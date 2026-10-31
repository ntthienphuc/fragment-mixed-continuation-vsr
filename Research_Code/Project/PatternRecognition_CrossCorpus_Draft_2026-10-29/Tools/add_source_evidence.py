"""Export only the completed three-state source contrast and draw its profile."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
ANALYSIS = ROOT / "Research_Source_Retention_2026-10-29/Analysis_01"
sys.path.insert(0, str(ANALYSIS))
from collect import metadata, model, plans


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    path = ANALYSIS / "Main_Contrast_02_Path_Migration/Summary.json"
    summary = json.loads(path.read_text())
    assert summary["status"] == "requested_complete_populations_independently_recounted"
    assert summary["all_requested_models_complete"]
    assert summary["population_clips"] == 1243 and summary["source_groups"] == 354
    names = ["EN_B0", "EN_T0_LAST", "EN_T1_LAST"]
    assert {row["model"] for row in summary["results"]} == set(names)
    assert all(row["units"] == 6660 for row in summary["results"])
    parent, assessment = plans()
    rows, pixels = metadata("lrs2")
    per_clip = []
    for name in names:
        status, records, provenance = model("lrs2", name, parent, assessment, rows, pixels)
        assert records is not None and len(records) == 1243
        for regime in ("common", "primary"):
            decoder = "beam_c0.1" if regime == "common" else parent["models"][name]["frozen_transfer_decoder"]
            for record in records:
                counts = record["predictions"][decoder]["counts"]
                per_clip.append({"corpus": "LRS2", "model": name, "regime": regime,
                                 "decoder": decoder, "clip_id": record["clip_id"],
                                 "source_id": record["source_id"], **counts})
    assert len(per_clip) == 7458
    data = HERE / "Data"
    write_csv(data / "Source_Per_Clip_Errors.csv", per_clip)
    write_csv(data / "Source_Results.csv", summary["results"])
    write_csv(data / "Source_Comparisons.csv", summary["comparisons"])
    write_csv(data / "Source_Destination_Profile.csv", summary["source_and_destination"])
    counts = {}
    for name in names:
        for regime in ("common", "primary"):
            selection = [row for row in per_clip if row["model"] == name and row["regime"] == regime]
            expected = next(row for row in summary["results"]
                            if row["model"] == name and row["regime"] == regime)
            for field in ("errors", "units", "substitutions", "deletions", "insertions"):
                assert sum(row[field] for row in selection) == expected[field]
            counts[name + "|" + regime] = expected["errors"]
    labels = {"EN_B0": "Initializer", "EN_T0_LAST": "FC", "EN_T1_LAST": "FM"}
    common = {row["model"]: row for row in summary["source_and_destination"] if row["regime"] == "common"}
    table = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{English source/destination performance at unchanged final states,",
             r"with the common decoder. Source evaluation covers all 1,243 LRS2 test clips",
             r"(6,660 words); destination evaluation covers all 2,854 WildVSR clips",
             r"(48,037 words). Lower WER is better.}", r"\label{tab:source}",
             r"\begin{tabular}{lrr}", r"\toprule",
             r"State & LRS2 WER (\%) & WildVSR WER (\%)\\", r"\midrule"]
    for name in names:
        row = common[name]
        table.append(f"{labels[name]} & {row['source_wer']:.3f} & {row['destination_wer']:.3f}" + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (HERE / "Source/tables/source_profile.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = ["#637587", "#246a9a", "#d96a3a"]
    for axis, key, title, limits in zip(axes, ["source_wer", "destination_wer"],
                                       ["LRS2 continuation source", "WildVSR destination"],
                                       [(49.5, 58), (66.5, 67.5)]):
        values = [common[name][key] for name in names]
        axis.scatter(range(3), values, c=colors, s=50, zorder=2)
        for index, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (index, value), xytext=(0, 9),
                          textcoords="offset points", ha="center", fontsize=9)
        axis.set_xticks(range(3), [labels[name] for name in names])
        axis.set_xlim(-0.3, 2.3)
        axis.set_ylim(*limits)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("WER (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["right", "top"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(HERE / "Source/figures/source_destination_profile.pdf", bbox_inches="tight")
    plt.close(fig)
    proof = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_three_state_source_population_recounted_and_exported",
        "summary": path.relative_to(ROOT).as_posix(),
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "per_clip_records": len(per_clip),
        "errors_by_state_and_regime": counts,
        "source_test_readouts_retuned": False,
        "full_seven_state_matrix_promoted": False,
        "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in sorted(data.glob("Source_*.csv"))},
    }
    (data / "Source_Evidence_Verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": proof["status"], "per_clip_records": len(per_clip)}))


if __name__ == "__main__":
    main()
