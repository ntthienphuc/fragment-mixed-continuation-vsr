"""Export the two completed source populations and their unchanged destination profiles."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ANALYSIS = ROOT / "Research_Source_Retention_2026-10-29/Analysis_01"
sys.path.insert(0, str(ANALYSIS))
from collect import metadata, model, plans


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_file(name, rows):
    assert rows
    with (PAPER / "Data" / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completed_sources():
    coverage = read(ROOT / "Research_Vast_Source_2026-10-29/Full_Source_Backup_Coverage.json")
    assert coverage["verified_shards"] == 48
    assert coverage["status"] == "full_coverage_backed_up_pending_joint_scientific_analysis"
    specifications = [
        ("lrs2", "Main_Contrast_02_Path_Migration", "EN", 1243, 6660, 354),
        ("vlr", "Vietnamese_Main_Vast_01", "VLR", 11307, 225960, 61),
    ]
    checked = []
    for corpus, folder, prefix, clips, units, groups in specifications:
        path = ANALYSIS / folder / "Summary.json"
        summary = read(path)
        assert summary["status"] == "requested_complete_populations_independently_recounted"
        assert summary["all_requested_models_complete"] and not summary["source_test_readouts_retuned"]
        assert summary["population_clips"] == clips and summary["source_groups"] == groups
        names = [prefix + suffix for suffix in ("_B0", "_T0_LAST", "_T1_LAST")]
        assert {row["model"] for row in summary["results"]} == set(names)
        assert all(row["units"] == units for row in summary["results"])
        if corpus == "vlr":
            independent = read(path.parent / "Independent_Verification.json")
            assert independent["summary_sha256"] == sha(path)
            assert independent["clips_per_model"] == clips and independent["contrasts"] == 6
            assert independent["paired_resamples_per_contrast"] == 10000
            complete = read(ROOT / "Research_Vast_Source_2026-10-29/Analysis_Adapter_01/Complete.json")
            assert complete["verification"] == independent
        checked.append((corpus, names, summary, path))
    return checked


def export_records(sources):
    parent, assessment = plans()
    records = []
    results = []
    comparisons = []
    profiles = []
    for corpus, names, summary, _ in sources:
        rows, pixels = metadata(corpus)
        for name in names:
            status, predictions, provenance = model(corpus, name, parent, assessment, rows, pixels)
            assert predictions is not None and len(predictions) == len(rows)
            for regime in ("common", "primary"):
                decoder = "beam_c0.1" if regime == "common" else parent["models"][name]["frozen_transfer_decoder"]
                selected = []
                for prediction in predictions:
                    counts = prediction["predictions"][decoder]["counts"]
                    selected.append({"corpus": corpus.upper(), "model": name, "regime": regime,
                                     "decoder": decoder, "clip_id": prediction["clip_id"],
                                     "source_id": prediction["source_id"], **counts})
                expected = next(row for row in summary["results"]
                                if row["model"] == name and row["regime"] == regime)
                for field in ("errors", "units", "substitutions", "deletions", "insertions"):
                    assert sum(row[field] for row in selected) == expected[field]
                records.extend(selected)
        results.extend(summary["results"])
        comparisons.extend(summary["comparisons"])
        profiles.extend(summary["source_and_destination"])
    assert len(records) == (1243 + 11307) * 3 * 2 == 75300
    csv_file("Source_Per_Clip_Errors.csv", records)
    csv_file("Source_Results.csv", results)
    csv_file("Source_Comparisons.csv", comparisons)
    csv_file("Source_Destination_Profile.csv", profiles)
    return profiles, len(records)


def render_table_and_figure(profiles):
    common = {row["model"]: row for row in profiles if row["regime"] == "common"}
    labels = ("Initializer", "FC", "FM")
    suffixes = ("_B0", "_T0_LAST", "_T1_LAST")
    table = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{Source and destination WER (\%) at the same fitted states and",
             r"common decoder. Source populations are 1,243 LRS2 clips (6,660 words)",
             r"and 11,307 VLR clips (225,960 whitespace units). Destination populations",
             r"are 2,854 WildVSR and 1,167 ViCocktail clips. Continuation results use seed 17;",
             r"each corpus is scored separately. Lower WER is better.}", r"\label{tab:source}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{English} & \multicolumn{2}{c}{Vietnamese}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"State & LRS2 & WildVSR & VLR & ViCocktail\\", r"\midrule"]
    for label, suffix in zip(labels, suffixes):
        values = [common[prefix + suffix][metric] for prefix in ("EN", "VLR")
                  for metric in ("source_wer", "destination_wer")]
        table.append(label + " & " + " & ".join(f"{value:.3f}" for value in values) + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (PAPER / "Source/tables/source_profile.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.9))
    colors = ["#637587", "#246a9a", "#d96a3a"]
    for row, (prefix, source, destination) in enumerate((("EN", "LRS2", "WildVSR"),
                                                        ("VLR", "VLR", "ViCocktail"))):
        for column, (metric, title) in enumerate((("source_wer", source + " source"),
                                                 ("destination_wer", destination + " destination"))):
            axis = axes[row, column]
            values = [common[prefix + suffix][metric] for suffix in suffixes]
            axis.scatter(range(3), values, c=colors, s=48, zorder=2)
            for index, value in enumerate(values):
                axis.annotate(f"{value:.3f}", (index, value), xytext=(0, 8),
                              textcoords="offset points", ha="center", fontsize=9)
            span = max(values) - min(values)
            margin = max(.25 * span, .3)
            axis.set_ylim(min(values) - margin, max(values) + 1.5 * margin)
            axis.set_xlim(-.4, 2.4)
            axis.set_xticks(range(3), labels)
            axis.set_title(title, fontsize=10)
            axis.set_ylabel("WER (%)")
            axis.grid(axis="y", alpha=.2)
            axis.spines[["right", "top"]].set_visible(False)
    figure.tight_layout(pad=1.2)
    figure.savefig(PAPER / "Source/figures/source_destination_profile.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    sources = completed_sources()
    profiles, count = export_records(sources)
    render_table_and_figure(profiles)
    receipt = {"utc": datetime.now(timezone.utc).isoformat(),
               "status": "two_complete_source_populations_recounted_and_exported",
               "per_clip_records": count, "source_test_readouts_retuned": False,
               "inputs": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
                          for _, _, _, path in sources],
               "files": {path.name: sha(path) for path in sorted((PAPER / "Data").glob("Source_*.csv"))}}
    (PAPER / "Data/Source_Evidence_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
