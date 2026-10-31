"""Check manuscript tables and important claims against the numerical supplement."""

import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import pymupdf

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "Data"
SOURCE = HERE / "Source"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table_values(name):
    text = (SOURCE / "tables" / name).read_text(encoding="utf-8")
    return [[value.strip() for value in line.removesuffix(r"\\").split("&")]
            for line in text.splitlines() if "&" in line and "Model &" not in line]


def main():
    has_short_only = (DATA / "Short_Only").exists()
    short_only_proof = None
    if has_short_only:
        sys.path.insert(0, str(HERE / "Tools"))
        from verify_short_only_manuscript import verify as verify_short_only

        short_only_proof = verify_short_only()
        assert short_only_proof["matched_models"] == 4
        assert short_only_proof["aggregate_wer_cells"] == 8 and short_only_proof["paired_intervals"] == 4
    rows = read_csv("Results.csv")
    indexed = {(row["model_id"], row["decoder_regime"]): row for row in rows}
    counts = 0
    for name, models in [
        ("vietnamese.tex", ["VLR_B0", "VLR_T0_LAST", "VLR_T1_LAST", "VLR_T2_LAST",
                            "VLR_T4_LAST", "VLR_T0_29_LAST", "VLR_T1_29_LAST", "VC_NATIVE"]),
        ("english.tex", ["EN_B0", "EN_T0_LAST", "EN_T1_LAST", "EN_T2_LAST",
                         "EN_T4_LAST", "EN_F0_LAST", "EN_B1"]),
    ]:
        values = table_values(name)
        assert len(values) == len(models)
        for values_row, model in zip(values, models):
            for column, regime in [(2, "primary"), (3, "common")]:
                expected = indexed[(model, regime)]
                assert values_row[column] == f"{float(expected['wer_percent']):.3f}"
                counts += 1
    paired = read_csv("Comparisons.csv")
    values = [row for row in table_values("comparisons.tex") if row[0] in ("VI", "EN")]
    assert len(values) == len(paired) == 6
    for actual, expected in zip(values, paired):
        assert actual[0] == expected["language"] and actual[1] == expected["seed"]
        assert actual[2].lower() == expected["decoder_regime"]
        assert actual[3] == f"{float(expected['gain_pp']):.3f}"
        assert actual[4] == f"[{float(expected['ci95_low_pp']):.3f}, {float(expected['ci95_high_pp']):.3f}]"
    source = read_csv("Source_Results.csv")
    values = [row for row in table_values("source_profile.tex") if row[0] in ("Initializer", "FC", "FM")]
    assert len(values) == 3 and len(source) == 12
    for actual, suffix in zip(values, ["_B0", "_T0_LAST", "_T1_LAST"]):
        assert len(actual) == 5
        for prefix, column in (("EN", 1), ("VLR", 3)):
            model = prefix + suffix
            expected = next(row for row in source if row["model"] == model and row["regime"] == "common")
            assert actual[column] == f"{float(expected['wer_percent']):.3f}"
            assert actual[column + 1] == f"{float(indexed[(model, 'common')]['wer_percent']):.3f}"
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SOURCE / "sections").glob("*.tex"))
    for row in read_csv("Source_Comparisons.csv"):
        if row["role"] == "matched_full_vs_fragment":
            for field in ("reference_minus_comparison_pp", "ci95_low_pp", "ci95_high_pp"):
                assert f"{float(row[field]):.3f}" in text
    exposure = read_csv("Training_Exposure.csv")
    for language, seed in [("VI", "17"), ("VI", "29"), ("EN", "17")]:
        values = []
        for field in ("student_encoder_frames", "target_tokens"):
            for method in ("T0", "T1"):
                row = next(row for row in exposure if (row["language"], row["seed"], row["method"])
                           == (language, seed, method))
                values.append(f"{int(row[field]) / 1000000:.3f}")
        assert " & ".join(values) in text
    group_gains = {}
    clips = read_csv("Per_Clip_Errors.csv")
    for seed, suffix in [(17, ""), (29, "_29")]:
        for regime in ("common", "primary"):
            errors = defaultdict(lambda: [0, 0])
            for row in clips:
                if row["decoder_regime"] != regime:
                    continue
                for index, method in enumerate(("T0", "T1")):
                    if row["model_id"] == "VLR_" + method + suffix + "_LAST":
                        errors[row["source_id"]][index] += int(row["errors"])
            assert len(errors) == 20 and all(left > right for left, right in errors.values())
            group_gains[f"seed{seed}_{regime}"] = len(errors)
    main = (SOURCE / "main.tex").read_text(encoding="utf-8")
    assert "Cross-Corpus Vietnamese" not in main
    assert "whole-word" in main and "new architecture" not in main
    crop_rows = []
    with (DATA / "Matched_Crops/Results.csv").open(encoding="utf-8", newline="") as stream:
        crop_rows = list(csv.DictReader(stream))
    crop_table = [row for row in table_values("crop_comparison.tex")
                  if row[0] in ("Full-only", "Fixed-time mixed", "Whole-word mixed")]
    assert len(crop_table) == len(crop_rows) == 3
    crop_exposure = json.loads((DATA / "Matched_Crops/Training_Exposure.json").read_text())
    crop_arms = {"MATCHED_FC17": "full", "MATCHED_FT17": "fixed_time", "MATCHED_WW17": "whole_word"}
    profile_path = DATA / "Matched_Source/Source_Destination_Profile.csv"
    profiles = {row['model']: row for row in read_csv('Matched_Source/Source_Destination_Profile.csv')} if (
        profile_path.exists()) else {}
    for actual, expected in zip(crop_table, crop_rows):
        destination_column = 2 if profiles else 1
        assert actual[destination_column] == f"{float(expected['wer_percent']):.3f}"
        assert f"{float(expected['wer_percent']):.2f}" in main
        if profiles:
            profile = profiles[expected['model']]
            assert actual[1] == f"{float(profile['source_wer_percent']):.3f}"
            assert float(profile['destination_wer_percent']) == float(expected['wer_percent'])
        else:
            for column, field in enumerate(("substitutions", "deletions", "insertions"), start=2):
                assert actual[column] == f"{int(expected[field]):,}"
        exposure = crop_exposure[crop_arms[expected["model"]]]["exposure"]
        frame_column = 3 if profiles else 5
        assert actual[frame_column] == f"{exposure['input_frames'] / 1e6:.3f}"
        assert actual[frame_column + 1] == f"{exposure['target_tokens'] / 1e6:.3f}"
    assert "0.459" in text and "[0.268, 0.674]" in text
    reduction = read_csv('Loss_Reduction/Results.csv')
    reduced_index = {row['model']: row for row in reduction}
    reduction_table = [row for row in table_values('loss_reduction.tex')
                       if row[0] in ('Native', 'Token-normalized')]
    assert len(reduction) == 4 and len(reduction_table) == 2
    for actual, prefix in zip(reduction_table, ['MATCHED', 'NORMALIZED']):
        full = float(reduced_index[prefix + '_FC17']['wer_percent'])
        mixed = float(reduced_index[prefix + '_FT17']['wer_percent'])
        assert actual[1:] == [f'{full:.3f}', f'{mixed:.3f}', f'{full - mixed:.3f}']
    aggregate = defaultdict(lambda: defaultdict(int))
    cohort = defaultdict(set)
    for name in ['Native_Per_Clip_Counts.csv', 'Normalized_Per_Clip_Counts.csv']:
        for row in read_csv('Loss_Reduction/' + name):
            if row['model'] not in reduced_index:
                continue
            assert row['clip_id'] not in cohort[row['model']]
            cohort[row['model']].add(row['clip_id'])
            for field in ['errors', 'substitutions', 'deletions', 'insertions', 'units']:
                aggregate[row['model']][field] += int(row[field])
    assert len({frozenset(ids) for ids in cohort.values()}) == 1
    for model, expected in reduced_index.items():
        assert len(cohort[model]) == 1167
        assert all(aggregate[model][field] == int(expected[field]) for field in aggregate[model])
    contrasts = {row['contrast']: row for row in read_csv('Loss_Reduction/Contrasts.csv')}
    for contrast in ['normalized_gain', 'normalized_gain_minus_native_gain']:
        for field in ['estimate_pp', 'ci95_low_pp', 'ci95_high_pp']:
            assert f"{float(contrasts[contrast][field]):.3f}" in text
    assert '3.46 points' in main
    source_text = main + text
    references = (SOURCE / "references.bib").read_text(encoding="utf-8")
    citation_keys = set(re.findall(r"@\w+\{([^,]+),", references))
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source_text)
             for key in group.split(",")}
    assert cited <= citation_keys
    document = pymupdf.open(HERE / "Manuscript.pdf")
    assert 20 <= len(document) <= 35
    assert "Temporal supervision" in document[0].get_text(), "Keep the complete front matter together"
    assert all("[?]" not in page.get_text() for page in document)
    fig_count = source_text.count(r"\begin{figure}")
    table_count = source_text.count(r"\begin{table}") + source_text.count(r"\input{tables/")
    assert fig_count == 7 and table_count == (9 if has_short_only else 8)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "manuscript_numeric_tables_and_key_claims_verified",
        "destination_wer_cells": counts,
        "paired_interval_rows": len(paired),
        "source_states": len({row["model"] for row in source}),
        "matched_crop_models": len(crop_rows) + int(has_short_only),
        "matched_crop_table_cells": len(crop_rows) * (4 if profiles else 6) + 4 * int(has_short_only),
        "new_native_source_profile_models": len(profiles) + int(has_short_only),
        "short_only_evidence_verified": has_short_only,
        "short_only_aggregate_wer_cells": short_only_proof["aggregate_wer_cells"] if short_only_proof else 0,
        "short_only_paired_intervals": short_only_proof["paired_intervals"] if short_only_proof else 0,
        "recording_group_improvements": group_gains,
        "references_cited": len(cited),
        "pages": len(document), "figures": fig_count, "tables": table_count,
        "pdf_sha256": hashlib.sha256((HERE / "Manuscript.pdf").read_bytes()).hexdigest(),
        "visual_inspection_required_separately": True,
    }
    (HERE / "QA/Manuscript_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
