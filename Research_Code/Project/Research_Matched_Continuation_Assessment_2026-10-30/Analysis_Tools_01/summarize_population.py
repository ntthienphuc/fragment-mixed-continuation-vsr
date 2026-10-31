"""Create one complete-population summary from explicit verified model bindings."""

import argparse
import csv
import json
from pathlib import Path

from collect_population import HERE, PHASE, collect
from common import read, save, sha, utc
from paired_statistics import FIELDS, paired, totals


PAIRS = (("MATCHED_FC17", "MATCHED_FT17"), ("MATCHED_FC17", "MATCHED_WW17"),
         ("MATCHED_WW17", "MATCHED_FT17"))


def write_csv(path, records):
    assert records
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(arguments):
    specification = read(arguments.inputs)
    stage = specification["stage"]
    readout = specification["readout"]
    assert stage in ("source", "assessment") and readout in ("common", "selected")
    assert specification["assessment_plan_sha256"] == sha(PHASE / "Plan.json")
    models = specification["models"]
    assert models and set(models).issubset(read(PHASE / "Plan.json")["models"])
    records = {}
    provenance = {}
    summaries = []
    per_clip = []
    for model, source in models.items():
        records[model], provenance[model] = collect(
            stage, model, readout, Path(source["binding"]), Path(source["shards"]),
            Path(source["selection"]) if source.get("selection") else None,
            Path(specification["destination_root"]),
            Path(source["development_shards"]) if source.get("development_shards") else None)
        decoder = provenance[model]["candidates"][0]
        result = totals(records[model], decoder)
        assert result["units"] == provenance[model]["reference_units"]
        summaries.append({"model": model, "stage": stage, "readout": readout, "decoder": decoder,
                          "clips": len(records[model]), **result})
        for row in records[model]:
            counts = row["predictions"][decoder]["counts"]
            per_clip.append({"model": model, "stage": stage, "readout": readout, "decoder": decoder,
                             "position": row["position"], "clip_id": row["clip_id"],
                             "source_id": row["source_id"], "frames": row["frames"],
                             **{field: counts[field] for field in FIELDS}})
    contrasts = []
    for left, right in PAIRS:
        if left in records and right in records:
            left_decoder = provenance[left]["candidates"][0]
            right_decoder = provenance[right]["candidates"][0]
            label = "|".join([stage, readout, left, right])
            contrasts.append({"reference": left, "comparison": right, "stage": stage, "readout": readout,
                              "reference_decoder": left_decoder, "comparison_decoder": right_decoder,
                              **paired(records[left], records[right], left_decoder, right_decoder, label)})
    output = arguments.output.resolve()
    assert output.is_relative_to(PHASE.resolve()) and output != PHASE.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "Results.csv", summaries)
    write_csv(output / "Per_Clip_Counts.csv", per_clip)
    if contrasts:
        scalar = [{key: value for key, value in row.items() if not isinstance(value, list)} for row in contrasts]
        write_csv(output / "Paired_Comparisons.csv", scalar)
    result = {"utc": utc(), "status": "complete_requested_matched_populations_recounted",
              "stage": stage, "readout": readout, "models": list(models), "results": summaries,
              "comparisons": contrasts, "provenance": provenance,
              "assessment_plan_sha256": sha(PHASE / "Plan.json"), "inputs_sha256": sha(arguments.inputs),
              "all_three_matched_models_complete": len(models) == 3,
              "both_source_and_destination_complete": False, "new_blind_test": False,
              "manuscript_updated": False, "code_sha256": {path.name: sha(path) for path in HERE.glob("*.py")},
              "output_sha256": {path.name: sha(path) for path in output.glob("*.csv")}}
    save(output / "Summary.json", result)
    print(json.dumps({"status": result["status"], "results": summaries, "comparisons": len(contrasts)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
