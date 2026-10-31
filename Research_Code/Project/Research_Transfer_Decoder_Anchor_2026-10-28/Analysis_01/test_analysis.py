"""Reject incomplete real matrices; exercise summaries with explicitly synthetic counts."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from analyze_anchor import HERE, build_tables, describe_source_counts
from collect_anchor import FULL, PHASE, ROOT, manifest, read, require_complete, sha
from anchor_core import lines, save, utc


def fails(action):
    try:
        action()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid case accepted")


def main(output_name="Verification_01.json"):
    population = lines(FULL / "Assessment.jsonl")
    models = read(PHASE / "Plan.json")["model_order"]["vi"]
    records = {model: {} for model in models}
    for model in models:
        for index, source in enumerate(population):
            substitutions = 1 if model in {"VLR_T0_LAST", "VLR_T0_29_LAST"} else 0
            count = {"errors": substitutions, "substitutions": substitutions, "deletions": 0,
                     "insertions": 0, "units": len(source["reference"].split())}
            records[model][index] = {"clip_id": source["clip_id"],
                                    "predictions": {"beam_c0.1": {"counts": count}}}
    sources, values, strata, exported = build_tables(population, records, models)
    assert len(exported) == 7002
    assert np.array_equal(sum(strata.values()), values.sum(axis=1))
    generator = np.random.default_rng(20261027)
    weights = generator.multinomial(20, np.full(20, .05), 10000)
    result = describe_source_counts(values, models, sources, weights)
    expected = 100 * 1167 / 30091
    assert all(abs(value["gain_pp"] - expected) < 1e-12 for value in result["pairs"].values())
    assert abs(result["fixed_two_seed_mean"]["gain_pp"] - expected) < 1e-12
    for case in ("missing", "wrong_clip", "wrong_count", "wrong_denominator", "negative"):
        broken = deepcopy(records)
        if case == "missing":
            broken[models[0]].pop(0)
        elif case == "wrong_clip":
            broken[models[0]][0]["clip_id"] = "wrong"
        else:
            count = broken[models[0]][0]["predictions"]["beam_c0.1"]["counts"]
            if case == "wrong_count":
                count["errors"] += 1
            elif case == "wrong_denominator":
                count["units"] += 1
            else:
                count["insertions"] = -1
        fails(lambda: build_tables(population, broken, models))
    fails(lambda: require_complete({"a": {}}, {"a", "b"}))
    fails(lambda: require_complete({"a": {}, "extra": {}}, {"a"}))
    require_complete({"a": {}}, {"a"})
    try:
        manifest()
    except ValueError as error:
        assert str(error).startswith("Incomplete task coverage:")
        incomplete_message = str(error)
    else:
        raise AssertionError("Expected the currently incomplete real matrix to remain incomplete")
    scripts = ("collect_anchor.py", "analyze_anchor.py", "test_analysis.py")
    proof = {"utc": utc(), "status": "passed", "synthetic_rows_checked": 7002,
             "scalar_bootstrap_checks": result["independent_scalar_bootstrap_checks"],
             "malformed_synthetic_cases_rejected": 7, "actual_incomplete_matrix_rejected": incomplete_message,
             "target_prediction_counts_parsed_by_this_test": False,
             "closed_prediction_file_bytes_read_for_hash_checks": True,
             "clarification": "Verification_01 meant no target counts were parsed; file bytes were hashed",
             "plan_sha256": sha(PHASE / "Plan.json"),
             "scripts": {name: sha(HERE / name) for name in scripts}}
    assert Path(output_name).name == output_name and output_name.endswith(".json")
    save(HERE / output_name, proof)
    print(proof, flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) == 2 else "Verification_01.json")
