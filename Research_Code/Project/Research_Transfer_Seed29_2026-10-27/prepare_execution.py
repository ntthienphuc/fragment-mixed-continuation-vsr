"""Prepare the native readout only after the admitted encoder caches are verified."""

from copy import deepcopy

from common import COHORT, ENGINE, HERE, PRIOR, ROOT, STORE, checked_plan, read, save, sha, utc


def main():
    scientific = checked_plan()
    design = read(HERE / "Execution_Design_Lock.json")
    for name, expected in design["files"].items():
        assert sha(ROOT / name) == expected, name
    registry = read(HERE / "model_registry.json")
    origin = PRIOR / "Beam_Execution/Plan.json"
    plan = deepcopy(read(origin))
    plan.update(utc=utc(), status="seed29_final_pair_beam_inputs_prepared", models={},
                protocol_sha256=sha(HERE / "Execution_Design.md"),
                seed29_input_plan_sha256=sha(HERE / "Plan.json"),
                primary="T0 versus T1, both final step 6000, training seed 29")
    for key in ("greedy_exposure_receipt_sha256", "initialization_plan_sha256", "gate"):
        plan.pop(key, None)
    for model in scientific["models"]:
        cache = STORE / "Cache" / model
        proof = read(cache / "Verification.json")
        assert proof["status"] == "passed" and proof["clips"] == 531
        assert proof["plan_sha256"] == sha(HERE / "Plan.json")
        assert proof["ctc_reconstructions"] == 531 and proof["tensor_export_checks"] == 163
        assert proof["independent_greedy_count_and_path_checks"] == 3717
        assert proof["index_sha256"] == sha(cache / "Index.jsonl")
        assert proof["predictions_sha256"] == sha(HERE / "Greedy_01" / model / "Predictions.jsonl")
        assert sha(cache / "Verification.json") == sha(HERE / "Greedy_01" / model / "Verification.json")
        plan["models"][model] = {"entry": registry["models"][model],
                                 "cache_index_sha256": proof["index_sha256"]}
    assert plan["cohort_sha256"] == sha(COHORT)
    assert plan["registry_sha256"] == sha(ENGINE / "model_registry.json")
    assert plan["candidate_tie_order"] == scientific["candidate_order"]
    save(HERE / "Beam_Execution/Plan.json", plan)
    print({"status": plan["status"], "models": list(plan["models"])}, flush=True)


if __name__ == "__main__":
    main()
