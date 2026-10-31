"""Deterministic complete-population tasks, independent of outcomes."""

from common import HERE, lines, read


def cohort_for(stage, model):
    if stage == "assessment":
        filename = "Assessment.jsonl"
    elif stage == "development":
        filename = "Development.jsonl" if model == "VC_NATIVE" else "Development_New.jsonl"
    else:
        raise ValueError("Only full development or assessment tasks are defined")
    return lines(HERE / filename)


def build_tasks(plan, chunks=8):
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise ValueError("chunks must be a positive integer")
    tasks = {}
    for stage in ("development", "assessment"):
        for model in [*plan["matched_models"], plan["practical_reference"]]:
            cohort = cohort_for(stage, model)
            ordered = sorted(range(len(cohort)), key=lambda index: (cohort[index]["frames"],
                                                                   cohort[index]["clip_id"]))
            for chunk in range(chunks):
                name = f"{stage}_{model}_{chunk:02d}"
                tasks[name] = {"stage": stage, "model": model, "positions": ordered[chunk::chunks]}
    return tasks


def selected_task(name, lock):
    task = lock["tasks"][name]
    rows = cohort_for(task["stage"], task["model"])
    return task, [(index, rows[index]) for index in task["positions"]]


def candidates_for(stage, model, plan, selection):
    if stage == "development":
        return list(plan["candidate_order"])
    if stage != "assessment" or selection is None:
        raise ValueError("Assessment requires a complete development selection")
    selected = selection["models"][model]["decoder"]
    if selected not in plan["candidate_order"]:
        raise ValueError("Unregistered decoder")
    return [selected]
