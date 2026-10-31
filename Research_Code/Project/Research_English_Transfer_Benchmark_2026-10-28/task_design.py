"""Outcome-independent complete-population tasks and the fixed assessment readouts."""

from english_common import HERE, lines, require


def cohort(stage):
    require(stage in {"development", "assessment"}, "Unknown data stage")
    return lines(HERE / (stage.capitalize() + ".jsonl"))


def build_tasks(plan, chunks=8):
    require(type(chunks) is int and chunks > 0, "Invalid shard count")
    tasks = {}
    for stage in ("development", "assessment"):
        rows = cohort(stage)
        ordered = sorted(range(len(rows)), key=lambda index: (rows[index]["frames"], rows[index]["clip_id"]))
        for model in plan["models"]:
            for chunk in range(chunks):
                tasks[f"{stage}_{model}_{chunk:02d}"] = {
                    "stage": stage, "model": model, "positions": ordered[chunk::chunks]}
    return tasks


def selected_task(name, lock):
    task = lock["tasks"][name]
    rows = cohort(task["stage"])
    return task, [(position, rows[position]) for position in task["positions"]]


def candidates_for(stage, model, plan, selection):
    if stage == "development":
        return list(plan["candidate_order"])
    require(stage == "assessment" and selection is not None, "Complete development selection required")
    require(set(selection["models"]) == set(plan["models"]), "Incomplete model selection")
    require(selection["test_predictions_read"] is False, "Selection used test outcomes")
    selected = selection["models"][model]["decoder"]
    require(selected in plan["candidate_order"], "Unregistered selection")
    return list(dict.fromkeys([selected, plan["anchor"]]))
