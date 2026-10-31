"""Source-population execution contract, separate from resource diagnostics."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from retention_common import ROOT, checked_plan, read, require, sha


def check(task_name):
    parent = checked_plan()
    plan = read(HERE / "Plan.json")
    require(plan["parent_plan_sha256"] == sha(HERE.parent / "Plan.json"), "Parent plan changed")
    for name, digest in plan["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Assessment code changed: " + name)
    require(task_name in plan["tasks"], "Unknown complete-cohort task")
    task = plan["tasks"][task_name]
    require(task["candidates"] == parent["models"][task["model"]]["execution_candidates"], "Decoder changed")
    return plan, task
