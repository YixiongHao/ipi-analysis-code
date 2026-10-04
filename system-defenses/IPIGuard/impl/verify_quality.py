#!/usr/bin/env python
"""Quality/well-formedness verifier for a hand-authored IPIGuard TDG (NOT a utility check).

Mirrors the CaMeL authoring workflow's validation step, but the bar is QUALITY, not AgentDojo
success: whether the defense ultimately completes the task is the over-defense signal we measure,
so we do NOT gate on utility=True. We gate on:

  STATIC (G1): valid schema; every function_name is a real suite tool; every args key is a real
    parameter of that tool ("<unknown>: type" placeholders allowed as values); depends_on ids exist
    and are acyclic; >=1 node; the plan is COMPLETE (if the task requires a consequential/mutation
    tool, at least one such tool is present — a read-only plan for an action task is an
    incomplete-DAG quality defect).
  MECHANICAL (G2): the TDG loads via _json_to_dag and traverses through the traverse-only replay
    with NO crash/replay-error, producing >=1 executed tool call.

Usage (master venv, from impl/):
  python verify_quality.py --suite banking --task user_task_0 [--plans-dir plans_agentdojo_authored]
Exit 0 + prints JSON {ok_static, ok_mechanical, ok, issues:[...], executed_calls, traversed}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import agentdojo_plan as P  # _fork_env, _json_to_dag, replay_saved_plan, make_client
from agentdojo.task_suite.load_suites import get_suite

HERE = Path(__file__).parent
VERSION = "v1.1.2"
# Mutation verbs. AgentDojo tool names are verb_noun (send_money, create_file, add_*); read tools
# start with get/list/search/read/check. So classify by the LEADING token — substring matching
# over-fires (e.g. "add" in get_hotels_ADDress, "schedule" in get_SCHEDULEd_transactions).
_MUTATION_VERBS = frozenset((
    "send", "create", "delete", "update", "add", "remove", "schedule", "reserve", "share", "post",
    "reply", "append", "invite", "cancel", "reschedule", "book", "transfer", "pay", "write", "move",
    "set", "make", "edit",
))
# verbs in the USER PROMPT that imply the task requires a consequential action (word-boundary
# matched so "invited"/"addresses"/"scheduled" don't spuriously match invite/add/schedule).
_TASK_ACTION_RE = re.compile(
    r"\b(send|create|delete|update|add|remove|schedule|reserve|share|post|reply|append|invite|"
    r"cancel|reschedule|book|transfer|pay|make a reservation|set up|write)\b", re.I)


def _is_mutation(tool_name: str) -> bool:
    return tool_name.split("_", 1)[0] in _MUTATION_VERBS


def verify(suite_name: str, task_id: str, plans_dir: Path, static_only: bool = False) -> dict:
    issues: list[str] = []
    suite = get_suite(VERSION, suite_name)
    tools = {t.name: t for t in suite.tools}
    ut = suite.get_user_task_by_id(task_id)

    path = plans_dir / suite_name / f"{task_id}.json"
    if not path.exists():
        return {"ok": False, "ok_static": False, "ok_mechanical": False, "issues": ["no_plan_file"]}
    rec = json.loads(path.read_text())
    dag = rec.get("dag", [])

    # --- G1 static ---
    if not dag:
        issues.append("empty_dag")
    ids = set()
    for n in dag:
        nid = str(n.get("id"))
        if nid in ids:
            issues.append(f"duplicate_id:{nid}")
        ids.add(nid)
        fn = n.get("function_name")
        if fn not in tools:
            issues.append(f"unknown_tool:{fn}")
            continue
        # arg keys must be real params (placeholders allowed as VALUES, not keys)
        valid_params = set(tools[fn].parameters.model_json_schema().get("properties", {}).keys())
        for k in (n.get("args") or {}):
            if k not in valid_params:
                issues.append(f"{fn}:bad_arg_key:{k}")
    # depends_on validity + acyclicity
    for n in dag:
        for dep in (n.get("depends_on") or []):
            if str(dep) not in ids:
                issues.append(f"{n.get('id')}:dangling_dep:{dep}")
    # cycle check (Kahn)
    graph = {str(n["id"]): [str(d) for d in (n.get("depends_on") or [])] for n in dag}
    seen, stack = set(), []
    def _visit(u, path_set):
        if u in path_set:
            issues.append("cycle_detected"); return
        if u in seen:
            return
        path_set = path_set | {u}
        for v in graph.get(u, []):
            _visit(v, path_set)
        seen.add(u)
    for u in list(graph):
        _visit(u, set())
    # completeness: task wants an action but no mutation node present
    task_wants_action = bool(_TASK_ACTION_RE.search(ut.PROMPT))
    has_mutation_node = any(_is_mutation(n.get("function_name", "")) for n in dag)
    if task_wants_action and not has_mutation_node:
        issues.append("incomplete_dag:task_requires_action_but_no_mutation_node")

    ok_static = len([i for i in issues if not i.startswith("warn:")]) == 0

    # --- G2 mechanical: traverse-only replay, must not crash; NOT utility-gated ---
    ok_mech, executed, traversed = False, [], False
    if static_only:
        return {"suite": suite_name, "task": task_id, "n_nodes": len(dag),
                "ok_static": ok_static, "ok_mechanical": None, "ok": ok_static, "issues": issues,
                "task_wants_action": task_wants_action, "has_mutation_node": has_mutation_node}
    try:
        util, messages = P.replay_saved_plan(suite, ut, rec)
        traversed = True
        executed = [m["tool_call"].function for m in messages
                    if m.get("role") == "tool" and m.get("tool_call")]
        ok_mech = len(executed) >= 1
        if not ok_mech:
            issues.append("mechanical:no_tool_calls_executed")
    except Exception as e:  # noqa: BLE001
        issues.append(f"mechanical:replay_error:{type(e).__name__}: {e}")

    return {"suite": suite_name, "task": task_id, "n_nodes": len(dag),
            "ok_static": ok_static, "ok_mechanical": ok_mech,
            "ok": ok_static and ok_mech, "issues": issues,
            "executed_calls": executed, "task_wants_action": task_wants_action,
            "has_mutation_node": has_mutation_node}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--plans-dir", default="plans_agentdojo_authored")
    args = ap.parse_args()
    out = verify(args.suite, args.task, HERE / args.plans_dir)
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
