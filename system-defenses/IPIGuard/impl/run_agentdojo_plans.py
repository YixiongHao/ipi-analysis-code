#!/usr/bin/env python
"""IPIGuard AgentDojo FP-utility: pre-author TDGs, then replay them (no live construction in eval).

Two phases, mirroring the arena reuse-plans split:
  --mode build : construct + save one TDG per user task to plans_agentdojo/<suite>/<task>.json.
      A TDG is a single static plan (no correction shards by construction). Retries construction
      only if it ERRORS or yields an empty DAG (well-formedness, <=3 rounds) — NOT to select for
      utility, so over-defense stays visible in the eval.
  --mode replay : load each saved TDG and replay it (traverse-only, DagToolsExecutionLoop skips
      construction); utility of the replay = the FP/over-defense measurement.
  --mode both : build then replay.

Agent = deepseek-v4-pro via OpenRouter (agentdojo_plan patches the fork's LLM calls with the
deepseek provider pin + reasoning-off). Suites parallelize as separate processes.
"""

import argparse
import json
from pathlib import Path

import agentdojo_plan as P
from agentdojo.task_suite.load_suites import get_suite

HERE = Path(__file__).parent
SUITES = ["workspace", "banking", "slack", "travel"]
VERSION = "v1.1.2"  # fork max; user-task set identical to master v1.2.2 (verified)


def _json_default(o):
    """Make transcript messages JSON-safe: AgentDojo tool calls are pydantic FunctionCall objects."""
    for attr in ("model_dump", "dict", "_asdict"):
        m = getattr(o, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:  # noqa: BLE001
                pass
    return str(o)


def build_one(suite, ut, out_dir, max_rounds=3):
    err = "empty_dag"
    for r in range(max_rounds):
        try:
            rec = P.build_and_save_plan(suite, ut, out_dir)
            if rec["dag"]:  # non-empty, well-formed
                rec["build_rounds"] = r + 1
                (out_dir / f"{ut.ID}.json").write_text(json.dumps(rec, indent=2))
                return rec, None
        except Exception as e:  # noqa: BLE001 (record construction failure, retry)
            err = f"{type(e).__name__}: {e}"
    return None, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="both", choices=["build", "replay", "both"])
    ap.add_argument("--suites", nargs="+", default=SUITES, choices=SUITES)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--run-idx", type=int, default=0)
    ap.add_argument("--plans-dir", default="plans_agentdojo",
                    help="dir of pre-authored plans (e.g. plans_agentdojo_authored for the Claude planner)")
    ap.add_argument("--tag", default=None, help="results_fp/<tag>/ subdir (default run-<run-idx>)")
    args = ap.parse_args()

    plans_root = HERE / args.plans_dir
    summ_dir = HERE / "results_fp" / (args.tag or f"run-{args.run_idx}")
    summ_dir.mkdir(parents=True, exist_ok=True)
    out = {"defense": "ipiguard", "model": P.DEFAULT_MODEL, "per_suite": {}}
    grand_ok = grand_n = 0
    for suite_name in args.suites:
        suite = get_suite(VERSION, suite_name)
        tasks = list(suite.user_tasks.values())
        if args.max_tasks:
            tasks = tasks[: args.max_tasks]
        out_dir = plans_root / suite_name
        results = {}
        records = []  # per-task transcript records (utility + messages), matches the IPI fork store
        for ut in tasks:
            if args.mode in ("build", "both"):
                rec, err = build_one(suite, ut, out_dir)
                if err:
                    results[ut.ID] = {"built": False, "err": err, "utility": None}
                    print(f"[{suite_name}/{ut.ID}] BUILD FAILED: {err}", flush=True)
                    continue
            plan_path = out_dir / f"{ut.ID}.json"
            if not plan_path.exists():
                results[ut.ID] = {"built": False, "err": "no_plan_file", "utility": None}
                continue
            saved = json.loads(plan_path.read_text())
            entry = {"built": True, "nodes": len(saved["dag"]),
                     "planned_tools": saved["planned_tools"],
                     "build_rounds": saved.get("build_rounds", 1)}
            if args.mode in ("replay", "both"):
                msgs = None
                try:
                    util, msgs = P.replay_saved_plan(suite, ut, saved)
                    entry["utility"] = bool(util)
                except Exception as e:  # noqa: BLE001
                    entry["utility"] = None
                    entry["replay_err"] = f"{type(e).__name__}: {e}"
                grand_n += 1
                grand_ok += 1 if entry.get("utility") else 0
                records.append({"user_task_id": ut.ID, "nodes": entry["nodes"],
                                "planned_tools": entry["planned_tools"],
                                "utility": entry.get("utility"),
                                "replay_err": entry.get("replay_err"), "messages": msgs or []})
                print(f"[{suite_name}/{ut.ID}] nodes={entry['nodes']} utility={entry.get('utility')}", flush=True)
            results[ut.ID] = entry
        if records:  # persist per-task transcripts (one JSONL per suite) for the unified FP store
            with (summ_dir / f"records_{suite_name}.jsonl").open("w") as f:
                for rec in records:
                    f.write(json.dumps(rec, default=_json_default) + "\n")
        ok = sum(1 for e in results.values() if e.get("utility"))
        n = sum(1 for e in results.values() if e.get("utility") is not None)
        out["per_suite"][suite_name] = {"ok": ok, "n": n, "tasks": results}
        if n:
            print(f"[{suite_name}] utility {ok}/{n} = {ok / n:.1%}", flush=True)
    if grand_n:
        print(f"[TOTAL] utility {grand_ok}/{grand_n} = {grand_ok / grand_n:.1%}")
    (summ_dir / f"summary_{'-'.join(args.suites)}.json").write_text(json.dumps(out, indent=2))
    print(f"summary -> {summ_dir}")


if __name__ == "__main__":
    main()
