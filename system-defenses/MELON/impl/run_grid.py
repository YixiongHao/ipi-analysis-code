"""MELON AgentDojo grid driver (Phase 4 fan-out).

Runs suites x attacks x arms and aggregates BU/UA/ASR into a grid. Resumable: per-cell results
are written to results/grid.json after every cell-arm, and the per-task agentdojo logs are cached
in results/runs (so re-running skips finished work). One cell-arm failing doesn't abort the grid.

Default: full 4x4 grid (4 suites x 4 attacks), arms baseline+melon, capped to 5 user tasks/suite
(all injection tasks). MELON ~doubles LLM calls, so capping keeps the shared box tractable.

    python run_grid.py                       # full grid
    python run_grid.py --suites slack \
        --attacks direct ignore_previous system_message important_instructions \
        --arms baseline melon melon-aug                                              # MELON-Aug row
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from agentdojo_adapter import make_bge_embed_fn, make_qwen_llm
from run_agentdojo import RESULTS, run_arm

SUITES = ["workspace", "slack", "banking", "travel"]
ATTACKS = ["direct", "ignore_previous", "system_message", "important_instructions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--attacks", nargs="+", default=ATTACKS)
    ap.add_argument("--arms", nargs="+", default=["baseline", "melon"])
    ap.add_argument("--user-tasks", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="grid.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / args.out
    grid = json.loads(out_path.read_text()) if out_path.exists() else {}

    llm = make_qwen_llm()
    embed_fn = make_bge_embed_fn()

    for suite_name in args.suites:
        suite = get_suite(args.version, suite_name)
        user_task_ids = list(suite.user_tasks)[: args.user_tasks]
        inj_task_ids = list(suite.injection_tasks)
        for attack in args.attacks:
            cell_key = f"{suite_name}|{attack}"
            cell = grid.setdefault(cell_key, {})
            for arm in args.arms:
                arm_key = f"{arm}@t{args.threshold}" if arm != "baseline" else "baseline"
                if arm_key in cell and not args.force:
                    print(f"[skip] {cell_key} {arm_key} (already in {args.out})", flush=True)
                    continue
                print(f"\n===== {cell_key} :: {arm_key} =====", flush=True)
                try:
                    res = run_arm(arm, suite, attack, user_task_ids, inj_task_ids, llm, embed_fn,
                                  args.threshold, args.version, args.force)
                    cell[arm_key] = {"BU": res["BU"], "UA": res["UA"], "ASR": res["ASR"],
                                     "n_pairs": res["n_pairs"]}
                except Exception as e:
                    cell[arm_key] = {"error": repr(e)[:300]}
                    print(f"[ERROR] {cell_key} {arm_key}: {e}", flush=True)
                out_path.write_text(json.dumps(grid, indent=2))  # incremental persist
                print(f"[done] {cell_key} {arm_key} -> {cell[arm_key]}", flush=True)

    write_table(grid, RESULTS / (Path(args.out).stem + "_table.md"), args)
    print(f"\nWrote {out_path} and table.")


def write_table(grid: dict, path: Path, args):
    lines = [f"# MELON AgentDojo grid (Qwen3-32B, theta={args.threshold}, "
             f"{args.user_tasks} user tasks/suite, all injection tasks)\n",
             "| suite | attack | arm | BU | UA | ASR | n_pairs |",
             "|---|---|---|---|---|---|---|"]
    for cell_key in sorted(grid):
        suite, attack = cell_key.split("|")
        for arm_key, v in grid[cell_key].items():
            if "error" in v:
                lines.append(f"| {suite} | {attack} | {arm_key} | ERR | ERR | ERR | - |")
            else:
                lines.append(f"| {suite} | {attack} | {arm_key} | {v['BU']} | {v['UA']} | "
                             f"{v['ASR']} | {v['n_pairs']} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
