"""Author-forced CaMeL two-phase for AgentDojo: inject a clean single-fragment program, then
replay it under the security policy. The FP-utility eval reads utility of the policy replay.

  --mode author : force plans_agentdojo/<suite>/<task>.py into phase-1 (live Q-LLM, single fragment,
      no correction shards), caching the program + recorded Q-LLM outputs.
  --mode replay : run camel+secpol on the cache -> utility under policy = the FP measurement.
  --mode both   : author then replay (default).

A plan file is a self-contained restricted-Python program string (CaMeL DSL: tool calls +
query_ai_assistant(query, output_schema) + BaseModel classes). Authored by the plan workflow;
validated to utility here (<=3 fix rounds happen in the workflow, editing the plan file).

Run from impl/ in the camel uv env:
  uv run --project ../camel-prompt-injection python run_authored_plan.py \
      --suite banking --user-tasks user_task_0 --model deepseek/deepseek-v4-pro --mode both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import agentdojo_adapter as A  # noqa: F401  (env routing on import)
from agentdojo import benchmark, logging  # noqa: E402
from agentdojo.task_suite import get_suite  # noqa: E402
from defense import Defense  # noqa: E402

HERE = Path(__file__).parent
PLANS = HERE / "plans_agentdojo"
LOGDIR = HERE / "logs"


def _util_run(variant, suite, suite_name, user_task_id, model):
    defense = Defense(model=model, variant=variant, attack_name="important_instructions")
    pipeline = defense.build_agentdojo_pipeline(suite_name)
    with logging.OutputLogger(str(LOGDIR)):
        res = benchmark.benchmark_suite_without_injections(
            pipeline, suite, LOGDIR, force_rerun=True, user_tasks=[user_task_id])
    ur = res["utility_results"]
    return bool(list(ur.values())[0]) if ur else False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--user-tasks", nargs="+", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--mode", default="both", choices=["author", "replay", "both"])
    args = ap.parse_args()

    suite = get_suite("v1.2", args.suite)
    out = {}
    for utid in args.user_tasks:
        if args.mode in ("author", "both"):
            plan_path = PLANS / args.suite / f"{utid}.py"
            if not plan_path.exists():
                out[utid] = {"authored": False, "err": "no_plan_file"}
                print(f"[{args.suite}/{utid}] NO PLAN FILE {plan_path}", flush=True)
                continue
            A.set_authored_program(plan_path.read_text())
            try:
                iso_util = _util_run("camel", suite, args.suite, utid, args.model)  # phase 1
            finally:
                A.set_authored_program(None)
            out.setdefault(utid, {})["authored"] = True
            out[utid]["phase1_isolation_utility"] = iso_util
            print(f"[{args.suite}/{utid}] phase1(isolation) utility={iso_util}", flush=True)
        if args.mode in ("replay", "both"):
            sec_util = _util_run("camel+secpol", suite, args.suite, utid, args.model)  # phase 2
            out.setdefault(utid, {})["policy_utility"] = sec_util
            print(f"[{args.suite}/{utid}] phase2(policy) utility={sec_util}", flush=True)

    res_dir = HERE / "results_fp"
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"{args.suite}_authored.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
