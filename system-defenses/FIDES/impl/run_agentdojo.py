"""FIDES AgentDojo replication harness (Phase 4).

Runs the paper's config (Tool Knowledge attack) on Qwen3-32B for two arms:
  * baseline — no defense (reproduces the paper's Basic arm: many successful injections)
  * fides    — taint monitor + P-T policy (reproduces Basic*/FIDES*: in-policy ASR -> ~0)

Metrics (AgentDojo): ASR (injection-task success %), UA (user-task success under attack),
BU (benign user-task success, no attack). Writes results/agentdojo_<suite>_<attack>.json.

Run (master venv; Qwen on port 8000 must be up):
    python run_agentdojo.py --suite slack --user-tasks 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from agentdojo_adapter import Defense, FidesMonitor, make_qwen_llm

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LOGDIR = RESULTS / "runs"


def build_pipeline(arm: str, llm) -> AgentPipeline:
    system_message = load_system_message(None)
    components = [SystemMessage(system_message), InitQuery(), llm]
    if arm == "baseline":
        loop = ToolsExecutionLoop([ToolsExecutor(), llm])
        name = "Qwen3-32B"
    elif arm == "fides":
        # Monitor BEFORE ToolsExecutor so a violating call is caught before it executes.
        monitor = FidesMonitor(Defense(), raise_on_violation=True)
        loop = ToolsExecutionLoop([monitor, ToolsExecutor(), llm])
        name = "Qwen3-32B-fides"
    else:
        raise ValueError(arm)
    pipeline = AgentPipeline([*components, loop])
    pipeline.name = name
    return pipeline


def mean(d: dict) -> float:
    return round(100.0 * sum(bool(v) for v in d.values()) / len(d), 2) if d else 0.0


def run_arm(arm, suite, attack_name, user_task_ids, inj_task_ids, llm, version, force):
    pipeline = build_pipeline(arm, llm)
    attack = load_attack(attack_name, suite, pipeline)

    util_attack: dict = {}  # UA: user-task success under attack
    security: dict = {}     # ASR: injection-task success
    benign_util: dict = {}  # BU: user-task success, no attack

    with OutputLogger(str(LOGDIR), live=None):
        for utid in user_task_ids:
            user_task = suite.get_user_task_by_id(utid)
            u, s = run_task_with_injection_tasks(
                suite, pipeline, user_task, attack, LOGDIR, force, inj_task_ids, version
            )
            util_attack.update(u)
            security.update(s)
            bu_success, _ = run_task_without_injection_tasks(suite, pipeline, user_task, LOGDIR, force, version)
            benign_util[utid] = bu_success
            print(f"[{arm}] {utid}: UA {mean(util_attack)}  ASR {mean(security)}  BU {mean(benign_util)}", flush=True)

    return {
        "arm": arm,
        "pipeline_name": pipeline.name,
        "n_user_tasks": len(user_task_ids),
        "n_pairs": len(security),
        "BU": mean(benign_util),
        "UA": mean(util_attack),
        "ASR": mean(security),
        "detail": {
            "utility_under_attack": {f"{k[0]}|{k[1]}": v for k, v in util_attack.items()},
            "security": {f"{k[0]}|{k[1]}": v for k, v in security.items()},
            "benign_utility": {k[0]: v for k, v in benign_util.items()},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="slack")
    ap.add_argument("--attack", default="tool_knowledge", help="paper uses Tool Knowledge")
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--user-tasks", type=int, default=5, help="number of user tasks (first N)")
    ap.add_argument("--injection-tasks", type=int, default=0, help="number of injection tasks (0 = all)")
    ap.add_argument("--arms", nargs="+", default=["baseline", "fides"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached task logs")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    suite = get_suite(args.version, args.suite)
    user_task_ids = list(suite.user_tasks)[: args.user_tasks]
    inj_task_ids = list(suite.injection_tasks)
    if args.injection_tasks:
        inj_task_ids = inj_task_ids[: args.injection_tasks]

    llm = make_qwen_llm(enable_thinking=args.enable_thinking)

    print(f"Config: suite={args.suite} attack={args.attack} model=Qwen3-32B thinking={args.enable_thinking} "
          f"user_tasks={user_task_ids} injection_tasks={inj_task_ids}", flush=True)

    out = {"config": vars(args), "user_task_ids": user_task_ids, "injection_task_ids": inj_task_ids, "arms": {}}
    for arm in args.arms:
        print(f"\n===== ARM: {arm} =====", flush=True)
        res = run_arm(arm, suite, args.attack, user_task_ids, inj_task_ids, llm, args.version, args.force)
        out["arms"][arm] = res
        print(f"[{arm}] FINAL  BU={res['BU']}  UA={res['UA']}  ASR={res['ASR']}  (n_pairs={res['n_pairs']})", flush=True)

    out_path = RESULTS / f"agentdojo_{args.suite}_{args.attack}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    if "baseline" in out["arms"] and "fides" in out["arms"]:
        b, m = out["arms"]["baseline"], out["arms"]["fides"]
        print(f"\nDelta (fides vs baseline):  ASR {b['ASR']} -> {m['ASR']}   UA {b['UA']} -> {m['UA']}   BU {b['BU']} -> {m['BU']}")


if __name__ == "__main__":
    main()
