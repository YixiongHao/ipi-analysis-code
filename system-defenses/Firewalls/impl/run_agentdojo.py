"""Replicate the Firewalls AgentDojo results (Phase 4).

Agent model: Qwen3-32B via local vLLM (port 8000, vllm_parsed/hermes), thinking OFF
(via `/no_think` appended to the system message). Attack: tool_knowledge (the attack
used in the paper's detailed Qwen3-32b tables 10-12). Defenses: none vs sanitizer
(headline); minimizer/combined also available.

Reports BU (utility, no attack), UA (utility under attack), ASR (security under
attack), matching the paper. Writes a JSON summary to results/.

Usage:
  python run_agentdojo.py --suite banking --defense sanitizer --max-user-tasks 8
  python run_agentdojo.py --suite banking --defense none      --max-user-tasks 8
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

# make `import defense` / `import agentdojo_adapter` work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentdojo.agent_pipeline.agent_pipeline import MODEL_PROVIDERS, AgentPipeline, get_llm, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite

from agentdojo_adapter import MinimizerElement, SanitizerElement
from defense import Defense

BENCHMARK_VERSION = "v1.2.2"


def build_pipeline(defense_name: str, model: str, no_think: bool):
    provider = MODEL_PROVIDERS[ModelsEnum(model)]
    llm = get_llm(provider, model, None, "tool")
    llm_name = getattr(llm, "name", None) or model

    system_message = load_system_message(None)
    if no_think:
        system_message = system_message + " /no_think"
    sys_msg = SystemMessage(system_message)
    init_query = InitQuery()
    fmt = tool_result_to_str

    d = Defense() if defense_name != "none" else None
    loop_elements = [ToolsExecutor(fmt)]
    if defense_name in ("minimizer", "combined"):
        # minimizer must run BEFORE execution -> it sits before ToolsExecutor; we
        # rebuild the loop element order accordingly.
        loop_elements = [MinimizerElement(d), ToolsExecutor(fmt)]
    if defense_name in ("sanitizer", "combined"):
        loop_elements.append(SanitizerElement(d))
    loop_elements.append(llm)

    pipeline = AgentPipeline([sys_msg, init_query, llm, ToolsExecutionLoop(loop_elements)])
    pipeline.name = f"{llm_name}-firewalls-{defense_name}"
    return pipeline, d


def rate(d: dict) -> float:
    return 100.0 * sum(1 for v in d.values() if v) / len(d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--defense", default="sanitizer", choices=["none", "sanitizer", "minimizer", "combined"])
    ap.add_argument("--attack", default="tool_knowledge")
    ap.add_argument("--model", default="vllm_parsed")
    ap.add_argument("--max-user-tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--max-injection-tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--think", dest="no_think", action="store_false")
    args = ap.parse_args()

    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    logdir = outdir / "logs"

    suite = get_suite(BENCHMARK_VERSION, args.suite)
    user_tasks = list(suite.user_tasks.keys())
    inj_tasks = list(suite.injection_tasks.keys())
    if args.max_user_tasks:
        user_tasks = user_tasks[: args.max_user_tasks]
    if args.max_injection_tasks:
        inj_tasks = inj_tasks[: args.max_injection_tasks]

    pipeline, d = build_pipeline(args.defense, args.model, args.no_think)

    with OutputLogger(str(logdir), live=None):
        # --- Benign Utility (no attack) ---
        bu_res = benchmark_suite_without_injections(
            pipeline, suite, logdir, force_rerun=False,
            user_tasks=user_tasks, benchmark_version=BENCHMARK_VERSION,
        )
        bu = rate(bu_res["utility_results"])

        # --- Under attack: UA (utility) + ASR (security) ---
        attack = load_attack(args.attack, suite, pipeline)
        atk_res = benchmark_suite_with_injections(
            pipeline, suite, attack, logdir, force_rerun=False,
            user_tasks=user_tasks, injection_tasks=inj_tasks,
            verbose=False, benchmark_version=BENCHMARK_VERSION,
        )
        ua = rate(atk_res["utility_results"])
        asr = rate(atk_res["security_results"])

    summary = {
        "suite": args.suite, "defense": args.defense, "attack": args.attack,
        "model": args.model, "no_think": args.no_think,
        "n_user_tasks": len(user_tasks), "n_injection_tasks": len(inj_tasks),
        "n_attack_pairs": len(atk_res["security_results"]),
        "BU": round(bu, 2), "UA": round(ua, 2), "ASR": round(asr, 2),
        "firewall_calls": getattr(d, "calls", 0),
        "firewall_parse_failures": getattr(d, "parse_failures", 0),
    }
    out = outdir / f"agentdojo_{args.suite}_{args.defense}_{args.attack}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
