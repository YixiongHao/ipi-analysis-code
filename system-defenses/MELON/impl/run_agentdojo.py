"""MELON AgentDojo replication harness (Phase 4).

Runs the paper's metrics (BU / UA / ASR) on a chosen suite x attack for two arms:
  * baseline  — Qwen3-32B, no defense
  * melon     — Qwen3-32B + MELON detector (aborts on detection)

Defaults to the confirmed first-pass cell: slack suite x important_instructions, first 5
user tasks x all 5 injection tasks. Agent = local Qwen3-32B vLLM (thinking OFF), embeddings =
local BGE vLLM. Writes results/agentdojo_<suite>_<attack>.json.

Run (master venv; servers on ports 8000/8001 must be up):
    python run_agentdojo.py --user-tasks 5
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

from agentdojo_adapter import MELONDetector, make_bge_embed_fn, make_qwen_llm

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LOGDIR = RESULTS / "runs"


def build_pipeline(arm: str, llm, embed_fn, sim_threshold: float) -> AgentPipeline:
    system_message = load_system_message(None)
    components = [SystemMessage(system_message), InitQuery(), llm]
    if arm == "baseline":
        loop = ToolsExecutionLoop([ToolsExecutor(), llm])
        name = "Qwen3-32B"
    elif arm == "melon":
        melon = MELONDetector(llm, embed_fn, sim_threshold=sim_threshold, raise_on_injection=True)
        loop = ToolsExecutionLoop([ToolsExecutor(), melon])
        name = f"Qwen3-32B-melon-t{sim_threshold}"
    elif arm == "melon-aug":
        # MELON-Aug = MELON + repeat_user_prompt: MELON re-appends the user query inside its
        # original run (augment=True), strengthening focus on the user task (paper §4.2). It stays
        # in the loop in place of the in-loop LLM so the tool message still triggers detection.
        melon = MELONDetector(llm, embed_fn, sim_threshold=sim_threshold, raise_on_injection=True, augment=True)
        loop = ToolsExecutionLoop([ToolsExecutor(), melon])
        name = f"Qwen3-32B-melon-aug-t{sim_threshold}"
    else:
        raise ValueError(arm)
    pipeline = AgentPipeline([*components, loop])
    pipeline.name = name
    return pipeline


def mean(d: dict) -> float:
    return round(100.0 * sum(bool(v) for v in d.values()) / len(d), 2) if d else 0.0


def run_arm(arm, suite, attack_name, user_task_ids, inj_task_ids, llm, embed_fn, threshold, version, force):
    pipeline = build_pipeline(arm, llm, embed_fn, threshold)
    attack = load_attack(attack_name, suite, pipeline)

    util_attack: dict = {}   # UA: user-task success under attack
    security: dict = {}      # ASR: injection-task success
    benign_util: dict = {}   # BU: user-task success, no attack

    with OutputLogger(str(LOGDIR), live=None):
        for utid in user_task_ids:
            user_task = suite.get_user_task_by_id(utid)
            # UA + ASR (user task x each injection task, attack injected)
            u, s = run_task_with_injection_tasks(
                suite, pipeline, user_task, attack, LOGDIR, force, inj_task_ids, version
            )
            util_attack.update(u)
            security.update(s)
            # BU (no injection) — returns (successful: bool, _)
            bu_success, _ = run_task_without_injection_tasks(suite, pipeline, user_task, LOGDIR, force, version)
            benign_util[utid] = bu_success
            print(f"[{arm}] {utid}: UA so far {mean(util_attack)}  ASR so far {mean(security)}  BU so far {mean(benign_util)}", flush=True)

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
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--user-tasks", type=int, default=5, help="number of user tasks (first N)")
    ap.add_argument("--injection-tasks", type=int, default=0, help="number of injection tasks (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--arms", nargs="+", default=["baseline", "melon"])
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
    embed_fn = make_bge_embed_fn()

    print(f"Config: suite={args.suite} attack={args.attack} model=Qwen3-32B thinking={args.enable_thinking} "
          f"theta={args.threshold} user_tasks={user_task_ids} injection_tasks={inj_task_ids}", flush=True)

    out = {"config": vars(args), "user_task_ids": user_task_ids, "injection_task_ids": inj_task_ids, "arms": {}}
    for arm in args.arms:
        print(f"\n===== ARM: {arm} =====", flush=True)
        res = run_arm(arm, suite, args.attack, user_task_ids, inj_task_ids, llm, embed_fn,
                      args.threshold, args.version, args.force)
        out["arms"][arm] = res
        print(f"[{arm}] FINAL  BU={res['BU']}  UA={res['UA']}  ASR={res['ASR']}  (n_pairs={res['n_pairs']})", flush=True)

    out_path = RESULTS / f"agentdojo_{args.suite}_{args.attack}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    if "baseline" in out["arms"] and "melon" in out["arms"]:
        b, m = out["arms"]["baseline"], out["arms"]["melon"]
        print(f"\nDelta (melon vs baseline):  ASR {b['ASR']} -> {m['ASR']}   UA {b['UA']} -> {m['UA']}   BU {b['BU']} -> {m['BU']}")


if __name__ == "__main__":
    main()
