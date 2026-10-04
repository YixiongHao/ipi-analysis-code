"""CausalArmor AgentDojo replication harness (Phase 4).

Runs BU / UA / ASR on a chosen suite x attack for two arms:
  * baseline    — Qwen3-32B, no defense
  * causalarmor — Qwen3-32B + CausalArmor (attribution-gated sanitize + CoT mask + re-gen)

Defaults to the first-pass cell: slack suite x important_instructions, first N user tasks
x all injection tasks. Agent + proxy + sanitizer all = local Qwen3-32B vLLM (port 8000),
thinking OFF. Writes results/agentdojo_<suite>_<attack>.json.

Run (master venv; vLLM Qwen3-32B on port 8000 must be up):
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

from agentdojo_adapter import CausalArmorElement, make_qwen_llm
from defense import Defense

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LOGDIR = RESULTS / "runs"


def build_pipeline(arm: str, llm, tau: float):
    system_message = load_system_message(None)
    components = [SystemMessage(system_message), InitQuery(), llm]
    ca_element = None
    if arm == "baseline":
        loop = ToolsExecutionLoop([ToolsExecutor(), llm])
        name = "Qwen3-32B"
    elif arm == "causalarmor":
        ca_element = CausalArmorElement(Defense(tau=tau), llm)
        # CausalArmor runs FIRST in the loop: guard the proposed action, then execute, then next LLM.
        loop = ToolsExecutionLoop([ca_element, ToolsExecutor(), llm])
        name = f"Qwen3-32B-causalarmor-tau{tau}"
    else:
        raise ValueError(arm)
    pipeline = AgentPipeline([*components, loop])
    pipeline.name = name
    return pipeline, ca_element


def mean(d: dict) -> float:
    return round(100.0 * sum(bool(v) for v in d.values()) / len(d), 2) if d else 0.0


def run_arm(arm, suite, attack_name, user_task_ids, inj_task_ids, llm, tau, version, force):
    pipeline, ca_element = build_pipeline(arm, llm, tau)
    attack = load_attack(attack_name, suite, pipeline)

    util_attack: dict = {}
    security: dict = {}
    benign_util: dict = {}

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

    res = {
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
            "benign_utility": {k[0] if isinstance(k, tuple) else k: v for k, v in benign_util.items()},
        },
    }
    if ca_element is not None:
        n_dec = len(ca_element.events)
        n_int = sum(e["intervened"] for e in ca_element.events)
        res["causalarmor_stats"] = {
            "privileged_decisions": n_dec,
            "interventions": n_int,
            "intervention_rate": round(n_int / n_dec, 3) if n_dec else 0.0,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="slack")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--user-tasks", type=int, default=5)
    ap.add_argument("--injection-tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--arms", nargs="+", default=["baseline", "causalarmor"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    suite = get_suite(args.version, args.suite)
    user_task_ids = list(suite.user_tasks)[: args.user_tasks]
    inj_task_ids = list(suite.injection_tasks)
    if args.injection_tasks:
        inj_task_ids = inj_task_ids[: args.injection_tasks]

    llm = make_qwen_llm(enable_thinking=args.enable_thinking)
    print(f"Config: suite={args.suite} attack={args.attack} model=Qwen3-32B thinking={args.enable_thinking} "
          f"tau={args.tau} user_tasks={user_task_ids} injection_tasks={inj_task_ids}", flush=True)

    out = {"config": vars(args), "user_task_ids": user_task_ids, "injection_task_ids": inj_task_ids, "arms": {}}
    for arm in args.arms:
        print(f"\n===== ARM: {arm} =====", flush=True)
        res = run_arm(arm, suite, args.attack, user_task_ids, inj_task_ids, llm, args.tau, args.version, args.force)
        out["arms"][arm] = res
        print(f"[{arm}] FINAL  BU={res['BU']}  UA={res['UA']}  ASR={res['ASR']}  (n_pairs={res['n_pairs']})", flush=True)

    tau_tag = f"_tau{args.tau}" if args.tau != 0.0 else ""
    out_path = RESULTS / f"agentdojo_{args.suite}_{args.attack}{tau_tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    if "baseline" in out["arms"] and "causalarmor" in out["arms"]:
        b, c = out["arms"]["baseline"], out["arms"]["causalarmor"]
        print(f"\nDelta (causalarmor vs baseline):  ASR {b['ASR']} -> {c['ASR']}   "
              f"UA {b['UA']} -> {c['UA']}   BU {b['BU']} -> {c['BU']}")


if __name__ == "__main__":
    main()
