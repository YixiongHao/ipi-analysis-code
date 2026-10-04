"""Pre-author + validate + cache IPIGuard TDGs for AgentDojo tasks, then reuse them at eval time.

Mirrors the arena reuse-plans split (ipi_suites/harness.py: construct-only vs traverse-only) but
sources runtime/env/prompt from an AgentDojo TaskSuite instead of an arena packet — so no live TDG
construction happens during the FP-utility eval (the plan is built ahead of time by the authoring
workflow and validated to utility). No fork edits.

  build_and_save_plan(suite, user_task, ...) -> construct-only pipeline captures extra_args["dag"],
      serialized to plans_agentdojo/<suite>/<user_task>.json ({suite, user_task_id, model,
      planned_tools, dag:[{id,function_name,args,depends_on}]}).
  replay_saved_plan(suite, user_task, saved, ...) -> traverse-only pipeline seeded with the saved
      dag via run_task_with_pipeline_msg(extra_args=...); DagToolsExecutionLoop skips construction.
      Returns (utility: bool, messages).

Agent (construct + traverse Argument-Estimation) = deepseek-v4-pro via OpenRouter. The fork's
ipiguard_llm.chat_completion_request sends no extra_body, so we monkeypatch it to add the deepseek
first-party provider pin + reasoning.enabled=false (same as every other arm). deepseek supports
response_format={"type":"json_object"} through OpenRouter (probed OK).
"""

from __future__ import annotations

import json
from pathlib import Path

import _fork_env  # noqa: F401  (must precede agentdojo imports; sets fork sys.path + stubs)
import networkx as nx
import openai

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.llms import ipiguard_llm
from agentdojo.agent_pipeline.llms.ipiguard_llm import OpenAIConstructLLM, OpenAITraverseLLM
from agentdojo.agent_pipeline.tool_execution import DagToolsExecutionLoop, DagToolsExecutor
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime

_SD = Path("system-defenses")
import sys
sys.path.insert(0, str(_SD / "fp_agentdojo"))
from deepseek_llm import DEFAULT_MODEL, openrouter_key  # noqa: E402
from ipi_arena_compat import openrouter_extra_body  # noqa: E402

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# --- deepseek routing: inject provider pin + reasoning-off into every IPIGuard LLM call ---------
def _patch_extra_body(model: str) -> None:
    if getattr(ipiguard_llm.chat_completion_request, "_ds_patched", False):
        return
    orig = ipiguard_llm.chat_completion_request
    eb = openrouter_extra_body(model, thinking=False)

    def patched(client, model, messages, temperature=0.0, json_format=False):
        create_kwargs = {"model": model, "messages": messages, "temperature": temperature,
                         "extra_body": eb}
        if json_format:
            create_kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**create_kwargs)

    patched._ds_patched = True
    ipiguard_llm.chat_completion_request = patched


def make_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=openrouter_key(), base_url=OPENROUTER_BASE, timeout=600.0, max_retries=2)


# --- dag (de)serialization (verbatim from ipi_suites/harness.py, packet-independent) ------------
def _dag_to_json(dag) -> list[dict]:
    out = []
    for n in dag.nodes:
        fc = dag.nodes[n]["function_call"]
        out.append({"id": fc.id, "function_name": fc.function, "args": fc.args,
                    "depends_on": dag.nodes[n].get("depends_on", [])})
    return out


def _json_to_dag(nodes: list[dict]):
    dag = nx.DiGraph()
    for d in nodes:
        fc = FunctionCall(function=d["function_name"], args=d.get("args") or {}, id=str(d["id"]))
        dag.add_node(fc.id, function_call=fc, depends_on=d.get("depends_on", []))
    for d in nodes:
        for dep in d.get("depends_on", []):
            dag.add_edge(str(dep), str(d["id"]))
    return dag


# --- build (construct-only) --------------------------------------------------------------------
def build_and_save_plan(suite, user_task, out_dir: Path, model: str = DEFAULT_MODEL) -> dict:
    _patch_extra_body(model)
    client = make_client()
    construct_pipe = AgentPipeline([
        SystemMessage(load_system_message(None)), InitQuery(), OpenAIConstructLLM(client, model),
    ])
    env = suite.load_and_inject_default_environment({})
    task_env = user_task.init_environment(env)
    runtime = FunctionsRuntime(suite.tools)
    *_, extra = construct_pipe.query(user_task.PROMPT, runtime, task_env, [],
                                     {"input_tokens": 0, "output_tokens": 0})
    dag_json = _dag_to_json(extra["dag"])
    rec = {"suite": suite.name, "user_task_id": user_task.ID, "model": model,
           "planned_tools": sorted({d["function_name"] for d in dag_json}), "dag": dag_json}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{user_task.ID}.json").write_text(json.dumps(rec, indent=2))
    return rec


# --- replay (traverse-only, no construction) ---------------------------------------------------
def replay_saved_plan(suite, user_task, saved: dict, model: str = DEFAULT_MODEL) -> tuple[bool, list]:
    _patch_extra_body(model)
    client = make_client()
    traverse_pipe = AgentPipeline([
        SystemMessage(load_system_message(None)), InitQuery(),
        DagToolsExecutionLoop(DagToolsExecutor(OpenAITraverseLLM(client, model))),
    ])
    dag = _json_to_dag(saved["dag"])
    extra = {"dag": dag, "input_tokens": 0, "output_tokens": 0}
    # This fork's run_task_with_pipeline_msg returns (utility, security, messages, args).
    utility, _sec, messages, _args = suite.run_task_with_pipeline_msg(
        traverse_pipe, user_task, injection_task=None, injections={}, extra_args=extra,
    )
    return utility, messages
