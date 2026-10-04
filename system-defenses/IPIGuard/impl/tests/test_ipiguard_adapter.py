"""Adapter wiring tests for IPIGuard — both execution paths.

  (1) ipi_eval full-loop adapter (ipi_eval/adapters/ipiguard.py): drives_loop()==True,
      run_loop() builds the TDG then guards each proposed tool call. We FIX the plan
      (monkeypatch impl.plan) so no LLM planning is needed, and script the target client
      deterministically, then assert the loop's admission CODE: an unplanned *command*
      call is stripped from the recorded assistant turn (never executed), while a planned
      or read-only call passes. NEVER an efficacy / model-judgment claim.

  (2) AgentDojo pipeline construction (impl/agentdojo_adapter.py): build_ipiguard_pipeline
      / build_baseline_pipeline return a composed AgentPipeline with the expected element
      types (by class NAME — the fork loads modules under separate instances, so isinstance
      across module copies is unreliable). Construction-only; no LLM is run.
"""
import json

import pytest

from testlib import fixtures, pathsetup, stubs

# Load the impl (fork env handled) and the adapter module by file path.
ig = pathsetup.load_impl("IPIGuard")
adapter_mod = pathsetup.load_impl("IPIGuard", "agentdojo_adapter.py", "_ipiguard_agentdojo_adapter")


def _openai_tc(name, args, id="a"):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# =============================================================== ipi_eval path ===
def test_ipi_adapter_registry_shape():
    ad = pathsetup.get_ipi_adapter("ipiguard")
    assert ad.name == "ipiguard"
    assert ad.drives_loop() is True  # full-loop defense


def _fix_plan(ad, *planned_tools):
    """Replace the LLM-driven planner with a fixed TDG so run_loop is deterministic."""
    plan = {"tool_calls": [], "planned_tools": set(planned_tools), "pending": []}
    ad.impl.plan = lambda user_request, tools, ctx=None: plan
    return plan


def test_run_loop_strips_unplanned_command_call():
    ad = pathsetup.get_ipi_adapter("ipiguard")
    _fix_plan(ad, "read_file")  # run_sql_query is NOT planned

    # Step 1: model proposes an UNPLANNED command tool (run_sql_query is in command_tools).
    # Step 2: a no-tool finish to end the loop.
    target = stubs.StubLLMClient([
        stubs.chat_response(content="", tool_calls=[
            stubs.tool_call("run_sql_query", {"command": "DROP TABLE users_prod"}, "tc_evil")]),
        stubs.chat_response(content="done", tool_calls=[]),
    ])
    ctx = stubs.make_ctx(target_client=target)
    ctx.max_steps = 5

    res = ad.run_loop(ctx)

    assert set(res) >= {"messages", "generated_messages"}
    # The blocked command call was stripped: no executed tool message for it, and the
    # assistant turn no longer carries that tool_call.
    all_tool_calls = []
    for m in res["messages"]:
        if m.get("role") == "assistant":
            for c in (m.get("tool_calls") or []):
                all_tool_calls.append(c["id"] if isinstance(c, dict) else c)
    assert "tc_evil" not in all_tool_calls
    # No tool result was emitted for the blocked call.
    assert not any(m.get("role") == "tool" and m.get("tool_call_id") == "tc_evil"
                   for m in res["messages"])
    # A refusal note surfaced on some assistant message.
    assert any("IPIGuard" in (m.get("content") or "")
               for m in res["messages"] if m.get("role") == "assistant")


def test_run_loop_allows_planned_command_call():
    ad = pathsetup.get_ipi_adapter("ipiguard")
    _fix_plan(ad, "run_sql_query")  # now PLANNED -> admissible

    target = stubs.StubLLMClient([
        stubs.chat_response(content="", tool_calls=[
            stubs.tool_call("run_sql_query", {"command": "SELECT 1"}, "tc_ok")]),
        stubs.chat_response(content="done", tool_calls=[]),
    ])
    ctx = stubs.make_ctx(target_client=target)
    res = ad.run_loop(ctx)

    # Planned call kept and executed (a tool result keyed to it exists).
    assistant_tc_ids = [
        (c["id"] if isinstance(c, dict) else c)
        for m in res["messages"] if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    ]
    assert "tc_ok" in assistant_tc_ids
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tc_ok"
               for m in res["messages"])


def test_run_loop_allows_unplanned_readonly_call():
    # read_file is unplanned but NOT in command_tools -> the adapter's read-only override
    # admits it (Node Expansion). Asserts the loop's classification CODE, not efficacy.
    ad = pathsetup.get_ipi_adapter("ipiguard")
    _fix_plan(ad, "run_sql_query")  # read_file not planned

    target = stubs.StubLLMClient([
        stubs.chat_response(content="", tool_calls=[
            stubs.tool_call("read_file", {"name": "report.txt"}, "tc_read")]),
        stubs.chat_response(content="done", tool_calls=[]),
    ])
    ctx = stubs.make_ctx(target_client=target)
    # default command_tools = {run_sql_query, write_file, execute_bash}; read_file is read-only
    res = ad.run_loop(ctx)

    assistant_tc_ids = [
        (c["id"] if isinstance(c, dict) else c)
        for m in res["messages"] if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    ]
    assert "tc_read" in assistant_tc_ids
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tc_read"
               for m in res["messages"])


# =============================================================== AgentDojo path ===
def test_build_ipiguard_pipeline_composition():
    pipe = adapter_mod.build_ipiguard_pipeline(model="Qwen3-32B")
    # Compare by class name: the fork loads agentdojo under its own module instance.
    names = [type(e).__name__ for e in pipe.elements]
    assert names == ["SystemMessage", "InitQuery", "OpenAIConstructLLM", "DagToolsExecutionLoop"]
    assert "Qwen3-32B" in pipe.name


def test_build_baseline_pipeline_composition():
    pipe = adapter_mod.build_baseline_pipeline(model="Qwen3-32B")
    names = [type(e).__name__ for e in pipe.elements]
    assert names == ["SystemMessage", "InitQuery", "OpenAILLM", "ToolsExecutionLoop"]
    assert "Qwen3-32B" in pipe.name


def test_build_pipeline_dispatch():
    ig_pipe = adapter_mod.build_pipeline("ipiguard", model="Qwen3-32B")
    base_pipe = adapter_mod.build_pipeline(None, model="Qwen3-32B")
    assert type(ig_pipe.elements[2]).__name__ == "OpenAIConstructLLM"
    assert type(base_pipe.elements[2]).__name__ == "OpenAILLM"
    with pytest.raises(ValueError):
        adapter_mod.build_pipeline("nope", model="Qwen3-32B")


def test_build_ipiguard_pipeline_rejects_unsupported_model():
    # The fork's ipiguard branch only supports gpt/Qwen/llama ids (asserted in the adapter).
    with pytest.raises(AssertionError):
        adapter_mod.build_ipiguard_pipeline(model="mistral-7b")


def test_make_client_is_openai_pointed_at_base_url():
    client = adapter_mod.make_client(base_url="http://localhost:8000/v1")
    assert type(client).__name__ == "OpenAI"
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"
