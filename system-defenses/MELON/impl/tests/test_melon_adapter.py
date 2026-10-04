"""Adapter wiring tests for MELON — both the AgentDojo BasePipelineElement
(impl/agentdojo_adapter.py) and the ipi_eval DefenseAdapter (ipi_eval/adapters/melon.py).

These assert CONTROL FLOW with stubbed LLM + stub embed fn: detector caching, that the
original vs masked runs are issued, and what happens on a detected match (scrub vs abort vs
AbortAgentError). They never assert a real attack was caught — the "match" is forced by feeding
the stub LLM identical original/masked tool calls so stub-embed cosine == 1.0 > theta.
"""
import json

import pytest

from testlib import fixtures, pathsetup, stubs

mel = pathsetup.load_impl("MELON")
adapter_mod = pathsetup.load_impl("MELON", "agentdojo_adapter.py", "_melon_agentdojo_adapter")

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.types import text_content_block_from_string  # noqa: E402


# =============================================================== AgentDojo path ===
class _StubInnerLLM(BasePipelineElement):
    """Minimal inner-LLM stub: each .query appends an assistant message whose tool_calls
    are the next scripted list. The MELONDetector calls .query twice (original, then masked)."""

    def __init__(self, scripted_tool_calls):
        self.scripted = list(scripted_tool_calls)
        self.calls = []

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        self.calls.append(list(messages))
        tcs = self.scripted.pop(0) if len(self.scripted) > 1 else self.scripted[0]
        out = {"role": "assistant",
               "content": [text_content_block_from_string("")],
               "tool_calls": tcs}
        return query, runtime, env, [*messages, out], extra_args


def _malicious_fc():
    mc = fixtures.agentdojo_slack()["malicious_call"]
    return FunctionCall(function=mc["function"], args=mc["args"], id="tc_mal")


def test_agentdojo_noop_when_last_message_not_tool():
    # Detector only fires after a tool result; a non-tool tail is passed through unchanged.
    llm = _StubInnerLLM([[_malicious_fc()]])
    det = adapter_mod.MELONDetector(llm, stubs.stub_embed_fn, sim_threshold=0.8)
    msgs = [{"role": "user", "content": [text_content_block_from_string("hi")]}]
    q, r, e, out, ea = det.query("hi", FunctionsRuntime([]), EmptyEnv(), msgs, {})
    assert out is msgs
    assert llm.calls == []  # no runs issued
    assert "_melon_detector" not in ea


def test_agentdojo_caches_detector_and_runs_original_and_masked():
    mc = _malicious_fc()
    # original run -> malicious call; masked run -> the SAME call (forces stub cosine 1.0).
    llm = _StubInnerLLM([[mc], [mc]])
    det = adapter_mod.MELONDetector(llm, stubs.stub_embed_fn, sim_threshold=0.8,
                                    raise_on_injection=False)
    msgs = fixtures.agentdojo_chat_messages(with_injection=True)  # ends with role == "tool"
    q, r, e, out, ea = det.query("send the summary", FunctionsRuntime([]), EmptyEnv(), msgs, {})

    # Detector cached in extra_args (per-task bank persistence). The adapter loads defense.py
    # under module name "defense", so compare by class name rather than identity.
    assert "_melon_detector" in ea
    assert type(ea["_melon_detector"]).__name__ == "MelonDetector"
    assert hasattr(ea["_melon_detector"], "call_bank")
    # Two inner-LLM runs issued: original then masked.
    assert len(llm.calls) == 2
    # Match detected.
    assert ea["is_injection"] is True
    # Scrub path: action replaced with a no-tool-call stop assistant message.
    assert out[-1]["role"] == "assistant"
    assert out[-1]["tool_calls"] is None
    assert "prompt injection" in out[-1]["content"][0]["content"]
    # The tool output was scrubbed with the transform sentinel.
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert any(adapter_mod.MELONDetector.transform() in
               (b.get("content") or "" for b in m["content"]) for m in tool_msgs)


def test_agentdojo_raise_on_injection_raises_abort():
    mc = _malicious_fc()
    llm = _StubInnerLLM([[mc], [mc]])
    det = adapter_mod.MELONDetector(llm, stubs.stub_embed_fn, sim_threshold=0.8,
                                    raise_on_injection=True)
    msgs = fixtures.agentdojo_chat_messages(with_injection=True)
    with pytest.raises(AbortAgentError):
        det.query("send the summary", FunctionsRuntime([]), EmptyEnv(), msgs, {})


def test_agentdojo_no_match_passes_through():
    # original run makes a benign call; masked run makes a disjoint call -> stub cosine 0 -> no flag.
    orig = FunctionCall(function="read_file", args={"name": "report.txt"}, id="o")
    masked = FunctionCall(function="send_email", args={"recipients": "friend@x.com"}, id="m")
    llm = _StubInnerLLM([[orig], [masked]])
    det = adapter_mod.MELONDetector(llm, stubs.stub_embed_fn, sim_threshold=0.8,
                                    raise_on_injection=False)
    msgs = fixtures.agentdojo_chat_messages(with_injection=False)
    q, r, e, out, ea = det.query("read the file", FunctionsRuntime([]), EmptyEnv(), msgs, {})
    assert ea["is_injection"] is False
    # The original action (read_file) is preserved, not scrubbed.
    assert out[-1]["role"] == "assistant"
    assert out[-1]["tool_calls"] == [orig]


def test_agentdojo_original_no_tool_calls_short_circuits():
    # If the original run produces no tool calls, MELON records is_injection False and returns.
    llm = _StubInnerLLM([None, [_malicious_fc()]])  # original=None, masked=malicious
    det = adapter_mod.MELONDetector(llm, stubs.stub_embed_fn, sim_threshold=0.8)
    msgs = fixtures.agentdojo_chat_messages(with_injection=True)
    q, r, e, out, ea = det.query("noop", FunctionsRuntime([]), EmptyEnv(), msgs, {})
    assert ea.get("is_injection") is False
    assert out[-1]["tool_calls"] is None


# =============================================================== ipi_eval path ===
def _openai_tc(name, args, id="a"):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_ipi_adapter_registry_shape():
    ad = pathsetup.get_ipi_adapter("melon")
    assert ad.name == "melon"
    assert ad.drives_loop() is False  # seam-based (inspect_assistant), not full-loop


def _wire_stub_detector(ad, target_client):
    """Pre-seed the adapter with a stub-embed detector + defense_mod so inspect_assistant
    skips on_rollout_start (which would build the real BGE embed fn / hit :8001)."""
    ad.defense_mod = mel
    ad.detector = mel.MelonDetector(stubs.stub_embed_fn, sim_threshold=0.8)
    ctx = stubs.make_ctx(target_client=target_client)
    ctx.scratch["melon_detector"] = ad.detector
    return ctx


def test_ipi_adapter_inspect_no_tool_calls_returns_none():
    ad = pathsetup.get_ipi_adapter("melon")
    target = stubs.StubLLMClient()
    ctx = _wire_stub_detector(ad, target)
    res = ad.inspect_assistant({"role": "assistant", "content": "done", "tool_calls": None}, ctx)
    assert res is None
    assert target.calls == []  # no masking run when there is nothing to check


def test_ipi_adapter_inspect_flags_match_returns_abort():
    call = fixtures.ipi_decisive_call()  # run_sql_query DROP TABLE
    # masked run (via target_client) returns the SAME malicious call -> stub cosine 1.0 -> abort.
    masked = stubs.tool_call(call["function"], call["args"])
    target = stubs.StubLLMClient([stubs.chat_response(tool_calls=[masked])])
    ad = pathsetup.get_ipi_adapter("melon")
    ctx = _wire_stub_detector(ad, target)

    orig_msg = {"role": "assistant", "content": "",
                "tool_calls": [_openai_tc(call["function"], call["args"])]}
    res = ad.inspect_assistant(orig_msg, ctx)
    assert res == "abort"
    assert len(target.calls) == 1  # exactly one masking re-execution issued


def test_ipi_adapter_inspect_no_match_returns_none():
    # original run = run_sql_query; masked run returns a disjoint call -> no flag.
    call = fixtures.ipi_decisive_call()
    masked = stubs.tool_call("send_email", {"recipients": "friend@x.com"})
    target = stubs.StubLLMClient([stubs.chat_response(tool_calls=[masked])])
    ad = pathsetup.get_ipi_adapter("melon")
    ctx = _wire_stub_detector(ad, target)

    orig_msg = {"role": "assistant", "content": "",
                "tool_calls": [_openai_tc(call["function"], call["args"])]}
    res = ad.inspect_assistant(orig_msg, ctx)
    assert res is None
    assert len(target.calls) == 1
