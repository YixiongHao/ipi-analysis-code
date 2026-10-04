"""Adapter tests for FIDES — both integration seams.

(1) ipi_eval seam (review_tool_call): build a RolloutContext via stubs.make_ctx with a
    message list whose taint we control, then assert review_tool_call returns the
    DETERMINISTIC FIDES decision — block on a P-T tool when the context is tool-tainted,
    allow when the prefix is trusted-only or the tool is neutral. We assert the adapter's
    routing, never that a real model behaved.

(2) AgentDojo seam (FidesMonitor): feed a content-block message list ending in an assistant
    tool-call message and assert the monitor's routing matches the policy — raise
    AbortAgentError when raise_on_violation=True, scrub the tool call when False, and pass
    through untouched for trusted contexts / neutral tools. No agent LLM is invoked
    (the monitor itself makes no model calls).
"""
from __future__ import annotations

import pytest

from testlib import fixtures, pathsetup, stubs

# The adapter takes per-behavior policies from ctx.tool_policies (empty = FIDES is a no-op), so
# the seam tests supply the AgentDojo Table-3 classes for the tools they call.
_POLICIES = {"send_money": "P-T", "send_direct_message": "P-F or P-T"}

impl = pathsetup.load_impl("FIDES")
IntegrityLabel = impl.IntegrityLabel

# Load FIDES's agentdojo_adapter under a unique module name. Loading by the bare name
# `agentdojo_adapter` collides with the other defenses' identically-named modules in
# sys.modules (whichever imported first wins), so we go through pathsetup.
_ad_adapter = pathsetup.load_impl("FIDES", "agentdojo_adapter.py")
FidesMonitor = _ad_adapter.FidesMonitor


# --------------------------------------------------------------------- helpers ---
def _clean_trusted_prefix() -> list[dict]:
    """A trusted-only prefix (system + user) built from the fixture — NO tool turns,
    so context_integrity is trusted (the fixture's own prefill is already tool-tainted)."""
    f = fixtures.ipi_transcript()
    return [
        {"role": "system", "content": f["system_prompt"]},
        {"role": "user", "content": f["user_request"]},
    ]


# ============================================================= (1) ipi_eval seam ===
class TestIpiEvalReviewToolCall:
    def test_block_pt_tool_when_context_tool_tainted(self):
        """Tool-tainted prefix (includes the injected tool msg) + a P-T tool -> blocked."""
        adapter = pathsetup.get_ipi_adapter("fides")
        tainted = fixtures.ipi_prefill_messages(include_injection=True)
        assert any(m.get("role") == "tool" for m in tainted)  # confirm taint present
        ctx = stubs.make_ctx(messages=tainted, tool_policies=_POLICIES)
        adapter.on_rollout_start(ctx)

        allow, new_args, replacement = adapter.review_tool_call("send_money", {"amount": 1}, ctx)
        assert allow is False
        assert new_args is None
        assert "FIDES blocked" in replacement
        assert "P-T violation" in replacement

    def test_allow_pt_tool_when_prefix_trusted_only(self):
        """Clean trusted-only prefix + same P-T tool -> allowed (the decision flips purely on
        taint, demonstrating it's a function of context integrity)."""
        adapter = pathsetup.get_ipi_adapter("fides")
        ctx = stubs.make_ctx(messages=_clean_trusted_prefix(), tool_policies=_POLICIES)
        adapter.on_rollout_start(ctx)

        allow, new_args, replacement = adapter.review_tool_call("send_money", {"amount": 1}, ctx)
        assert allow is True
        assert new_args is None
        assert replacement is None

    def test_neutral_tool_allowed_even_when_tainted(self):
        """The IPI fixture's decisive tool (execute_sql) is neutral under these policies, so FIDES
        passes it through regardless of taint — verifies neutral routing on real fixture data."""
        adapter = pathsetup.get_ipi_adapter("fides")
        ctx = stubs.make_ctx(messages=fixtures.ipi_prefill_messages(include_injection=True),
                              tool_policies=_POLICIES)
        adapter.on_rollout_start(ctx)

        call = fixtures.ipi_decisive_call()  # {'function': 'execute_sql', 'args': {...}}
        allow, new_args, replacement = adapter.review_tool_call(call["function"], call["args"], ctx)
        assert allow is True
        assert (new_args, replacement) == (None, None)

    def test_pf_or_pt_tool_blocked_when_tainted(self):
        adapter = pathsetup.get_ipi_adapter("fides")
        ctx = stubs.make_ctx(messages=fixtures.ipi_prefill_messages(include_injection=True),
                              tool_policies=_POLICIES)
        adapter.on_rollout_start(ctx)
        allow, _, replacement = adapter.review_tool_call(
            "send_direct_message", {"recipient": "x"}, ctx)
        assert allow is False
        assert "FIDES blocked" in replacement

    def test_review_lazy_inits_without_explicit_on_rollout_start(self):
        # review_tool_call must self-initialize the impl if on_rollout_start was skipped.
        adapter = pathsetup.get_ipi_adapter("fides")
        ctx = stubs.make_ctx(messages=fixtures.ipi_prefill_messages(include_injection=True),
                              tool_policies=_POLICIES)
        allow, _, _ = adapter.review_tool_call("send_money", {}, ctx)
        assert allow is False

    def test_drives_loop_is_false(self):
        # FIDES is a per-call seam, not a full-loop defense.
        assert pathsetup.get_ipi_adapter("fides").drives_loop() is False

    def test_returns_review_result_shape(self):
        adapter = pathsetup.get_ipi_adapter("fides")
        ctx = stubs.make_ctx(messages=_clean_trusted_prefix())
        result = adapter.review_tool_call("read_file", {}, ctx)
        assert isinstance(result, tuple) and len(result) == 3
        allow, new_args, replacement = result
        assert isinstance(allow, bool)


# ============================================================= (2) AgentDojo seam ===
def _assistant_with_call(name: str, args: dict):
    from agentdojo.functions_runtime import FunctionCall
    return {"role": "assistant", "content": None,
            "tool_calls": [FunctionCall(function=name, args=args, id="tc_mal")]}


def _tainted_messages_then_call(name: str, args: dict) -> list[dict]:
    """AgentDojo content-block messages ending at an injected tool result, then an assistant
    message proposing `name`(args). The tool result before it taints the context."""
    msgs = fixtures.agentdojo_chat_messages(with_injection=True)  # ends with a tool result
    assert msgs[-1]["role"] == "tool"
    msgs.append(_assistant_with_call(name, args))
    return msgs


class TestAgentDojoMonitor:
    def test_constructs_with_default_defense(self):
        mon = FidesMonitor()  # no LLM needed — the monitor makes no model calls
        assert mon.raise_on_violation is True
        assert mon.defense is not None

    def test_raises_abort_on_pt_violation(self):
        from agentdojo.agent_pipeline.errors import AbortAgentError
        from agentdojo.functions_runtime import FunctionsRuntime

        mon = FidesMonitor(raise_on_violation=True)
        msgs = _tainted_messages_then_call("send_direct_message", {"recipient": "attacker", "body": "x"})
        with pytest.raises(AbortAgentError) as exc:
            mon.query("q", FunctionsRuntime([]), messages=msgs, extra_args={})
        assert "FIDES policy abort" in str(exc.value)

    def test_scrubs_tool_call_when_not_raising(self):
        from agentdojo.functions_runtime import FunctionsRuntime

        mon = FidesMonitor(raise_on_violation=False)
        msgs = _tainted_messages_then_call("send_direct_message", {"recipient": "attacker", "body": "x"})
        _, _, _, out_msgs, extra = mon.query("q", FunctionsRuntime([]), messages=msgs, extra_args={})
        scrubbed = out_msgs[-1]
        assert scrubbed["role"] == "assistant"
        assert scrubbed["tool_calls"] is None          # the proposed action was removed
        assert "fides_blocked" in extra
        assert any(d["allowed"] is False for d in extra["fides_decisions"])

    def test_passthrough_for_trusted_context(self):
        """No tool output before the call (trusted ctx) -> P-T satisfied -> passthrough."""
        from agentdojo.functions_runtime import FunctionsRuntime
        from agentdojo.types import text_content_block_from_string

        mon = FidesMonitor(raise_on_violation=True)
        msgs = [
            {"role": "system", "content": [text_content_block_from_string("sys")]},
            {"role": "user", "content": [text_content_block_from_string("send money")]},
            _assistant_with_call("send_money", {"amount": 1}),
        ]
        q, rt, env, out_msgs, extra = mon.query("q", FunctionsRuntime([]), messages=msgs, extra_args={})
        assert out_msgs is msgs                          # untouched passthrough
        assert "fides_blocked" not in extra
        assert all(d["allowed"] for d in extra.get("fides_decisions", []))

    def test_passthrough_for_neutral_tool_even_when_tainted(self):
        from agentdojo.functions_runtime import FunctionsRuntime

        mon = FidesMonitor(raise_on_violation=True)
        msgs = _tainted_messages_then_call("read_file", {"path": "/tmp/x"})
        _, _, _, out_msgs, extra = mon.query("q", FunctionsRuntime([]), messages=msgs, extra_args={})
        assert out_msgs is msgs
        assert "fides_blocked" not in extra

    def test_no_tool_calls_passthrough(self):
        from agentdojo.functions_runtime import FunctionsRuntime
        from agentdojo.types import text_content_block_from_string

        mon = FidesMonitor()
        msgs = [
            {"role": "user", "content": [text_content_block_from_string("hi")]},
            {"role": "assistant",
             "content": [text_content_block_from_string("done")], "tool_calls": None},
        ]
        _, _, _, out_msgs, _ = mon.query("q", FunctionsRuntime([]), messages=msgs, extra_args={})
        assert out_msgs is msgs

    def test_empty_messages_passthrough(self):
        from agentdojo.functions_runtime import FunctionsRuntime
        _, _, _, out_msgs, _ = FidesMonitor().query("q", FunctionsRuntime([]), messages=[], extra_args={})
        assert out_msgs == []

    def test_explicit_extra_args_isolates_decisions(self):
        """Contract / footgun guard: query() records decisions into the caller-supplied
        extra_args (default param is a shared mutable {} — see FidesMonitor.query). Passing a
        fresh dict per call keeps each query's decisions isolated, which is what the real
        AgentDojo loop does."""
        from agentdojo.functions_runtime import FunctionsRuntime

        mon = FidesMonitor(raise_on_violation=False)
        e1: dict = {}
        mon.query("q", FunctionsRuntime([]),
                  messages=_tainted_messages_then_call("send_direct_message", {"recipient": "x"}),
                  extra_args=e1)
        assert e1.get("fides_blocked")                       # blocked recorded in e1

        e2: dict = {}
        mon.query("q", FunctionsRuntime([]),
                  messages=_tainted_messages_then_call("read_file", {"path": "/tmp/x"}),
                  extra_args=e2)
        # Fresh dict -> no leakage of the previous query's block; only this query's decision.
        assert "fides_blocked" not in e2
        assert [d["tool"] for d in e2["fides_decisions"]] == ["read_file"]
