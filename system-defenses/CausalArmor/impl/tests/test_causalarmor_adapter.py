"""Adapter wiring tests for CausalArmor — both execution paths.

  (1) AgentDojo CausalArmorElement.query (impl/agentdojo_adapter.py): inspects the
      proposed privileged action, runs LOO attribution, and on a flag sanitizes the
      flagged tool span(s) + masks downstream CoT + RE-GENERATES via the inner LLM.
  (2) ipi_eval CausalArmorDefense.review_tool_call (ipi_eval/adapters/causalarmor.py):
      the same flow mapped onto the review_tool_call seam.

CORE PRINCIPLE: we assert CONTROL FLOW, not efficacy. The "flag" is forced by stubbing
`analyze` (or `_score_batch`) on the element's internal Defense and stubbing `sanitize`
to a KNOWN string. We assert: privileged+flagged -> sanitize called + CoT masked +
re-generation triggered; non-privileged / non-flagged -> passthrough.

The adapter loads defense.py under its own module name, so the Attribution/SpanScore it
produces are a SEPARATE class identity from the test's `ca` module — we therefore build
those objects from the adapter module's own classes (and duck-type, never isinstance).
"""
import importlib.util
import sys

import pytest

from testlib import fixtures, pathsetup, stubs


# ---- load the adapter under a unique module name (bare `import agentdojo_adapter`
#      would collide across all six defenses). -------------------------------------------
def _load_agentdojo_adapter():
    pathsetup.add_impl("CausalArmor")
    path = pathsetup.impl_dir("CausalArmor") / "agentdojo_adapter.py"
    modname = "_causalarmor_agentdojo_adapter"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_adojo = _load_agentdojo_adapter()

from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.types import (  # noqa: E402
    get_text_content_as_str,
    text_content_block_from_string,
)

# The adapter's OWN Defense class (separate `defense` module identity from `ca`).
# Attribution lives in that same `defense` module (the adapter only re-exports Defense
# and COT_MASK_PLACEHOLDER), so pull it from Defense's module rather than the adapter ns.
_Defense = _adojo.Defense
_Attribution = sys.modules[_Defense.__module__].Attribution
COT = _adojo.COT_MASK_PLACEHOLDER


def _attr(flagged):
    """Build an adapter-module Attribution with a forced `flagged` list."""
    return _Attribution(action="write_file({})", n_action_tokens=10, user_index=1,
                        delta_u=5.0, delta_u_norm=0.5, spans=[], flagged=list(flagged))


class _StubInnerLLM:
    """Records the messages it is asked to re-generate on; appends a benign assistant msg."""
    def __init__(self):
        self.calls = []

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        self.calls.append(list(messages))
        out = {"role": "assistant",
               "content": [text_content_block_from_string("REGENERATED")],
               "tool_calls": None}
        return query, runtime, env, [*messages, out], extra_args


def _privileged_assistant_msgs():
    """history = system,user,asst(get_webpage),tool(injected),asst(reasoning) ;
    + a trailing assistant proposing a PRIVILEGED write_file call (the guarded action).

    The extra assistant at index 4 (AFTER the injected tool span at index 3) exists so
    the CoT-masking step has a downstream reasoning message to redact."""
    base = fixtures.agentdojo_chat_messages(with_injection=True)
    base.append({"role": "assistant",
                 "content": [text_content_block_from_string("Let me follow that note.")],
                 "tool_calls": None})
    call = FunctionCall(function="write_file",
                        args={"path": "/tmp/x", "content": "data"}, id="tc_write")
    base.append({"role": "assistant", "content": None, "tool_calls": [call]})
    return base


# ============================================================= AgentDojo path ===
class TestAgentDojoElement:
    def test_flagged_privileged_sanitizes_masks_and_regenerates(self, monkeypatch):
        d = _Defense(tau=0.0)
        # Force a flag on the injected tool span (index 3 in the fixture history).
        monkeypatch.setattr(d, "analyze", lambda action, ctx: _attr([3]))
        san_calls = []

        def _fake_sanitize(content, user_request, tool_name):
            san_calls.append((content, user_request, tool_name))
            return "SANITIZED_SPAN"
        monkeypatch.setattr(d, "sanitize", _fake_sanitize)

        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)
        messages = _privileged_assistant_msgs()

        q, rt, env, out_msgs, extra = elem.query(
            "do the task", FunctionsRuntime(), EmptyEnv(), messages, {})

        # Re-generation was triggered (inner LLM called exactly once) and its output is the tail.
        assert len(llm.calls) == 1
        assert get_text_content_as_str(out_msgs[-1]["content"]) == "REGENERATED"
        # sanitize() called on the flagged tool span exactly once, with its content.
        assert len(san_calls) == 1
        assert san_calls[0][0] == fixtures.agentdojo_tool_result_content(True)
        # The regenerated-on history had the flagged span replaced with the stub string...
        regen_history = llm.calls[0]
        assert get_text_content_as_str(regen_history[3]["content"]) == "SANITIZED_SPAN"
        # ...and the assistant CoT AFTER the first injection was masked.
        assert get_text_content_as_str(regen_history[4]["content"]) == COT
        # Telemetry records the intervention.
        assert elem.events[-1]["intervened"] is True
        assert elem.events[-1]["flagged"] == [3]

    def test_flagged_but_cot_masking_off_skips_mask(self, monkeypatch):
        d = _Defense(tau=0.0, cot_masking=False)
        monkeypatch.setattr(d, "analyze", lambda action, ctx: _attr([3]))
        monkeypatch.setattr(d, "sanitize", lambda c, u, t: "SANITIZED_SPAN")
        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)

        elem.query("t", FunctionsRuntime(), EmptyEnv(), _privileged_assistant_msgs(), {})

        regen_history = llm.calls[0]
        # Span sanitized but the downstream assistant message is NOT masked.
        assert get_text_content_as_str(regen_history[3]["content"]) == "SANITIZED_SPAN"
        assert get_text_content_as_str(regen_history[4]["content"]) != COT

    def test_not_flagged_passes_through_original_action(self, monkeypatch):
        d = _Defense(tau=0.0)
        monkeypatch.setattr(d, "analyze", lambda action, ctx: _attr([]))   # nothing flagged
        sanitize_called = []
        monkeypatch.setattr(d, "sanitize",
                            lambda c, u, t: sanitize_called.append(1) or "X")
        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)
        messages = _privileged_assistant_msgs()

        q, rt, env, out_msgs, extra = elem.query(
            "t", FunctionsRuntime(), EmptyEnv(), messages, {})

        # No sanitize, no re-generation; original messages returned unchanged.
        assert sanitize_called == []
        assert llm.calls == []
        assert out_msgs is messages
        assert elem.events[-1]["intervened"] is False

    def test_non_privileged_action_is_gated_out(self, monkeypatch):
        # Proposed call is read_file (not privileged) -> attribution never runs.
        d = _Defense(tau=0.0)
        analyze_called = []
        monkeypatch.setattr(d, "analyze",
                            lambda a, c: analyze_called.append(1) or _attr([3]))
        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)
        msgs = fixtures.agentdojo_chat_messages(with_injection=True)
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [FunctionCall(function="read_file",
                                                 args={"name": "a"}, id="r")]})

        q, rt, env, out_msgs, extra = elem.query(
            "t", FunctionsRuntime(), EmptyEnv(), msgs, {})

        assert analyze_called == []     # gated out before attribution
        assert llm.calls == []
        assert out_msgs is msgs

    def test_no_prior_tool_span_is_gated_out(self, monkeypatch):
        # Privileged proposal but NO prior tool output span -> gate passes through.
        d = _Defense(tau=0.0)
        monkeypatch.setattr(d, "analyze",
                            lambda a, c: (_ for _ in ()).throw(AssertionError("called")))
        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)
        msgs = [
            {"role": "system", "content": [text_content_block_from_string("s")]},
            {"role": "user", "content": [text_content_block_from_string("u")]},
            {"role": "assistant", "content": None,
             "tool_calls": [FunctionCall(function="write_file", args={"p": "x"}, id="w")]},
        ]
        q, rt, env, out_msgs, extra = elem.query(
            "t", FunctionsRuntime(), EmptyEnv(), msgs, {})
        assert out_msgs is msgs
        assert llm.calls == []

    def test_last_message_not_assistant_passthrough(self):
        d = _Defense(tau=0.0)
        llm = _StubInnerLLM()
        elem = _adojo.CausalArmorElement(d, llm)
        msgs = fixtures.agentdojo_chat_messages(with_injection=True)  # ends on tool
        q, rt, env, out_msgs, extra = elem.query(
            "t", FunctionsRuntime(), EmptyEnv(), msgs, {})
        assert out_msgs is msgs
        assert llm.calls == []


# =============================================================== ipi_eval path ===
def _openai_tc(name, args, id="a"):
    import json
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class TestIpiEvalAdapter:
    def test_registry_shape(self):
        ad = pathsetup.get_ipi_adapter("causalarmor")
        assert ad.name == "causalarmor"
        assert ad.drives_loop() is False     # seam-based (review_tool_call)

    def _adapter(self):
        ad = pathsetup.get_ipi_adapter("causalarmor")
        ad.on_rollout_start(stubs.make_ctx())   # builds ad.impl (tokenizer falls back if missing)
        return ad

    def test_non_privileged_allows(self):
        ad = self._adapter()
        ctx = stubs.make_ctx()
        # read_file is not privileged -> (True, None, None) without touching the model.
        assert ad.review_tool_call("read_file", {"name": "x"}, ctx) == (True, None, None)

    def test_privileged_no_tool_span_allows(self):
        ad = self._adapter()
        # Messages with no tool span at all.
        ctx = stubs.make_ctx(messages=[{"role": "system", "content": "s"},
                                       {"role": "user", "content": "u"}])
        assert ad.review_tool_call("write_file", {"p": "x"}, ctx) == (True, None, None)

    def test_attribution_failure_fails_open(self, monkeypatch):
        ad = self._adapter()
        monkeypatch.setattr(ad.impl, "analyze",
                            lambda a, m: (_ for _ in ()).throw(RuntimeError("server down")))
        ctx = stubs.make_ctx()  # default messages contain tool spans
        assert ad.review_tool_call("write_file", {"p": "x"}, ctx) == (True, None, None)

    def test_not_flagged_allows(self, monkeypatch):
        ad = self._adapter()
        monkeypatch.setattr(ad.impl, "analyze", lambda a, m: _attr([]))
        ctx = stubs.make_ctx()
        assert ad.review_tool_call("write_file", {"p": "x"}, ctx) == (True, None, None)

    def test_flagged_regenerates_same_tool_returns_new_args(self, monkeypatch):
        ad = self._adapter()
        ctx = stubs.make_ctx()
        flag_idx = next(i for i, m in enumerate(ctx.messages) if m.get("role") == "tool")
        monkeypatch.setattr(ad.impl, "analyze", lambda a, m: _attr([flag_idx]))
        san_calls = []
        monkeypatch.setattr(ad.impl, "sanitize",
                            lambda c, u, t: san_calls.append((c, t)) or "SANITIZED")
        # target_client re-generates the SAME privileged tool with new args.
        regen = stubs.tool_call("write_file", {"p": "clean"})
        ctx.target_client = stubs.StubLLMClient([stubs.chat_response(tool_calls=[regen])])

        allow, new_args, reason = ad.review_tool_call("write_file", {"p": "x"}, ctx)
        assert allow is True
        assert new_args == {"p": "clean"}       # regenerated args substituted
        assert reason is None
        assert len(san_calls) == 1              # flagged span sanitized
        assert len(ctx.target_client.calls) == 1

    def test_flagged_regen_drops_call_blocks(self, monkeypatch):
        ad = self._adapter()
        ctx = stubs.make_ctx()
        flag_idx = next(i for i, m in enumerate(ctx.messages) if m.get("role") == "tool")
        monkeypatch.setattr(ad.impl, "analyze", lambda a, m: _attr([flag_idx]))
        monkeypatch.setattr(ad.impl, "sanitize", lambda c, u, t: "SANITIZED")
        # Re-generation makes NO call to the privileged tool -> BLOCK.
        ctx.target_client = stubs.StubLLMClient([stubs.chat_response(content="refused")])

        allow, new_args, reason = ad.review_tool_call("write_file", {"p": "x"}, ctx)
        assert allow is False
        assert new_args is None
        assert "CausalArmor" in reason

    def test_flagged_regen_failure_fails_open(self, monkeypatch):
        ad = self._adapter()
        ctx = stubs.make_ctx()
        flag_idx = next(i for i, m in enumerate(ctx.messages) if m.get("role") == "tool")
        monkeypatch.setattr(ad.impl, "analyze", lambda a, m: _attr([flag_idx]))
        monkeypatch.setattr(ad.impl, "sanitize", lambda c, u, t: "SANITIZED")

        class _Boom:
            def chat(self, *a, **k):
                raise RuntimeError("regen down")
        ctx.target_client = _Boom()
        assert ad.review_tool_call("write_file", {"p": "x"}, ctx) == (True, None, None)
