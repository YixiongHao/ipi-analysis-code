"""Adapter wiring tests for the Firewalls defense.

Two execution paths:
  (1) ipi_eval seams  -> FirewallsDefense.transform_tool_output / review_tool_call
  (2) AgentDojo pipeline elements -> SanitizerElement / MinimizerElement .query

In both, the auxiliary firewall LLM is replaced by a scripted stub returning KNOWN
JSON; we assert the adapter routes inputs to the impl and writes the stub's output
back into the message/args, and otherwise passes the pipeline tuple through.
"""
import importlib.util
import sys

import pytest

from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.types import get_text_content_as_str

from testlib import fixtures, pathsetup, stubs


def _load_agentdojo_adapter():
    pathsetup.add_impl("Firewalls")
    path = pathsetup.impl_dir("Firewalls") / "agentdojo_adapter.py"
    modname = "_firewalls_agentdojo_adapter"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_adojo = _load_agentdojo_adapter()


# ============================================================== ipi_eval path ===
class TestIpiEvalAdapter:
    def _adapter_with(self, chat_responses):
        a = pathsetup.get_ipi_adapter("firewalls")
        a.impl._client = stubs.StubOpenAIClient(chat_responses=chat_responses)
        return a

    def test_transform_tool_output_routes_and_extracts(self):
        a = self._adapter_with('{"sanitized_output": "KNOWN_CLEAN"}')
        ctx = stubs.make_ctx(user_request="run the migration")
        content = fixtures.ipi_injected_tool_content()
        assert a.transform_tool_output(content, ctx) == "KNOWN_CLEAN"

    def test_transform_tool_output_fail_open(self):
        a = self._adapter_with(RuntimeError("down"))
        ctx = stubs.make_ctx()
        content = fixtures.ipi_injected_tool_content()
        assert a.transform_tool_output(content, ctx) == content

    def test_review_tool_call_returns_minimized_triple(self):
        a = self._adapter_with('{"minimized_args": {"database": "prod_db"}}')
        ctx = stubs.make_ctx(user_request="list tables")
        args = fixtures.ipi_decisive_call()["args"]
        result = a.review_tool_call("run_sql_query", args, ctx)
        assert result == (True, {"database": "prod_db"}, None)

    def test_review_tool_call_fail_open_keeps_args(self):
        a = self._adapter_with("garbage-not-json")
        ctx = stubs.make_ctx()
        args = fixtures.ipi_decisive_call()["args"]
        allow, new_args, reason = a.review_tool_call("run_sql_query", args, ctx)
        assert allow is True
        assert new_args == args  # original args preserved on fail-open
        assert reason is None


# ============================================================= AgentDojo path ===
def _stubbed_defense(chat_responses):
    d = _adojo.Defense()
    d._client = stubs.StubOpenAIClient(chat_responses=chat_responses)
    return d


class TestSanitizerElement:
    def test_rewrites_trailing_tool_message(self):
        d = _stubbed_defense('{"sanitized_output": "SANITIZED_BODY"}')
        elem = _adojo.SanitizerElement(d)
        messages = fixtures.agentdojo_chat_messages(with_injection=True)
        query = "summarize the page"

        q, rt, env, out_msgs, extra = elem.query(
            query, FunctionsRuntime(), EmptyEnv(), messages, {})

        # Tuple passthrough (query/runtime/env/extra unchanged).
        assert q == query
        # The trailing tool message content was replaced with the stub output.
        assert get_text_content_as_str(out_msgs[-1]["content"]) == "SANITIZED_BODY"

    def test_no_tool_message_is_passthrough(self):
        d = _stubbed_defense('{"sanitized_output": "SHOULD_NOT_APPLY"}')
        elem = _adojo.SanitizerElement(d)
        # Slice ending at the assistant message => last role != "tool".
        messages = fixtures.agentdojo_chat_messages(with_injection=True)[:3]

        q, rt, env, out_msgs, extra = elem.query(
            "q", FunctionsRuntime(), EmptyEnv(), messages, {})

        assert out_msgs == messages          # untouched
        assert d._client.calls == []         # LLM never called

    def test_fail_open_leaves_tool_content(self):
        d = _stubbed_defense(RuntimeError("aux llm down"))
        elem = _adojo.SanitizerElement(d)
        messages = fixtures.agentdojo_chat_messages(with_injection=True)
        original = get_text_content_as_str(messages[-1]["content"])

        _, _, _, out_msgs, _ = elem.query(
            "q", FunctionsRuntime(), EmptyEnv(), messages, {})

        assert get_text_content_as_str(out_msgs[-1]["content"]) == original


class TestMinimizerElement:
    def test_rewrites_tool_call_args(self):
        d = _stubbed_defense('{"minimized_args": {"url": "www.informations.com"}}')
        elem = _adojo.MinimizerElement(d)
        # Slice to end on the assistant message holding the tool call.
        messages = fixtures.agentdojo_chat_messages(with_injection=True)[:3]
        assert messages[-1]["role"] == "assistant"

        q, rt, env, out_msgs, extra = elem.query(
            "fetch the page", FunctionsRuntime(), EmptyEnv(), messages, {})

        assert q == "fetch the page"
        tc = out_msgs[-1]["tool_calls"][0]
        assert tc.args == {"url": "www.informations.com"}

    def test_no_assistant_message_is_passthrough(self):
        d = _stubbed_defense('{"minimized_args": {"x": 1}}')
        elem = _adojo.MinimizerElement(d)
        messages = fixtures.agentdojo_chat_messages(with_injection=True)  # ends on tool

        _, _, _, out_msgs, _ = elem.query(
            "q", FunctionsRuntime(), EmptyEnv(), messages, {})

        assert out_msgs == messages
        assert d._client.calls == []

    def test_fail_open_keeps_original_args(self):
        d = _stubbed_defense("not json")
        elem = _adojo.MinimizerElement(d)
        messages = fixtures.agentdojo_chat_messages(with_injection=True)[:3]
        original_args = dict(messages[-1]["tool_calls"][0].args)

        _, _, _, out_msgs, _ = elem.query(
            "q", FunctionsRuntime(), EmptyEnv(), messages, {})

        assert out_msgs[-1]["tool_calls"][0].args == original_args
