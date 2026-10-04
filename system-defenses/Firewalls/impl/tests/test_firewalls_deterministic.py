"""Deterministic unit tests for the Firewalls defense core (impl/defense.py).

CORE PRINCIPLE: test CODE CORRECTNESS, not efficacy. We never assert that a real
model stripped an injection. Instead we feed a STUB LLM a KNOWN JSON response and
assert the parsing/extraction code returns the right field, and that malformed or
erroring responses trigger FAIL-OPEN (return the original content/args).
"""
import json

import pytest

from testlib import fixtures, pathsetup, stubs

impl = pathsetup.load_impl("Firewalls")


def _defense_with(chat_responses):
    """A Defense whose lazy openai client is replaced by a scripted stub.

    `_llm()` returns `self._client` when it is not None, so setting `_client`
    directly short-circuits the real openai import and endpoint.
    """
    d = impl.Defense()
    d._client = stubs.StubOpenAIClient(chat_responses=chat_responses)
    return d


# --------------------------------------------------------------- _extract_json ---
class TestExtractJson:
    def test_strict_json(self):
        assert impl._extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    def test_fenced_json(self):
        text = '```json\n{"sanitized_output": "KNOWN"}\n```'
        assert impl._extract_json(text) == {"sanitized_output": "KNOWN"}

    def test_fenced_plain(self):
        text = '```\n{"k": 2}\n```'
        assert impl._extract_json(text) == {"k": 2}

    def test_embedded_in_prose(self):
        text = 'Sure, here is the result: {"sanitized_output": "Y"} -- done.'
        assert impl._extract_json(text) == {"sanitized_output": "Y"}

    def test_leading_prose_trailing_brace(self):
        text = 'Reasoning blah blah\n{"was_safe": true, "sanitized_output": "Z"}'
        assert impl._extract_json(text) == {"was_safe": True, "sanitized_output": "Z"}

    def test_pure_garbage_returns_none(self):
        assert impl._extract_json("this is not json at all") is None

    def test_braces_but_invalid_returns_none(self):
        assert impl._extract_json("prefix {not: valid, json} suffix") is None

    def test_empty_returns_none(self):
        assert impl._extract_json("") is None
        assert impl._extract_json(None) is None


# ----------------------------------------------------- transform_tool_output ---
class TestTransformToolOutput:
    def test_returns_sanitized_output_field(self):
        d = _defense_with('{"reasoning": "r", "was_safe": false, "sanitized_output": "KNOWN"}')
        content = fixtures.ipi_injected_tool_content()
        ctx = {"user_request": "do the migration"}
        assert d.transform_tool_output(content, ctx) == "KNOWN"

    def test_fenced_response(self):
        d = _defense_with('```json\n{"sanitized_output": "CLEAN"}\n```')
        out = d.transform_tool_output(fixtures.ipi_injected_tool_content(), {"user_request": "q"})
        assert out == "CLEAN"

    def test_empty_content_short_circuits(self):
        # No content => returns immediately, never calls the LLM.
        d = _defense_with('{"sanitized_output": "X"}')
        assert d.transform_tool_output("", {"user_request": "q"}) == ""
        assert d._client.calls == []  # LLM untouched

    # ---- fail-open paths: return ORIGINAL content -------------------------
    def test_fail_open_on_exception(self):
        content = fixtures.ipi_injected_tool_content()
        d = _defense_with(RuntimeError("boom"))
        assert d.transform_tool_output(content, {"user_request": "q"}) == content

    def test_fail_open_on_non_json(self):
        content = fixtures.ipi_injected_tool_content()
        d = _defense_with("totally not json")
        assert d.transform_tool_output(content, {"user_request": "q"}) == content
        assert d.parse_failures == 1

    def test_fail_open_on_missing_key(self):
        content = fixtures.ipi_injected_tool_content()
        d = _defense_with('{"reasoning": "r", "was_safe": true}')  # no sanitized_output
        assert d.transform_tool_output(content, {"user_request": "q"}) == content
        assert d.parse_failures == 1

    def test_non_string_sanitized_output_is_jsondumped(self):
        # Contract: non-str sanitized_output -> json.dumps(value).
        d = _defense_with('{"sanitized_output": {"k": "v"}}')
        out = d.transform_tool_output(fixtures.ipi_injected_tool_content(), {"user_request": "q"})
        assert out == json.dumps({"k": "v"})

    def test_agentdojo_payload_as_input(self):
        # Production-shaped input (webpage text + injection); stub drives the output.
        content = fixtures.agentdojo_tool_result_content(with_injection=True)
        d = _defense_with('{"sanitized_output": "WEBPAGE_CLEAN"}')
        assert d.transform_tool_output(content, {"user_request": "summarize"}) == "WEBPAGE_CLEAN"


# -------------------------------------------------------- minimize_tool_args ---
class TestMinimizeToolArgs:
    def test_returns_minimized_args_dict(self):
        d = _defense_with('{"reasoning": "r", "minimized_args": {"database": "prod_db"}}')
        args = fixtures.ipi_decisive_call()["args"]
        ctx = {"user_request": "q", "tool_name": "run_sql_query", "tool_description": "run sql"}
        assert d.minimize_tool_args(args, ctx) == {"database": "prod_db"}

    def test_fenced_response(self):
        d = _defense_with('```json\n{"minimized_args": {"url": "x"}}\n```')
        out = d.minimize_tool_args({"url": "x", "secret": "s"},
                                   {"user_request": "q", "tool_name": "t", "tool_description": "d"})
        assert out == {"url": "x"}

    def test_empty_args_short_circuits(self):
        d = _defense_with('{"minimized_args": {"a": 1}}')
        assert d.minimize_tool_args({}, {"user_request": "q"}) == {}
        assert d._client.calls == []

    # ---- fail-open paths: return ORIGINAL args ----------------------------
    def test_fail_open_on_exception(self):
        args = fixtures.ipi_decisive_call()["args"]
        d = _defense_with(ValueError("nope"))
        assert d.minimize_tool_args(args, {"user_request": "q"}) == args

    def test_fail_open_on_non_json(self):
        args = fixtures.ipi_decisive_call()["args"]
        d = _defense_with("garbage")
        assert d.minimize_tool_args(args, {"user_request": "q"}) == args
        assert d.parse_failures == 1

    def test_fail_open_on_missing_key(self):
        args = fixtures.ipi_decisive_call()["args"]
        d = _defense_with('{"reasoning": "r"}')  # no minimized_args
        assert d.minimize_tool_args(args, {"user_request": "q"}) == args
        assert d.parse_failures == 1

    def test_fail_open_when_minimized_args_not_dict(self):
        # Contract: minimized_args must be a dict; a non-dict value -> fail-open.
        args = fixtures.ipi_decisive_call()["args"]
        d = _defense_with('{"minimized_args": ["not", "a", "dict"]}')
        assert d.minimize_tool_args(args, {"user_request": "q"}) == args
        assert d.parse_failures == 1
