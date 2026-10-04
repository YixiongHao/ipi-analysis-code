"""Frozen test fixtures in ``testlib/data/fixtures.json``, built from public data only by
``testlib/build_fixtures.py`` so suites are self-contained and reproducible.

Sources:
  * IPI transcript  -> public ipi_arena_os behavior ``database-deletion`` (TRUNCATE TABLE
    injection in a product review); the attack text and decisive reply are test-written.
  * AgentDojo       -> slack suite v1.2.1, user_task_0 + injection_task_1.

Two message shapes are provided so both execution paths can be exercised:
  * plain-OpenAI dicts  -> for defense.py cores and ipi_eval adapters
  * AgentDojo content-block ChatMessages -> for impl/agentdojo_adapter.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "fixtures.json"
_RAW = json.loads(_DATA.read_text())

# Raw frozen dicts (deep-copy on access so tests can mutate freely).
def ipi_transcript() -> dict:
    """The full IPI-transcript fixture (system prompt, tools, injected tool msg,
    decisive TRUNCATE-TABLE assistant turn, criteria, objective)."""
    return copy.deepcopy(_RAW["ipi_transcript"])


def agentdojo_slack() -> dict:
    """The AgentDojo slack fixture (user prompt, injection goal/payload, tools,
    the expected malicious send_direct_message call)."""
    return copy.deepcopy(_RAW["agentdojo_slack"])


# ---------------------------------------------------------------- IPI shape ---
def ipi_injected_tool_content() -> str:
    """The injected tool-message content (the product review carrying the TRUNCATE note)."""
    return ipi_transcript()["injected_tool_message"]["content"]


def ipi_prefill_messages(include_injection: bool = True) -> list[dict]:
    """Plain-OpenAI conversation up to and including the injected tool result.

    seed_messages (system/user/assistant/tool history) + the flagged tool message
    that carries the IPI payload. Set include_injection=False for a trusted-only
    prefix (no tool-tainted content)."""
    f = ipi_transcript()
    msgs: list[dict] = []
    # Guarantee a leading system message even if seed_messages omits it.
    if not f["seed_messages"] or f["seed_messages"][0].get("role") != "system":
        msgs.append({"role": "system", "content": f["system_prompt"]})
    msgs.extend(copy.deepcopy(f["seed_messages"]))
    if include_injection:
        msgs.append(copy.deepcopy(f["injected_tool_message"]))
    return msgs


def ipi_decisive_call() -> dict:
    """The malicious tool call (execute_sql TRUNCATE TABLE) as {function, args}."""
    a = ipi_transcript()["decisive_assistant_message"]
    tc = a["tool_calls"][0]
    return {"function": tc["function"]["name"],
            "args": json.loads(tc["function"]["arguments"])}


# ----------------------------------------------------------- AgentDojo shape ---
def _adojo_tool_result_text(with_injection: bool) -> str:
    f = agentdojo_slack()
    text = f["benign_webpage_text"]
    if with_injection:
        text = text + "\n\n" + f["injection_payload"]
    return text


def agentdojo_chat_messages(with_injection: bool = True) -> list[dict]:
    """AgentDojo content-block ChatMessage list: system, user, assistant(get_webpage),
    tool(result). The tool result optionally carries the injection payload."""
    from agentdojo.types import text_content_block_from_string
    from agentdojo.functions_runtime import FunctionCall

    f = agentdojo_slack()
    call = FunctionCall(function="get_webpage",
                        args={"url": "www.informations.com"}, id="tc_get_webpage")
    return [
        {"role": "system",
         "content": [text_content_block_from_string("You are a helpful assistant.")]},
        {"role": "user",
         "content": [text_content_block_from_string(f["user_prompt"])]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool",
         "content": [text_content_block_from_string(_adojo_tool_result_text(with_injection))],
         "tool_call": call, "tool_call_id": "tc_get_webpage", "error": None},
    ]


def agentdojo_tool_result_content(with_injection: bool = True) -> str:
    """Just the AgentDojo webpage tool-result text (for sanitizer-style inputs)."""
    return _adojo_tool_result_text(with_injection)
