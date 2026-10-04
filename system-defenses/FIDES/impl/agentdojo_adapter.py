"""AgentDojo wiring for FIDES (taint + P-T policy monitor).

FIDES is a planning/taint defense, so there is no drop-in pipeline slot — instead we wrap the
existing tool loop with a *taint monitor*. ``FidesMonitor`` is a ``BasePipelineElement`` placed
**before** ``ToolsExecutor`` inside the ``ToolsExecutionLoop``: it inspects the tool calls the
LLM just proposed, computes the integrity label of the context that produced them (system/user
trusted, tool outputs untrusted), and enforces each tool's Table-3 policy via ``defense.py``. A
consequential/egress call generated in an untrusted context is a P-T violation → the agent is
aborted, so the injected action never executes.

Agent LLM = Qwen3-32B served by local vLLM (OpenAI-compatible, port 8000), thinking OFF (repo
standard; FIDES's security mechanism does not rely on agent CoT). The QwenLLM subclass mirrors
``system-defenses/MELON/impl/agentdojo_adapter.py``.

Compose (see run_agentdojo.py)::

    llm = make_qwen_llm()
    loop = ToolsExecutionLoop([FidesMonitor(Defense(), raise_on_violation=True), ToolsExecutor(), llm])
    pipeline = AgentPipeline([SystemMessage(sys), InitQuery(), llm, loop])
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import openai

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from defense import Defense, IntegrityLabel

QWEN_MODEL = "Qwen3-32B"


# --------------------------------------------------------------------------------------------
# Agent LLM factory (local vLLM, thinking OFF)
# --------------------------------------------------------------------------------------------
class QwenLLM(OpenAILLM):
    """OpenAILLM that disables Qwen3 thinking via ``chat_template_kwargs.enable_thinking``."""

    def __init__(self, client: openai.OpenAI, model: str, enable_thinking: bool = False, temperature: float | None = 0.0):
        super().__init__(client, model, temperature=temperature)
        self.enable_thinking = enable_thinking

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        openai_messages = [_message_to_openai(m, self.model) for m in messages]
        openai_tools = [_function_to_openai(t) for t in runtime.functions.values()]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools or openai.NOT_GIVEN,
            tool_choice="auto" if openai_tools else openai.NOT_GIVEN,
            temperature=self.temperature if self.temperature is not None else openai.NOT_GIVEN,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args


def make_qwen_llm(enable_thinking: bool = False) -> QwenLLM:
    port = os.getenv("LOCAL_LLM_PORT", "8000")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    return QwenLLM(client, QWEN_MODEL, enable_thinking=enable_thinking)


# --------------------------------------------------------------------------------------------
# The FIDES taint monitor
# --------------------------------------------------------------------------------------------
class FidesMonitor(BasePipelineElement):
    """Taint + per-tool-policy check. Place BEFORE ``ToolsExecutor`` in ``ToolsExecutionLoop``.

    Args:
        defense: a configured ``Defense`` (Table 3 policies + P-T enforcement).
        raise_on_violation: if True (default), raise ``AbortAgentError`` on a P-T violation so
            the benchmark records the task as aborted (injected action blocked). If False, scrub
            the offending assistant tool calls into a stop message (loop terminates quietly).
    """

    def __init__(self, defense: Defense | None = None, raise_on_violation: bool = True) -> None:
        super().__init__()
        self.defense = defense if defense is not None else Defense()
        self.raise_on_violation = raise_on_violation

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        last = messages[-1]
        if last["role"] != "assistant" or not last.get("tool_calls"):
            return query, runtime, env, messages, extra_args

        # Integrity of the context that produced these tool calls (everything before this msg).
        ctx_integrity = self.defense.context_integrity(messages[:-1])
        ctx = {"context_integrity": ctx_integrity}

        decisions = extra_args.setdefault("fides_decisions", [])
        for tc in last["tool_calls"]:
            allowed, reason = self.defense.allow_call(tc.function, tc.args or {}, ctx)
            decisions.append({"tool": tc.function, "allowed": allowed, "reason": reason})
            if not allowed:
                extra_args["fides_blocked"] = reason
                if self.raise_on_violation:
                    raise AbortAgentError(f"FIDES policy abort: {reason}", list(messages), env)
                # scrub: replace the proposed action with a terminating assistant message
                scrubbed = list(messages)
                scrubbed[-1] = {
                    "role": "assistant",
                    "content": [text_content_block_from_string(f"Blocked by FIDES policy: {reason}")],
                    "tool_calls": None,
                }
                return query, runtime, env, scrubbed, extra_args

        return query, runtime, env, messages, extra_args


__all__ = ["FidesMonitor", "QwenLLM", "make_qwen_llm", "QWEN_MODEL", "Defense", "IntegrityLabel"]
