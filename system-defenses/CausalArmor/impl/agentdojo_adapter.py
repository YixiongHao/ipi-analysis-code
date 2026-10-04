"""AgentDojo wiring for CausalArmor.

CausalArmor guards the *proposed privileged action* before it is executed, so the
adapter is a ``BasePipelineElement`` placed **first** in the ``ToolsExecutionLoop``
(before ``ToolsExecutor``). Each loop iteration it inspects ``messages[-1]`` (the
assistant message the LLM just produced):

  * Gate (Alg. 2 l.2): only act if a proposed tool call is privileged AND there is at
    least one prior tool-output span. Otherwise pass through (zero overhead).
  * Step 1: LOO attribution over the context (``defense.analyze``).
  * If any span is flagged (Eq. 5): sanitize the flagged tool spans (Appendix D.1),
    retroactively mask assistant CoT after the first injection (Alg. 2 l.19-25), then
    **re-generate** the action on the sanitized context and let ``ToolsExecutor`` run
    the new action.

Compose (see run_agentdojo.py)::

    llm = make_qwen_llm()
    ca  = CausalArmorElement(Defense(tau=0.0), llm)
    loop = ToolsExecutionLoop([ca, ToolsExecutor(), llm])
    pipeline = AgentPipeline([system_message, init_query, llm, loop])

Agent + proxy + sanitizer all = Qwen3-32B on the local vLLM endpoint (port 8000),
thinking OFF. Paper used Gemini/Gemma/Claude — a deliberate deviation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import openai

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string

from defense import COT_MASK_PLACEHOLDER, Defense

QWEN_MODEL = "Qwen3-32B"


# --------------------------------------------------------------------------------------------
# Agent LLM (Qwen3-32B, thinking OFF) — same pattern as MELON's adapter.
# --------------------------------------------------------------------------------------------
class QwenLLM(OpenAILLM):
    def __init__(self, client: openai.OpenAI, model: str, enable_thinking: bool = False,
                 temperature: float | None = 0.0):
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


def make_qwen_llm(enable_thinking: bool = False, port_env: str = "LOCAL_LLM_PORT") -> QwenLLM:
    port = os.getenv(port_env, "8000")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    return QwenLLM(client, QWEN_MODEL, enable_thinking=enable_thinking)


# --------------------------------------------------------------------------------------------
# Message <-> attribution-context helpers
# --------------------------------------------------------------------------------------------
def _text(m: ChatMessage) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return get_text_content_as_str(c)


def _flatten(messages: Sequence[ChatMessage], defense: Defense) -> list[dict]:
    """1:1 map AgentDojo messages -> {role, content} dicts for LOO attribution.

    Length/order preserved, so attribution span indices == indices into `messages`.
    Assistant tool calls are serialized into the text so the context is faithful.
    """
    out = []
    for m in messages:
        role, txt = m["role"], _text(m)
        if role == "assistant" and m.get("tool_calls"):
            calls = " ".join(defense.serialize_action(tc.function, tc.args) for tc in m["tool_calls"])
            txt = (txt + "\n" + calls).strip() if txt else calls
        out.append({"role": role, "content": txt})
    return out


def _proposed_action(assistant_msg: ChatMessage, defense: Defense) -> str:
    return " ".join(defense.serialize_action(tc.function, tc.args)
                    for tc in (assistant_msg.get("tool_calls") or []))


# --------------------------------------------------------------------------------------------
# CausalArmor pipeline element
# --------------------------------------------------------------------------------------------
class CausalArmorElement(BasePipelineElement):
    def __init__(self, defense: Defense, llm: BasePipelineElement):
        self.defense = defense
        self.llm = llm                      # used to re-generate on the sanitized context
        self.events: list[dict] = []        # per-decision telemetry for the harness

    def query(self, query, runtime, env=EmptyEnv(), messages: Sequence[ChatMessage] = (),
              extra_args: dict = {}):
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        proposed = messages[-1]
        calls = proposed.get("tool_calls") or []
        history = list(messages[:-1])

        # -- Gate (Alg. 2 l.2): a privileged proposed call + a prior tool span ----------
        privileged = [tc for tc in calls if self.defense.is_privileged(tc.function)]
        span_indices = [i for i, m in enumerate(history) if m["role"] == "tool"]
        if not privileged or not span_indices:
            return query, runtime, env, messages, extra_args

        # -- Step 1: LOO attribution ----------------------------------------------------
        action = _proposed_action(proposed, self.defense)
        ctx = _flatten(history, self.defense)
        attr = self.defense.analyze(action, ctx)
        event = {
            "action": action, "delta_u_norm": attr.delta_u_norm,
            "max_span_norm": attr.max_span_norm, "flagged": list(attr.flagged),
            "intervened": False,
        }

        if not attr.flagged:
            self.events.append(event)
            return query, runtime, env, messages, extra_args      # execute original action

        # -- Stage 1: selective sanitization of flagged tool spans ----------------------
        user_request = next((_text(m) for m in history if m["role"] == "user"), "")
        new_history = [dict(m) for m in history]
        for i in attr.flagged:
            tool_msg = new_history[i]
            tool_name = getattr(tool_msg.get("tool_call"), "function", "tool")
            cleaned = self.defense.sanitize(_text(tool_msg), user_request, tool_name)
            tool_msg["content"] = [text_content_block_from_string(cleaned)]

        # -- Stage 2: retroactive CoT masking after first injection (Alg. 2 l.19-25) ----
        if self.defense.cot_masking:
            k_min = min(attr.flagged)
            for j in range(k_min + 1, len(new_history)):
                if new_history[j]["role"] == "assistant":
                    new_history[j] = {**new_history[j],
                                      "content": [text_content_block_from_string(COT_MASK_PLACEHOLDER)]}

        # -- Re-generate the action on the sanitized/masked context ---------------------
        event["intervened"] = True
        self.events.append(event)
        return self.llm.query(query, runtime, env, new_history, extra_args)
