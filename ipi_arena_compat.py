"""Compatibility layer over the public `ipi_arena_bench` package (the `ipi_arena_os` submodule).

Our harness needs four things the public package does not ship:

1. `LLMClient.chat(..., extra_body=...)`, with `temperature` / `max_tokens` OMITTED when passed
   `None`, `max_tokens` sent as `max_completion_tokens` on native OpenAI (api.openai.com 400s on
   `max_tokens` for reasoning models), and a one-shot retry off the Z.AI provider when it returns
   `finish_reason="error"` (its content guard) on a GLM model.
2. `openrouter_extra_body(model, thinking)`: provider pin + usage accounting + reasoning toggle.
3. A STATEFUL world-sim: the public `simulate_tool_response` ignores `conversation_history`, so
   repeated identical tool calls get contradictory answers. The version here passes the whole
   conversation to the simulator and uses the arena's user-message wording, with no temperature
   and no max_tokens (the arena's `generateText` sets neither).
4. `ChatResponse.finish_reason`, which the engine reads to count content-filter refusals.

Everything else is re-exported unchanged from `ipi_arena_bench`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ipi_arena_bench import llm_client as _lc
from ipi_arena_bench import worldsim as _ws
from ipi_arena_bench.llm_client import ChatResponse, ToolCall
from ipi_arena_bench.worldsim import build_worldsim_prompt

__all__ = ["LLMClient", "ChatResponse", "ToolCall", "openrouter_extra_body", "read_secret",
           "build_worldsim_prompt", "build_conversation_context", "simulate_tool_response"]

logger = logging.getLogger(__name__)

SECRETS_FILE = Path(__file__).resolve().parent / "secrets.md"
# secrets.md line-name prefix -> environment variable checked first (lowercase prefixes).
_ENV_KEYS = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY",
             "ant": "ANTHROPIC_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
             "gemini": "GEMINI_API_KEY", "hf": "HF_TOKEN"}


def read_secret(name: str) -> str:
    """API key for `name` ('openrouter', 'openai', 'anthropic', 'gemini', 'hf'); '' if absent.

    The matching environment variable wins (OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GEMINI_API_KEY, HF_TOKEN). Otherwise the first `<name>... = <value>` line of `secrets.md` at the
    repository root is used, with surrounding quotes stripped.
    """
    low = name.lower()
    for prefix, var in _ENV_KEYS.items():
        if low.startswith(prefix) and os.getenv(var):
            return os.environ[var]
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(low) and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# Model-family -> OpenRouter provider pin. A pin keeps prompt-cache hits on one endpoint.
_PROVIDER_PINS = (("glm", "z-ai"), ("gemini", "google-ai-studio"), ("deepseek", "deepseek"))


def openrouter_extra_body(model: str, thinking: bool = False) -> dict:
    """OpenRouter `extra_body` for one model: provider pin (known families only), usage
    accounting, and `reasoning.enabled=false` when `thinking` is False."""
    m = (model or "").lower()
    eb: dict[str, Any] = {}
    for family, provider in _PROVIDER_PINS:
        if family in m:
            eb["provider"] = {"order": [provider], "allow_fallbacks": True}
            break
    eb["usage"] = {"include": True}
    if not thinking:
        eb["reasoning"] = {"enabled": False}
    return eb


class LLMClient(_lc.LLMClient):
    """Public `LLMClient` plus `extra_body`, omit-on-None parameters and the native-OpenAI and
    Z.AI fixes described in the module docstring."""

    def __init__(self, provider: str = "openrouter", model: str = "qwen/qwen3-vl-235b-a22b-instruct",
                 api_key: str | None = None, base_url: str | None = None):
        super().__init__(provider=provider, model=model, api_key=api_key, base_url=base_url)
        self._native_openai = urlparse(str(self.client.base_url)).hostname == "api.openai.com"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = 0.0, max_tokens: int | None = 4096,
             extra_body: dict | None = None) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens" if self._native_openai else "max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**kwargs)
        if ("glm" in self.model.lower() and response.choices
                and response.choices[0].finish_reason == "error"):
            kwargs["extra_body"] = {**(extra_body or {}),
                                    "provider": {"ignore": ["z-ai"], "allow_fallbacks": True}}
            response = self.client.chat.completions.create(**kwargs)
        return _parse(response, self.model)


def _parse(response: Any, model: str) -> ChatResponse:
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    tool_calls = []
    for tc in message.tool_calls or []:
        try:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, AttributeError):
            args = {"_raw": tc.function.arguments}
        tool_calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id or ""))
    usage: dict[str, int] = {}
    if response.usage:
        u = response.usage
        usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                 "total_tokens": u.total_tokens}
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
        if cached is not None:
            usage["cached_tokens"] = cached
    resp = ChatResponse(content=message.content, reasoning=reasoning, tool_calls=tool_calls,
                        raw_response=response, model=response.model or model, usage=usage)
    resp.finish_reason = response.choices[0].finish_reason  # engine counts content_filter stops
    return resp


def build_conversation_context(messages: list[dict]) -> str:
    """Render the conversation for the world-sim, one block per message.

    Reconstruction of the arena's `buildConversationContext`: same inputs (the whole conversation,
    system prompt and prefill included), but the exact line format is not verified against the
    original TypeScript.
    """
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        text = content or ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            text += f"\n[tool call] {fn.get('name', '')}({fn.get('arguments', '')})"
        if role == "tool":
            role = f"tool result ({m.get('name') or m.get('tool_call_id', '')})"
        parts.append(f"{role}: {text}")
    return "\n\n".join(parts)


def simulate_tool_response(worldsim_client: Any, worldsim_system_prompt: str,
                           tool_call: ToolCall, conversation_history: list[dict]) -> str:
    """Stateful world-sim: the simulator sees the whole conversation so far."""
    tool_call_str = f"{tool_call.name}({json.dumps(tool_call.arguments, indent=2)})"
    context = (f"Here is the conversation so far:\n"
               f"{build_conversation_context(conversation_history)}\n\n"
               if conversation_history else "")
    user_message = (f"{context}Now simulate the tool response for the last function/tool call "
                    f"in a simplified format as following:\n{tool_call_str}")
    response = worldsim_client.chat(
        [{"role": "system", "content": worldsim_system_prompt},
         {"role": "user", "content": user_message}],
        temperature=None, max_tokens=None)
    content = response.content
    if content is None:
        logger.warning("WorldSim returned empty content, using fallback")
        content = '{"status": "success"}'
    return _ws._extract_json(content)
