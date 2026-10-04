"""Deterministic stand-ins for the LLM / embedding backends, server probes, and a
RolloutContext builder.

The stubs let tests assert how a defense's *code* handles a KNOWN model/embedding
output — never the judgment of a real model. Live wiring tests use ``qwen_up`` /
``bge_up`` to skip when servers are down.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


# --------------------------------------------------------------- embeddings ---
def stub_embed_fn(text: str, dim: int = 64) -> list[float]:
    """Deterministic hashing bag-of-words embedding. Identical text -> identical
    vector (cosine 1.0); disjoint token sets -> orthogonal (cosine 0.0)."""
    vec = [0.0] * dim
    for tok in str(text).split():
        vec[hash_token(tok) % dim] += 1.0
    return vec


def hash_token(tok: str) -> int:
    # stable across processes (Python's hash() is salted for str), so roll our own.
    h = 0
    for ch in tok:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


# ------------------------------------------------------------- OpenAI client ---
@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg
    text: str = ""
    logprobs: Any = None


@dataclass
class _Resp:
    choices: list


class _ChatCompletions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls.append(("chat", kwargs))
        content = self._outer._next_chat()
        if isinstance(content, Exception):
            raise content
        return _Resp(choices=[_Choice(message=_Msg(content=content))])


class _Completions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls.append(("completion", kwargs))
        item = self._outer._next_completion()
        if isinstance(item, Exception):
            raise item
        text, logprobs = item if isinstance(item, tuple) else (item, None)
        return _Resp(choices=[_Choice(message=_Msg(content=text), text=text, logprobs=logprobs)])


class StubOpenAIClient:
    """Mimics ``openai.OpenAI``. Feed scripted chat / completion responses.

    chat_responses / completion_responses may each be a list (consumed in order,
    last value repeats) or a single value. An ``Exception`` value is raised to
    exercise fail-open paths. Completion responses may be ``(text, logprobs)``."""

    def __init__(self, chat_responses: Any = None, completion_responses: Any = None):
        self._chat = list(chat_responses) if isinstance(chat_responses, list) else (
            [chat_responses] if chat_responses is not None else [])
        self._comp = list(completion_responses) if isinstance(completion_responses, list) else (
            [completion_responses] if completion_responses is not None else [])
        self.calls: list = []
        self.chat = type("ChatNS", (), {})()
        self.chat.completions = _ChatCompletions(self)
        self.completions = _Completions(self)

    def _next_chat(self):
        if not self._chat:
            return ""
        return self._chat.pop(0) if len(self._chat) > 1 else self._chat[0]

    def _next_completion(self):
        if not self._comp:
            return ("", None)
        return self._comp.pop(0) if len(self._comp) > 1 else self._comp[0]


# ----------------------------------------- ipi_arena_bench LLMClient stand-in ---
@dataclass
class _StubToolCall:
    name: str
    arguments: dict
    id: str = "tc_stub"


@dataclass
class _StubChatResponse:
    content: str = ""
    tool_calls: list = field(default_factory=list)


class StubLLMClient:
    """Mimics ipi_arena_bench.llm_client.LLMClient: ``.chat(messages, tools=...)``
    returns a response with ``.content`` and ``.tool_calls``. Scripted per call."""

    def __init__(self, scripted: Sequence[_StubChatResponse] | None = None):
        self.scripted = list(scripted or [])
        self.calls: list = []

    def chat(self, messages, tools=None, temperature: float = 0.0, max_tokens: int = 4096):
        self.calls.append({"messages": messages, "tools": tools})
        if self.scripted:
            return self.scripted.pop(0) if len(self.scripted) > 1 else self.scripted[0]
        return _StubChatResponse()


def tool_call(name: str, args: dict, id: str = "tc_stub") -> _StubToolCall:
    return _StubToolCall(name=name, arguments=args, id=id)


def chat_response(content: str = "", tool_calls=None) -> _StubChatResponse:
    return _StubChatResponse(content=content, tool_calls=list(tool_calls or []))


# ----------------------------------------------------------- server probes ---
def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def qwen_up(host: str = "localhost", port: int = 8000) -> bool:
    return _port_open(host, port)


def bge_up(host: str = "localhost", port: int = 8001) -> bool:
    return _port_open(host, port)


# ----------------------------------------------------- RolloutContext builder ---
def make_ctx(user_request: str | None = None, tools: list | None = None,
             command_tools: set | None = None, messages: list | None = None,
             target_client=None, worldsim_client=None, **overrides):
    """Build an ipi_eval RolloutContext for seam tests, defaulting to the IPI
    fixture. Stub clients are supplied (most seams never call them)."""
    from ipi_eval.defense_base import RolloutContext  # type: ignore
    from . import fixtures

    f = fixtures.ipi_transcript()
    ctx = RolloutContext(
        behavior=overrides.pop("behavior", {"id": f["behavior_id"]}),
        system_prompt=overrides.pop("system_prompt", f["system_prompt"]),
        user_request=user_request if user_request is not None else f["user_request"],
        tools=tools if tools is not None else f["openai_tools"],
        command_tools=command_tools if command_tools is not None
        # the fixture's consequential tool + the generic command tools the seam tests use
        else {"execute_sql", "run_sql_query", "write_file", "execute_bash"},
        target_client=target_client or StubLLMClient(),
        worldsim_client=worldsim_client or StubLLMClient(),
        worldsim_prompt=overrides.pop("worldsim_prompt", "world-sim"),
        messages=messages if messages is not None else fixtures.ipi_prefill_messages(),
        **overrides,
    )
    return ctx
