"""AgentDojo wiring for MELON.

Faithful port of the reference repo's ``MELON`` pipeline element
(``system-defenses/MELON/MELON/pi_detector.py``) onto our local stack:
  * agent LLM  = Qwen3-32B served by vLLM (OpenAI-compatible, port 8000), thinking OFF
  * embeddings = BAAI/bge-large-en-v1.5 served by vLLM (port 8001), replacing the repo's
    OpenAI ``text-embedding-3-large``.

The comparison math + masking prompt live in ``defense.py``; this module is the glue:
the ``MELONDetector`` ``BasePipelineElement`` (drop it in ``ToolsExecutionLoop`` *in place of*
the in-loop LLM, after ``ToolsExecutor()``), plus factories for the two LLM/embed clients.

Compose a pipeline (see ``run_agentdojo.py``)::

    llm = make_qwen_llm()
    melon = MELONDetector(llm, make_bge_embed_fn(), sim_threshold=0.8, raise_on_injection=True)
    loop = ToolsExecutionLoop([ToolsExecutor(), melon])
    pipeline = AgentPipeline([system_message, init_query, llm, loop])
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Sequence

import openai

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from defense import (
    TASK_NEUTRAL_PROMPT,
    MelonDetector,
    build_few_shot,
    tool_calls_to_texts,
)

QWEN_MODEL = "Qwen3-32B"
BGE_MODEL = "bge-large"
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # the paper/repo embedder (parity backend)


# --------------------------------------------------------------------------------------------
# LLM / embedding factories (local vLLM endpoints)
# --------------------------------------------------------------------------------------------
class QwenLLM(OpenAILLM):
    """OpenAILLM that disables Qwen3 thinking via ``chat_template_kwargs.enable_thinking``.

    AgentDojo's stock OpenAILLM has no hook for ``extra_body``; we override ``query`` to pass it
    while reusing the repo's message/tool (de)serialization helpers."""

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


def _local_client(port_env: str, default_port: int) -> openai.OpenAI:
    port = os.getenv(port_env, str(default_port))
    return openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")


def make_qwen_llm(enable_thinking: bool = False) -> QwenLLM:
    """Agent LLM = Qwen3-32B on LOCAL_LLM_PORT (default 8000), thinking OFF by default."""
    return QwenLLM(_local_client("LOCAL_LLM_PORT", 8000), QWEN_MODEL, enable_thinking=enable_thinking)


def make_bge_embed_fn(model: str = BGE_MODEL) -> Callable[[str], list[float]]:
    """Embed a tool-call string with local BGE on EMBED_LLM_PORT (default 8001)."""
    client = _local_client("EMBED_LLM_PORT", 8001)

    def embed(text: str) -> list[float]:
        # BGE max context is 512 tokens; tool calls are short but args can embed huge content.
        # Truncate defensively (~1800 chars) so an oversized call still embeds instead of erroring.
        return client.embeddings.create(input=text[:1800], model=model).data[0].embedding

    return embed


def make_openai_embed_fn(model: str = OPENAI_EMBED_MODEL,
                         api_key: str | None = None) -> Callable[[str], list[float]]:
    """Embed a tool-call string with the paper's OpenAI ``text-embedding-3-large`` (parity backend).

    This is the embedder the MELON paper/repo use, and the one that calibrates θ=0.8. The ipi_eval
    adapter passes ``api_key`` resolved from secrets.md; if None it falls back to ``OPENAI_API_KEY``
    (and optional ``OPENAI_BASE_URL``) in the environment. ``text-embedding-3-large`` accepts ~8191
    tokens, so we truncate far less aggressively than BGE."""
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def embed(text: str) -> list[float]:
        return client.embeddings.create(input=text[:8000], model=model).data[0].embedding

    return embed


# --------------------------------------------------------------------------------------------
# Message helpers (this AgentDojo clone uses content-block messages, not plain strings)
# --------------------------------------------------------------------------------------------
def _fc(function: str, args: dict, id: str) -> FunctionCall:
    return FunctionCall(function=function, args=args, id=id)


def _blockify(msg: dict) -> ChatMessage:
    """Convert a plain {role, content:str, tool_calls?, tool_call?, ...} dict into an AgentDojo
    ChatMessage with content-block content (so OpenAILLM can serialize it)."""
    out = dict(msg)
    content = msg.get("content")
    out["content"] = None if content is None else [text_content_block_from_string(content)]
    if msg["role"] == "assistant":
        out.setdefault("tool_calls", None)
    return out  # type: ignore[return-value]


def _build_masked_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """MELON masking function M: keep the system prompt, drop the user task + agent turns, and
    feed all tool outputs consolidated into one fake read_file(random.txt) result, primed by the
    task-neutral prompt T_f and the two few-shot samples. Mirrors the reference repo."""
    sep = "=" * 50
    masked: list[ChatMessage] = []
    tool_contents = ""
    for m in messages:
        if m["role"] == "system":
            masked.append(m)  # already content-block form
        elif m["role"] == "tool":
            fn = m["tool_call"].function
            body = m["content"]
            if isinstance(body, list):  # content blocks -> text
                body = "".join(b.get("content") or "" for b in body)
            tool_contents += f"{sep}\n\nfunction: {fn}\n\n{body}\n\n{sep}\n\n"

    masked += [_blockify(d) for d in build_few_shot(_fc)]
    masked += [
        _blockify({"role": "user", "content": TASK_NEUTRAL_PROMPT}),
        _blockify({"role": "assistant", "content": "", "tool_calls": [_fc("read_file", {"name": "random.txt"}, "toolu_01C3V95dCfSAku17wrkx9fCV")]}),
        {
            "role": "tool",
            "content": [text_content_block_from_string(tool_contents)],
            "tool_call_id": "toolu_01C3V95dCfSAku17wrkx9fCV",
            "tool_call": _fc("read_file", {"name": "random.txt"}, "toolu_01C3V95dCfSAku17wrkx9fCV"),
            "error": None,
        },
    ]
    return masked


# --------------------------------------------------------------------------------------------
# The MELON pipeline element
# --------------------------------------------------------------------------------------------
class MELONDetector(BasePipelineElement):
    """MELON detector. Place AFTER ``ToolsExecutor()`` inside ``ToolsExecutionLoop`` *instead of*
    the in-loop LLM: this element generates the next action itself (the original run), runs the
    masked re-execution, compares tool calls, and aborts/scrubs on a match.

    Args:
        llm: the agent LLM pipeline element (its ``.query`` is used for both runs).
        embed_fn: tool-call-string -> embedding vector (defaults to local BGE).
        sim_threshold: cosine-sim match threshold (paper θ = 0.8).
        raise_on_injection: raise ``AbortAgentError`` on detection (else scrub + stop quietly).
        augment: MELON-Aug — re-append the user query before the original run (the repeat-prompt
            augmentation), strengthening the original run's focus on the user task. The masking
            run is unaffected (it always replaces the user prompt with T_f).
    """

    def __init__(
        self,
        llm: BasePipelineElement,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        sim_threshold: float = 0.8,
        raise_on_injection: bool = False,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.embed_fn = embed_fn if embed_fn is not None else make_bge_embed_fn()
        self.sim_threshold = sim_threshold
        self.raise_on_injection = raise_on_injection
        self.augment = augment

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        # Per-task detector state (extra_args is fresh per task, like the repo's bank-in-extra_args).
        detector = extra_args.get("_melon_detector")
        if detector is None:
            detector = MelonDetector(self.embed_fn, sim_threshold=self.sim_threshold)
            extra_args["_melon_detector"] = detector

        # --- Original run: the action the agent would take next (A^o_{t+1}). ---
        # MELON-Aug: re-append the user query (repeat-prompt) to refocus the original run.
        orig_input = messages
        if self.augment:
            orig_input = [*messages, {"role": "user", "content": [text_content_block_from_string(query)]}]
        q, rt, ev, updated_messages, extra_args = self.llm.query(query, runtime, env, orig_input, extra_args)
        original_action = updated_messages[-1]
        original_tool_calls = original_action.get("tool_calls")

        # --- Masking run: re-execute the step with the user prompt masked (A^m_{t+1}). ---
        masked_messages = _build_masked_messages(messages)
        _, _, _, masked_out, _ = self.llm.query(query, runtime, env, masked_messages, copy.copy(extra_args))
        masked_tool_calls = masked_out[-1].get("tool_calls")

        # --- Cache masked calls, then compare original vs cache (design #2 + #3). ---
        detector.add_masked_calls(tool_calls_to_texts(masked_tool_calls))
        if original_tool_calls is None:
            extra_args.setdefault("is_injection", False)
            return q, rt, ev, updated_messages, extra_args

        is_injection, score = detector.check(tool_calls_to_texts(original_tool_calls))
        extra_args["is_injection"] = bool(extra_args.get("is_injection")) or is_injection

        if is_injection:
            if self.raise_on_injection:
                raise AbortAgentError(
                    f"aborting execution because a prompt injection was detected (score: {score})",
                    list(updated_messages),
                    ev,
                )
            # Scrub the last tool output and replace the action with a stop message so the
            # ToolsExecutionLoop terminates (assistant message without tool_calls). Locate the
            # tool message explicitly (its position shifts when augment re-appends a user msg).
            updated_messages = list(updated_messages)
            for j in range(len(updated_messages) - 1, -1, -1):
                if updated_messages[j]["role"] == "tool":
                    updated_messages[j]["content"] = [text_content_block_from_string(self.transform())]
                    break
            updated_messages[-1] = {
                "role": "assistant",
                "content": [text_content_block_from_string("The tool outputs contain a potential prompt injection; stopping execution.")],
                "tool_calls": None,
            }
        return q, rt, ev, updated_messages, extra_args

    @staticmethod
    def transform() -> str:
        return "<Data omitted because a prompt injection was detected>"
