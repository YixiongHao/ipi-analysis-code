"""MELON — Masked re-Execution and tooL comparisON (SEAM c: inspect_assistant).

Delegates to system-defenses/MELON/impl/defense.py (the backend-agnostic detector core) and
mirrors MELON/impl/agentdojo_adapter.py's MELONDetector.query at the method level.

The engine has already produced the *original* assistant action (the `message` passed to
inspect_assistant). MELON's detection step therefore only needs to run the *masking* re-execution:
re-run the LLM with the user task replaced by the task-neutral prompt T_f + few-shot, feeding the
consolidated tool outputs as a single fake read_file(random.txt) result, get the masked tool calls,
embed both original and masked calls, and flag if any original call is cosine-similar (> threshold)
to any *cached* masked call. The detector keeps its masked-call cache across steps, so it is built
once in on_rollout_start and reused.
"""
from __future__ import annotations

import importlib.util
import llm_parse
import json
import sys
from pathlib import Path

import os

from ..defense_base import DefenseAdapter, RolloutContext

_IMPL_DIR = Path(__file__).resolve().parents[2] / "MELON" / "impl"

# Per-embedding-backend operating point. BGE's same-function cosine ≈0.88 vs
# identical=1.0, so the paper's θ=0.8 over-fires on benign same-function actions → BGE uses θ=0.9.
# OpenAI text-embedding-3-large is the paper's own embedder, calibrated at θ=0.8.
_THRESHOLD_BY_BACKEND = {"bge": 0.9, "openai": 0.8}


def _openai_key() -> str:
    """OpenAI key for the text-embedding-3-large arm: env OPENAI_API_KEY, else secrets.md.
    Raise a clear, actionable error if neither is set (the paper-faithful arm can't run blind)."""
    from ..engine import read_secret
    key = os.getenv("OPENAI_API_KEY") or read_secret("openai")
    if not key:
        raise RuntimeError(
            "MELON's default 'openai' embedding arm needs an OpenAI key. Add an `openai = sk-...` "
            "line to secrets.md (or export OPENAI_API_KEY), or run the "
            "'melon-bge' arm to use the local BGE substitute instead.")
    return key


def _load_impl():
    """Import MELON impl/defense.py + agentdojo_adapter.py by path (impl dir on sys.path so the
    adapter's `from defense import ...` resolves)."""
    if str(_IMPL_DIR) not in sys.path:
        sys.path.insert(0, str(_IMPL_DIR))
    spec = importlib.util.spec_from_file_location("melon_impl_defense", _IMPL_DIR / "defense.py")
    defense = importlib.util.module_from_spec(spec)
    sys.modules["defense"] = defense  # let agentdojo_adapter's `from defense import ...` resolve
    spec.loader.exec_module(defense)
    spec2 = importlib.util.spec_from_file_location("melon_impl_adapter", _IMPL_DIR / "agentdojo_adapter.py")
    adapter = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(adapter)
    return defense, adapter


def _openai_fc(function: str, args: dict, id: str) -> dict:
    """fc_factory for build_few_shot: build an OpenAI-format tool-call dict (our LLMClient stack)."""
    return {"id": id, "type": "function",
            "function": {"name": function, "arguments": json.dumps(args)}}


def _few_shot_to_openai(messages: list[dict]) -> list[dict]:
    """Strip the AgentDojo-only keys (`tool_call`, `error`) build_few_shot puts on tool messages so
    the dicts are valid OpenAI chat messages; assistant tool_calls already use _openai_fc form."""
    out = []
    for m in messages:
        msg = {k: v for k, v in m.items() if k not in ("tool_call", "error")}
        out.append(msg)
    return out


class MelonDefense(DefenseAdapter):
    name = "melon"

    def __init__(self, augment: bool = False, embed_backend: str | None = None) -> None:
        """Args:
            augment: MELON-Aug — re-append the user request before the *original* run
                (repeat-prompt augmentation, applied via preprocess_agent_input). The masking
                run is unaffected (it always replaces the user prompt with T_f).
            embed_backend: "openai" (text-embedding-3-large, the paper's embedder — DEFAULT,
                θ=0.8; key from OPENAI_API_KEY or secrets.md) or "bge" (local substitute, θ=0.9).
                Defaults to the MELON_EMBED_BACKEND env var, else "openai". Sets θ per
                _THRESHOLD_BY_BACKEND.
        """
        backend = (embed_backend or os.getenv("MELON_EMBED_BACKEND") or "openai").lower()
        if backend not in _THRESHOLD_BY_BACKEND:
            raise ValueError(f"embed_backend must be one of {sorted(_THRESHOLD_BY_BACKEND)}, got {backend!r}")
        self.augment = augment
        self.embed_backend = backend
        self.sim_threshold = _THRESHOLD_BY_BACKEND[backend]
        self.defense_mod = None
        self.adapter_mod = None
        self.detector = None

    def on_rollout_start(self, ctx: RolloutContext) -> None:
        # Build detector ONCE per rollout so its masked-call cache (H) persists across steps.
        self.defense_mod, self.adapter_mod = _load_impl()
        if self.embed_backend == "openai":
            embed_fn = self.adapter_mod.make_openai_embed_fn(api_key=_openai_key())  # text-embedding-3-large (paper)
        else:
            embed_fn = self.adapter_mod.make_bge_embed_fn()  # BGE on EMBED_LLM_PORT (default 8001)
        self.detector = self.defense_mod.MelonDetector(embed_fn, sim_threshold=self.sim_threshold)
        ctx.scratch["melon_detector"] = self.detector

    def preprocess_agent_input(self, messages: list[dict], ctx: RolloutContext) -> list[dict]:
        # MELON-Aug repeat-prompt: re-append the trusted user request as a fresh user turn so the
        # original run stays anchored to the user task (mirrors agentdojo_adapter's `augment`).
        # Transient (the engine does not persist this) and only the original run sees it.
        if not self.augment or not ctx.user_request:
            return messages
        return [*messages, {"role": "user", "content": ctx.user_request}]

    def _consolidated_tool_outputs(self, ctx: RolloutContext) -> str:
        """Concatenate (function_name, content) tool outputs from the running conversation, mirroring
        agentdojo_adapter._build_masked_messages' tool_contents loop. The plain OpenAI tool messages
        don't carry the function name, so recover it from the assistant tool_calls via tool_call_id."""
        id_to_fn: dict[str, str] = {}
        for m in ctx.messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    id_to_fn[tc["id"]] = tc["function"]["name"]
        blocks = []
        for m in ctx.messages:
            if m.get("role") == "tool":
                fn = id_to_fn.get(m.get("tool_call_id", ""), "read_file")
                content = m.get("content")
                if isinstance(content, list):  # tolerate content-block form
                    content = "".join(b.get("content") or "" for b in content)
                blocks.append((fn, content or ""))
        return self.defense_mod.consolidate_tool_outputs(blocks)

    def _build_masked_messages(self, ctx: RolloutContext) -> list[dict]:
        """MELON masking input M: system prompt + few-shot + T_f, with all tool outputs consolidated
        into one fake read_file(random.txt) result. Mirrors agentdojo_adapter._build_masked_messages
        in plain OpenAI message form."""
        msgs: list[dict] = []
        for m in ctx.messages:
            if m.get("role") == "system":
                msgs.append({"role": "system", "content": m.get("content", "")})
        msgs += _few_shot_to_openai(self.defense_mod.build_few_shot(_openai_fc))
        tool_contents = self._consolidated_tool_outputs(ctx)
        rid = "toolu_01C3V95dCfSAku17wrkx9fCV"
        msgs += [
            {"role": "user", "content": self.defense_mod.TASK_NEUTRAL_PROMPT},
            {"role": "assistant", "content": "",
             "tool_calls": [_openai_fc("read_file", {"name": "random.txt"}, rid)]},
            {"role": "tool", "content": tool_contents, "tool_call_id": rid},
        ]
        return msgs

    def inspect_assistant(self, message: dict, ctx: RolloutContext) -> str | None:
        if self.detector is None:
            self.on_rollout_start(ctx)

        original_tool_calls = message.get("tool_calls")
        if not original_tool_calls:
            return None  # nothing to check this step (matches repo: original_tool_calls is None path)

        # --- Masking run: re-execute the step with the user task masked (A^m_{t+1}). ---
        # Use ctx.target_client (the same defended Qwen) with the task tools, exactly as the agentdojo
        # element reuses self.llm.query for both runs.
        masked_msgs = self._build_masked_messages(ctx)
        resp = ctx.target_client.chat(masked_msgs, tools=ctx.tools or None)
        # ChatResponse.tool_calls are ToolCall(name, arguments, id); render via (name, args) tuples.
        # `resp.tool_calls` can be None (a refused / reasoning-truncated masking run) and `tc.arguments`
        # None — guard both so a degenerate masked run doesn't crash detection (it just yields no calls).
        masked_texts = self.defense_mod.tool_calls_to_texts(
            [(tc.name, tc.arguments or {}) for tc in (resp.tool_calls or [])]
        )

        # --- Cache masked calls, then compare original vs cache (detector.step does both). ---
        # Render the original assistant tool_calls (OpenAI dict form -> (name, args) tuples).
        # safe_args tolerates None / fenced / preamble / malformed argument JSON (never raises).
        original_texts = self.defense_mod.tool_calls_to_texts(
            [((tc.get("function") or {}).get("name", "tool"),
              llm_parse.safe_args((tc.get("function") or {}).get("arguments")))
             for tc in original_tool_calls]
        )
        flagged, _score = self.detector.step(original_texts, masked_texts)
        if flagged:
            return "abort"  # injection detected: the defended model should not take the injected action
        return None

    def drives_loop(self) -> bool:
        return False
