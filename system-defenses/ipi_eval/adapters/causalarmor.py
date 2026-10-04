"""CausalArmor defense adapter — Kim et al., arXiv:2602.07918.

Maps the CausalArmor pipeline element onto the review_tool_call seam (seam b),
which — like the AgentDojo element — fires BEFORE a proposed tool call executes.

Method (mirrors CausalArmor/impl/agentdojo_adapter.py CausalArmorElement):
  * Gate (Alg. 2 l.2): only act on a PRIVILEGED proposed call when there is at
    least one prior tool-output span in the context. Otherwise pass through.
  * Step 1: LOO log-prob attribution over the context (defense.analyze).
  * If any span is flagged (Eq. 5): sanitize the flagged tool span(s) (Appendix
    D.1), retroactively mask assistant CoT after the first injection
    (Alg. 2 l.19-25), then RE-GENERATE the action on the sanitized/masked context.

In this harness review_tool_call fires per proposed tool call with the
pre-assistant history already in ctx.messages (the engine appends the proposed
assistant message only after the review loop), so ctx.messages == the element's
`history`. We delegate all logic to CausalArmor/impl/defense.py.

Model wiring (paper-faithful, per the methodology check):
  * Agent re-generation = ctx.target_client (the model under attack; whatever the harness pins).
  * Proxy (LOO logprob scorer) = paper M_proxy Gemma-3-12B-IT, served LOCALLY on a logprob-capable
    vLLM endpoint (Gemini cannot return prompt logprobs). Set CAUSALARMOR_PROXY_URL (and a matching
    tokenizer_path) to the served Gemma endpoint; a missing tokenizer now raises LOUDLY rather than
    silently falling back to a Qwen tokenizer (which would corrupt the Gemma chat template).
  * Sanitizer = paper M_san Gemini-2.5-flash -> Gemini-3-Flash via OpenRouter (key from secrets.md).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from ..defense_base import DefenseAdapter, RolloutContext, ReviewResult

_IMPL_DIR = Path(__file__).resolve().parents[2] / "CausalArmor" / "impl"
_IMPL_PATH = _IMPL_DIR / "defense.py"

# Add CausalArmor/impl to sys.path first so any sibling imports resolve.
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

_spec = importlib.util.spec_from_file_location("_causalarmor_impl", _IMPL_PATH)
_impl = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses can resolve cls.__module__ during class creation.
sys.modules["_causalarmor_impl"] = _impl
_spec.loader.exec_module(_impl)

def _openrouter_key() -> str:
    """OpenRouter API key for the Gemini-3-Flash sanitizer (OPENROUTER_API_KEY or secrets.md)."""
    from ipi_arena_compat import read_secret
    return read_secret("openrouter")


class CausalArmorDefense(DefenseAdapter):
    name = "causalarmor"

    def __init__(self, proxy_model: str | None = None,
                 sanitizer_model: str = "google/gemini-3-flash-preview",
                 base_url: str | None = None,
                 sanitizer_base_url: str = "https://openrouter.ai/api/v1",
                 sanitizer_api_key: str | None = None, cot_masking: bool = True,
                 tokenizer_path: str | None = None):
        # base_url None -> impl resolves CAUSALARMOR_PROXY_URL / local :8002 (proxy endpoint).
        # proxy_model / tokenizer_path default to the paper Gemma-3-12B but can be overridden via
        # CAUSALARMOR_PROXY_MODEL / CAUSALARMOR_TOKENIZER_PATH (e.g. point at a stable local Qwen
        # logprob endpoint when Gemma OOMs under batched LOO load — must match: same model+tokenizer).
        import os as _os
        self.proxy_model = proxy_model or _os.environ.get("CAUSALARMOR_PROXY_MODEL", "gemma-3-12b-it")
        self.tokenizer_path = tokenizer_path or _os.environ.get("CAUSALARMOR_TOKENIZER_PATH",
                                                                "google/gemma-3-12b-it")
        self.sanitizer_model = sanitizer_model
        self.base_url = base_url
        self.sanitizer_base_url = sanitizer_base_url
        self.sanitizer_api_key = sanitizer_api_key
        self.cot_masking = cot_masking
        self.impl = None

    def on_rollout_start(self, ctx: RolloutContext) -> None:
        if self.impl is not None:
            return
        # Proxy/sanitizer are now different endpoints+models (paper-faithful). The sanitizer key
        # comes from secrets.md unless the caller passed one. We do NOT swap a missing tokenizer
        # for a Qwen one: a Gemma proxy needs Gemma's chat template, so a bad tokenizer_path must
        # fail loudly (AutoTokenizer.from_pretrained raises) rather than corrupt the LOO context.
        key = self.sanitizer_api_key if self.sanitizer_api_key is not None else _openrouter_key()
        kwargs = dict(proxy_model=self.proxy_model, sanitizer_model=self.sanitizer_model,
                      sanitizer_base_url=self.sanitizer_base_url, sanitizer_api_key=key,
                      cot_masking=self.cot_masking, tokenizer_path=self.tokenizer_path)
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self.impl = _impl.Defense(**kwargs)

    def review_tool_call(self, tool_name: str, args: dict, ctx: RolloutContext) -> ReviewResult:
        if self.impl is None:
            self.on_rollout_start(ctx)

        # -- Gate (Alg. 2 l.2): only defend privileged/state-changing actions. ----------
        # F2: use the harness-resolved privileged set (command_tools / P-T tool_policies),
        # which is the paper's "user-configured actions to defend"; fall back to the impl
        # keyword grep only when the harness provides no signal.
        if not self._is_privileged(tool_name, ctx):
            return (True, None, None)

        # Need at least one prior untrusted (tool) span to attribute to.
        span_indices = [i for i, m in enumerate(ctx.messages) if m.get("role") == "tool"]
        if not span_indices:
            return (True, None, None)

        action = self.impl.serialize_action(tool_name, args)

        # F1+F3: build a faithful LOO context. F1: analyze()->_render_prefix keeps only
        # {role, content}, so serialize assistant tool_calls into text first. F3: ~94% of
        # our prefills carry the user request U only in ctx.user_request (no user-role
        # turn), which makes Delta_U vacuous (-inf) and degenerates the flag rule to
        # "flag everything"; inject U as a user turn (mirrors impl/ipi_adapter). `nmap`
        # maps each flat index back to its ctx.messages index (None = the synthetic U).
        flat, nmap = self._loo_context(ctx)

        # -- Step 1: LOO attribution. Fail OPEN on any scoring backend failure. ----------
        try:
            attr = self.impl.analyze(action, flat)
        except Exception:
            # NOTE: LOO scoring uses vLLM /v1/completions echo+logprobs; if that call
            # fails (server down / endpoint unsupported) we allow the action rather
            # than crash the eval — fail-open, matching "zero overhead when no signal".
            return (True, None, None)

        # Map flagged flat-indices back onto ctx.messages (drop the synthetic U turn).
        flagged_native = [nmap[j] for j in attr.flagged if nmap[j] is not None]
        if not flagged_native:
            return (True, None, None)

        # -- Stage 1: sanitize the flagged tool span(s) in a context copy. --------------
        user_request = next((m.get("content") or "" for m in ctx.messages
                             if m.get("role") == "user"), ctx.user_request)
        new_messages = [dict(m) for m in ctx.messages]
        for i in flagged_native:
            tmsg = new_messages[i]
            # Recover the tool name for this span from its assistant tool_call, else "tool".
            span_tool = self._span_tool_name(new_messages, i)
            cleaned = self.impl.sanitize(tmsg.get("content") or "", user_request, span_tool)
            tmsg["content"] = cleaned

        # -- Stage 2: retroactive CoT masking after the first injection (Alg. 2 l.19-25). -
        if self.impl.cot_masking:
            k_min = min(flagged_native)
            for j in range(k_min + 1, len(new_messages)):
                if new_messages[j].get("role") == "assistant":
                    masked = dict(new_messages[j])
                    masked["content"] = _impl.COT_MASK_PLACEHOLDER
                    new_messages[j] = masked

        # -- Re-generate the action on the sanitized/masked context. ---------------------
        try:
            resp = ctx.target_client.chat(new_messages, tools=ctx.tools or None)
        except Exception:
            # NOTE: if re-generation fails, fall back to allowing the original action
            # (the sanitize step already removed the injected text from the span copy,
            # but we cannot mutate the real observation here — fail-open on regen error).
            return (True, None, None)

        # Find a regenerated call to the SAME privileged tool. `resp.tool_calls` can be None (a
        # reasoning-truncated / refused regeneration) and `regen.arguments` None — guard both.
        regen = next((tc for tc in (resp.tool_calls or []) if tc.name == tool_name), None)
        if regen is not None:
            return (True, regen.arguments or {}, None)

        # Regeneration dropped/refused the privileged call -> BLOCK it.
        return (False, None, "[CausalArmor: action removed after sanitizing injected span]")

    def _is_privileged(self, tool_name: str, ctx: RolloutContext) -> bool:
        """Privileged = in the harness-resolved command-tool / P-T set (the paper's
        configured 'actions to defend'). Fall back to the impl keyword grep only when the
        harness gives no signal at all."""
        cmd = ctx.command_tools or set()
        pol = (ctx.tool_policies or {}).get(tool_name)
        if cmd or ctx.tool_policies:
            return tool_name in cmd or pol in ("P-T", "P-F or P-T")
        return self.impl.is_privileged(tool_name)

    @staticmethod
    def _parse_tool_call(tc: dict) -> tuple[str, dict]:
        """(name, args) from an OpenAI-style recorded/proposed tool call."""
        fn = tc.get("function") or {}
        name, raw = (fn.get("name") or "tool", fn.get("arguments")) if isinstance(fn, dict) \
            else (fn, tc.get("args"))
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
        else:
            args = raw or {}
        return name, (args if isinstance(args, dict) else {"_value": args})

    def _flat_msg(self, m: dict) -> dict:
        """One {role, content} message, serializing assistant tool_calls into the content
        text so the proxy sees the agent's prior actions (mirrors impl/ipi_adapter)."""
        role = m.get("role", "user")
        c = m.get("content")
        txt = c if isinstance(c, str) else ("" if c is None else str(c))
        if role == "assistant" and m.get("tool_calls"):
            calls = " ".join(self.impl.serialize_action(*self._parse_tool_call(tc))
                             for tc in m["tool_calls"])
            txt = (txt + "\n" + calls).strip() if txt else calls
        return {"role": role, "content": txt}

    def _loo_context(self, ctx: RolloutContext) -> tuple[list[dict], list[int | None]]:
        """Build the LOO context C plus a flat->native index map. Serializes tool_calls
        (F1) and ensures the user request U is present as a user turn (F3) so Delta_U is
        well-defined; U is inserted before the first non-system message. nmap[j] is the
        ctx.messages index of flat[j], or None for the synthetic U turn."""
        msgs = ctx.messages
        ureq = (ctx.user_request or "").strip()
        inject = ureq and not any(m.get("role") == "user" for m in msgs)
        flat: list[dict] = []
        nmap: list[int | None] = []
        injected = False
        for i, m in enumerate(msgs):
            if inject and not injected and m.get("role") != "system":
                flat.append({"role": "user", "content": ureq})
                nmap.append(None)
                injected = True
            flat.append(self._flat_msg(m))
            nmap.append(i)
        if inject and not injected:           # context was all system messages
            flat.append({"role": "user", "content": ureq})
            nmap.append(None)
        return flat, nmap

    @staticmethod
    def _span_tool_name(messages: list[dict], span_index: int) -> str:
        """Best-effort tool name for a tool-output span: match its tool_call_id to the
        preceding assistant tool_calls. Falls back to 'tool'."""
        call_id = messages[span_index].get("tool_call_id")
        if call_id:
            for m in messages[:span_index]:
                if m.get("role") != "assistant":
                    continue
                for c in m.get("tool_calls") or []:
                    if c.get("id") == call_id:
                        return (c.get("function") or {}).get("name", "tool")
        return "tool"

    def drives_loop(self) -> bool:
        return False
