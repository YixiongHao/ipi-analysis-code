"""Firewalls defense adapter — Bhagwatkar et al., arXiv:2510.05244.

Maps the two LLM firewalls onto the interception seams:
  - transform_tool_output (seam a) -> Sanitizer (Tool-Output Firewall, F_O)
  - review_tool_call      (seam b) -> Minimizer (Tool-Input Firewall, F_I)

Delegates to system-defenses/Firewalls/impl/defense.py (loaded by file path, like
judges.py loads tool_judge.py). The impl auxiliary LLM defaults to the local
Qwen3-32B vLLM endpoint; the openai client is created lazily on first call.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from ..defense_base import DefenseAdapter, RolloutContext, ReviewResult

_IMPL_PATH = (Path(__file__).resolve().parents[2]
              / "Firewalls" / "impl" / "defense.py")
_spec = importlib.util.spec_from_file_location("_firewalls_impl", _IMPL_PATH)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)


class FirewallsDefense(DefenseAdapter):
    name = "firewalls"

    def __init__(self, model: str = "Qwen3-32B",
                 base_url: str = "http://localhost:8000/v1", api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        if api_key is None:
            # OpenRouter aux (e.g. the Gemini Minimizer/Sanitizer) needs the real key; a local
            # vLLM endpoint is happy with a dummy. Auto-load so the config only needs model+base_url.
            from ..engine import read_secret
            api_key = read_secret("openrouter") if "openrouter" in base_url else "EMPTY"
        self.impl = _impl.Defense(model=model, base_url=base_url, api_key=api_key)

    def preprocess_prefill(self, ctx: RolloutContext) -> None:
        # Run the Sanitizer over the injected prefill tool message so the firewall acts
        # on the injection (which the agent may consume on the very first generated step,
        # before any new tool output exists). Targeted: only ctx.prefill_end_index.
        i = ctx.prefill_end_index
        if i is None or not (0 <= i < len(ctx.messages)):
            return
        msg = ctx.messages[i]
        if msg.get("role") != "tool":
            return
        try:
            cleaned = self.transform_tool_output(msg.get("content") or "", ctx)
        except Exception:
            return  # fail-open: leave original content
        new_msg = dict(msg)
        new_msg["content"] = cleaned
        ctx.messages[i] = new_msg

    def transform_tool_output(self, content: str, ctx: RolloutContext) -> str:
        return self.impl.transform_tool_output(content, {"user_request": ctx.user_request})

    def review_tool_call(self, tool_name: str, args: dict, ctx: RolloutContext) -> ReviewResult:
        minimized = self.impl.minimize_tool_args(args, {
            "user_request": ctx.user_request,
            "tool_name": tool_name,
            "tool_description": ctx.tool_description(tool_name),
        })
        return (True, minimized, None)
