"""FIDES — taint + per-tool P-T policy monitor (SEAM b: review_tool_call).

Delegates to system-defenses/FIDES/impl/defense.py (taint tracking + Table-3 policy
enforcement). Mirrors FidesMonitor.query from FIDES/impl/agentdojo_adapter.py: compute the
integrity label of the context that produced the proposed tool call, then call
defense.allow_call(tool, args, {"context_integrity": ...}). FIDES makes NO LLM calls.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..defense_base import DefenseAdapter, RolloutContext, ReviewResult

_IMPL_DIR = Path(__file__).resolve().parents[2] / "FIDES" / "impl"


def _load_impl():
    """Import FIDES impl/defense.py by file path (add impl dir to sys.path for sibling imports)."""
    if str(_IMPL_DIR) not in sys.path:
        sys.path.insert(0, str(_IMPL_DIR))
    spec = importlib.util.spec_from_file_location("fides_impl_defense", _IMPL_DIR / "defense.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FidesDefense(DefenseAdapter):
    name = "fides"

    def __init__(self) -> None:
        self.impl_mod = None
        self.impl = None

    def on_rollout_start(self, ctx: RolloutContext) -> None:
        if self.impl is None:
            self.impl_mod = _load_impl()
            # Use the per-behavior arena policy map (consequential/egress tools) instead of the
            # AgentDojo Table-3 default, which doesn't cover arena tool names. An empty map keeps
            # FIDES a safe no-op (no consequential tools were classified for this behavior).
            self.impl = self.impl_mod.Defense(tool_policies=ctx.tool_policies)

    def review_tool_call(self, tool_name: str, args: dict, ctx: RolloutContext) -> ReviewResult:
        if self.impl is None:
            self.on_rollout_start(ctx)

        # Mirror FidesMonitor.query: integrity of the context that produced this tool call.
        # The proposed assistant tool-call message is not yet in ctx.messages, so ctx.messages
        # is exactly the FidesMonitor `messages[:-1]` (system/user trusted, tool outputs untrusted).
        ctx_integrity = self.impl.context_integrity(ctx.messages)
        allow_ctx = {"context_integrity": ctx_integrity}

        allowed, reason = self.impl.allow_call(tool_name, args or {}, allow_ctx)
        if allowed:
            return (True, None, None)
        return (False, None, f"[FIDES blocked: {reason}]")

    def drives_loop(self) -> bool:
        return False
