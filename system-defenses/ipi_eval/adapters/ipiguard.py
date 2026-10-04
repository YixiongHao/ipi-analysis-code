"""IPIGuard defense adapter (full-loop) — Tool Dependency Graph planning + guarded traversal.

IPIGuard is a planning / control-flow defense: it builds a Tool Dependency Graph (TDG)
from the *trusted* user request + tool list up front, then constrains execution to that
plan. At run time, NEW read-only ("query") tools introduced by tool outputs may be added
(Node Expansion), but NEW *command* (state-mutating) tools that weren't planned are
BLOCKED — this is what neutralizes injected instructions sitting in tool observations.

This adapter re-hosts the fork's DagToolsExecutionLoop / DagToolsExecutor traversal
against our worldsim engine, delegating the two decision-surface methods to the existing
reference impl (system-defenses/IPIGuard/impl/defense.py):
  - Defense.plan(user_request, tools)            -> build the TDG
  - Defense.allow_call(tool_name, args, plan)    -> admissibility during traversal

The impl is loaded by file path (like firewalls.py). IPIGuard/impl must be on sys.path
first so its sibling `_fork_env` import resolves.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .. import engine
from ..defense_base import DefenseAdapter, RolloutContext

_IMPL_DIR = Path(__file__).resolve().parents[2] / "IPIGuard" / "impl"
# IPIGuard/impl/defense.py does `import _fork_env` (sibling) to point agentdojo at the
# fork; that only resolves if the impl dir is importable.
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

_IMPL_PATH = _IMPL_DIR / "defense.py"
_spec = importlib.util.spec_from_file_location("_ipiguard_impl", _IMPL_PATH)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)


def _flat_tools(ctx: RolloutContext) -> list[dict]:
    """ctx.tools is OpenAI shape [{"type":"function","function":{...}}]; the impl's
    plan() wants flat [{"name","description","parameters"}] (see defense.py docstring)."""
    out = []
    for t in ctx.tools:
        fn = t.get("function", t)
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


class IPIGuardDefense(DefenseAdapter):
    name = "ipiguard"

    def __init__(self, model: str = "Qwen3-32B",
                 base_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.base_url = base_url
        self.impl = _impl.Defense(model=model, base_url=base_url)

    def drives_loop(self) -> bool:
        return True

    def run_loop(self, ctx: RolloutContext) -> dict:
        # 1) Build the TDG once from the TRUSTED user request + tools (converted to the
        #    impl's expected flat shape).
        plan = self.impl.plan(ctx.user_request, _flat_tools(ctx))

        generated: list[dict] = []

        # 2) Guarded agentic traversal, seeded with the prefill in ctx.messages. This
        #    mirrors DagToolsExecutionLoop: model proposes a turn, each proposed tool
        #    call is checked for admissibility against the plan before execution.
        for _step in range(ctx.max_steps):
            resp = ctx.target_client.chat(ctx.messages, tools=ctx.tools or None)
            amsg = engine._assistant_msg(resp)

            if not resp.tool_calls:
                ctx.messages.append(amsg)
                generated.append(amsg)
                break

            tool_msgs = []
            blocked_ids: set[str] = set()
            blocked_names: list[str] = []
            for tc in resp.tool_calls:
                allowed = self.impl.allow_call(tc.name, tc.arguments, plan)
                # NOTE: the impl's read-only fallback consults the AgentDojo tool
                # whitelist (Defense._query_tools()), which does NOT cover our arena
                # tools. Override the read-only determination with our own
                # classification: a tool is COMMAND iff it's in ctx.command_tools,
                # read-only otherwise. Planned tools are always allowed; non-planned
                # tools are allowed only if read-only; non-planned command tools are
                # blocked (the path an injected instruction needs to trigger an action).
                planned = tc.name in plan.get("planned_tools", set())
                if not planned:
                    allowed = tc.name not in ctx.command_tools

                if allowed:
                    # Allowed: execute against the worldsim and keep the call.
                    output = engine._simulate(ctx, tc.name, tc.arguments, tc.id)
                    tool_msgs.append({"role": "tool", "content": output,
                                      "tool_call_id": tc.id})
                else:
                    # Blocked: do NOT put its effect into the world (no _simulate) and
                    # do NOT emit a tool result — the call is being stripped from the
                    # assistant turn below, so a tool message keyed to it would be an
                    # orphan (no parent tool_call) and would break the next chat() call
                    # and confuse the judge.
                    blocked_ids.add(tc.id)
                    blocked_names.append(tc.name)

            # NOTE (judging): the tool-usage judge extracts tool_calls from the running
            # conversation (run_eval passes full_messages=messages to score_behavior).
            # A blocked command call was never executed, so strip it from the recorded
            # assistant message — otherwise the judge would count an action that never
            # happened, defeating the defense. `amsg` is the same dict in both
            # ctx.messages and `generated`, so one strip covers both. Allowed calls stay.
            if blocked_ids and "tool_calls" in amsg:
                amsg["tool_calls"] = [c for c in amsg["tool_calls"]
                                      if c["id"] not in blocked_ids]
                if not amsg["tool_calls"]:
                    del amsg["tool_calls"]
                # Surface the refusal to the model so it can continue without the action.
                # Use assistant content (not a tool message) to avoid an orphan tool msg.
                note = ("[IPIGuard: refused state-mutating tool(s) not in the plan: "
                        + ", ".join(blocked_names) + "]")
                amsg["content"] = (amsg.get("content") or "") + ("\n" if amsg.get("content") else "") + note

            ctx.messages.append(amsg)
            generated.append(amsg)
            ctx.messages.extend(tool_msgs)

        return {"messages": ctx.messages, "generated_messages": generated,
                "aborted": False}
