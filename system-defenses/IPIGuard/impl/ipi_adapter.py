"""Load our recorded IPI rollouts and reduce each to the inputs IPIGuard's planner
needs: the trusted user request, the available tool catalog, and the tool calls the
(compromised) agent actually made.

Reuses `classifier-defenses/loader.py` for all corpus plumbing. Because IPIGuard is a
planning/control-flow defense that changes the *action sequence*, it cannot be scored
post-hoc the way a detector is; instead `run_ipi.py` uses these scenarios to run the
real planner and ask whether IPIGuard's plan-admissibility rule would have blocked the
injected (command) action. The methodology is necessarily partial.
"""
import json
import os
import sys

_CLF = "classifier-defenses"
if _CLF not in sys.path:
    sys.path.insert(0, _CLF)
import loader  # noqa: E402

# Read-only "Query Tool" name heuristic (the IPI corpora use real-world tool names like
# `send_email`, not AgentDojo's whitelist). Documented limitation: this is a
# name-based proxy for IPIGuard's query/command split, not the paper's exact whitelist.
_QUERY_PREFIXES = (
    "get_", "read_", "list_", "search_", "find_", "fetch_", "view_", "check_",
    "lookup_", "browse_", "query_", "describe_", "show_", "load_", "scan_",
)
_QUERY_SUBSTR = ("search", "list_", "_get_", "read", "retrieve", "lookup")


def is_command_tool(name: str) -> bool:
    """True if the tool likely mutates state (a 'Command Tool'); False if read-only."""
    n = (name or "").lower()
    if n.startswith(_QUERY_PREFIXES):
        return False
    if any(s in n for s in _QUERY_SUBSTR):
        return False
    return True


def _called_tools(agent_messages):
    """Ordered list of (fn_name, raw_args_str) the agent invoked across the transcript."""
    calls = []
    for m in agent_messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
                name = fn.get("name") if fn else tc.get("name")
                args = fn.get("arguments") if fn else tc.get("arguments")
                if name:
                    calls.append((name, args))
    return calls


def extract_scenario(rec: dict) -> dict:
    r = rec["record"]
    calls = _called_tools(r["agent_messages"])
    tool_names = sorted({c[0] for c in calls})
    tools = [{"name": n, "description": "", "parameters": {}} for n in tool_names]
    return {
        "attack_id": rec["attack_id"],
        "behavior_id": rec["behavior_id"],
        "corpus": rec["corpus"],
        "user_request": r["user_request"] or "",
        "system_prompt": r.get("system_prompt") or "",
        "tools": tools,
        "called": [{"name": n, "is_command": is_command_tool(n)} for (n, _) in calls],
        "attack_text": (rec.get("attack_text") or "")[:500],
    }


def load_slice(corpus="ipi_2026_q1", target=12, per_behavior_cap=2, max_scan=4000,
               keep_areas=None):
    streamer = loader.CORPORA[corpus]
    picked, _counts = loader.stratified(
        streamer, target=target, per_behavior_cap=per_behavior_cap,
        max_scan=max_scan, keep_areas=keep_areas,
    )
    return [extract_scenario(r) for r in picked if (r["record"]["user_request"] and
            _called_tools(r["record"]["agent_messages"]))]
