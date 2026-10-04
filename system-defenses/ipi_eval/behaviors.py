"""Behavior-definition loader for the Arena IPI defense eval.

Single source of truth for task context (structured tools, criteria, judge config)
across all three corpora. Records in attacks.jsonl only carry tools as markdown inside
the system prompt, so structured tools/criteria/judges are joined from here by behavior_id.

Behavior defs (same schema across corpora):
  arena-data/ipi_2025/behaviors_detailed.json      (corpus ipi_2025)
  arena-data/ipi_2026_q1/behaviors_detailed.json   (corpus ipi_2026_q1)
  arena-data/ipi_2026_q2/behaviors_detailed.json   (corpus ipi_2026_q2)

Per entry the fields we use:
  system_prompt_config.content                  -> system prompt (fallback when record lacks one)
  tools_config.custom_tool_schema (JSON string) -> [{name, description, input_schema}]
  criteria[]                                    -> {id, name, description, threshold, includeAllMessages}
  judging.submit.{judge_id, criteria_config, panel[]}
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ARENA_DATA = _REPO / "arena-data"
_CORPUS_DIRS = {
    "ipi_2025": "ipi_2025",
    "ipi_2026_q1": "ipi_2026_q1",
    "ipi_2026_q2": "ipi_2026_q2",
}

# Tool-name/description keywords that mark a STATE-MUTATING ("command") tool.
# Used by the IPIGuard adapter (query-expansion allows new read-only tools, never new
# command tools). Read-only is the default; a tool is "command" if any keyword matches.
_COMMAND_KEYWORDS = (
    "send", "delete", "remove", "create", "update", "modify", "edit", "write",
    "post", "put", "patch", "execute", "run", "deploy", "install", "cancel",
    "transfer", "pay", "purchase", "book", "approve", "publish", "upload",
    "activate", "deactivate", "set_", "add_", "move", "share", "invite",
    "schedule", "submit", "reset", "drop", "truncate", "merge", "push",
    # shell / code-execution tools (e.g. Bash, shell, terminal, exec, interpret) —
    # consequential because they run arbitrary commands (the harmful vector for coding agents).
    "bash", "shell", "terminal", "exec", "subprocess", "cmd", "interpret", "compile",
)


@lru_cache(maxsize=1)
def _load_all() -> dict:
    """Flat {behavior_id: behavior_doc} across all corpora (ids are unique across corpora).

    Each doc gets an injected `_corpus` key.
    """
    out: dict[str, dict] = {}
    for corpus, dirname in _CORPUS_DIRS.items():
        path = _ARENA_DATA / dirname / "behaviors_detailed.json"
        if not path.exists():   # arena store not present (public release): see get_behavior
            continue
        for b in json.loads(path.read_text()):
            b = dict(b)
            b["_corpus"] = corpus
            # Key by the stripped id: some defs carry stray surrounding whitespace
            # (e.g. ' hotel-frontdesk-premature-checkout'), and get_behavior() strips
            # its lookup key, so an un-stripped store key would never match.
            out[b["id"].strip()] = b
    return out


@lru_cache(maxsize=1)
def _load_public() -> dict:
    """Public ipi_arena_os behaviors (tool + coding), converted to the internal doc shape."""
    try:
        from ipi_arena_bench.behavior import load_behaviors
    except ImportError:
        return {}
    from .public_data import SUPPORTED, to_internal
    return {bid: to_internal(b) for bid, b in load_behaviors().items()
            if b.get("category") in SUPPORTED}


def get_behavior(behavior_id: str) -> dict | None:
    """Look up a behavior doc by id (tolerates stray surrounding whitespace).

    Arena behaviors come first; an id not in the arena store resolves from the public
    ipi_arena_os data."""
    bid = (behavior_id or "").strip()
    return _load_all().get(bid) or _load_public().get(bid)


def system_prompt(behavior: dict) -> str:
    spc = behavior.get("system_prompt_config")
    if isinstance(spc, str):
        return spc
    if isinstance(spc, dict):
        return spc.get("content") or ""
    return ""


def openai_tools(behavior: dict) -> list[dict]:
    """Parse tools_config.custom_tool_schema -> OpenAI function-calling tool list."""
    tc = behavior.get("tools_config") or {}
    raw = tc.get("custom_tool_schema")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if not isinstance(raw, list):
        raw = []
    tools = []
    for t in raw:
        params = t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": params,
            },
        })
    return tools


def command_tools_from_tools(tools: list[dict]) -> set[str]:
    """Heuristic set of state-mutating tool names over a resolved OpenAI tool list."""
    cmd = set()
    for tool in tools:
        name = tool["function"]["name"] or ""
        if any(kw in name.lower() for kw in _COMMAND_KEYWORDS):
            cmd.add(name)
    return cmd


def command_tools(behavior: dict) -> set[str]:
    """Heuristic set of state-mutating tool names (read-only is the complement)."""
    return command_tools_from_tools(openai_tools(behavior))


def is_read_only(behavior: dict, tool_name: str) -> bool:
    return tool_name not in command_tools(behavior)
