"""First-principles tool resolver for the IPI re-rollout harness.

`behaviors.openai_tools` only parses the Format-A `tools_config.custom_tool_schema`, so
behaviors that declare tools via the Format-B `tools_config.tools` array (or via no schema
at all) resolve to an empty / parameter-less tool list. This module reconstructs a complete
OpenAI tool list per behavior by MERGING declared tools (Format A + Format B) with the
parameter shapes observed in the recorded trajectory's tool calls (the ground truth), and
classifies each tool for FIDES.

Entry point: `resolve_tools(behavior, trajectory_messages) -> (openai_tools, tool_policies)`.

Consumers:
  - openai_tools (name+description+parameters): defended agent, worldsim, IPIGuard planner, CaMeL.
  - tool_policies (name -> "P-T" | "P-F or P-T"): FIDES (`FIDES/impl/defense.py` Defense(tool_policies=)).
"""
from __future__ import annotations

import copy
import json
import re
from typing import Iterator

from . import behaviors as B

# FIDES policy classes — MUST match FIDES/impl/defense.py constants (PT / PF_OR_PT).
_PT = "P-T"               # consequential / state-changing action
_PF_OR_PT = "P-F or P-T"  # egress / exfiltration sink

# Egress = data leaves the agent's boundary. ACTION verbs only, so reads such as
# email_read / email_search / read_emails / listEmails stay neutral.
_EGRESS_KEYWORDS = (
    "send", "post", "share", "upload", "publish", "forward", "export",
    "reply", "tweet", "broadcast", "leak", "email_send", "direct_message",
)

# Leading-verb classification of a tool by its DESCRIPTION, used only when the NAME carries no
# verb (domain tools whose names are nouns: `course_registration_system`, `markEmailAsRead`,
# `food_beverage_pos`). Tool blurbs start with their action verb ("Register for…", "Mark an…",
# "Process … transactions"), so we match the FIRST word only — incidental later words
# ("…modified time", "…reported posts", "…process it") must NOT trigger a (mis)classification.
_DESC_EGRESS_VERBS = frozenset({
    "send", "sends", "post", "posts", "share", "shares", "upload", "uploads", "publish",
    "publishes", "forward", "forwards", "email", "emails", "reply", "broadcast", "transmit",
    "dispatch", "deliver", "leak", "tweet",
})
_DESC_CONSEQUENTIAL_VERBS = frozenset({
    "register", "registers", "enroll", "enrolls", "process", "processes", "charge", "charges",
    "mark", "marks", "reserve", "reserves", "deduct", "withdraw", "modify", "modifies", "cancel",
    "cancels", "deactivate", "activate", "schedule", "schedules", "submit", "submits", "create",
    "creates", "update", "updates", "delete", "deletes", "edit", "edits", "write", "writes",
    "execute", "executes", "run", "runs", "deploy", "install", "book", "books", "purchase",
    "pay", "approve", "approves", "place", "places", "issue", "add", "remove", "set", "move",
    "merge", "reset", "transfer", "order", "orders",
})


# --- JSON-schema type inference from observed argument values --------------------------

def _is_unknown(schema: dict) -> bool:
    return not schema.get("type") and "properties" not in schema and "items" not in schema


def _infer_type(value) -> dict:
    """A single JSON value -> minimal JSON-schema fragment ({} = unknown/unconstrained)."""
    if value is None:
        return {}
    if isinstance(value, bool):          # before int: isinstance(True, int) is True
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        items: dict = {}
        for el in value:
            items = _merge_type(items, _infer_type(el))
        return {"type": "array", "items": items}
    if isinstance(value, dict):
        return {"type": "object",
                "properties": {k: _infer_type(v) for k, v in value.items()}}
    return {}


def _merge_type(a: dict, b: dict) -> dict:
    """Least-upper-bound of two inferred fragments across multiple observations."""
    if _is_unknown(a):
        return b
    if _is_unknown(b):
        return a
    ta, tb = a.get("type"), b.get("type")
    if ta and tb and ta != tb:
        if {ta, tb} == {"integer", "number"}:
            return {"type": "number"}
        return {}  # conflicting scalar types -> any
    if ta == "object" or tb == "object":
        props = dict(a.get("properties") or {})
        for k, v in (b.get("properties") or {}).items():
            props[k] = _merge_type(props.get(k, {}), v)
        return {"type": "object", "properties": props}
    if ta == "array" or tb == "array":
        return {"type": "array",
                "items": _merge_type(a.get("items") or {}, b.get("items") or {})}
    return {"type": ta or tb}


def _decode_arguments(raw) -> dict:
    """tool_calls[].function.arguments (a JSON string) -> dict; {} on anything malformed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return v if isinstance(v, dict) else {}
    return {}


def _iter_tool_calls(messages: list[dict]) -> Iterator[tuple[str, dict]]:
    """Yield (tool_name, args_dict) for every assistant tool call across messages.

    Handles plain `m["tool_calls"]` (OpenAI) and content-block (`tool_use`/`function`) forms.
    """
    for m in messages or []:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name")
            raw = fn.get("arguments") if "arguments" in fn else tc.get("args")
            if name:
                yield name, _decode_arguments(raw)
        content = m.get("content")
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict) or blk.get("type") not in ("tool_use", "function"):
                    continue
                fn = blk.get("function") if isinstance(blk.get("function"), dict) else {}
                name = blk.get("name") or fn.get("name")
                raw = blk.get("input")
                if raw is None:
                    raw = blk.get("arguments") if "arguments" in blk else fn.get("arguments")
                if name:
                    yield name, _decode_arguments(raw)


def _params_from_trajectory(messages: list[dict]) -> dict[str, dict]:
    """tool_name -> object-schema inferred from observed calls (union of keys, merged types)."""
    calls: dict[str, list[dict]] = {}
    for name, args in _iter_tool_calls(messages):
        calls.setdefault(name, []).append(args)
    out: dict[str, dict] = {}
    for name, arg_dicts in calls.items():
        props: dict[str, dict] = {}
        for args in arg_dicts:
            for k, v in args.items():
                props[k] = _merge_type(props.get(k, {}), _infer_type(v))
        schema: dict = {"type": "object", "properties": props}
        # `required` only when we have >=2 observations agreeing — avoids marking a one-off
        # call's optional args as required.
        if len(arg_dicts) >= 2:
            required = sorted(set.intersection(*[set(a.keys()) for a in arg_dicts]))
            if required:
                schema["required"] = required
        out[name] = schema
    return out


# --- declared tool parsers (Format A + Format B) ---------------------------------------

def _declared_format_a(behavior: dict) -> dict[str, dict]:
    """tools_config.custom_tool_schema (JSON string) -> {name: {description, parameters}}."""
    tc = behavior.get("tools_config") or {}
    raw = tc.get("custom_tool_schema")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if not isinstance(raw, list):
        raw = []
    out: dict[str, dict] = {}
    for t in raw:
        name = t.get("name")
        if not name:
            continue
        out[name] = {"description": t.get("description", ""),
                     "parameters": t.get("input_schema") or t.get("parameters")}
    return out


def _declared_format_b(behavior: dict) -> dict[str, dict]:
    """tools_config.tools (array) -> {name: {description, parameters}} (parameters often {})."""
    tc = behavior.get("tools_config") or {}
    raw = tc.get("tools")
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict] = {}
    for t in raw:
        fn = t.get("function") or {}
        name = fn.get("name") or t.get("name")
        if not name:
            continue
        out[name] = {"description": t.get("description", "") or fn.get("description", ""),
                     "parameters": fn.get("parameters")}
    return out


def _has_props(params) -> bool:
    return isinstance(params, dict) and bool(params.get("properties"))


def _scrape_description(system_prompt: str, name: str) -> str:
    """Best-effort: first system-prompt line mentioning the tool name."""
    if not system_prompt or not name:
        return ""
    for line in system_prompt.splitlines():
        if name in line:
            return line.strip()[:300]
    return ""


# --- classification (for FIDES) --------------------------------------------------------

def classify_tool(name: str, description: str = "") -> str:
    """-> 'neutral' | 'consequential' | 'egress'. Egress checked first (verbs outrank create/update)."""
    nl = (name or "").lower()
    if any(kw in nl for kw in _EGRESS_KEYWORDS):
        return "egress"
    if any(kw in nl for kw in B._COMMAND_KEYWORDS):
        return "consequential"
    # Fallback for a neutral-named (noun) tool: classify by the LEADING verb of its description.
    m = re.match(r"\s*([a-zA-Z]+)", description or "")
    verb = m.group(1).lower() if m else ""
    if verb in _DESC_EGRESS_VERBS:
        return "egress"
    if verb in _DESC_CONSEQUENTIAL_VERBS:
        return "consequential"
    return "neutral"


def tool_policies_for(tools: list[dict]) -> dict[str, str]:
    """OpenAI tool list -> {name: 'P-T' | 'P-F or P-T'} for FIDES (neutral tools omitted)."""
    out: dict[str, str] = {}
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        cls = classify_tool(name, fn.get("description", ""))
        if cls == "egress":
            out[name] = _PF_OR_PT
        elif cls == "consequential":
            out[name] = _PT
    return out


# --- entry point -----------------------------------------------------------------------

def resolve_tools(behavior: dict | None,
                  trajectory_messages: list[dict] | None = None) -> tuple[list[dict], dict[str, str]]:
    """Resolve a complete OpenAI tool list + FIDES policy map for a behavior.

    Merges declared Format-A ∪ Format-B tools with parameter schemas observed in the
    recorded trajectory; reconstructs trajectory-only (undeclared) tools.
    """
    behavior = behavior or {}
    fa = _declared_format_a(behavior)
    fb = _declared_format_b(behavior)
    traj = _params_from_trajectory(trajectory_messages or [])
    sysp = B.system_prompt(behavior)

    # Ordered union: Format-A (declared order), then Format-B extras, then trajectory-only.
    names: list[str] = list(fa)
    names += [n for n in fb if n not in fa]
    names += [n for n in traj if n not in fa and n not in fb]

    tools: list[dict] = []
    for name in names:
        decl_desc = (fa.get(name, {}).get("description")
                     or fb.get(name, {}).get("description") or "")
        description = decl_desc or _scrape_description(sysp, name)

        declared_params = fa.get(name, {}).get("parameters")
        if not _has_props(declared_params):
            declared_params = fb.get(name, {}).get("parameters")

        if _has_props(declared_params):
            params = copy.deepcopy(declared_params)
            params.setdefault("type", "object")
            # Augment with trajectory-observed keys the declaration missed (never overwrite).
            tparams = traj.get(name) or {}
            for k, v in (tparams.get("properties") or {}).items():
                params["properties"].setdefault(k, v)
        else:
            params = traj.get(name) or {"type": "object", "properties": {}}

        tools.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": params},
        })

    return tools, tool_policies_for(tools)
