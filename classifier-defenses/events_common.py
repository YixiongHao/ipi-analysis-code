#!/usr/bin/env python3
"""Single source of truth for the detector smoke harnesses: WHICH trajectory
boundaries get monitored (`_events`) and HOW a monitored message prefix is
serialized to text (`_render`).

Imported by every detector in this experiment so the monitored event set is
identical across guards — the only variable is the guard itself. Cygnal is ground
truth, so this is its event definition; the others conform to it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # co-located cygnal helper
import cygnal_monitor_posthoc as cm  # noqa: E402  (pure helpers; aiohttp imported lazily)

ROLE_TAG = {"system": "[SYSTEM]", "user": "[USER]",
            "assistant": "[ASSISTANT]", "tool": "[TOOL_RESPONSE]"}


def _events(record):
    """Monitoring events, in chronological order, driven by the per-turn ground-truth
    tags written by `tag_attack_turns.py`:
      - every injection site (`containsIPI == true` turn) -> after_tool_response
      - the decisive harmful action (`isHarmfulAction == true` turn) ->
        after_assistant_tool_call

    Gating injection sites to `containsIPI` turns is the correction over the old
    "every tool turn is a candidate" heuristic: `isIndirectPromptToolMessage` fires on
    *every* tool turn that echoes the payload (e.g. turns [1,4,8]), which 2-3x inflated
    the Q1/Q2 site count. `containsIPI` marks only the true injection turn(s).

    The after_assistant_tool_call event is consumed ONLY by Cygnal's policy-violation
    head — the encoder/KAD detectors ignore every non-after_tool_response stage — so
    monitoring the harmful action is effectively Cygnal-only. It is emitted only when
    the harmful action was isolated (an `isHarmfulAction == true` turn exists).

    Fallback for untagged / `failed` records (no `containsIPI == true` turn): the
    `isIndirectPromptToolMessage` marker if present, else every tool-role message, so an
    un-tagged trajectory is still fully monitored. `injection_known` reports whether the
    sites are ground-truth (`containsIPI`/marker) vs the all-tool-turn assumption.
    Returns (agent_norm, events); each event = {"msg_index", "stage",
    "injection_known", "tool_call_id"/"tool_calls", optional "is_harmful_action"}."""
    agent_norm = [cm._normalize_message(m) for m in record.get("agent_messages") or []]
    sites = [i for i, m in enumerate(agent_norm) if m.get("containsIPI") is True]
    injection_known = bool(sites)
    if not sites:  # untagged / failed record: fall back to marker, else all tool turns
        marked = [i for i, m in enumerate(agent_norm) if m.get("isIndirectPromptToolMessage")]
        injection_known = bool(marked)
        sites = marked or [i for i, m in enumerate(agent_norm) if m.get("role") == "tool"]
    if not sites:
        return agent_norm, []
    evs = []
    for i in sites:
        evs.append({"msg_index": i, "stage": cm.STAGE_AFTER_TOOL_RESPONSE,
                    "tool_call_id": agent_norm[i].get("tool_call_id"),
                    "injection_known": injection_known})
    for i, m in enumerate(agent_norm):
        if m.get("isHarmfulAction") is True:
            names = [(tc.get("function") or {}).get("name") for tc in (m.get("tool_calls") or [])]
            evs.append({"msg_index": i, "stage": cm.STAGE_AFTER_ASSISTANT_TOOL_CALL,
                        "tool_calls": names, "injection_known": injection_known,
                        "is_harmful_action": True})
    evs.sort(key=lambda e: e["msg_index"])
    return agent_norm, evs


def _render(messages):
    """Serialize the same message prefix Cygnal receives into one text blob."""
    parts = []
    for m in messages:
        role = m.get("role", "unknown")
        tag = ROLE_TAG.get(role, "[" + role.upper() + "]")
        seg = tag + "\n" + (m.get("content") or "")
        tcs = m.get("tool_calls")
        if role == "assistant" and tcs:
            try:
                rendered = cm._stable_json_dumps([
                    {"name": (tc.get("function") or {}).get("name"),
                     "arguments": (tc.get("function") or {}).get("arguments")}
                    for tc in tcs])
            except Exception:
                rendered = str(tcs)
            seg += "\n[TOOL_CALLS] " + rendered
        parts.append(seg)
    return "\n\n".join(parts)
