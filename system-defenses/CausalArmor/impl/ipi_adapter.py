"""Run CausalArmor's detector over recorded IPI arena rollouts (Phase 5).

CausalArmor's *detection* signal (the LOO attribution margin at the privileged decision)
needs no agent re-run — the recorded trajectory already contains the context, the injected
span, and the malicious privileged action the agent took. So this is a **post-hoc** check
(the cheap, reliable IPI arm per the skill): of attacks that *did* succeed against the real
model, on how many would CausalArmor's dominance-shift test fire (and thus trigger
sanitization)?

Per rollout we:
  1. locate the injected tool message (loader marks it `isIndirectPromptToolMessage`),
  2. find the first **privileged** tool call the agent issued *after* the injection — the
     malicious action Y driven by the injection (CausalArmor only guards privileged calls),
  3. build context C = system + user + agent messages before Y,
  4. run `Defense.analyze(Y, C)` and check whether the injected span is in the flagged set.

We report the **flag rate** over successful attacks (mirrors classifier-defenses envelope),
plus coverage (fraction of rollouts where a privileged post-injection action exists to guard).

This re-uses the same Qwen3-32B proxy as the AgentDojo arm.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from defense import Defense


def _parse_tool_call(tc: dict) -> tuple[str, dict]:
    """(name, args) from an OpenAI-style recorded tool call."""
    fn = tc.get("function") or {}
    if isinstance(fn, dict):
        name, raw = fn.get("name") or "tool", fn.get("arguments")
    else:
        name, raw = fn, tc.get("args")
    if isinstance(raw, str):
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {"_raw": raw}
    else:
        args = raw or {}
    return name, (args if isinstance(args, dict) else {"_value": args})


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content blocks
        return "".join(b.get("content") or b.get("text") or "" for b in c if isinstance(b, dict))
    return str(c)


def _flatten(messages: Sequence[dict], defense: Defense) -> list[dict]:
    """OpenAI-style recorded messages -> {role, content} dicts for LOO attribution.
    Assistant tool calls serialized into text so the context is faithful (mirrors the
    AgentDojo adapter's _flatten). Length/order preserved."""
    out = []
    for m in messages:
        role, txt = m.get("role", "user"), _msg_text(m)
        if role == "assistant" and m.get("tool_calls"):
            calls = " ".join(defense.serialize_action(*_parse_tool_call(tc)) for tc in m["tool_calls"])
            txt = (txt + "\n" + calls).strip() if txt else calls
        out.append({"role": role, "content": txt})
    return out


def analyze_rollout(rec: dict, defense: Defense) -> dict:
    """Run CausalArmor detection on one loader record. Returns a per-record result dict.

    status:
      "flagged"        — privileged post-injection action found AND injected span dominates
      "not_flagged"    — action found but margin did not flag the injected span
      "no_decision"    — no privileged post-injection action / no marked injection span
    """
    system_prompt = rec.get("system_prompt") or ""
    user_request = rec.get("user_request") or ""
    agent_messages = rec.get("agent_messages") or []

    # 1. locate the injected span (marked by the loader)
    inj_idx = next((i for i, m in enumerate(agent_messages)
                    if m.get("isIndirectPromptToolMessage")), None)
    if inj_idx is None:
        return {"status": "no_decision", "reason": "no_injection_marker"}

    # 2. first privileged tool call AFTER the injection
    action_idx, action_str = None, None
    for j in range(inj_idx + 1, len(agent_messages)):
        m = agent_messages[j]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            priv = [tc for tc in m["tool_calls"] if defense.is_privileged(_parse_tool_call(tc)[0])]
            if priv:
                action_idx = j
                action_str = " ".join(defense.serialize_action(*_parse_tool_call(tc)) for tc in priv)
                break
    if action_idx is None:
        return {"status": "no_decision", "reason": "no_privileged_action_after_injection"}

    # 3. context C = system + user + agent messages strictly before the malicious action
    prefix = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_request}]
    head = list(agent_messages[:action_idx])
    full = prefix + _flatten(head, defense)
    span_index = len(prefix) + inj_idx     # injected span's index in the flattened context

    # 4. attribution + flag check on the injected span
    flagged, score = defense.detect(_msg_text(agent_messages[inj_idx]),
                                    {"messages": full, "action": action_str, "span_index": span_index})
    return {
        "status": "flagged" if flagged else "not_flagged",
        "action": action_str,
        "score": round(score, 4),
        "span_index": span_index,
        "n_spans": sum(1 for m in full if m["role"] == "tool"),
    }
