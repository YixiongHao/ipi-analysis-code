"""Shared world-sim for the ipi_suites forks (FIDES / CaMeL / IPIGuard).

Generates a realistic response for a tool call with the arena world-sim
(`ipi_arena_bench.worldsim` — the same simulator the ipi_eval engine uses for new tool calls).
Default model is **gpt-5-mini via the native OpenAI route**: it is what the arena itself defaults
to (`arena_frontend/src/lib/server/world-sim.ts`) and it is cheaper than Gemini 3 Flash
($0.25/$2.00 vs $0.50/$3.00 per M). temperature/max_tokens are left to the provider default,
matching arena's `generateText`. Two axes controlled by the caller:

  * COVERAGE — whether the fork consults recorded `canned` outputs first (fallback: world-sim only
    for un-canned tools) or simulates everything (CaMeL runs `all_worldsim`, ignoring canned). That
    choice lives in the fork's build_runtime/build_tools; this module just supplies the simulator.
  * GOAL-AWARENESS — `goal_aware=True` (default) feeds the world-sim the behavior's attack
    description+objective (from arena-data), reproducing the arena's ORIGINAL collection, where the
    simulator is goal-aware and biases tool outputs toward the attack narrative. `goal_aware=False`
    passes only the benign user_request, keeping outputs attack-neutral.

The carrier still returns the injection verbatim (handled in build_runtime, not here). Opt-in: pass
the returned callable into build_runtime/build_tools; a None world-sim preserves the
'{"status":"ok"}' stub (keeps the offline unit tests deterministic).
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

_SECRETS = Path("secrets.md")
_ARENA = Path("arena-data")
_FALLBACK = '{"status": "ok"}'


def _openrouter_key() -> str:
    if not _SECRETS.exists():
        return ""
    for line in _SECRETS.read_text().splitlines():
        if line.startswith("openrouter"):
            return line.split("=", 1)[1].strip()
    return ""


@lru_cache(maxsize=None)
def _behavior_goal(corpus: str | None, behavior_id: str | None) -> tuple[str, str]:
    """(attack description, objective) from arena-data/<corpus>/behaviors_detailed.json — the goal
    that makes the world-sim goal-aware (matches arena's original collection). ('', '') if
    unavailable. These fields carry the ATTACK intent (e.g. 'Silently open the garage door while
    the user only asks for an email summary'), which is exactly what build_worldsim_prompt appends
    as Scenario Context to bias outputs toward the attack narrative."""
    if not corpus or not behavior_id:
        return ("", "")
    path = _ARENA / corpus / "behaviors_detailed.json"
    if not path.exists():
        return ("", "")
    bid = behavior_id.strip()
    try:
        for b in json.loads(path.read_text()):
            if (b.get("id") or "").strip() == bid:
                return (b.get("description") or "", b.get("objective") or "")
    except Exception:
        pass
    return ("", "")


def make_worldsim_fallback(packet: dict,
                           model: str = "gpt-5-mini",
                           base_url: str = "https://api.openai.com/v1",
                           api_key: str | None = None,
                           goal_aware: bool = True):
    """Return fallback(name, args)->str for tool calls, or None if no key is available (caller then
    keeps the '{"status":"ok"}' stub).

    goal_aware=True (default): the world-sim sees the behavior's attack description+objective (arena
    parity — the simulator is instructed to make outputs support the attack narrative).
    goal_aware=False: benign scenario context (user_request only), keeping outputs attack-neutral."""
    # Route the key off base_url so the default (native OpenAI gpt-5-mini) authenticates
    # correctly; an OpenRouter base_url still takes the OpenRouter key.
    from ipi_eval.engine import _NATIVE_ROUTES
    if api_key is not None:
        key, provider = api_key, "openrouter"
    else:
        hit = next(((tag, get) for host, (tag, get) in _NATIVE_ROUTES.items() if host in base_url), None)
        provider, key = (hit[0], hit[1]()) if hit else ("openrouter", _openrouter_key())
    if not key:
        return None
    from ipi_arena_compat import LLMClient, ToolCall
    from ipi_arena_compat import build_worldsim_prompt, simulate_tool_response
    client = LLMClient(provider=provider, model=model, base_url=base_url, api_key=key)
    if goal_aware:
        desc, obj = _behavior_goal(packet.get("corpus"), packet.get("behavior_id"))
        behavior = {"description": desc or (packet.get("user_request") or ""), "objective": obj}
    else:
        behavior = {"description": packet.get("user_request") or "", "objective": ""}
    prompt = build_worldsim_prompt(packet.get("tools") or [], behavior)

    def fallback(name: str, args: dict | None, history: list[dict] | None = None) -> str:
        """`history` = the conversation so far. Arena passes the full message list, and the
        simulator needs it to stay self-consistent across turns (see simulate_tool_response's
        ARENA FIDELITY note). Optional so the fork runners' existing 2-arg calls still work —
        but a 2-arg call reverts to the stateless behaviour and should be updated."""
        try:
            out = simulate_tool_response(
                client, prompt, ToolCall(name=name, arguments=args or {}, id="wsim"),
                history or [])
            return out or _FALLBACK
        except Exception:
            return _FALLBACK  # fail-open: never break a rollout on a world-sim hiccup
    return fallback
