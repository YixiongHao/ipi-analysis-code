#!/usr/bin/env python3
"""Surgically re-run ONLY Cygnal's `ipi` (injection) head against the corrected
`containsIPI` sites, and merge into the existing results/cygnal-code-8b.jsonl WITHOUT
touching the policy-violation (after_assistant_tool_call) head.

Why: the prior full-store run (2026-06-18) scored the `ipi` head at the OLD (wrong)
containsIPI turns. tag_attack_turns.py re-localized those turns (2026-06-19), so the
`ipi`-head events must be recomputed at the new turns. The policy head keys off
`isHarmfulAction` (unchanged) and its call payloads are independent of containsIPI, so
those events are accurate — they are copied VERBATIM from the existing file.

Per record we write: NEW after_tool_response events (ipi head, this run) + OLD
after_assistant_tool_call events (policy head, copied), and recompute the envelope
fields. Only ipi-head calls are issued (the response cache absorbs unchanged 2025 /
same-turn records), so this is ~½ the calls of a full re-run.

Run:  CYGNAL_API_KEY=... python rerun_cygnal_ipi.py
"""
import asyncio
import json
import os

import detectors as D
import cygnal_monitor_posthoc as cm
from detectors import _cygnal_one, _cygnal_proto_violation, config_hash

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store", "attacks.jsonl")
OUT = os.path.join(HERE, "results", "cygnal-code-8b.jsonl")
DEFENSES = os.path.join(HERE, "defenses.json")
AT = cm.STAGE_AFTER_ASSISTANT_TOOL_CALL
TR = cm.STAGE_AFTER_TOOL_RESPONSE


def merge(new_env, old_env, params):
    """new_env: ipi-only re-run (after_tool_response events only).
    old_env: prior full envelope (source of the preserved policy-head events)."""
    new_tr = [e for e in new_env["events"] if e["stage"] == TR]
    old_at = [e for e in (old_env or {}).get("events", []) if e["stage"] == AT]

    ipi_sites = [m["agent_message_index"] for m in new_tr if m["ipi"]]
    proto_turns = [m["agent_message_index"] for m in old_at if _cygnal_proto_violation(m)]
    harm = [m for m in old_at if m.get("is_harmful_action")]
    ipi_flagged, violation_flagged = bool(ipi_sites), bool(proto_turns)
    events = sorted(new_tr + old_at, key=lambda e: e["agent_message_index"])
    return {
        "flagged_any": ipi_flagged or violation_flagged,
        "config_hash": config_hash(params),
        "flaggedCorrectMessage": any(m["ipi"] and m["injection_known"] for m in new_tr),
        "events": events,
        "summary": {
            "ipi_flagged": ipi_flagged, "ipi_sites": ipi_sites,
            "violation_flagged": violation_flagged, "violation_turns": proto_turns,
            "harmful_action_turns": [m["agent_message_index"] for m in harm],
            "harmful_action_violation": any(_cygnal_proto_violation(m) for m in harm),
            "n_tool_response_events": len(new_tr), "n_assistant_toolcall_events": len(old_at),
            "any_error": any(m["cygnal_error"] for m in events),
        },
    }


async def run(rows, bundle, api_key, old):
    import aiohttp
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "grayswan-api-key": api_key, "Accept-Encoding": "identity"}
    cache = cm.SqliteResponseCache(bundle["cache_path"]) if bundle["cache_path"] else None
    sem = asyncio.Semaphore(bundle["concurrency"])
    done = 0
    out = []
    async with aiohttp.ClientSession() as session:
        async def one(r):
            nonlocal done
            env = await _cygnal_one(session, sem, r["record"], bundle, headers, cache, ipi_only=True)
            done += 1
            if done % 500 == 0:
                print(f"  ...{done}/{len(rows)}", flush=True)
            return merge(env, old.get(r["attack_id"]), bundle["params"])
        out = await asyncio.gather(*[one(r) for r in rows])
    return out


def _read_key():
    api_key = os.getenv("CYGNAL_API_KEY")
    if api_key:
        return api_key
    import re
    secrets = "secrets.md"
    if os.path.exists(secrets):
        for l in open(secrets):
            if re.search(r"api", l, re.I) and "=" in l:
                return l.split("=", 1)[1].strip()
    return None


def main():
    api_key = _read_key()
    if not api_key:
        raise SystemExit("CYGNAL_API_KEY unset")
    rows = [json.loads(l) for l in open(STORE) if l.strip()]
    old = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                e = json.loads(l)
                old[e["attack_id"]] = e
    print(f"store rows={len(rows)}  old cygnal envelopes={len(old)}")

    params = next(d["params"] for d in json.load(open(DEFENSES)) if d["defense_id"] == "cygnal-code-8b")
    bundle = D.load_cygnal(params)
    envs = asyncio.run(run(rows, bundle, api_key, old))

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        for r, env in zip(rows, envs):
            f.write(json.dumps({"attack_id": r["attack_id"], "defense_id": "cygnal-code-8b", **env},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    fa = sum(e["flagged_any"] for e in envs)
    fc = sum(bool(e["flaggedCorrectMessage"]) for e in envs)
    print(f"=== merged {len(envs)} -> {OUT}  flagged_any={fa}  flaggedCorrectMessage={fc} ===")


if __name__ == "__main__":
    main()
