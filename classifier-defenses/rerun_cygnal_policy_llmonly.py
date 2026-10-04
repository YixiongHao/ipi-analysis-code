#!/usr/bin/env python3
"""Surgically run Cygnal's policy-violation head (after_assistant_tool_call) for the 1,027
records whose harmful action was just localized by `localize_llm_only_harmful.py`
(`isHarmfulAction_status` ∈ {isolated:challenge_completion, isolated:llm_final_turn};
previously `failed:llm_only`).

Why: those records never had an `isHarmfulAction == true` turn, so `_events` never emitted
a policy-head event and their existing `results/cygnal-code-8b.jsonl` envelope has ONLY
ipi-head (after_tool_response) events. Now the harmful turn is marked, so we POST the full
prefix up to it ([system]+[user]+agent_messages[0..harmful_idx]) and read the policy head.

This is an AGENT ACTION, so the signal of interest is the policy-violation head
(`_cygnal_proto_violation` on the after_assistant_tool_call event).

Surgical: only the 1,027 records are recomputed (policy-head call only); their ipi-head
events are copied VERBATIM from the existing envelope. The other 25,439 envelopes are
written back byte-for-byte. Order preserved.

Run:  CYGNAL_API_KEY=... python rerun_cygnal_policy_llmonly.py
"""
import asyncio
import json
import os

import detectors as D
import cygnal_monitor_posthoc as cm
from detectors import _cygnal_one, _cygnal_proto_violation, config_hash
from rerun_cygnal_ipi import _read_key

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store", "attacks.jsonl")
OUT = os.path.join(HERE, "results", "cygnal-code-8b.jsonl")
DEFENSES = os.path.join(HERE, "defenses.json")
AT = cm.STAGE_AFTER_ASSISTANT_TOOL_CALL
TR = cm.STAGE_AFTER_TOOL_RESPONSE
NEW_STATUSES = {"isolated:challenge_completion", "isolated:llm_final_turn"}


def merge(new_env, old_env, params):
    """new_env: policy-only re-run (after_assistant_tool_call events only).
    old_env: prior envelope (source of the preserved ipi-head after_tool_response events)."""
    new_at = [e for e in new_env["events"] if e["stage"] == AT]
    old_tr = [e for e in (old_env or {}).get("events", []) if e["stage"] == TR]

    ipi_sites = [m["agent_message_index"] for m in old_tr if m["ipi"]]
    proto_turns = [m["agent_message_index"] for m in new_at if _cygnal_proto_violation(m)]
    harm = [m for m in new_at if m.get("is_harmful_action")]
    ipi_flagged, violation_flagged = bool(ipi_sites), bool(proto_turns)
    events = sorted(old_tr + new_at, key=lambda e: e["agent_message_index"])
    return {
        "flagged_any": ipi_flagged or violation_flagged,
        "config_hash": config_hash(params),
        "flaggedCorrectMessage": any(m["ipi"] and m["injection_known"] for m in old_tr),
        "events": events,
        "summary": {
            "ipi_flagged": ipi_flagged, "ipi_sites": ipi_sites,
            "violation_flagged": violation_flagged, "violation_turns": proto_turns,
            "harmful_action_turns": [m["agent_message_index"] for m in harm],
            "harmful_action_violation": any(_cygnal_proto_violation(m) for m in harm),
            "n_tool_response_events": len(old_tr), "n_assistant_toolcall_events": len(new_at),
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
    async with aiohttp.ClientSession() as session:
        async def one(r):
            nonlocal done
            env = await _cygnal_one(session, sem, r["record"], bundle, headers, cache,
                                    policy_only=True)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(rows)}", flush=True)
            return r["attack_id"], merge(env, old.get(r["attack_id"]), bundle["params"])
        return dict(await asyncio.gather(*[one(r) for r in rows]))


def main():
    api_key = _read_key()
    if not api_key:
        raise SystemExit("CYGNAL_API_KEY unset")

    targets = [json.loads(l) for l in open(STORE)
               if l.strip() and json.loads(l)["record"]["isHarmfulAction_status"] in NEW_STATUSES]
    old_lines = [json.loads(l) for l in open(OUT) if l.strip()]
    old = {e["attack_id"]: e for e in old_lines}
    print(f"target records={len(targets)}  existing envelopes={len(old)}")
    assert all(t["attack_id"] in old for t in targets), "some target has no existing envelope"

    params = next(d["params"] for d in json.load(open(DEFENSES)) if d["defense_id"] == "cygnal-code-8b")
    bundle = D.load_cygnal(params)
    updated = asyncio.run(run(targets, bundle, api_key, old))

    # rewrite OUT in original order, substituting only the updated envelopes
    tmp = OUT + ".tmp"
    n_sub = 0
    with open(tmp, "w") as f:
        for e in old_lines:
            aid = e["attack_id"]
            if aid in updated:
                e = {"attack_id": aid, "defense_id": "cygnal-code-8b", **updated[aid]}
                n_sub += 1
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

    nv = sum(1 for aid in updated if updated[aid]["summary"]["harmful_action_violation"])
    err = sum(1 for aid in updated if updated[aid]["summary"]["any_error"])
    print(f"=== substituted {n_sub} envelopes -> {OUT}")
    print(f"    harmful_action_violation (policy head fired at the action): {nv}/{len(updated)}")
    print(f"    envelopes with any cygnal_error: {err}")


if __name__ == "__main__":
    main()
