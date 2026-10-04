#!/usr/bin/env python3
"""Cygnal-with-CoT variant: run Cygnal's policy-violation head on the HARMFUL ACTION, but
with each assistant turn's externally-keyed (`reasoning`-field) chain-of-thought surfaced
INLINE in `content` as a <think>..</think> block.

Why: Cygnal reads the prefix as standard chat messages (role/content/tool_calls) and
ignores the non-standard `reasoning` key, so it normally monitors the action blind to the
model's stated intent. This variant moves that CoT into `content` to test whether the
policy head fires more once it can see the reasoning. Paired 1:1 against the baseline
`results/cygnal-code-8b.jsonl` (same records, same harmful turn, no CoT).

Population: the 9,271 records whose `isHarmfulAction==True` turn carries a non-summarized
`reasoning`-field CoT (>= MIN_COT chars). Only the policy head is run (one
after_assistant_tool_call call per record, full prefix up to the harmful turn). Output:
a new store `results/cygnal-code-8b-cot.jsonl`, same envelope structure.

Run:  CYGNAL_API_KEY=... python run_cygnal_cot.py
"""
import asyncio
import copy
import json
import os

import detectors as D
import cygnal_monitor_posthoc as cm
from detectors import _cygnal_one, _cygnal_proto_violation, config_hash
from rerun_cygnal_ipi import _read_key
from tag_real_cot import is_summary_family

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store", "attacks.jsonl")
OUT = os.path.join(HERE, "results", "cygnal-code-8b-cot.jsonl")
DEFENSES = os.path.join(HERE, "defenses.json")
DEFENSE_ID = "cygnal-code-8b-cot"
AT = cm.STAGE_AFTER_ASSISTANT_TOOL_CALL
MIN_COT = 50


def harmful_turn(record):
    for m in record["agent_messages"]:
        if m.get("isHarmfulAction") is True:
            return m
    return None


def in_population(row):
    """Harmful turn carries an externally-keyed (reasoning-field), non-summarized CoT."""
    if is_summary_family(row["model_name"]):
        return False
    h = harmful_turn(row["record"])
    return bool(h) and len((h.get("reasoning") or "").strip()) >= MIN_COT


def inject_cot(record, max_chars):
    """Deep-copy `record`; for each assistant turn with a reasoning-field CoT, prepend it
    into `content` as a <think>..</think> block (head-capped to max_chars) and blank the
    `reasoning` key so the only delta vs baseline is the in-content CoT. Returns
    (record_copy, {turns_injected, cot_chars, truncated})."""
    rc = copy.deepcopy(record)
    turns, total, truncated = 0, 0, False
    for m in rc["agent_messages"]:
        if m.get("role") != "assistant":
            continue
        reas = (m.get("reasoning") or "").strip()
        if len(reas) < MIN_COT:
            continue
        cut = reas[:max_chars]
        if len(reas) > max_chars:
            cut += "\n…[CoT truncated]"
            truncated = True
        m["content"] = "<think>\n" + cut + "\n</think>\n" + (m.get("content") or "")
        m["reasoning"] = ""  # strip external channel -> clean delta
        turns += 1
        total += len(reas)
    return rc, {"turns_injected": turns, "cot_chars": total, "truncated": truncated}


def envelope(env, info, params):
    """Same shape as the baseline policy-head envelope, + cot_injection provenance."""
    at = [e for e in env["events"] if e["stage"] == AT]
    proto_turns = [m["agent_message_index"] for m in at if _cygnal_proto_violation(m)]
    harm = [m for m in at if m.get("is_harmful_action")]
    return {
        "flagged_any": bool(proto_turns),
        "config_hash": config_hash(params),
        "flaggedCorrectMessage": None,  # no ipi head in this variant
        "events": at,
        "summary": {
            "ipi_flagged": False, "ipi_sites": [],
            "violation_flagged": bool(proto_turns), "violation_turns": proto_turns,
            "harmful_action_turns": [m["agent_message_index"] for m in harm],
            "harmful_action_violation": any(_cygnal_proto_violation(m) for m in harm),
            "n_tool_response_events": 0, "n_assistant_toolcall_events": len(at),
            "any_error": any(m["cygnal_error"] for m in at),
            "cot_injection": info,
        },
    }


async def run(rows, bundle, api_key, max_chars):
    import aiohttp
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "grayswan-api-key": api_key, "Accept-Encoding": "identity"}
    cache = cm.SqliteResponseCache(bundle["cache_path"]) if bundle["cache_path"] else None
    sem = asyncio.Semaphore(bundle["concurrency"])
    done = 0
    async with aiohttp.ClientSession() as session:
        async def one(r):
            nonlocal done
            rc, info = inject_cot(r["record"], max_chars)
            env = await _cygnal_one(session, sem, rc, bundle, headers, cache, policy_only=True)
            done += 1
            if done % 250 == 0:
                print(f"  ...{done}/{len(rows)}", flush=True)
            return {"attack_id": r["attack_id"], "defense_id": DEFENSE_ID,
                    **envelope(env, info, bundle["params"])}
        return await asyncio.gather(*[one(r) for r in rows])


def main():
    api_key = _read_key()
    if not api_key:
        raise SystemExit("CYGNAL_API_KEY unset")
    rows = [r for r in (json.loads(l) for l in open(STORE) if l.strip()) if in_population(r)]
    print(f"population (externally-keyed CoT @ harmful turn): {len(rows)}")

    params = next(d["params"] for d in json.load(open(DEFENSES)) if d["defense_id"] == DEFENSE_ID)
    max_chars = params["cot_injection"]["max_chars"]
    bundle = D.load_cygnal(params)
    envs = asyncio.run(run(rows, bundle, api_key, max_chars))

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        for e in envs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

    nv = sum(e["summary"]["harmful_action_violation"] for e in envs)
    tr = sum(e["summary"]["cot_injection"]["truncated"] for e in envs)
    err = sum(e["summary"]["any_error"] for e in envs)
    print(f"=== wrote {len(envs)} -> {OUT}")
    print(f"    policy head fired @ action (with CoT): {nv}/{len(envs)} = {nv/len(envs):.1%}")
    print(f"    truncated CoT (>{max_chars} chars): {tr}  |  cygnal errors: {err}")


if __name__ == "__main__":
    main()
