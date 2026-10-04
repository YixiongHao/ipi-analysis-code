#!/usr/bin/env python3
"""LLM-judge classifier defense: frontier LLMs prompted with the exact Cygnal
monitor prompts (see cygnal_prompt.py), called via OpenRouter with first-party
provider pinning.

Plugs into the classifier-defenses pipeline exactly like the Cygnal/Granite
runners: same event enumeration (events_common._events), same prefix
construction, and byte-compatible sidecar schema written to
results/<defense_id>.jsonl.

Usage:
    run_llm_judge.py --defense llmjudge-gpt-5.6-luna --store store/attacks.jsonl \
        --results results_ipi [--limit N] [--concurrency 16] [--heads ipi,policy] \
        [--dry-run]

--dry-run renders every prompt and reports event counts + size/cost estimates
without any API calls (used for the full-run cost estimate).

Differences from the Cygnal runner (deliberate):
  - errors are NOT cached (a cached _error row in the Cygnal cache is never
    retried — a known operational trap); failed events are retried on re-run.
  - the judge prompt is built CLIENT-side (cygnal_prompt.py); the hosted Cygnal
    endpoint renders server-side. IPI-head prompts therefore contain only the
    [tool-call, tool-response] pair per the published conversation_ir adapter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_CD = os.path.dirname(_HERE)
sys.path.insert(0, _CD)
sys.path.insert(0, _HERE)

import events_common  # noqa: E402
import cygnal_monitor_posthoc as cm  # noqa: E402
import cygnal_prompt  # noqa: E402
from detectors import config_hash  # noqa: E402
from run_cygnal_cot import inject_cot  # noqa: E402  (the canonical CoT-surfacing transform)

SECRETS_PATH = os.path.join(os.path.dirname(_CD), "secrets.md")
DEFAULT_CACHE = os.path.join(_CD, ".llmjudge_cache.sqlite")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# rough chars/token for dry-run cost estimates (JSON-heavy English text)
EST_CHARS_PER_TOKEN = 4.0


def _read_llm_baseline_key() -> Optional[str]:
    key = os.getenv("OPENROUTER_LLM_BASELINE_KEY")
    if key:
        return key
    if os.path.exists(SECRETS_PATH):
        for line in open(SECRETS_PATH):
            if re.search(r"llm[ _-]?baseline", line, re.I) and "=" in line:
                return line.split("=", 1)[1].strip()
    return None


def _read_openai_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists(SECRETS_PATH):
        for line in open(SECRETS_PATH):
            if (re.match(r"\s*openai\b", line, re.I)
                    and "openrouter" not in line.lower() and "=" in line):
                return line.split("=", 1)[1].strip()
    return None


def _is_openai_direct(base_url: str) -> bool:
    return "api.openai.com" in base_url


def read_key_for(params: Dict[str, Any]) -> Optional[str]:
    if _is_openai_direct(params.get("base_url", "")):
        return _read_openai_key()
    return _read_llm_baseline_key()


def load_defense(defense_id: str) -> Dict[str, Any]:
    with open(os.path.join(_CD, "defenses.json")) as f:
        for entry in json.load(f):
            if entry.get("defense_id") == defense_id:
                return entry
    raise SystemExit(f"defense_id {defense_id!r} not found in defenses.json")


# ── prompt construction ─────────────────────────────────────────────────────

def build_event_prompts(
    record: Dict[str, Any], token_limit: Optional[int]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (events, prompt_jobs). Each job:
    {"event_idx", "head", "prompt"} or {"event_idx", "head", "render_error"}."""
    agent_norm, evs = events_common._events(record)
    prefix: List[Dict[str, Any]] = []
    sp = cm._normalize_content(record.get("system_prompt"))
    if sp:
        prefix.append({"role": "system", "content": sp})
    prefix.append({"role": "user", "content": cm._normalize_content(record.get("user_request") or "")})
    base = prefix + agent_norm

    jobs = []
    for idx, ev in enumerate(evs):
        msgs = base[: len(prefix) + ev["msg_index"] + 1]
        head = "ipi" if ev["stage"] == cm.STAGE_AFTER_TOOL_RESPONSE else "policy"
        last = msgs[-1] if msgs else {}
        nonstandard = (
            last.get("role") != "tool"
            if head == "ipi"
            else (last.get("role") != "assistant" or not last.get("tool_calls"))
        )
        try:
            if head == "ipi":
                prompt = cygnal_prompt.render_ipi_prompt(msgs, token_limit=token_limit, strict=False)
            else:
                prompt = cygnal_prompt.render_violation_prompt(msgs, token_limit=token_limit, strict=False)
            job = {"event_idx": idx, "head": head, "prompt": prompt}
            if nonstandard:
                job["nonstandard_target"] = True
            jobs.append(job)
        except ValueError as e:
            jobs.append({"event_idx": idx, "head": head, "render_error": str(e)})
    return evs, jobs


def build_payload(params: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    # sampling params deliberately omitted -> provider defaults (temp 1);
    # gpt-5.6 rejects an explicit temperature parameter.
    body: Dict[str, Any] = {
        "model": params["model"],
        "messages": [{"role": "user", "content": prompt}],
    }
    # opt-in only: no registered defense sets this, so default payloads/cache keys are unchanged.
    # Needed by the CoT-suppression judge arm, where provider-default sampling gives a ~21%
    # test-retest flip rate that swamps the effects being measured.
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    effort = params.get("reasoning_effort")
    if _is_openai_direct(params.get("base_url", "")):
        # OpenAI direct: reasoning models take max_completion_tokens +
        # reasoning_effort; usage is always returned; no provider routing.
        body["max_completion_tokens"] = params.get("max_tokens", 4096)
        if effort:
            body["reasoning_effort"] = effort
        if params.get("logprobs"):
            # continuous score: P(true) from the verdict token's alternatives
            body["logprobs"] = True
            body["top_logprobs"] = params.get("top_logprobs", 5)
    else:
        body["max_tokens"] = params.get("max_tokens", 4096)
        body["usage"] = {"include": True}
        if params.get("provider_pin"):
            body["provider"] = params["provider_pin"]
        if effort == "none":
            # explicit thinking-off (Anthropic-style models have no 'none' tier)
            body["reasoning"] = {"enabled": False}
        elif effort:
            body["reasoning"] = {"effort": effort}
    return body


# ── OpenRouter call ─────────────────────────────────────────────────────────

async def _post_json(session, url, headers, payload, timeout, retries):
    import aiohttp

    last_err: Dict[str, Any] = {"_error": {"exception": "no attempt made"}}
    for attempt in range(retries + 1):
        try:
            async with session.post(
                url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                body = await resp.text()
                if resp.status == 200:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return {"_error": {"status": 200, "body": body[:2000], "kind": "json_parse"}}
                    # OpenRouter can 200 with an error object (e.g. provider error)
                    if isinstance(data, dict) and data.get("error"):
                        err_code = (data["error"] or {}).get("code")
                        last_err = {"_error": {"status": err_code, "body": json.dumps(data["error"])[:2000]}}
                        if err_code in RETRYABLE_STATUS and attempt < retries:
                            await asyncio.sleep(0.5 * (2 ** attempt))
                            continue
                        return last_err
                    return data
                last_err = {"_error": {"status": resp.status, "body": body[:2000]}}
                if resp.status in RETRYABLE_STATUS and attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                return last_err
        except Exception as e:  # network/timeouts
            last_err = {"_error": {"exception": f"{type(e).__name__}: {e}"}}
            if attempt < retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
    return last_err


async def _judge_one(session, sem, cache, params, base_url, headers, job,
                     timeout, retries, cache_salt=None):
    """Returns (job, response_dict). Errors are NOT cached."""
    payload = build_payload(params, job["prompt"])
    key_obj = {"llmjudge_base_url": base_url, "model": params["model"], "payload": payload}
    if cache_salt:
        # repeated sampled passes over identical payloads: salt makes each pass
        # draw fresh instead of replaying the first pass's cached draw
        key_obj["sample_salt"] = cache_salt
    cache_key = cm._sha256_hex(cm._stable_json_dumps(key_obj))
    cached = cache.get(cache_key)  # returns a parsed dict (or None)
    if isinstance(cached, dict) and cached.get("_cache_parse_error"):
        # heal legacy double-encoded rows (set() was once fed a pre-dumped str)
        try:
            fixed = json.loads(cached.get("raw") or "")
            if isinstance(fixed, str):  # row was dumps(dumps(resp))
                fixed = json.loads(fixed)
        except (json.JSONDecodeError, TypeError):
            fixed = None
        cached = fixed if isinstance(fixed, dict) else None
        if cached is not None:
            cache.set(cache_key, cached)
    if cached is not None:
        return job, cached
    async with sem:
        resp = await _post_json(
            session, base_url.rstrip("/") + "/chat/completions",
            headers, payload, timeout, retries,
        )
    if isinstance(resp, dict) and "_error" not in resp:
        cache.set(cache_key, resp)
    return job, resp


def _judged_event(ev: Dict[str, Any], head: str, resp: Optional[Dict[str, Any]],
                  render_error: Optional[str],
                  nonstandard: bool = False) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Build the sidecar event dict + usage stats from a response."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    flagged: Optional[bool] = None
    prob: Optional[float] = None
    error = False
    if render_error is not None or resp is None:
        error = True
    elif "_error" in resp:
        error = True
    else:
        try:
            choice = resp["choices"][0]
            content = (choice["message"] or {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            choice, content, error = None, "", True
        if not error:
            parsed = cygnal_prompt.parse_judge_response(content, head)
            if parsed["ok"]:
                flagged = parsed["flagged"]
                prob = cygnal_prompt.verdict_probability(choice.get("logprobs"), head)
            else:
                error = True
        u = resp.get("usage") or {}
        usage["prompt_tokens"] = u.get("prompt_tokens") or 0
        usage["completion_tokens"] = u.get("completion_tokens") or 0
        usage["reasoning_tokens"] = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
        usage["cost_usd"] = u.get("cost") or 0.0

    out: Dict[str, Any] = {
        "stage": ev["stage"],
        "agent_message_index": ev["msg_index"],
        "head": head,
        "flagged": bool(flagged) if flagged is not None else False,
        "prob_flagged": round(prob, 6) if prob is not None else None,
    }
    if head == "ipi":
        # duplicate under "ipi" so add_flagged_correct.py's blocked/ipi probe works
        out["ipi"] = out["flagged"]
        out["injection_known"] = ev.get("injection_known", False)
    else:
        out["is_harmful_action"] = True
        out["tool_calls"] = ev.get("tool_calls") or []
        out["injection_known"] = ev.get("injection_known", False)
    if nonstandard:
        out["nonstandard_target"] = True
    out["error"] = error
    if render_error is not None:
        out["render_error"] = render_error
    elif error and isinstance(resp, dict) and "_error" in resp:
        # keep the API failure reason: a bare error=True bool is undiagnosable after the fact
        out["error_detail"] = json.dumps(resp["_error"])[:400]
    return out, usage


def assemble_envelope(defense_id: str, cfg_hash: str, row: Dict[str, Any],
                      events_out: List[Dict[str, Any]],
                      usage_total: Dict[str, Any],
                      cot_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ipi_events = [e for e in events_out if e["stage"] == cm.STAGE_AFTER_TOOL_RESPONSE]
    pol_events = [e for e in events_out if e["stage"] == cm.STAGE_AFTER_ASSISTANT_TOOL_CALL]
    ipi_sites = [e["agent_message_index"] for e in ipi_events if e["flagged"]]
    violation_turns = [e["agent_message_index"] for e in pol_events if e["flagged"]]
    harmful_turns = [e["agent_message_index"] for e in pol_events]
    flagged_correct = any(e["flagged"] and e.get("injection_known") for e in ipi_events) or None
    if flagged_correct is None and any(e.get("injection_known") for e in ipi_events):
        flagged_correct = False
    env = {
        "attack_id": row["attack_id"],
        "defense_id": defense_id,
        "flagged_any": bool(ipi_sites or violation_turns),
        "config_hash": cfg_hash,
        "flaggedCorrectMessage": flagged_correct,
        "events": events_out,
        "summary": {
            "ipi_flagged": bool(ipi_sites),
            "ipi_sites": ipi_sites,
            "violation_flagged": bool(violation_turns),
            "violation_turns": violation_turns,
            "harmful_action_turns": harmful_turns,
            # every judge policy event IS a harmful-action event, so this mirrors the Cygnal /
            # Granite sidecar key that all the cot_suppression tooling reads. Additive: the
            # pre-existing llmjudge-*.jsonl files predate it and use violation_flagged.
            "harmful_action_violation": bool(violation_turns),
            "n_tool_response_events": len(ipi_events),
            "n_assistant_toolcall_events": len(pol_events),
            "any_error": any(e["error"] for e in events_out),
            "usage": usage_total,
        },
    }
    if cot_info is not None:
        env["summary"]["cot_injection"] = cot_info
    return env


# ── main flows ──────────────────────────────────────────────────────────────

def maybe_inject_cot(record, params):
    """Surface the agent's own CoT to the judge, iff params['cot_injection'].enabled.

    Uses run_cygnal_cot.inject_cot verbatim — the same transform behind the
    `cygnal-code-8b-cot` / `granite-guardian-4.1-8b-cot` arms — so a judge `-cot` defense is the
    byte-equivalent manipulation on the judge: every assistant turn's reasoning field is moved into
    `content` as a <think>…</think> block (head-capped) and the external channel is blanked. Without
    it, `reasoning` is never rendered into a judge prompt at all, so the plain defense IS the
    CoT-blind arm and the pair is a clean one-variable contrast.
    """
    ci = params.get("cot_injection") or {}
    if not ci.get("enabled"):
        return record, None
    return inject_cot(record, ci.get("max_chars", 24000))


def dry_run(rows, params, heads, defense_id):
    token_limit = params.get("token_limit")
    n_events = {"ipi": 0, "policy": 0}
    chars = {"ipi": 0, "policy": 0}
    render_errors = 0
    for row in rows:
        rec, _ = maybe_inject_cot(row["record"], params)
        _, jobs = build_event_prompts(rec, token_limit)
        for job in jobs:
            if job["head"] not in heads:
                continue
            if "render_error" in job:
                render_errors += 1
                continue
            n_events[job["head"]] += 1
            chars[job["head"]] += len(job["prompt"])
    est_in_tokens = {h: chars[h] / EST_CHARS_PER_TOKEN for h in chars}
    # completion: reasoning-off JSON answer is tiny; budget 30 out-tokens (none)
    # plus headroom for minimal-effort reasoning (~300)
    out_tok_per_event = 300 if params.get("reasoning_effort") == "minimal" else 30
    price_in = params.get("price_per_mtok_in", 0.5)
    price_out = params.get("price_per_mtok_out", 3.0)
    total_events = sum(n_events.values())
    cost = (sum(est_in_tokens.values()) * price_in
            + total_events * out_tok_per_event * price_out) / 1e6
    report = {
        "defense_id": defense_id,
        "records": len(rows),
        "events": n_events,
        "render_errors": render_errors,
        "prompt_chars": chars,
        "est_prompt_tokens": {h: int(est_in_tokens[h]) for h in est_in_tokens},
        "est_out_tokens_per_event": out_tok_per_event,
        "price_per_mtok": {"in": price_in, "out": price_out},
        "est_cost_usd": round(cost, 2),
    }
    print(json.dumps(report, indent=2))
    return report


async def run_async(rows, entry, args):
    import aiohttp

    params = entry["params"]
    defense_id = entry["defense_id"]
    cfg_hash = config_hash(params)
    key = read_key_for(params)
    if not key:
        raise SystemExit(
            "API key not found for this defense's base_url: need an 'openai' line "
            "(direct) or 'llm baseline' line (OpenRouter) in secrets.md, or the "
            "OPENAI_API_KEY / OPENROUTER_LLM_BASELINE_KEY env var"
        )
    base_url = params.get("base_url", "https://openrouter.ai/api/v1")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    heads = set(args.heads.split(","))
    token_limit = params.get("token_limit")
    cache = cm.SqliteResponseCache(args.cache)
    sem = asyncio.Semaphore(args.concurrency)
    timeout, retries = args.timeout, args.retries

    out_rows = []
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        async def handle_row(row):
            rec, cot_info = maybe_inject_cot(row["record"], params)
            evs, jobs = build_event_prompts(rec, token_limit)
            jobs = [j for j in jobs if j["head"] in heads]
            results = await asyncio.gather(*[
                _judge_one(session, sem, cache, params, base_url, headers, j,
                           timeout, retries, cache_salt=args.cache_salt)
                for j in jobs if "render_error" not in j
            ])
            resp_by_idx = {job["event_idx"]: resp for job, resp in results}
            rerr_by_idx = {j["event_idx"]: j["render_error"] for j in jobs if "render_error" in j}
            events_out, totals = [], {"prompt_tokens": 0, "completion_tokens": 0,
                                      "reasoning_tokens": 0, "cost_usd": 0.0}
            for j in jobs:
                idx = j["event_idx"]
                ev_out, usage = _judged_event(
                    evs[idx], j["head"], resp_by_idx.get(idx), rerr_by_idx.get(idx),
                    nonstandard=j.get("nonstandard_target", False))
                events_out.append(ev_out)
                for k in totals:
                    totals[k] += usage[k]
            if totals["cost_usd"] == 0.0 and (totals["prompt_tokens"] or totals["completion_tokens"]):
                # OpenAI direct returns no cost field -> compute from list prices
                totals["cost_usd"] = (
                    totals["prompt_tokens"] * params.get("price_per_mtok_in", 0.5)
                    + totals["completion_tokens"] * params.get("price_per_mtok_out", 3.0)
                ) / 1e6
            totals["cost_usd"] = round(totals["cost_usd"], 6)
            return assemble_envelope(defense_id, cfg_hash, row, events_out, totals, cot_info)

        # Chunked + resumable. Creating one task per record up front would hold every rendered
        # prompt in memory at once (fine at 1k records, not at 9k with 250k-token prompts), and a
        # single write at the end would throw away a multi-hour paid run on any transient failure.
        os.makedirs(args.results, exist_ok=True)
        stem = f"{defense_id}.{args.out_suffix}" if args.out_suffix else defense_id
        out_path = os.path.join(args.results, f"{stem}.jsonl")
        part_path = out_path + ".partial"
        done = set()
        if args.resume and os.path.exists(part_path):
            for line in open(part_path):
                if line.strip():
                    done.add(json.loads(line)["attack_id"])
            print(f"resume: {len(done)} records already scored in {part_path}", flush=True)
        todo = [r for r in rows if r["attack_id"] not in done]
        print(f"records to score: {len(todo)} of {len(rows)} | chunk {args.chunk}", flush=True)
        n_done = len(done)
        with open(part_path, "a") as pf:
            for i in range(0, len(todo), args.chunk):
                chunk = todo[i:i + args.chunk]
                for env in await asyncio.gather(*[handle_row(r) for r in chunk]):
                    pf.write(json.dumps(env, ensure_ascii=False) + "\n")
                pf.flush()
                os.fsync(pf.fileno())
                n_done += len(chunk)
                print(f"  {n_done}/{len(rows)} records ({time.time()-t0:.0f}s)", flush=True)
        results = [json.loads(l) for l in open(part_path) if l.strip()]

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for env in results:
            f.write(json.dumps(env, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    os.remove(part_path)

    n_ev = sum(len(e["events"]) for e in results)
    n_err = sum(1 for e in results for ev in e["events"] if ev["error"])
    cost = round(sum(e["summary"]["usage"]["cost_usd"] for e in results), 4)
    print(f"wrote {out_path}: {len(results)} records, {n_ev} events, "
          f"{n_err} errors, ${cost} ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--results", default=os.path.join(_CD, "results_ipi"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heads", default="ipi,policy")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chunk", type=int, default=400,
                    help="records per gather batch (bounds peak prompt memory)")
    ap.add_argument("--resume", action="store_true",
                    help="skip attack_ids already in results/<defense_id>.jsonl.partial")
    ap.add_argument("--out-suffix", default=None,
                    help="write results/<defense_id>.<suffix>.jsonl (multi-sample passes)")
    ap.add_argument("--cache-salt", default=None,
                    help="salt the cache key so a repeated pass draws fresh samples "
                         "instead of replaying cached draws (use with --out-suffix)")
    args = ap.parse_args()

    entry = load_defense(args.defense)
    if entry.get("defense_type") != "llm_judge_openrouter":
        raise SystemExit(f"{args.defense} is not an llm_judge_openrouter defense")

    rows = []
    with open(args.store) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    if args.dry_run:
        dry_run(rows, entry["params"], set(args.heads.split(",")), entry["defense_id"])
        return
    asyncio.run(run_async(rows, entry, args))


if __name__ == "__main__":
    main()
