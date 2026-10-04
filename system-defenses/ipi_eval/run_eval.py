"""Run a defended (or baseline) rollout eval over the val set, then judge.

For each val record: replay the prefill, continue the rollout on local Qwen-32B with the
defense active, then grade with the ported Arena panel judge. Writes per-record results and
prints ASR overall + per corpus.

NOTE: the JUDGE defaults to Gemini 3 Flash via OpenRouter (one shared frontier judge across all
harnesses; key from secrets.md) — override with --judge-model/--judge-base-url. World-sim still
runs on the local target model (it simulates this agent's environment, not the grader).

Usage:
  python -m ipi_eval.run_eval --defense baseline --limit 50 --workers 8
  python -m ipi_eval.run_eval --defense firewalls
"""
from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # system-defenses/ (for results_io)
import results_io

from . import behaviors as B
from . import engine
from . import judges
from .adapters import get_defense

VALSET = Path(__file__).parent / "valset" / "ipi_defense_valset.jsonl"


def _load_valset(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def _eval_one(rec: dict, defense_name: str, target_client, worldsim_client, judge_chat,
              max_steps: int, live_carrier: bool, sid: str, run_idx: int,
              defense_id: str, target_model: str, defense_kwargs: dict,
              arm: str | None = None) -> dict:
    import time
    # public-format records (public_data.py) carry their converted behavior inline
    behavior = rec.get("behavior") or B.get_behavior(rec["behavior_id"])
    defense = get_defense(defense_name, **defense_kwargs)
    t0 = time.monotonic()
    # `arm` labels which experimental arm this record belongs to (e.g. one strategy cell of a
    # multi-arm panel). results_io.make_record has always had the slot; nothing populated it, so
    # every row read `arm: null` and the arm had to be recovered from a side-car hash->arm map.
    common = dict(defense_id=defense_id, target_model=target_model, run_idx=run_idx,
                  attack_id=rec["attack_id"], behavior_id=rec["behavior_id"], corpus=rec["corpus"],
                  attack_hash=rec.get("attack_hash"), arm=arm or rec.get("arm"))
    roll: dict | None = None   # bound before the try so the error path can still save partial work
    try:
        roll = engine.run_rollout(
            behavior, rec["prefill_messages"], rec["user_request"],
            defense, target_client, worldsim_client, max_steps=max_steps, record=rec,
            live_carrier=live_carrier,
        )
        # The arena tool judge scans the whole conversation; the public benchmark judges only the
        # calls made in the rollout, so public-format records exclude the prefill.
        verdict = judges.score_behavior(
            behavior, roll["generated_messages"], judge_chat,
            full_messages=roll["generated_messages"] if "behavior" in rec else roll["messages"],
        )
        # non-live-carrier = injection replayed in the prefill -> delivered by construction.
        delivered = roll.get("carrier_delivered") if live_carrier else True
        # `status` is "empty" whenever the model generated no text, which conflates two very
        # different things. `content_filtered` separates them: a safety-classifier stop is a
        # REFUSED attack (a real non-break that must stay in the denominator), whereas an empty
        # generation with no filter stop is a measurement loss. Without this the two are
        # indistinguishable in the saved record, and excluding refusals inflates ASR for exactly
        # the models that refuse most.
        return results_io.make_record(
            sid, **common, is_break=verdict["is_break"], delivered=bool(delivered),
            status=verdict["status"], stop_reason=roll.get("stop_reason"),
            criteria=verdict["criteria"], defense_actions=roll.get("decisions", []),
            transcript=roll["messages"], generated_messages=roll["generated_messages"],
            detail={"finish_reasons": roll.get("finish_reasons", []),
                    "content_filtered": bool(roll.get("content_filtered"))},
            elapsed_s=round(time.monotonic() - t0, 2))
    except Exception as e:  # one bad rollout shouldn't kill the run
        # A moderation block is a REFUSAL, not a harness failure, and different vendors deliver it
        # differently: Anthropic returns finish_reason="content_filter" (handled in run_rollout),
        # while Azure/OpenAI return HTTP 200 with `choices: null` and "Request blocked." in the body,
        # which reaches us as an exception. Classifying the second as an `error` would EXCLUDE it
        # while the first stays in the denominator -- an asymmetry that inflates ASR for whichever
        # vendor happens to signal refusals as errors. Both are non-breaks that must be counted.
        msg = f"{e}"
        moderated = any(s in msg.lower() for s in (
            "request blocked", "content_filter", "content management policy",
            "responsibleaipolicyviolation", "content filter"))
        return results_io.make_record(
            sid, **common, is_break=False,
            status="empty" if moderated else "error",
            stop_reason="content_filter" if moderated else "error",
            # keep whatever the rollout produced before it died -- previously discarded, which made
            # the failure impossible to reclassify or diagnose after the fact.
            transcript=(roll or {}).get("messages"),
            generated_messages=(roll or {}).get("generated_messages"),
            detail={"content_filtered": moderated, "finish_reasons": []},
            error=msg, trace=traceback.format_exc()[-1500:],
            elapsed_s=round(time.monotonic() - t0, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--valset", default=str(VALSET))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="agent rollout sampling temperature (default 0.0 = paper parity; pass 0.6 for a "
                         "deployment-realism arm; world-sim+judge always 0.0)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="target-only nucleus sampling (OpenRouter extra_body); e.g. Qwen non-thinking 0.8")
    ap.add_argument("--top-k", type=int, default=None,
                    help="target-only top-k sampling (OpenRouter extra_body); e.g. Qwen non-thinking 20")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--arena-judge", action="store_true",
                    help="judge each behavior with the provider+model its OWN behaviors_detailed "
                         "panel entry declares (2025 openrouter/gemini-2.5-flash, Q1/Q2 "
                         "anthropic/claude-haiku-4-5) instead of one shared judge. Use for arena "
                         "fidelity; keep the shared judge when comparing defenses to each other.")
    ap.add_argument("--worldsim-model", default="gpt-5-mini",
                    help="tool-output world-sim model (a non-target aux). Defaults to gpt-5-mini — "
                         "the arena's own world-sim default, and cheaper than Gemini 3 Flash "
                         "($0.25/$2.00 vs $0.50/$3.00 per M). It is not the model-under-attack; pass "
                         "the target model+base-url to reproduce 'world-sim = target' behavior.")
    ap.add_argument("--worldsim-base-url", default="https://api.openai.com/v1",
                    help="matches --worldsim-model's default: OpenAI models go native (see "
                         "engine._NATIVE_ROUTES), not via OpenRouter.")
    ap.add_argument("--target-thinking", action="store_true",
                    help="enable native reasoning on the attacked target (default OFF to save cost). "
                         "OFF matches most papers' headline non-thinking target; ON ~ CaMeL/FIDES "
                         "reasoning arms.")
    ap.add_argument("--target-reasoning-effort", default=None,
                    choices=["floor", "off", "low", "medium", "high"],
                    help="graded reasoning effort on the target via OpenRouter's unified "
                         "reasoning.effort (native for OpenAI/Grok; mapped to a thinking-budget for "
                         "Gemini/Qwen). 'off' = reasoning disabled (== --target-thinking absent); "
                         "low/medium/high imply reasoning ON + max_tokens bumped to 8192. Used by the "
                         "reasoning-effort ablation on union381. "
                         "'floor' = the LOWEST level that actually works for this model, resolved "
                         "per-model from reasoning_floors.py — use this instead of 'off' for any "
                         "multi-model sweep: a uniform 'off' 400s on some endpoints and is SILENTLY "
                         "IGNORED on others (deepseek-r1, qwen3-235b-a22b-thinking-2507), which "
                         "would mislabel a reasoning-ON arm as reasoning-off.")
    ap.add_argument("--provider-only", default=None,
                    help="comma-separated OpenRouter provider SLUGS to pin the target to, e.g. "
                         "'azure' or 'siliconflow,deepinfra'. OpenRouter picks an endpoint per "
                         "REQUEST, so without a pin one run can be answered at several different "
                         "quantizations (measured: glm-5.2 drifted across 3 providers in 8 identical "
                         "calls) and can land on an endpoint lacking tool support entirely. Pin for "
                         "any multi-record sweep whose ASR is pooled into one number.")
    ap.add_argument("--provider-allow-fallbacks", action="store_true", default=False,
                    help="let OpenRouter fall back outside --provider-only. Default OFF so a pin "
                         "that stops being served fails loudly instead of silently drifting.")
    ap.add_argument("--strict-messages", action="store_true", default=False,
                    help="send the target ONLY the OpenAI-legal message keys (role/content/"
                         "tool_calls/tool_call_id/name), dropping the replayed arena bookkeeping "
                         "(id, parentId, created_at, display, done, containsIPI, isHarmfulAction, "
                         "isIndirectPromptToolMessage). Forced ON for openai/* targets, whose "
                         "Responses API 400s on our UUID message ids. Default OFF so every "
                         "existing arm's payload stays byte-identical.")
    ap.add_argument("--cache-prefix", action="store_true", default=False,
                    help="mark the last message with an ephemeral cache_control breakpoint so the "
                         "provider caches the replayed prefix (~10x cheaper on re-read). Only "
                         "needed for ANTHROPIC targets, which do no implicit caching; OpenAI/xAI/"
                         "Google cache automatically. Default OFF (payload-identical to prior runs).")
    ap.add_argument("--worldsim-thinking", action="store_true",
                    help="enable native reasoning on the tool-output world-sim (default OFF — the "
                         "Arena world-sim externalizes its reasoning into the prompt/content, so "
                         "native thinking is unnecessary and just burns tokens).")
    ap.add_argument("--live-carrier", action="store_true", default=False,
                    help="make the CARRIER tool call the defended model's first LIVE action instead "
                         "of replaying it as fixed prefill (exposes it to the review/inspect seams, "
                         "esp. Firewalls' Minimizer). Default OFF = guaranteed-delivery replay path. "
                         "ON => delivery is no longer guaranteed; read ASR delivered-gated via "
                         "carrier_delivered.")
    ap.add_argument("--arm", default=None,
                    help="experimental-arm label stamped on every record (results_io 'arm'). Use "
                         "for multi-arm panels so each row self-describes which arm it came from, "
                         "instead of needing a side-car attack_hash->arm map.")
    ap.add_argument("--store-id", default=None,
                    help="results store id (default: <model_key>__<defense-base>, e.g. glm__causalarmor). "
                         "Results go to results/<store-id>/run-<k>.jsonl.")
    ap.add_argument("--run-idx", type=int, default=None,
                    help="run index for repeated full-sample runs (default: next free index). "
                         "Each run is its own run-<k>.jsonl; summarize() merges toward n=4.")
    ap.add_argument("--defense-kwarg", action="append", default=[], metavar="K=V",
                    help="extra kwarg forwarded to the defense constructor (repeatable). Firewalls "
                         "aux=Gemini: --defense-kwarg model=google/gemini-3-flash-preview "
                         "--defense-kwarg base_url=https://openrouter.ai/api/v1 . CausalArmor proxy: "
                         "--defense-kwarg proxy_model=... --defense-kwarg base_url=http://localhost:8002/v1 .")
    ap.add_argument("--out", default=None, help="override the run file path (else results/<store-id>/run-<k>.jsonl)")
    args = ap.parse_args()
    defense_kwargs = dict(kv.split("=", 1) for kv in args.defense_kwarg)

    if args.target_reasoning_effort == "floor":
        # Record WHY this arm ran at the level it did — an unknown model silently defaults to
        # 'off', which is the one case that can mislabel a reasoning-ON arm as reasoning-off.
        from .reasoning_floors import describe
        print(f"[reasoning] {describe(args.model)}")

    target_client = engine.make_client(args.model, args.base_url, temperature=args.temperature,
                                       thinking=args.target_thinking, top_p=args.top_p, top_k=args.top_k,
                                       reasoning_effort=args.target_reasoning_effort,
                                       strict_messages=args.strict_messages,
                                       cache_prefix=args.cache_prefix,
                                       provider_only=(args.provider_only.split(",")
                                                      if args.provider_only else None),
                                       provider_allow_fallbacks=args.provider_allow_fallbacks)
    worldsim_client = engine.make_client(args.worldsim_model, args.worldsim_base_url,
                                         thinking=args.worldsim_thinking)
    judge_chat = (engine.ArenaJudgeRouter() if args.arena_judge else
                  engine.make_judge_chat(engine.make_judge_client(args.judge_model, args.judge_base_url)))

    defense_base = args.defense.split("-")[0]  # melon-aug->melon, causalarmor-nocot->causalarmor
    sid = args.store_id or results_io.store_id(args.model, defense_base)
    run_idx = args.run_idx if args.run_idx is not None else results_io.next_run_idx(sid)
    writer = results_io.RunWriter(sid, run_idx)
    if args.out:
        writer.path = Path(args.out)
        writer.path.parent.mkdir(parents=True, exist_ok=True)
    print(f"store={sid} run={run_idx} -> {writer.path}\n", flush=True)

    rows = _load_valset(Path(args.valset), args.limit)
    n_break = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_eval_one, r, args.defense, target_client, worldsim_client,
                          judge_chat, args.max_steps, args.live_carrier, sid, run_idx,
                          defense_base, args.model, defense_kwargs, args.arm) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            writer.append(rec)  # incremental + flock + fsync -> crash-safe, resumable
            n_break += int(rec["is_break"])
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} done | breaks so far: {n_break}", flush=True)

    s = results_io.summarize(sid)  # merges ALL runs present for this store
    print(f"\n=== {sid} (run {run_idx}) ===  n_runs={s['n_runs']}  n_records={s['n_records']}  "
          f"ASR(pooled)={s['asr_pooled']:.3f}  delivered-gated={s['asr_delivered_gated']}  errors={s['errors']}")
    for c, d in sorted(s["by_corpus"].items()):
        print(f"  {c:14s} ASR={d['asr']:.3f} (n={d['n']})")
    print(f"  stop_reasons: {s['stop_reason_hist']}")
    print(f"wrote {writer.path}  +  {writer.dir / 'summary.json'}")


if __name__ == "__main__":
    main()
