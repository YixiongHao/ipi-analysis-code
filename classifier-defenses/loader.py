#!/usr/bin/env python3
"""Build the detector-store manifest: stream the 3 arena corpora, keep
originally-successful injection-bearing trajectories, normalize each into one
combined record, and write a stratified sample (corpus x behavior).

Combined record (the unit of the detector store — see
planning/detector-results-store.md). One self-contained row per attack; detector
outputs are NOT stored here — each detector writes a per-defense sidecar
`results/<defense_id>.jsonl` keyed by `attack_id` (see run_detector.py):

  {
    "attack_id": "ipi_2026_q1:<submission_id>",   # PK, globally unique
    "corpus": "ipi_2025|ipi_2026_q1|ipi_2026_q2",
    "behavior_id": ..., "model_name": ..., "orig_success": true,
    "attack_hash": ...,
    "source": {"file": ..., "submission_id": ..., "byte_offset": <int|null>},
    "record": {                                   # what the detectors consume
        "behavior_id","model_name","attack_hash","submission_id","successful",
        "system_prompt","user_request","agent_messages"   # agent_messages carry
    }                                             #   isIndirectPromptToolMessage
  }

Per-corpus normalization (the 3 corpora have evolved schemas):
  - Q1/Q2 mark the injected tool message with `isIndirectPromptToolMessage`.
  - 2025 has no marker; the injected message is `messages[fill_message_index]`,
    which we mark synthetically so the shared `_events` finds it.
"""
import argparse
import hashlib
import json
import os

ARENA = "arena-data"


def _md5(s):
    return hashlib.md5((s or "").encode("utf-8", "replace")).hexdigest()


def _sys_prompts(behaviors_path):
    """behavior id -> system prompt string (from spec system_prompt_config.content)."""
    out = {}
    for b in json.load(open(behaviors_path)):
        bid = b.get("id") or b.get("behavior_id") or b.get("name")
        spc = b.get("system_prompt_config") or {}
        out[bid] = spc.get("content") if isinstance(spc, dict) else None
    return out


def _area_map(behaviors_path):
    """behavior id -> area_name (Tool Use / Coding[ Agent] / Computer Use). Q1/Q2 only."""
    return {b.get("id"): b.get("area_name") for b in json.load(open(behaviors_path))}


def _combined(corpus, sid, behavior_id, model_name, attack_hash, system_prompt,
              user_request, agent_messages, src_file, byte_offset, attack_text,
              marker_present):
    return {
        "attack_id": f"{corpus}:{sid}",
        "corpus": corpus,
        "behavior_id": behavior_id,
        "model_name": model_name,
        "orig_success": True,
        "attack_hash": attack_hash,
        "attack_text": attack_text,
        "source": {"file": src_file, "submission_id": sid, "byte_offset": byte_offset},
        "record": {
            "behavior_id": behavior_id, "model_name": model_name,
            "attack_hash": attack_hash, "submission_id": sid, "successful": True,
            # Whether a ground-truth injection site is known. When False (e.g. Q1
            # Computer Use, or Q2 coding/CU where the payload couldn't be located),
            # the grader has no single site to check -> assume any block is correct.
            "injection_marker_present": marker_present,
            "text_injection_present": _text_injection_present(agent_messages, attack_text),
            "system_prompt": system_prompt, "user_request": user_request,
            "agent_messages": agent_messages,
        },
        # detector outputs live in per-defense sidecars (results/<defense_id>.jsonl),
        # joined back by attack_id — see run_detector.py / validate.py.
    }


def _injected_content(agent_messages):
    for m in agent_messages:
        if m.get("isIndirectPromptToolMessage"):
            return m.get("content") or ""
    return ""


def _text_injection_present(agent_messages, attack_text):
    """True iff the MARKED injection message actually contains the payload as text.
    False for image-only injections (e.g. 2025 computer-use, where the payload was
    rendered into a screenshot and the transcript carries only a stub) -> the text
    detectors can't see those, so they're excluded from scoring downstream."""
    probe = (attack_text or "").strip()[:80]
    if not probe:
        return False
    for m in agent_messages:
        if m.get("isIndirectPromptToolMessage") and isinstance(m.get("content"), str) and probe in m["content"]:
            return True
    return False


# ---- per-corpus streamers: yield combined records for successful injected trajectories ----

def stream_q1(max_scan, keep_areas=None):
    """Q1: JSONL; success = corrected_grade_status. Tool-Use/Coding behaviors carry
    isIndirectPromptToolMessage; Computer Use does NOT — it loads as-is (marker
    optional), with every tool response treated as a candidate site downstream.
    keep_areas (set of area_names) filters by behavior category if given."""
    f_path = f"{ARENA}/ipi_2026_q1/dedup_submissions_with_messages.jsonl"
    sysp = _sys_prompts(f"{ARENA}/ipi_2026_q1/behaviors_detailed.json")
    amap = _area_map(f"{ARENA}/ipi_2026_q1/behaviors_detailed.json") if keep_areas else None
    with open(f_path) as f:
        for n, line in enumerate(f):
            if max_scan is not None and n >= max_scan:
                break
            off = None  # f.tell() unreliable mid-iteration; resolve by id if needed
            if '"corrected_grade_status": "success"' not in line:
                continue
            r = json.loads(line)
            if keep_areas and amap.get(r.get("behavior_id")) not in keep_areas:
                continue
            if r.get("corrected_grade_status") != "success":
                continue
            msgs = r.get("messages") or []
            if not msgs or msgs[0].get("role") != "user":
                continue
            agent = msgs[1:]
            marker_present = any(m.get("isIndirectPromptToolMessage") for m in agent)
            bid = r.get("behavior_id")
            yield bid, _combined(
                "ipi_2026_q1", r.get("submission_id"), bid, r.get("model_name"),
                r.get("attack_hash"), sysp.get(bid), msgs[0].get("content") or "",
                agent, f_path, off, _injected_content(agent), marker_present)


_Q2_PAYLOADS = None  # lazy chat_id -> [injection slot strings], from chats_slim


def _q2_payload_map():
    """chat_id -> list of potemkin_payload slot strings (the injected texts).
    Built once from chats_slim.jsonl; used to mark the injection site in Q2
    coding/computer-use records that lack `isIndirectPromptToolMessage`."""
    global _Q2_PAYLOADS
    if _Q2_PAYLOADS is None:
        _Q2_PAYLOADS = {}
        with open(f"{ARENA}/ipi_2026_q2/chats_slim.jsonl") as f:
            for line in f:
                if '"potemkin_payload"' not in line:
                    continue
                c = json.loads(line)
                pp = c.get("potemkin_payload") or {}
                vals = [v for v in pp.values() if isinstance(v, str) and v.strip()]
                if vals:
                    _Q2_PAYLOADS[c.get("_id")] = vals
    return _Q2_PAYLOADS


def _q2_synthesize(agent_msgs, payloads):
    """Coding/CU path: the rollout is one assistant turn whose `tool_calls` embed
    both the agent's ACTIONS (function.arguments) and its OBSERVATIONS
    (displayMessage = the tool output). Re-materialize the conventional
    [assistant tool_call -> tool result] sequence so the shared `_events` works:
    each non-scaffold tool_call becomes an assistant action message + a tool
    message whose content is its displayMessage; a tool message is marked
    isIndirectPromptToolMessage when its output contains an injection payload."""
    probes = [p.strip()[:80] for p in (payloads or []) if p and p.strip()]
    out = []
    for m in agent_msgs:
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {}).get("name")
            if fn == "status":            # opencode scaffold step marker, not a tool
                continue
            tcid = tc.get("id")
            out.append({"role": "assistant", "tool_calls": [
                {"id": tcid, "type": "function",
                 "function": {"name": fn, "arguments": (tc.get("function") or {}).get("arguments")}}]})
            dm = tc.get("displayMessage") or ""
            tool_msg = {"role": "tool", "content": dm, "tool_call_id": tcid}
            if probes and any(p in dm for p in probes):
                tool_msg["isIndirectPromptToolMessage"] = True
            out.append(tool_msg)
    return out


def stream_q2(max_scan, keep_areas=None):
    """Q2: JSONL; success = dedup.corrected_grade_status. Two shapes:
      - Tool-Use behaviors carry real role:tool messages with isIndirectPromptToolMessage.
      - Coding/Computer-Use behaviors collapse the rollout into one assistant turn
        whose tool_calls embed actions + observations (displayMessage); we synthesize
        the transcript and mark the injection site via potemkin_payload (see _q2_synthesize).
    keep_areas (set of area_names) filters by behavior category if given."""
    f_path = f"{ARENA}/ipi_2026_q2/dedup_submissions_full.jsonl"
    sysp = _sys_prompts(f"{ARENA}/ipi_2026_q2/behaviors_detailed.json")
    amap = _area_map(f"{ARENA}/ipi_2026_q2/behaviors_detailed.json") if keep_areas else None
    with open(f_path) as f:
        for n, line in enumerate(f):
            if max_scan is not None and n >= max_scan:
                break
            if '"corrected_grade_status": "success"' not in line:
                continue
            r = json.loads(line)
            if (r.get("dedup") or {}).get("corrected_grade_status") != "success":
                continue
            if keep_areas and amap.get(r.get("behavior")) not in keep_areas:
                continue
            msgs = r.get("messages") or []
            if not msgs or msgs[0].get("role") != "user":
                continue
            agent = msgs[1:]
            if any(m.get("isIndirectPromptToolMessage") for m in agent):
                final_agent = agent                       # tool-use path: marked already
            else:                                          # coding/CU path: synthesize from displayMessage
                asst = [m for m in agent if m.get("role") == "assistant" and m.get("tool_calls")]
                # synthesize the interleaved transcript when there are inline tool_calls;
                # otherwise keep the rollout as-is (never drop).
                final_agent = _q2_synthesize(asst, _q2_payload_map().get(r.get("chat_id"))) if asst else agent
            # marker may be native (tool-use), payload-located (coding/CU), or absent
            # (payload not found in any displayMessage) -> load anyway, marker optional.
            marker_present = any(m.get("isIndirectPromptToolMessage") for m in final_agent)
            bid = r.get("behavior")
            yield bid, _combined(
                "ipi_2026_q2", r.get("_id"), bid, r.get("model_name"),
                (r.get("dedup") or {}).get("attack_hash"),
                r.get("system_prompt") or sysp.get(bid), msgs[0].get("content") or "",
                final_agent, f_path, None, _injected_content(final_agent), marker_present)


def stream_2025(max_scan, keep_areas=None):
    """2025: JSON array; no marker -> synthesize on messages[fill_message_index].
    keep_areas (set) filters on the submission's `modality` field (2025's category
    signal: Tool Use / Coding Agent / Computer Use); None = all."""
    import ijson
    f_path = f"{ARENA}/ipi_2025/submissions.json"
    sysp = _sys_prompts(f"{ARENA}/ipi_2025/behaviors_detailed.json")
    with open(f_path, "rb") as f:
        for n, r in enumerate(ijson.items(f, "item")):
            if max_scan is not None and n >= max_scan:
                break
            if r.get("grade_status") != "success":
                continue
            if keep_areas and r.get("modality") not in keep_areas:
                continue
            msgs = r.get("messages") or []
            fi = r.get("fill_message_index")
            if not msgs or msgs[0].get("role") != "user":
                continue
            attack = r.get("attack") or ""
            # Mark the message that actually CONTAINS the payload (prefer
            # fill_message_index when it does; else content-match the transcript).
            # If no text message contains it -> image-only injection (screenshot);
            # keep the fill_message_index marker so the site still exists, but
            # text_injection_present will be False and it's excluded from scoring.
            probe = attack.strip()[:80]
            inj_idx = None
            if probe and isinstance(fi, int) and 1 <= fi < len(msgs) \
                    and isinstance(msgs[fi].get("content"), str) and probe in msgs[fi]["content"]:
                inj_idx = fi
            elif probe:
                for j in range(1, len(msgs)):
                    c = msgs[j].get("content")
                    if isinstance(c, str) and probe in c:
                        inj_idx = j
                        break
            if inj_idx is None:  # image-only / unlocatable: fall back to fill_message_index
                inj_idx = fi if (isinstance(fi, int) and 1 <= fi < len(msgs)) else None
            marker_present = inj_idx is not None
            if marker_present:
                msgs[inj_idx] = {**msgs[inj_idx], "isIndirectPromptToolMessage": True}
            agent = msgs[1:]
            bid = r.get("behavior_id")
            yield bid, _combined(
                "ipi_2025", r.get("submission_id"), bid, r.get("model"),
                r.get("attack_hash") or _md5(attack), sysp.get(bid),
                msgs[0].get("content") or "", agent, f_path, None, attack, marker_present)


def stratified(streamer, target, per_behavior_cap, max_scan, keep_areas=None):
    """Collect successes bucketed by behavior, then round-robin to `target`."""
    buckets = {}
    for bid, rec in streamer(max_scan, keep_areas=keep_areas):
        b = buckets.setdefault(bid, [])
        if len(b) < per_behavior_cap:
            b.append(rec)
    # round-robin across behaviors for diversity
    picked, order = [], sorted(buckets)
    i = 0
    while len(picked) < target and any(buckets[b] for b in order):
        b = order[i % len(order)]
        if buckets[b]:
            picked.append(buckets[b].pop(0))
        i += 1
    return picked, {b: len(v) for b, v in buckets.items()}


CORPORA = {"ipi_2025": stream_2025, "ipi_2026_q1": stream_q1, "ipi_2026_q2": stream_q2}


def _areas(spec):
    return {a.strip() for a in spec.split(",") if a.strip()} or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="total trajectories across all 3 corpora")
    ap.add_argument("--per-behavior-cap", type=int, default=6)
    ap.add_argument("--max-scan", type=int, default=80000, help="max records scanned per corpus")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "store", "attacks.jsonl"))
    ap.add_argument("--all", action="store_true",
                    help="take EVERY eligible record (ignore --n / --per-behavior-cap; scan all)")
    ap.add_argument("--q1-areas", default="", help="comma-sep area_names to INCLUDE for Q1 (default: all)")
    ap.add_argument("--q2-areas", default="", help="comma-sep area_names to INCLUDE for Q2 (default: all)")
    ap.add_argument("--2025-modalities", dest="m2025", default="",
                    help="comma-sep modalities to INCLUDE for 2025 (default: all)")
    args = ap.parse_args()

    keep = {"ipi_2025": _areas(args.m2025), "ipi_2026_q1": _areas(args.q1_areas),
            "ipi_2026_q2": _areas(args.q2_areas)}
    from collections import Counter

    # --all: stream every eligible record straight to disk (low memory, no cap).
    if args.all:
        by = Counter()
        with open(args.out, "w") as f:
            for corpus, streamer in CORPORA.items():
                for _bid, rec in streamer(None, keep_areas=keep[corpus]):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    by[corpus] += 1
                print(f"  {corpus:14} {by[corpus]} trajectories"
                      f"{' (areas ' + ','.join(sorted(keep[corpus])) + ')' if keep[corpus] else ''}")
        print(f"=== wrote {sum(by.values())} combined records -> {args.out} ===")
        return

    # Gather a diverse pool per corpus, then round-robin across corpora to N total.
    pools = {}
    for corpus, streamer in CORPORA.items():
        pool, _ = stratified(streamer, args.n, args.per_behavior_cap, args.max_scan, keep_areas=keep[corpus])
        pools[corpus] = pool
        print(f"  {corpus:14} pool {len(pool):>3} trajectories across "
              f"{len({r['behavior_id'] for r in pool})} behaviors")

    all_records, order, i = [], list(CORPORA), 0
    while len(all_records) < args.n and any(pools[c] for c in order):
        c = order[i % len(order)]
        if pools[c]:
            all_records.append(pools[c].pop(0))
        i += 1

    with open(args.out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_corpus = Counter(r["corpus"] for r in all_records)
    print(f"=== wrote {len(all_records)} combined records -> {args.out} ===")
    print(f"    by corpus: {dict(by_corpus)} | "
          f"behaviors: {len({(r['corpus'], r['behavior_id']) for r in all_records})}")


if __name__ == "__main__":
    main()
