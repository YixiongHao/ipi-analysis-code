"""Static-plan eval: replay each behavior's SAVED program across many attack strings.

The saved program (build_plans.py) is injection-independent, so for each behavior we swap every
attack string into the carrier's output and replay the fixed program (cheap: no P-LLM codegen,
just interpret + live Q-LLM). Delivery is determined by the plan (consistent across a behavior's
attacks); what varies per attack is the injected content the Q-LLM processes and what the judge
sees. Writes one transcript per (behavior, attack) to plan_eval/<bid>.<arm>.jsonl; grade with
grade_plan_eval.py (master venv).

Runs in the reference repo's uv env:
    cd system-defenses/CaMeL/ipi_suites
    uv run --project ../camel-prompt-injection python run_plan_eval.py \
        [--arm isolation|policy] [--model gemini-3-flash] [--per-behavior 25] \
        [--workers 6] [--delivered-only] [--only bid1,bid2]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import harness

sys.path.insert(0, "system-defenses")
from ipi_suites_worldsim import make_worldsim_fallback  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"
_PACKETS = _HERE / "packets"
_STORE = Path("classifier-defenses/store/attacks.jsonl")
_MANIFEST = _HERE.parent.parent / "verification_2026_06_28" / "canonical_attacks.json"
_OUT = _HERE / "plan_eval"


def _attacks_by_behavior(behaviors: set[str], per: int) -> dict[str, list[dict]]:
    """Up to `per` attack strings per behavior from the store (first-N, reproducible)."""
    out: dict[str, list[dict]] = {b: [] for b in behaviors}
    for line in open(_STORE):
        r = json.loads(line)
        b = r.get("behavior_id")
        if b in out and len(out[b]) < per and r.get("attack_text"):
            out[b].append({"attack_id": r.get("attack_id"), "attack_text": r["attack_text"]})
    return out


def _packet_attack(behaviors: set[str], pkts: dict) -> dict[str, list[dict]]:
    """Single canonical attack per behavior = the packet's own injection (identical to what
    the FIDES/IPIGuard forks use; == the ipi_eval canonical for 86/88, carrier-appropriate for
    the 2 log-rotation exceptions). attack_id taken from the shared canonical manifest."""
    man = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else {}
    out = {}
    for b in behaviors:
        aid = (man.get(b, {}).get("forks", {}) or {}).get("attack_id") or f"packet:{b}"
        out[b] = [{"attack_id": aid, "attack_text": pkts[b]["injection"]}]
    return out


def _attacks_from_json(path: str, behaviors: set[str]) -> dict[str, list[dict]]:
    """Explicit {behavior_id: [injection_string, ...]} source (e.g. the hard-sample 381 set).
    attack_hash = md5(injection) is the cross-arm join key (matches build_transfer_store's ahash and
    the adapter valset); attack_id is a stable per-behavior index. The saved plan is injection-blind,
    so all of a behavior's strings replay the same program with the carrier output swapped."""
    import hashlib
    raw = json.loads(Path(path).read_text())
    out: dict[str, list[dict]] = {b: [] for b in behaviors}
    for b, strings in raw.items():
        if b not in out:
            continue
        for i, s in enumerate(strings):
            out[b].append({"attack_id": f"{b}#u{i:03d}", "attack_text": s,
                           "attack_hash": hashlib.md5(s.encode()).hexdigest()})
    return out


def _worldsim(pkt: dict):
    """Per-packet world-sim fallback; None on any failure (no key, or deps missing in this env —
    the camel replay venv lacks ipi_arena_bench) -> keep the '{"status":"ok"}' stub."""
    try:
        return make_worldsim_fallback(pkt)
    except Exception:
        return None


def _eval_one(plan: dict, pkt: dict, attack: dict, model_cfg: dict, arm: str,
              cap_s: float = 120.0, worldsim=None, all_worldsim: bool = False) -> dict:
    """Replay the saved program with one attack's text swapped into the carrier output.

    Hard wall-clock cap: pydantic_ai's Q-LLM (`run_sync`) has NO request timeout, so a stalled
    Gemini call would block a worker forever (this is what hung the first 22h run). Run the replay
    on a worker thread and on timeout record a clean `status=timeout` result. The orphaned thread
    can't be killed; it dies with the process (the openai SDK's own retry/timeout bounds it)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FT
    pkt2 = {**pkt, "injection": attack["attack_text"]}
    base = {"behavior_id": plan["behavior_id"], "corpus": pkt.get("corpus"),
            "attack_id": attack["attack_id"], "attack_hash": attack.get("attack_hash"),
            "arm": arm, "carrier": plan["carrier"], "model": model_cfg["model"]}
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        rep = ex.submit(harness.replay_programs, plan["programs"], pkt2, model_cfg, arm,
                        worldsim=worldsim, all_worldsim=all_worldsim).result(timeout=cap_s)
        ex.shutdown(wait=False)
    except _FT:
        ex.shutdown(wait=False)
        return {**base, "executed_calls": [], "injection_delivered": False,
                "policy_blocked": False, "final_text": None, "status": f"timeout>{cap_s:.0f}s"}
    return {**base, "executed_calls": rep["executed_calls"],
            "injection_delivered": rep.get("injection_delivered", False),
            "policy_blocked": rep.get("policy_blocked", False), "final_text": rep.get("final_text")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="isolation", choices=["isolation", "policy"])
    ap.add_argument("--model", default="gemini-3-flash")
    ap.add_argument("--per-behavior", type=int, default=25)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delivered-only", action="store_true",
                    help="only eval behaviors whose saved plan replay-delivered")
    ap.add_argument("--use-packet-injection", action="store_true",
                    help="use the packet's own injection as the single canonical attack per behavior "
                         "(matches the FIDES/IPIGuard forks); overrides --per-behavior store sampling")
    ap.add_argument("--strings-json", default=None,
                    help="explicit {behavior_id:[injection,...]} source (e.g. hard_sample/eval_strings_381union.json); "
                         "runs every provided string per behavior (attack_hash=md5); overrides store sampling")
    ap.add_argument("--out-dir", default=None, help="override plan_eval output dir (isolate a run)")
    ap.add_argument("--thinking-off", action="store_true",
                    help="disable reasoning for the replay Q-LLM (e.g. GLM's `glm` preset defaults "
                         "thinking on for the planner)")
    ap.add_argument("--only", default="")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="quarantined-LLM (the only GLM call at replay) sampling temperature "
                         "(default 0.0 = greedy/paper parity; pass 1.0 for a hot-agent arm — NB the "
                         "replay outcome is temperature-invariant, the plan is frozen)")
    ap.add_argument("--no-worldsim", action="store_true",
                    help="disable the arena world-sim fallback for un-canned non-carrier tools "
                         "(keep the '{\"status\":\"ok\"}' stub); world-sim is ON by default. NB the "
                         "camel replay venv lacks ipi_arena_bench, so world-sim degrades to the stub "
                         "there unless that dep is installed.")
    ap.add_argument("--all-worldsim", dest="all_worldsim", action="store_true", default=True,
                    help="fully-simulated env (DEFAULT): answer EVERY non-carrier tool with the "
                         "goal-aware arena world-sim (IGNORE the recorded canned outputs); the "
                         "carrier still returns the injection. Implies world-sim ON. Opt out with "
                         "--no-all-worldsim. (Degrades to canned when ipi_arena_bench is absent "
                         "from the replay venv.)")
    ap.add_argument("--no-all-worldsim", dest="all_worldsim", action="store_false",
                    help="restore the recorded-canned replay: canned outputs primary, world-sim only "
                         "as a fallback for un-canned non-carrier tools.")
    args = ap.parse_args()

    if args.all_worldsim and args.no_worldsim:
        print("NB: --all-worldsim overrides --no-worldsim (world-sim ON, fully simulated)", flush=True)

    cfg = harness.model_preset(args.model)
    cfg["temperature"] = args.temperature
    if args.thinking_off:
        # replay runs only the frozen-plan Q-LLM (the attacked target) — reasoning off is faithful
        # + ~2.4x faster; planner reasoning is untouched because build_plans has its own preset path.
        cfg["thinking_off"] = True
    harness.configure_model(cfg["base_url"], cfg["api_key"])
    global _OUT
    if args.out_dir:
        _OUT = Path(args.out_dir)
    _OUT.mkdir(parents=True, exist_ok=True)

    only = {b for b in args.only.split(",") if b}
    if args.strings_json:  # restrict to the behaviors present in the explicit strings source
        sj_behaviors = set(json.loads(Path(args.strings_json).read_text()).keys())
        only = (only & sj_behaviors) if only else sj_behaviors
    plans = []
    for fp in sorted(_PLANS.glob("*.json")):
        plan = json.loads(fp.read_text())
        if only and plan["behavior_id"] not in only:
            continue
        if not plan.get("programs"):
            continue
        if args.delivered_only and not plan.get("replay_delivered"):
            continue
        plans.append(plan)

    behaviors = {p["behavior_id"] for p in plans}
    pkts = {b: json.loads((_PACKETS / f"{b}.json").read_text()) for b in behaviors}
    # One world-sim fallback per behavior (shared across its attack replays); None when disabled or
    # unavailable in this env -> the '{"status":"ok"}' stub. Threads share these safely. all_worldsim
    # implies world-sim ON (overrides --no-worldsim).
    _ws_off = args.no_worldsim and not args.all_worldsim
    ws_by_bid = {b: (None if _ws_off else _worldsim(pkts[b])) for b in behaviors}
    attacks = (_attacks_from_json(args.strings_json, behaviors) if args.strings_json
               else _packet_attack(behaviors, pkts) if args.use_packet_injection
               else _attacks_by_behavior(behaviors, args.per_behavior))
    jobs = [(p, pkts[p["behavior_id"]], a) for p in plans for a in attacks[p["behavior_id"]]]
    print(f"replaying {len(jobs)} (behavior,attack) pairs over {len(plans)} plans | arm={args.arm} "
          f"model={cfg['model']} per-behavior<={args.per_behavior} workers={args.workers}\n", flush=True)

    # fresh output files (truncate any stale per-behavior jsonl for this arm)
    for b in behaviors:
        (_OUT / f"{b}.{args.arm}.jsonl").write_text("")

    # Write each record to its per-behavior jsonl AS IT COMPLETES (under a lock) so a kill never
    # loses progress — the old "collect all, write at end" lost everything when the run hung.
    import threading
    lock = threading.Lock()
    counts = {"done": 0, "deliv": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_eval_one, p, pk, a, cfg, args.arm,
                          worldsim=ws_by_bid.get(p["behavior_id"]), all_worldsim=args.all_worldsim):
                p["behavior_id"] for (p, pk, a) in jobs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                bid = futs[fut]
                rec = {"behavior_id": bid, "attack_id": "?", "arm": args.arm,
                       "executed_calls": [], "injection_delivered": False, "error": str(e)[:160]}
            with lock:
                with open(_OUT / f"{rec['behavior_id']}.{args.arm}.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                counts["done"] += 1
                counts["deliv"] += int(rec.get("injection_delivered", False))
                if counts["done"] % 25 == 0:
                    print(f"  {counts['done']}/{len(jobs)} done (delivered {counts['deliv']})", flush=True)

    print(f"\n=== replayed {counts['done']} transcripts | delivered {counts['deliv']} "
          f"=== wrote plan_eval/*.{args.arm}.jsonl")


if __name__ == "__main__":
    main()
