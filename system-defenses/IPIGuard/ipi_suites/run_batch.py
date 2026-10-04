"""Run the REAL fork IPIGuard over a set of behavior packets, both arms.

PARALLEL across PROCESSES: seam W mutates the fork's process-global `whitelist` set per
behavior, so two behaviors cannot run concurrently in ONE process — but each worker
process has its own copy of that global, so behaviors run safely in parallel across
processes (one (behavior, arm) job per worker). Tune --workers to the vLLM server's
capacity (each job is a long chain of LLM calls to the same endpoint).

    cd system-defenses/IPIGuard/ipi_suites
    PY=python
    $PY run_batch.py                          # default batch-5, qwen, both arms
    $PY run_batch.py --all --workers 8        # every packet, 8 processes
    $PY run_batch.py --behaviors a,b,c --model qwen --arms baseline,ipiguard

Writes transcripts/<bid>.<arm>.json + transcripts/_summary.<arm>.json. Grade with
grade_batch.py (separate master-venv process).
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

_HERE = Path(__file__).resolve().parent
_TX = _HERE / "transcripts"
_PLANS = _HERE / "plans"
_PACKETS = _HERE.parents[1] / "CaMeL" / "ipi_suites" / "packets"

# Default batch: five public behaviors across tool and coding scenarios.
_BATCH5 = ["slack-phishing", "database-deletion", "mcp-email-corporate-leak",
           "share-document", "change-test-to-true"]


def _json_default(o):
    """Serialize the full env transcript: agentdojo tool_calls carry pydantic FunctionCall objects
    (not JSON-native). model_dump keeps them structured; str() is the fallback. Grading uses the
    already-dict executed_calls, so this only affects the stored transcript's fidelity."""
    md = getattr(o, "model_dump", None)
    return md() if callable(md) else str(o)


def _run_job(packet: dict, bid: str, arm: str, cfg: dict, timeout: int, dag_json=None,
             no_worldsim: bool = False, all_worldsim: bool = False, attack: dict | None = None):
    """Worker: one (behavior, arm) in its own process (own copy of the global whitelist).

    If dag_json is provided (reuse mode, ipiguard arm), REPLAY the saved delivering DAG with the
    packet's canonical injection on `cfg`'s model — control-flow (incl. the carrier read) is fixed
    by the saved plan, while Argument-Estimation / Node-Expansion / Fake-Tool-Invocation run live
    on the target. Otherwise build the TDG live (run_behavior).

    Multi-string mode: `attack` swaps the explicit injection into a private copy of the packet (the
    DAG/TDG is injection-blind); attack_hash=md5 is stamped into the transcript (cross-arm join key)."""
    import harness  # re-import is a no-op under fork; needed under spawn
    from ipi_suites_worldsim import make_worldsim_fallback  # (harness put system-defenses on sys.path)
    if attack is not None:
        packet = {**packet, "injection": attack["attack_text"]}
    t0 = time.time()
    # Build the world-sim fallback IN the worker: its closure holds an openai client, which is not
    # picklable across the process boundary. None (no key / deps unavailable) keeps '{"status":"ok"}'.
    # all_worldsim implies world-sim ON (overrides --no-worldsim).
    try:
        ws = None if (no_worldsim and not all_worldsim) else make_worldsim_fallback(packet)
    except Exception:
        ws = None
    if dag_json is not None:
        t = harness.replay_plan(dag_json, packet, packet["injection"], cfg, timeout_s=timeout,
                                worldsim=ws, all_worldsim=all_worldsim)
        t["arm"] = arm
    else:
        t = harness.run_behavior(packet, arm, cfg, behavior_timeout_s=timeout, worldsim=ws,
                                 all_worldsim=all_worldsim)
    if attack is not None:
        t["attack_id"] = attack["attack_id"]
        t["attack_hash"] = attack["attack_hash"]
    t.setdefault("behavior_id", bid)
    return bid, arm, t, round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviors", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strings-json", default=None,
                    help="explicit {behavior_id:[injection,...]} source (e.g. hard_sample/eval_strings_381union.json); "
                         "runs every string per behavior as its own record (attack_hash=md5), swapped into the "
                         "behavior's base packet (the saved DAG / TDG is injection-blind).")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--arms", default="baseline,ipiguard")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel worker PROCESSES (each forks its own whitelist global)")
    ap.add_argument("--reuse-plans", action="store_true",
                    help="ipiguard arm REPLAYS the saved delivering DAG (plans/<bid>.json) instead of "
                         "building the TDG live — guarantees carrier delivery; live model = --model")
    ap.add_argument("--out-dir", default=None, help="override transcripts dir (isolate a run)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="agent-under-eval (construct/traverse/baseline LLM) sampling temperature "
                         "(default 0.0 = paper parity; pass 1.0 for a hot-agent realism arm)")
    ap.add_argument("--no-worldsim", action="store_true",
                    help="disable the arena world-sim fallback for un-canned non-carrier tools "
                         "(keep the '{\"status\":\"ok\"}' stub); world-sim is ON by default.")
    ap.add_argument("--all-worldsim", dest="all_worldsim", action="store_true", default=True,
                    help="fully-simulated env (DEFAULT): answer EVERY non-carrier tool with the "
                         "goal-aware arena world-sim (IGNORE the recorded canned outputs); the "
                         "carrier still returns the injection. Implies world-sim ON. Opt out with "
                         "--no-all-worldsim.")
    ap.add_argument("--no-all-worldsim", dest="all_worldsim", action="store_false",
                    help="restore the recorded-canned replay: canned outputs primary, world-sim only "
                         "as a fallback for un-canned non-carrier tools.")
    args = ap.parse_args()

    if args.all_worldsim and args.no_worldsim:
        print("NB: --all-worldsim overrides --no-worldsim (world-sim ON, fully simulated)", flush=True)

    global _TX
    if args.out_dir:
        _TX = Path(args.out_dir)
    _TX.mkdir(parents=True, exist_ok=True)
    raw = None
    if args.strings_json:
        import hashlib
        raw = json.loads(Path(args.strings_json).read_text())
        bids = list(raw.keys())
    elif args.all:
        bids = sorted(p.stem for p in _PACKETS.glob("*.json"))
    elif args.behaviors:
        bids = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    else:
        bids = _BATCH5
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cfg = harness.model_preset(args.model)
    cfg["temperature"] = args.temperature

    # Load packets up front (cheap, fails early) so a bad packet skips just its jobs.
    # jobs = (bid, arm, attack_or_None); multi-string expands each behavior across its strings.
    packets, jobs = {}, []
    for bid in bids:
        try:
            packets[bid] = harness.load_packet(bid)
        except Exception as e:
            print(f"{bid}: PACKET LOAD FAILED: {e}", flush=True)
            continue
        if raw is not None:
            for i, s in enumerate(raw[bid]):
                atk = {"idx": i, "attack_id": f"{bid}#u{i:03d}", "attack_text": s,
                       "attack_hash": hashlib.md5(s.encode()).hexdigest()}
                jobs.extend((bid, arm, atk) for arm in arms)
        else:
            jobs.extend((bid, arm, None) for arm in arms)

    print(f"running {len(packets)} behaviors x {arms} = {len(jobs)} jobs | "
          f"model={cfg['model']} | {args.workers} processes\n", flush=True)

    # Reuse mode: preload each behavior's saved delivering DAG (ipiguard arm only).
    saved_dag = {}
    if args.reuse_plans:
        for bid in packets:
            pf = _PLANS / f"{bid}.json"
            if pf.exists():
                d = json.loads(pf.read_text())
                if d.get("dag"):
                    saved_dag[bid] = d["dag"]
        print(f"reuse-plans: loaded {len(saved_dag)} saved DAGs (ipiguard arm replays them)\n", flush=True)

    def _dag_for(bid, arm):
        return saved_dag.get(bid) if (args.reuse_plans and arm == "ipiguard") else None

    summary = {arm: {} for arm in arms}  # arm -> {bid: row}, sorted at the end
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_job, packets[bid], bid, arm, cfg, args.timeout, _dag_for(bid, arm),
                          args.no_worldsim, args.all_worldsim, attack): (bid, arm, attack)
                for bid, arm, attack in jobs}
        for fut in cf.as_completed(futs):
            bid, arm, attack = futs[fut]
            done += 1
            try:
                _bid, _arm, t, dt = fut.result()
            except Exception as e:
                print(f"[{done}/{len(jobs)}] {bid:42s} {arm:8s} JOB FAILED: {e}", flush=True)
                continue
            stem = f"{bid}#{attack['idx']:03d}" if attack else bid
            (_TX / f"{stem}.{arm}.json").write_text(json.dumps(t, indent=2, default=_json_default))
            row = {"behavior": bid, "corpus": t.get("corpus"), "status": t["status"],
                   "delivered": t["injection_delivered"],
                   "n_executed": len(t["executed_calls"]), "n_faked": len(t["faked_calls"]),
                   "secs": dt}
            summary[arm][stem] = row
            print(f"[{done}/{len(jobs)}] {bid:42s} {arm:8s} {t['status']:10s} "
                  f"deliv={t['injection_delivered']!s:5s} exec={row['n_executed']:2d} "
                  f"faked={row['n_faked']:2d} {dt:5.1f}s", flush=True)

    for arm in arms:
        rows = [summary[arm][bid] for bid in sorted(summary[arm])]
        (_TX / f"_summary.{arm}.json").write_text(json.dumps(rows, indent=2))
        deliv = sum(r["delivered"] for r in rows)
        ok = sum(r["status"] == "ok" for r in rows)
        faked = sum(r["n_faked"] for r in rows)
        print(f"\n{arm}: {ok}/{len(rows)} ok, delivered {deliv}/{len(rows)}, "
              f"{faked} FTI-faked calls -> transcripts/_summary.{arm}.json", flush=True)


if __name__ == "__main__":
    main()
