"""Verify the Firewalls Sanitizer on our IPI rollouts (Phase 5, post-hoc).

Loads a small stratified slice of originally-successful injection trajectories across
the 3 corpora via classifier-defenses/loader.py, runs the Sanitizer over each marked
injected tool message, and reports the neutralize rate. Writes per-record events to
results/ipi_sanitizer.jsonl and prints the {flagged_any, events, summary} envelope.

Usage:
  python run_ipi.py --n 60 --per-behavior-cap 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "classifier-defenses")

import loader  # noqa: E402

from defense import Defense  # noqa: E402
from ipi_adapter import apply_sanitizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="total trajectories across 3 corpora")
    ap.add_argument("--per-behavior-cap", type=int, default=4)
    ap.add_argument("--max-scan", type=int, default=8000)
    args = ap.parse_args()

    # stratified slice across all 3 corpora (round-robin across behaviors)
    records = []
    per_corpus = max(1, args.n // 3)
    for corpus, streamer in loader.CORPORA.items():
        picked, _ = loader.stratified(
            streamer, target=per_corpus, per_behavior_cap=args.per_behavior_cap,
            max_scan=args.max_scan,
        )
        records.extend(picked)
        print(f"[load] {corpus}: {len(picked)} records")

    d = Defense()
    events = [apply_sanitizer(d, r) for r in records]

    scored = [e for e in events if e["scored"]]
    neutralized = [e for e in scored if e["neutralized"]]
    summary = {
        "n_records": len(events),
        "n_scored": len(scored),
        "n_skipped": len(events) - len(scored),
        "n_neutralized": len(neutralized),
        "neutralize_rate": round(100.0 * len(neutralized) / len(scored), 2) if scored else 0.0,
        "n_changed": sum(1 for e in scored if e.get("changed")),
        "firewall_calls": d.calls,
        "firewall_parse_failures": d.parse_failures,
        "by_corpus": {},
    }
    for c in ("ipi_2025", "ipi_2026_q1", "ipi_2026_q2"):
        cs = [e for e in scored if e["corpus"] == c]
        cn = [e for e in cs if e["neutralized"]]
        summary["by_corpus"][c] = {
            "scored": len(cs),
            "neutralized": len(cn),
            "rate": round(100.0 * len(cn) / len(cs), 2) if cs else 0.0,
        }

    outdir = Path(HERE) / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "ipi_sanitizer.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    envelope = {"flagged_any": len(neutralized) > 0, "events": events, "summary": summary}
    (outdir / "ipi_sanitizer_summary.json").write_text(json.dumps({"flagged_any": envelope["flagged_any"], "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
