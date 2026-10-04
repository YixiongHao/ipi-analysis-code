"""Aggregate results/<defense>.jsonl into an ASR summary table + results/summary.json.

Usage: python -m ipi_eval.summarize
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
ORDER = ["baseline", "firewalls", "fides", "melon", "causalarmor", "ipiguard", "camel"]


def _summ(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    breaks = sum(r["is_break"] for r in rows)
    errors = sum(r.get("status") == "error" for r in rows)
    aborted = sum(bool(r.get("aborted")) for r in rows)
    by_corpus = defaultdict(lambda: [0, 0])
    for r in rows:
        by_corpus[r["corpus"]][0] += int(r["is_break"])
        by_corpus[r["corpus"]][1] += 1
    return {"n": n, "breaks": breaks, "asr": breaks / n if n else 0.0,
            "errors": errors, "aborted": aborted,
            "by_corpus": {c: {"asr": b / t, "n": t} for c, (b, t) in by_corpus.items()}}


def main() -> None:
    summary = {}
    for d in ORDER:
        p = RESULTS / f"{d}.jsonl"
        if p.exists():
            summary[d] = _summ(p)

    base = summary.get("baseline", {}).get("asr")
    print(f"{'defense':12s} {'n':>5} {'ASR':>7} {'Δvs base':>9} {'err':>4} {'abort':>6}  per-corpus")
    for d in ORDER:
        if d not in summary:
            continue
        s = summary[d]
        delta = "" if base is None else f"{s['asr'] - base:+.3f}"
        pc = "  ".join(f"{c.replace('ipi_','')}={v['asr']:.2f}({v['n']})" for c, v in sorted(s["by_corpus"].items()))
        print(f"{d:12s} {s['n']:>5} {s['asr']:>7.3f} {delta:>9} {s['errors']:>4} {s['aborted']:>6}  {pc}")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
