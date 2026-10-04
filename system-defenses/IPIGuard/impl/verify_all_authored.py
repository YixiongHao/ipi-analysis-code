#!/usr/bin/env python
"""Independently re-verify ALL hand-authored IPIGuard TDGs (do not trust the workflow's self-reports).

Iterates every plan under plans_agentdojo_authored/ and runs verify_quality.verify() with the
corrected (word-boundary) completeness heuristic. --static-only skips the live traverse (fast,
deterministic ground truth); default also runs the mechanical traverse (live, deepseek).
Writes results_fp/authored_quality[_static].json.
"""
import argparse
import json
from pathlib import Path

from verify_quality import verify

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default="plans_agentdojo_authored")
    ap.add_argument("--static-only", action="store_true")
    args = ap.parse_args()
    root = HERE / args.plans_dir
    rows = []
    for suite in ["workspace", "banking", "slack", "travel"]:
        for p in sorted((root / suite).glob("*.json")):
            r = verify(suite, p.stem, root, static_only=args.static_only)
            rows.append(r)
            flag = "OK " if r["ok"] else "FAIL"
            print(f"[{flag}] {suite}/{p.stem} static={r['ok_static']} mech={r.get('ok_mechanical')} "
                  f"nodes={r['n_nodes']} issues={r['issues'][:2]}", flush=True)
    n = len(rows)
    ok_static = sum(1 for r in rows if r["ok_static"])
    ok = sum(1 for r in rows if r["ok"])
    fails = [(r["suite"], r["task"], r["issues"]) for r in rows if not r["ok"]]
    print(f"\n=== {n} plans | static-OK {ok_static}/{n} | overall-OK {ok}/{n} ===")
    for s, t, iss in fails:
        print(f"  FAIL {s}/{t}: {iss}")
    out = HERE / "results_fp" / ("authored_quality_static.json" if args.static_only else "authored_quality.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "static_ok": ok_static, "ok": ok,
                               "fails": [{"suite": s, "task": t, "issues": i} for s, t, i in fails],
                               "rows": rows}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
