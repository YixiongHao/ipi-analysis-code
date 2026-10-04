"""Persist re-authored CaMeL plans.

For each accepted bid: back up plans/<bid>.json -> plans/<bid>.json.bak (once), then overwrite with the
new clean single-fragment program (read from <cand-dir>/<bid>.py), setting n_frags=1 + reauthored=True and
keeping behavior_id/carrier/corpus. Only touches bids listed in --accept.

    cd system-defenses/CaMeL/ipi_suites
    python reauthor_plans.py --cand-dir <dir> --accept <accepted.json>
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--accept", required=True, help="JSON list of accepted bids")
    args = ap.parse_args()

    accept = json.loads(Path(args.accept).read_text())
    cand = Path(args.cand_dir)
    done, missing = [], []
    for bid in accept:
        cf = cand / f"{bid}.py"
        if not cf.exists():
            missing.append(bid); continue
        prog = cf.read_text().strip()
        pf = _PLANS / f"{bid}.json"
        plan = json.loads(pf.read_text())
        bak = _PLANS / f"{bid}.json.bak"
        if not bak.exists():  # preserve the ORIGINAL frozen plan; never clobber an existing backup
            shutil.copy(pf, bak)
        plan["programs"] = [prog]
        plan["program"] = prog
        plan["n_frags"] = 1
        plan["reauthored"] = True
        pf.write_text(json.dumps(plan, indent=2))
        done.append(bid)

    print(f"persisted {len(done)} re-authored plans (originals backed up to .bak):")
    for b in done:
        print("  ", b)
    if missing:
        print(f"\nMISSING candidate files ({len(missing)}):")
        for b in missing:
            print("  ", b)


if __name__ == "__main__":
    main()
