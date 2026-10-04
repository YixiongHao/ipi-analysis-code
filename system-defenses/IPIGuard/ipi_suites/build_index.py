"""Assemble plans/_index.json — the behavior -> delivering-plan mapping summary.

Scans plans/<bid>.json, folds in the Stage-1 routing (_failures.json) and the reuse check
(_reuse_check.json) if present, and writes a sorted index + coverage summary.

    cd system-defenses/IPIGuard/ipi_suites
    python build_index.py
"""
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"
_PACKETS = _HERE.parents[1] / "CaMeL" / "ipi_suites" / "packets"


def main() -> None:
    allb = sorted(p.stem for p in _PACKETS.glob("*.json"))
    fail = json.loads((_PLANS / "_failures.json").read_text()) if (_PLANS / "_failures.json").exists() else {}
    subq = set(fail.get("subagent_queue", []))
    reuse = {}
    if (_PLANS / "_reuse_check.json").exists():
        reuse = {r["behavior"]: r["delivered"] for r in json.loads((_PLANS / "_reuse_check.json").read_text())}

    index = {}
    for bid in allb:
        fp = _PLANS / f"{bid}.json"
        if not fp.exists():
            index[bid] = {"status": "MISSING"}
            continue
        p = json.loads(fp.read_text())
        index[bid] = {
            "status": "saved", "corpus": p.get("corpus"), "carrier": p.get("carrier"),
            "k_delivered": p.get("k_delivered"), "hint_level": p.get("hint_level"),
            "delivered": p.get("delivered"), "route": "subagent" if bid in subq else "main",
            "n_dag_nodes": len(p.get("dag") or []),
            "reuse_delivered": reuse.get(bid),  # None if not in the reuse sample
        }

    saved = [b for b, v in index.items() if v["status"] == "saved"]
    out = {
        "model": "google/gemini-3-flash-preview",
        "summary": {
            "total": len(allb), "saved": len(saved), "missing": len(allb) - len(saved),
            "by_corpus": dict(Counter(v["corpus"] for v in index.values() if v["status"] == "saved")),
            "by_hint_level": dict(Counter(v["hint_level"] for v in index.values() if v["status"] == "saved")),
            "by_route": dict(Counter(v["route"] for v in index.values() if v["status"] == "saved")),
            "reuse_checked": sum(1 for v in index.values() if v.get("reuse_delivered") is not None),
            "reuse_delivered": sum(1 for v in index.values() if v.get("reuse_delivered") is True),
        },
        "plans": index,
    }
    (_PLANS / "_index.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    miss = [b for b in allb if index[b]["status"] != "saved"]
    if miss:
        print("MISSING:", miss)
    print("-> plans/_index.json")


if __name__ == "__main__":
    main()
