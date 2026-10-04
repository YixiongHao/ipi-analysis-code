"""Repair shared packets whose `system_prompt` is a corrupt arena-export placeholder — a 24-hex
MongoDB ObjectId (e.g. '0123456789abcdef01234567', shared across several behaviors) or empty —
by RECOVERING the real prompt from a clean sibling with the IDENTICAL toolset.

This is faithful recovery, not fabrication: arena system prompts are canonical PER AGENT TYPE, and
agent type is fully determined by the tool set (e.g. the OpenHands coding agent =
{execute_bash, str_replace_editor, task_tracker, execute_ipython_cell, think, finish}). Every clean
packet with a given toolset carries a byte-identical prompt, so a corrupt packet with that same
toolset used that same prompt. Only recovers when the toolset maps to exactly ONE clean prompt
(unambiguous); otherwise it leaves the packet and reports it.

    cd system-defenses
    python -m CaMeL.ipi_suites.repair_corrupt_prompts [--apply]
"""
from __future__ import annotations
import json, re, sys, hashlib
from pathlib import Path
from collections import defaultdict

_PACKETS = Path(__file__).resolve().parent / "packets"
_OBJID = re.compile(r"^[0-9a-f]{24}$")


def _toolset(pkt: dict) -> frozenset[str]:
    return frozenset(t["function"]["name"] for t in pkt.get("tools", []))


def _is_corrupt(sp: str) -> bool:
    sp = (sp or "").strip()
    return (not sp) or bool(_OBJID.match(sp))


def recover() -> dict[str, str]:
    """behavior_id -> recovered prompt (only for repairable corrupt packets)."""
    pkts = {f.stem: json.loads(f.read_text()) for f in _PACKETS.glob("*.json")}
    # canonical prompt per toolset, from CLEAN packets (real prompt, unique per toolset)
    by_toolset: dict[frozenset, set[str]] = defaultdict(set)
    for p in pkts.values():
        sp = p.get("system_prompt") or ""
        if not _is_corrupt(sp) and len(sp) > 80:
            by_toolset[_toolset(p)].add(sp)
    canonical = {ts: next(iter(sps)) for ts, sps in by_toolset.items() if len(sps) == 1}
    out = {}
    for bid, p in pkts.items():
        if _is_corrupt(p.get("system_prompt") or ""):
            ts = _toolset(p)
            if ts in canonical:
                out[bid] = canonical[ts]
    return out, pkts, by_toolset


def main() -> None:
    apply = "--apply" in sys.argv
    recovered, pkts, by_toolset = recover()
    corrupt = [b for b, p in pkts.items() if _is_corrupt(p.get("system_prompt") or "")]
    print(f"corrupt packets: {len(corrupt)} | recoverable by unique toolset-match: {len(recovered)}")
    for bid in sorted(corrupt):
        if bid in recovered:
            h = hashlib.md5(recovered[bid].encode()).hexdigest()[:8]
            head = recovered[bid].split("\n", 1)[0][:60]
            print(f"  [RECOVER] {bid:44s} <- toolset prompt md5={h} :: {head!r}")
        else:
            ts = frozenset(t["function"]["name"] for t in pkts[bid].get("tools", []))
            n = len(by_toolset.get(ts, set()))
            print(f"  [SKIP   ] {bid:44s} no unique toolset donor ({n} candidate prompts)")
    if apply:
        for bid, sp in recovered.items():
            f = _PACKETS / f"{bid}.json"
            p = json.loads(f.read_text()); p["system_prompt"] = sp
            f.write_text(json.dumps(p, indent=2))
        print(f"\nAPPLIED: wrote recovered prompt into {len(recovered)} packets.")
    else:
        print("\n(dry run — pass --apply to write)")


if __name__ == "__main__":
    main()
