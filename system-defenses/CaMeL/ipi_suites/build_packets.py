"""Step 0: auto-derive a CaMeL run packet per behavior from recorded data (master venv).

For each behavior in store/attacks.jsonl, build a self-contained packet:
  - tools, system_prompt, user_request, injection  (from the valset record)
  - carrier: the tool whose recorded output carries the injection (matched in the trajectory)
  - canned: {tool_name -> recorded output} harvested from the recorded trajectory
            (carrier -> the full injection text)
  - harmful_tool: the decisive tool call (isHarmfulAction), for labelling/the delivered gate
Writes packets/<behavior_id>.json and prints auto-derivation coverage so we can decide where
(if anywhere) subagents are still needed to author plausible outputs for unexercised tools.

    cd system-defenses
    python -m CaMeL.ipi_suites.build_packets
    # public data: records from `python -m ipi_eval.public_data ... --out records.jsonl`
    python -m CaMeL.ipi_suites.build_packets --records records.jsonl

With --records, each record's own conversation (the public behavior's messages) stands in for the
recorded trajectory, and the first record per behavior is used.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ipi_eval import behaviors as B

_HERE = Path(__file__).resolve().parent
_VALSET = Path("system-defenses/ipi_eval/valset/ipi_defense_valset.jsonl")
_STORE = Path("classifier-defenses/store/attacks.jsonl")
_OUT = _HERE / "packets"


def _valset_by_behavior(path: Path = _VALSET) -> dict[str, dict]:
    """First valset record per behavior (has resolved tools + injection + request)."""
    out: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.setdefault(r["behavior_id"], r)
    return out


def _store_traj_by_behavior() -> dict[str, dict]:
    """First successful store record's trajectory per behavior."""
    out: dict[str, dict] = {}
    for line in open(_STORE):
        r = json.loads(line)
        b = r.get("behavior_id")
        if b in out:
            continue
        rec = r.get("record") or {}
        if rec.get("agent_messages"):
            out[b] = rec
    return out


def _toolname_by_call_id(messages: list[dict]) -> dict[str, str]:
    m = {}
    for msg in messages:
        for tc in (msg.get("tool_calls") or []):
            cid = tc.get("id")
            name = (tc.get("function") or {}).get("name")
            if cid and name:
                m[cid] = name
    return m


def _tool_sequence(messages: list[dict]) -> list[str]:
    """Ordered tool-call names the recorded agent made (the expected control flow)."""
    seq = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            n = (tc.get("function") or {}).get("name")
            if n:
                seq.append(n)
    return seq


def _harvest(messages: list[dict], injection: str) -> tuple[dict[str, str], str | None]:
    """Map tool outputs to tool names; identify the carrier (output holding the injection)."""
    id2name = _toolname_by_call_id(messages)
    canned: dict[str, str] = {}
    carrier = None
    inj_head = (injection or "")[:120].strip()
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        content = content if isinstance(content, str) else json.dumps(content)
        name = msg.get("name") or id2name.get(msg.get("tool_call_id", ""), None)
        if not name:
            continue
        canned.setdefault(name, content)
        if carrier is None and inj_head and (inj_head in content or content[:120].strip() == inj_head):
            carrier = name
    return canned, carrier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=None,
                    help="public-format records (ipi_eval.public_data); default: the arena valset + store")
    args = ap.parse_args()
    _OUT.mkdir(exist_ok=True)
    if args.records:
        vs = _valset_by_behavior(args.records)
        traj = {bid: {"agent_messages": v["prefill_messages"]} for bid, v in vs.items()}
    else:
        vs = _valset_by_behavior()
        traj = _store_traj_by_behavior()

    stats = defaultdict(int)
    rows = []
    for bid, v in vs.items():
        rec = traj.get(bid)
        msgs = (rec or {}).get("agent_messages") or []
        canned, _carrier_sub = _harvest(msgs, v.get("attack_text", ""))

        # Deterministic carrier from the valset injection marker: the tool message AT
        # prefill_end_index is the injected carrier; its tool name comes from the
        # tool_call that produced it (mapped within the prefill).
        carrier = None
        carrier_args = {}
        pf = v.get("prefill_messages") or []
        idx = v.get("prefill_end_index")
        if isinstance(idx, int) and 0 <= idx < len(pf):
            cmsg = pf[idx]
            if cmsg.get("role") == "tool":
                cid = cmsg.get("tool_call_id", "")
                carrier = cmsg.get("name") or _toolname_by_call_id(pf).get(cid)
                ccontent = cmsg.get("content")
                if isinstance(ccontent, str) and ccontent:
                    canned[carrier] = ccontent  # carrier returns the exact injected output
                # The arguments the recorded agent used to call the carrier — so the hint
                # can call it directly in ONE step instead of relying on the planner to
                # discover the required id/path via a prior tool call.
                for m in pf:
                    for tc in (m.get("tool_calls") or []):
                        if tc.get("id") == cid:
                            raw = (tc.get("function") or {}).get("arguments")
                            try:
                                carrier_args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                            except Exception:
                                carrier_args = {}
        if carrier is None:
            carrier = _carrier_sub

        # forced_delivery is decided EMPIRICALLY, not by a static heuristic: default False
        # (faithful chaining). After a faithful run, the few behaviors that still don't
        # deliver get this flipped (mark_forced.py) so framed_query hands the carrier args
        # directly — a flagged fallback for the genuinely non-reproducible cases.
        forced_delivery = False

        # fallback: if no carrier matched in trajectory, the injection is the valset attack_text
        # and we mark the carrier unknown (subagent / heuristic needed).
        try:
            beh = v.get("behavior") or B.get_behavior(bid)
            n_criteria = len(beh.get("criteria", []))
        except Exception:
            n_criteria = 0
        packet = {
            "behavior_id": bid, "corpus": v["corpus"],
            "system_prompt": v["system_prompt"], "user_request": v["user_request"],
            "injection": v["attack_text"], "tools": v["tools"],
            "tool_policies": v.get("tool_policies", {}),
            "carrier": carrier, "carrier_args": carrier_args,
            "recorded_tool_sequence": _tool_sequence(msgs),
            "forced_delivery": forced_delivery, "canned": canned,
            "n_tools": len(v["tools"]), "n_canned": len(canned), "n_criteria": n_criteria,
        }
        if v.get("behavior"):   # public records: the harmful action is the tool_judge target
            for e in v["behavior"]["judging"]["submit"]["panel"]:
                if e.get("judge_id") == "toolUsageJudge":
                    p = e["parameters"]
                    packet["harmful_tool"] = p["target_tool"]
                    packet["harmful_args"] = {k: x["value"] for k, x in p["expected_parameter_values"].items()
                                              if not x.get("isRegex")}
        (_OUT / f"{bid.strip()}.json").write_text(json.dumps(packet, indent=2))
        stats["total"] += 1
        stats["has_carrier"] += int(carrier is not None)
        stats["has_canned"] += int(len(canned) > 0)
        stats["has_traj"] += int(len(msgs) > 0)
        rows.append((bid, v["corpus"], carrier is not None, len(canned), len(v["tools"])))

    print(f"\n=== Step 0 packet coverage (n={stats['total']}) ===")
    print(f"  has recorded trajectory : {stats['has_traj']}")
    print(f"  carrier auto-identified : {stats['has_carrier']}")
    print(f"  >=1 canned tool output  : {stats['has_canned']}")
    miss = [r for r in rows if not r[2]]
    print(f"\n  {len(miss)} behaviors WITHOUT an auto-identified carrier (need heuristic/subagent):")
    for bid, corp, _c, nc, nt in sorted(miss)[:40]:
        print(f"    {corp:14s} {bid:45s} canned={nc} tools={nt}")
    print(f"\nwrote {stats['total']} packets -> {_OUT}/")

    if args.records:   # the repair below targets an arena-export defect; public data has none
        return
    # Self-heal arena-export corruption: the valset ships a placeholder 24-hex ObjectId in place of
    # the real system_prompt for a few behaviors (a shared prompt referenced by _id, never exported).
    # Recover it faithfully from a clean sibling with the IDENTICAL toolset (agent type == toolset).
    from CaMeL.ipi_suites.repair_corrupt_prompts import recover
    recovered, _pkts, _bt = recover()
    for bid, sp in recovered.items():
        f = _OUT / f"{bid.strip()}.json"
        p = json.loads(f.read_text()); p["system_prompt"] = sp
        f.write_text(json.dumps(p, indent=2))
    if recovered:
        print(f"  repaired {len(recovered)} corrupt-prompt packets via toolset-match "
              f"({', '.join(sorted(recovered))})")


if __name__ == "__main__":
    main()
