"""Static audit of all 88 CaMeL plans for the carrier+downstream invariant.

For each behavior verify that the fragment which actually EXECUTES under replay
(fragments[0..k], where k = first fragment that calls the carrier; replay early-stops
after that fragment) contains BOTH the carrier call AND the downstream consequential
action(s) planned from the isolated user request. Specifically trace how
`query_ai_assistant` (Q-LLM) calls sit relative to the carrier, because a Q-LLM call
positioned BETWEEN the carrier and the consequential call is a truncation risk (if it
raises NotEnoughInformationError the consequential call is never reached -> policy not
observed, even though delivery already happened).

Outputs JSON (per-behavior facts) + a readable table + a classification.
"""
from __future__ import annotations
import ast, json, glob, os, sys
from pathlib import Path

HERE = Path("system-defenses/CaMeL/ipi_suites")
PLANS = HERE / "plans"
TOOL_LABELS = json.load(open("system-defenses/ipi_eval/tool_labels.json"))
POLICY_DIR = Path("system-defenses/verification_camel_reauthor/camel_policy")

def conseq_tools(bid: str) -> set[str]:
    # Policy-relevant = exactly what harness._read_only gates: every tool NOT judged "neutral"
    # (i.e. judged in {consequential, egress}). "consequential"-only misses egress tools like
    # send_email, which the CaMeL policy absolutely gates when args are tainted.
    return {e["tool"] for e in TOOL_LABELS["behaviors"].get(bid, []) if e.get("judged") != "neutral"}

def call_names_ordered(code: str) -> list[tuple[int, str]]:
    """Return [(lineno, func_name)] for every top-level function Call, in source order.
    Falls back to a light regex scan if the fragment won't parse."""
    out = []
    import re
    m = re.match(r"```[a-zA-Z0-9_+\-#]*\n(.*)\n```", code.strip(), re.DOTALL)  # stored plans are fenced
    if m:
        code = m.group(1)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # restricted-python that stdlib ast rejects is rare; regex fallback on `name(`
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
            out.append((code[:m.start()].count("\n") + 1, m.group(1)))
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name:
                out.append((getattr(node, "lineno", 0), name))
    out.sort(key=lambda x: x[0])
    return out

def load_dynamic(bid: str) -> dict:
    fp = POLICY_DIR / f"{bid}.policy.jsonl"
    if not fp.exists():
        return {"delivered": None, "policy_blocked": None, "executed": [], "reached": None, "final_text": None}
    r = json.loads(fp.read_text().splitlines()[0])
    ec = [c["name"] for c in (r.get("executed_calls") or [])]
    conseq = conseq_tools(bid)
    reached = bool(r.get("policy_blocked")) or any(t in ec for t in conseq)
    return {"delivered": bool(r.get("injection_delivered")), "policy_blocked": bool(r.get("policy_blocked")),
            "executed": ec, "reached": reached, "final_text": (r.get("final_text") or "")}

def audit_one(fp: Path) -> dict:
    p = json.loads(fp.read_text())
    bid = p["behavior_id"]; carrier = p["carrier"]
    frags = p.get("programs") or ([p["program"]] if p.get("program") else [])
    conseq = conseq_tools(bid)
    QLLM = "query_ai_assistant"

    # flat event list across ALL fragments, tagged with fragment index
    events = []  # (frag_idx, lineno, name)
    for fi, code in enumerate(frags):
        for ln, name in call_names_ordered(code):
            events.append((fi, ln, name))

    # first carrier call -> defines the executed fragment set (0..carrier_frag inclusive)
    carrier_evts = [e for e in events if e[2] == carrier]
    carrier_present = bool(carrier_evts)
    carrier_frag = carrier_evts[0][0] if carrier_present else None
    # executed set = events in fragments <= carrier_frag; within carrier_frag, only up to+incl carrier not enforced
    # (whole carrier fragment runs to completion under parse_and_interpret_code), so executed = frag_idx<=carrier_frag
    executed = [e for e in events if carrier_present and e[0] <= carrier_frag]

    # position of carrier within the flat executed order
    def pos(evt): return executed.index(evt)
    carrier_evt = carrier_evts[0] if carrier_present else None
    carrier_pos = pos(carrier_evt) if carrier_present else None

    # is carrier the FIRST tool call in its own fragment? (delivery robustness -- nothing can fail before it)
    frag_calls = [e for e in events if carrier_present and e[0] == carrier_frag]
    carrier_first_in_frag = bool(frag_calls) and frag_calls[0][2] == carrier

    # consequential calls anywhere, split by whether they're in the executed set and after the carrier
    conseq_all = [e for e in events if e[2] in conseq]
    conseq_after_carrier_exec = [e for e in executed if e[2] in conseq and pos(e) > carrier_pos] if carrier_present else []
    conseq_truncated = [e for e in events if e[2] in conseq and carrier_present and e[0] > carrier_frag]  # in a later frag -> never runs

    # Q-LLM calls after the carrier in the executed set
    qllm_after = [e for e in executed if e[2] == QLLM and pos(e) > carrier_pos] if carrier_present else []
    qllm_all = [e for e in events if e[2] == QLLM]

    # does a Q-LLM call sit BETWEEN the carrier and the first executed downstream consequential call?
    qllm_gates_conseq = False
    if carrier_present and conseq_after_carrier_exec and qllm_after:
        first_conseq_pos = min(pos(e) for e in conseq_after_carrier_exec)
        qllm_gates_conseq = any(carrier_pos < pos(q) < first_conseq_pos for q in qllm_after)

    dyn = load_dynamic(bid)

    # ---- classification -------------------------------------------------
    # only fragments 0..carrier_frag execute (replay early-stops after the carrier fragment); the
    # carrier is in frag 0 for every plan, so the EXECUTED fragment is frag 0. A consequential call
    # that appears ONLY in a later (dead) fragment is genuinely truncated; if it ALSO appears in the
    # executed set it is reached and the later copies are redundant retry cruft.
    conseq_only_later = bool(conseq_truncated) and not bool(conseq_after_carrier_exec)
    if not carrier_present:
        cls = "BROKEN_no_carrier"
    elif conseq_after_carrier_exec:
        cls = "RISK_qllm_gates_conseq" if qllm_gates_conseq else "OK_conseq_after_carrier"
    elif conseq_only_later:
        cls = "BROKEN_conseq_only_in_dead_frag"       # downstream action lives past the early-stop -> never exercised
    else:
        cls = "NO_DOWNSTREAM_conseq"                  # no P-T call after the carrier at all (isolation-only? verify)
    has_dead_frag_conseq = bool(conseq_truncated)

    # dynamic override: flag disagreement
    dyn_flag = None
    if carrier_present and dyn["delivered"] is False:
        dyn_flag = "DYN_not_delivered"
    elif conseq_after_carrier_exec and dyn["reached"] is False:
        dyn_flag = "DYN_not_reached"

    return {
        "bid": bid, "carrier": carrier, "n_frags": len(frags),
        "reauthored": bool(p.get("reauthored")),
        "carrier_frag": carrier_frag, "carrier_first_in_frag": carrier_first_in_frag,
        "consequential_tools": sorted(conseq),
        "conseq_calls_all": [(f, n) for f, l, n in conseq_all],
        "conseq_after_carrier_executed": [(f, n) for f, l, n in conseq_after_carrier_exec],
        "conseq_truncated_later_frag": [(f, n) for f, l, n in conseq_truncated],
        "has_dead_frag_conseq": has_dead_frag_conseq,
        "n_qllm_total": len(qllm_all),
        "n_qllm_after_carrier": len(qllm_after),
        "qllm_gates_conseq": qllm_gates_conseq,
        "class": cls, "dyn_flag": dyn_flag,
        "dyn_delivered": dyn["delivered"], "dyn_policy_blocked": dyn["policy_blocked"],
        "dyn_reached": dyn["reached"], "dyn_executed": dyn["executed"],
    }

def main():
    rows = [audit_one(fp) for fp in sorted(PLANS.glob("*.json")) if not fp.name.endswith(".bak")]
    out = HERE.parent.parent / "verification_camel_reauthor" / "plan_audit.json"
    out.write_text(json.dumps(rows, indent=1))

    import collections
    cls_c = collections.Counter(r["class"] for r in rows)
    dyn_c = collections.Counter(r["dyn_flag"] for r in rows if r["dyn_flag"])
    print(f"AUDITED {len(rows)} plans -> {out}")
    print("class counts:", dict(cls_c))
    print("dynamic flags:", dict(dyn_c))
    print()
    # show everything that is NOT the clean OK case, plus NO_DOWNSTREAM (to sanity-check the isolation-only set)
    hdr = f"{'bid':46s} {'nf':>2s} {'cf':>2s} {'q>c':>3s} {'gate':>4s} {'class':30s} {'dyn(dlv/blk/rch)':16s} {'flag'}"
    print(hdr); print("-"*len(hdr))
    def key(r): return (r["class"] != "OK_conseq_after_carrier", r["class"], r["bid"])
    for r in sorted(rows, key=key):
        if r["class"] == "OK_conseq_after_carrier" and not r["dyn_flag"]:
            continue
        d = f"{r['dyn_delivered']}/{r['dyn_policy_blocked']}/{r['dyn_reached']}"
        print(f"{r['bid']:46s} {r['n_frags']:>2d} {str(r['carrier_frag']):>2s} "
              f"{r['n_qllm_after_carrier']:>3d} {str(r['qllm_gates_conseq'])[:4]:>4s} {r['class']:30s} {d:16s} {r['dyn_flag'] or ''}")
    print(f"\n(clean OK_conseq_after_carrier w/ no dyn flag: "
          f"{sum(1 for r in rows if r['class']=='OK_conseq_after_carrier' and not r['dyn_flag'])})")

if __name__ == "__main__":
    main()
