"""Forced-delivery A/B, THINKING-OFF, same session (isolates the hint effect from GLM drift).
Runs lean-OFF (control) then forced-OFF over all 88, diffs. BLOCKED until the OpenRouter key limit
is restored (2026-07-01 hit HTTP 403 'Key limit exceeded'). Run:
    cd system-defenses
    python -m FIDES.ipi_suites.forced_delivery_ab_thinkoff
"""
import sys, json
sys.path.insert(0, "system-defenses")
from FIDES.ipi_suites import run_batch as B, run_pilot as R
from ipi_suites_worldsim import make_worldsim_fallback
from concurrent.futures import ThreadPoolExecutor, as_completed

cfg = R.model_preset("glm"); cfg["thinking"] = False  # <-- thinking OFF
ALL_WORLDSIM = "--no-all-worldsim" not in sys.argv  # fully-simulated env by DEFAULT (opt out: --no-all-worldsim)
NO_WORLDSIM = "--no-worldsim" in sys.argv and not ALL_WORLDSIM  # world-sim fallback ON by default

def _one(fp, forced):
    pkt = json.loads(fp.read_text())
    if forced:
        pkt["forced_delivery"] = True
    carrier = pkt.get("carrier")
    ws = None if NO_WORLDSIM else B._worldsim(pkt)
    try:
        t = R.run(pkt, carrier, dict(pkt.get("canned", {})), cfg, worldsim=ws, all_worldsim=ALL_WORLDSIM)
        return pkt["behavior_id"], bool(t["injection_delivered"]), t["called_names"], "ok"
    except Exception as e:
        return pkt["behavior_id"], False, [], f"ERROR {type(e).__name__}: {str(e)[:100]}"

def sweep(forced):
    files = sorted(B._PACKETS.glob("*.json"))
    _one(files[0], forced)  # warm up
    res = {}
    errs = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for b, deliv, calls, status in ex.map(lambda f: _one(f, forced), files):
            res[b] = deliv
            errs += status != "ok"
    return res, errs

print("LEAN-OFF control ...", flush=True)
lean, e1 = sweep(False)
print(f"  lean-OFF: {sum(lean.values())}/{len(lean)} delivered (errors={e1})", flush=True)
print("FORCED-OFF ...", flush=True)
forced, e2 = sweep(True)
print(f"  forced-OFF: {sum(forced.values())}/{len(forced)} delivered (errors={e2})", flush=True)

up = sorted(b for b in forced if forced[b] and not lean.get(b))
dn = sorted(b for b in forced if not forced[b] and lean.get(b))
print(f"\n=== FORCED vs LEAN (thinking-OFF, same session) ===")
print(f"lean-OFF {sum(lean.values())}/88  ->  forced-OFF {sum(forced.values())}/88")
print(f"FLIPPED up (non-deliver -> deliver) [{len(up)}]:", up)
print(f"REGRESSED (deliver -> non-deliver) [{len(dn)}]:", dn)
json.dump({"lean": lean, "forced": forced}, open("_forced_ab_thinkoff.json", "w"), indent=2)
