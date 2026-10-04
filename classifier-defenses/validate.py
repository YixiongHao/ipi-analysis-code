#!/usr/bin/env python3
"""Validate the detector store + per-defense result sidecars: structural checks
+ DuckDB analysis. The store (store/attacks.jsonl) holds one combined record per
attack; detector outputs live in results/<defense_id>.jsonl keyed by attack_id
and are joined here at read time (see run_detector.py).
"""
import glob
import json
import os

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "store", "attacks.jsonl")
RESULTS = os.path.join(HERE, "results")
ENVELOPE_KEYS = {"flagged_any", "config_hash", "events", "summary"}


def main():
    rows = [json.loads(l) for l in open(STORE) if l.strip()]
    by_id = {r["attack_id"]: r for r in rows}

    # load per-defense sidecars: {defense_id: {attack_id: envelope}}
    sidecars = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
        did = os.path.splitext(os.path.basename(path))[0]
        sidecars[did] = {x["attack_id"]: x for x in (json.loads(l) for l in open(path) if l.strip())}
    run_defenses = sorted(sidecars)
    print(f"store: {len(rows)} combined records | defenses with sidecars: {run_defenses}\n")

    # ---- structural checks ----
    problems = []
    for r in rows:
        for k in ("attack_id", "corpus", "behavior_id", "record", "source"):
            if k not in r:
                problems.append(f"{r.get('attack_id')}: missing {k}")
        if not (r.get("record") or {}).get("agent_messages"):
            problems.append(f"{r['attack_id']}: empty trajectory")
    for did, m in sidecars.items():
        extra = set(m) - set(by_id)
        if extra:
            problems.append(f"{did}: {len(extra)} result rows with no matching record")
        for aid, env in m.items():
            if not ENVELOPE_KEYS <= set(env):
                problems.append(f"{aid}/{did}: envelope missing {ENVELOPE_KEYS - set(env)}")
    print(f"structural check: {'OK' if not problems else problems[:5]}")
    print(f"  (records carry manifest+trajectory; every sidecar envelope >= {sorted(ENVELOPE_KEYS)})\n")

    # ---- runs view (one row per attack x defense), joined store <- sidecars ----
    runs = [(aid, by_id[aid]["corpus"], by_id[aid]["behavior_id"], did,
             bool(env["flagged_any"]), env.get("flaggedCorrectMessage"))
            for did, m in sidecars.items() for aid, env in m.items() if aid in by_id]
    con = duckdb.connect()
    con.execute("CREATE TABLE runs(attack_id VARCHAR, corpus VARCHAR, behavior_id VARCHAR, "
                "defense_id VARCHAR, flagged_any BOOLEAN, flagged_correct BOOLEAN)")
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?)", runs)

    # flagged_any = fired on ANY monitored turn; flagged_correct = fired on the true
    # injection turn (containsIPI). flagged_correct is NULL where containsIPI couldn't
    # be located, so its denominator counts only determinable records.
    print("flag rate by defense (overall):")
    print(con.sql("""SELECT defense_id, count(*) n, sum(flagged_any::int) flagged,
                     round(100.0*sum(flagged_any::int)/count(*),1) pct_any,
                     count(flagged_correct) n_det,
                     round(100.0*sum(flagged_correct::int)/count(flagged_correct),1) pct_correct
                     FROM runs GROUP BY defense_id ORDER BY pct_any DESC""").df().to_string(index=False))

    print("\nflag rate by corpus x defense (pct_any vs pct_correct):")
    print(con.sql("""SELECT corpus, defense_id, count(*) n,
                     round(100.0*sum(flagged_any::int)/count(*),1) pct_any,
                     round(100.0*sum(flagged_correct::int)/count(flagged_correct),1) pct_correct
                     FROM runs GROUP BY corpus, defense_id ORDER BY corpus, defense_id""").df().to_string(index=False))

    print("\nconsensus: trajectories flagged by k of the run detectors:")
    print(con.sql("""WITH pa AS (SELECT attack_id, sum(flagged_any::int) k FROM runs GROUP BY attack_id)
                     SELECT k n_detectors_flagging, count(*) n_trajectories
                     FROM pa GROUP BY k ORDER BY k""").df().to_string(index=False))

    # ---- prove the sidecars are directly queryable + joinable (no ETL) ----
    print("\ndirect join store <- sidecars (read_json_auto, no flattening):")
    pa, ds = os.path.join(RESULTS, "protectai-v2.jsonl"), os.path.join(RESULTS, "datasentinel-mistral7b.jsonl")
    print(con.sql(f"""SELECT s.corpus,
                        round(100.0*avg(p.flagged_any::int),1) protectai_pct,
                        round(100.0*avg(d.flagged_any::int),1) datasentinel_pct
                      FROM read_json_auto('{STORE}') s
                      LEFT JOIN read_json_auto('{pa}') p USING (attack_id)
                      LEFT JOIN read_json_auto('{ds}') d USING (attack_id)
                      GROUP BY s.corpus ORDER BY s.corpus""").df().to_string(index=False))


if __name__ == "__main__":
    main()
