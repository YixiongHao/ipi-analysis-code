# IPI verification for CaMeL: characterize a stratified slice of originally-successful
# arena attacks against CaMeL's threat model (see ipi_adapter.py for the rationale and
# the proxy's limits). No model re-run — this is a scope characterization, not a block
# rate. Run with the master venv (CPU only):
#   python run_ipi.py [--n-per-corpus 25]

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ipi_adapter import classify_attack

# Stratified attack store produced by classifier-defenses/loader.py (combined records).
STORE = Path("classifier-defenses/store/attacks.jsonl")
RESULTS_DIR = Path(__file__).parent / "results"


def load_slice(n_per_corpus: int):
    """Stream the combined-record store, keep up to n_per_corpus per corpus."""
    by_corpus = defaultdict(list)
    with STORE.open() as f:
        for line in f:
            rec = json.loads(line)
            c = rec["corpus"]
            if len(by_corpus[c]) < n_per_corpus:
                by_corpus[c].append(rec)
            if all(len(v) >= n_per_corpus for v in by_corpus.values()) and len(by_corpus) >= 3:
                break
    return [r for v in by_corpus.values() for r in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-corpus", type=int, default=25)
    args = ap.parse_args()

    records = load_slice(args.n_per_corpus)
    per_record = []
    flow = Counter()
    by_corpus = defaultdict(Counter)
    for rec in records:
        cls = classify_attack(rec["record"])
        flow[cls["flow_type"]] += 1
        by_corpus[rec["corpus"]][cls["flow_type"]] += 1
        per_record.append({
            "attack_id": rec["attack_id"], "corpus": rec["corpus"],
            "behavior_id": rec["behavior_id"], **cls,
        })

    n = len(records)
    in_model = flow["control_or_data_flow"]
    summary = {
        "n_attacks": n,
        "in_camel_threat_model": in_model,
        "in_camel_threat_model_frac": round(in_model / n, 3) if n else 0.0,
        "text_to_text_out_of_model": flow["text_to_text"],
        "by_corpus": {c: dict(v) for c, v in by_corpus.items()},
        "note": ("Proxy = presence of an attacker-driven tool call after the injection. "
                 "Coarse scope characterization, NOT a guaranteed block rate; CaMeL would "
                 "neutralize the in-model fraction by design (isolation + capabilities). "
                 "See ipi_adapter.py."),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"summary": summary, "per_record": per_record}
    (RESULTS_DIR / "ipi_characterization.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> wrote {RESULTS_DIR / 'ipi_characterization.json'}")


if __name__ == "__main__":
    main()
