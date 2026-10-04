"""Rebuild the saved delivering DAGs to honor the manual tool-label audit command set.

WHY (FINDING-5 residual): a planned COMMAND-tool node in the saved DAG executes for real in
topological traversal (DagToolsExecutor._run_tool_call_with_reflection), BYPASSING Fake Tool
Invocation — so an injected command that matches a pre-authorized node ran instead of being faked.
The construct LLM (OpenAIConstructLLM) never reads the whitelist, so the query/command split must be
applied POST-construction: drop command-tool nodes (audit `command_tools(bid)`, carrier exempt) from
the saved plan. The carrier (a read that delivers the injection) is retained, so delivery is preserved.

This rebuild is DETERMINISTIC: the construct DAG is independent of the command set, so the
audit-consistent plan = `_strip_command_nodes(existing Gemini DAG)`. Stripping the already
delivery-validated DAGs (88/88, hint-escalated where needed) preserves that hard-won delivery with
zero re-plan drift — equivalent to a Gemini rebuild through the now-patched plan_behavior, but safe.
(The patched plan_behavior strips identically for any FUTURE live Gemini/GLM rebuild.)

    cd system-defenses/IPIGuard/ipi_suites
    python rebuild_plans_audit.py            # rebuild all
    python rebuild_plans_audit.py --dry-run  # report only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report changes, do not rewrite")
    args = ap.parse_args()

    changed, unchanged = [], []
    for fp in sorted(_PLANS.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        plan = json.loads(fp.read_text())
        bid = plan["behavior_id"]
        carrier = plan.get("carrier")
        old_dag = plan.get("dag") or []
        packet = harness.load_packet(bid)
        cmd = harness._command_tools(packet)
        new_dag = harness._strip_command_nodes(old_dag, cmd, carrier)
        stripped = sorted({d["function_name"] for d in old_dag}
                          - {d["function_name"] for d in new_dag})
        if not stripped:
            unchanged.append(bid)
            continue
        changed.append((bid, carrier, stripped))
        if not args.dry_run:
            plan["dag"] = new_dag
            plan["planned_tools"] = sorted({d["function_name"] for d in new_dag})
            plan["audit_stripped_command_nodes"] = stripped  # provenance of the rebuild
            fp.write_text(json.dumps(plan, indent=2))

    print(f"{'DRY-RUN: would change' if args.dry_run else 'rebuilt'} {len(changed)} plans; "
          f"{len(unchanged)} unchanged (no pre-authorized command node beyond the carrier)\n")
    for bid, carrier, stripped in sorted(changed):
        print(f"  {bid:42s} carrier={carrier:34s} stripped={stripped}")


if __name__ == "__main__":
    main()
