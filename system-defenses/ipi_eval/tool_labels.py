"""Manual tool-label audit (ground truth) shared by FIDES / IPIGuard / CausalArmor / CaMeL.

Per-behavior human judgement of each tool's policy class (neutral / consequential / egress),
produced by the 18-subagent audit. Canonical data: `tool_labels.json` (this dir), which covers
all 33 public `ipi_arena_os` tool/coding behaviors (`flashing-lights-seizure` and
`stealth-dependency-append` were labeled by hand with the same rules). A behavior without labels
falls back to the keyword heuristics in `behaviors.py`. `rationale` is present where the audit
changed the original (`current`) label, and on every hand-labeled row.

Every defense that needs a tool taxonomy reduces to this one mapping:
  - FIDES      : `tool_policies(bid)`  consequential->P-T, egress->P-F or P-T (neutral omitted)
  - IPIGuard   : `command_tools(bid)`  state-mutating actions = consequential ∪ egress
  - CausalArmor: `command_tools(bid)`  (same privileged/"actions to defend" set)
  - CaMeL      : `read_only(bid)`      no-side-effect tools = neutral

Stdlib-only (no heavy imports) so it loads from any venv (incl. the CaMeL uv env).
"""
from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "tool_labels.json"
_DATA: dict | None = None


def _load() -> dict:
    global _DATA
    if _DATA is None:
        raw = json.load(open(_PATH))["behaviors"]
        _DATA = {k.strip(): v for k, v in raw.items()}  # keys are clean, but normalise defensively
    return _DATA


def judged_labels(behavior_id: str) -> dict[str, str]:
    """{tool -> 'neutral'|'consequential'|'egress'} for a behavior; {} if not audited."""
    return {r["tool"]: r["judged"] for r in _load().get((behavior_id or "").strip(), [])}


_POLICY = {"consequential": "P-T", "egress": "P-F or P-T"}


def tool_policies(behavior_id: str) -> dict[str, str]:
    """FIDES per-tool policy map (neutral tools omitted)."""
    return {t: _POLICY[c] for t, c in judged_labels(behavior_id).items() if c in _POLICY}


def command_tools(behavior_id: str) -> set[str]:
    """State-mutating action tools (consequential ∪ egress) — IPIGuard / CausalArmor."""
    return {t for t, c in judged_labels(behavior_id).items() if c in ("consequential", "egress")}


def read_only(behavior_id: str) -> set[str]:
    """No-side-effect tools (neutral) — CaMeL `no_side_effect_tools`."""
    return {t for t, c in judged_labels(behavior_id).items() if c == "neutral"}


def has(behavior_id: str) -> bool:
    """True iff this behavior is covered by the audit (else callers fall back to the heuristic)."""
    return bool(_load().get((behavior_id or "").strip()))
