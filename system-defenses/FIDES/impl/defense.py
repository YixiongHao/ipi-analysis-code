"""FIDES — taint-tracking + per-tool policy core (scaffold-callable).

Paper: "Securing AI Agents with Information-Flow Control" (Costa et al., arXiv 2505.23643).
Archetype: planning / taint (information-flow control). This module implements the *security*
mechanism that produces the paper's headline result — dynamic taint-tracking + deterministic
per-tool policy checks (Algorithm 6 + §4.3 policies P-T / P-F). The utility-recovery layer
(variable hiding, quarantined_llm, expand_variables, constrained decoding) is intentionally NOT
implemented.

The `Lattice` / `IntegrityLabel` / `ConfidentialityLabel` / `PowersetLattice` / `InverseLattice`
/ `ProductLabel` classes below are **vendored verbatim** from the reference repo's tutorial
notebook (`system-defenses/FIDES/fides/Tutorial.ipynb`, cell 21) — that notebook is the ground
truth for the IFC core. Everything from `# === FIDES defense ===` down is new glue.

Generic scaffold use (no AgentDojo needed)::

    from defense import Defense, IntegrityLabel
    d = Defense()
    # Track context integrity as data flows in (system/user trusted, tool output untrusted):
    ctx_label = IntegrityLabel.trusted()
    for msg in conversation:
        ctx_label = ctx_label.join(d.label_of_message(msg))
    # Before executing a proposed tool call, check the policy:
    ok, reason = d.allow_call("send_money", {"recipient": "..."}, {"context_integrity": ctx_label})
    if not ok:
        raise RuntimeError(f"blocked: {reason}")   # P-T violation -> the injected action is stopped
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, FrozenSet, Generic, TypeVar

# ============================================================================================
# IFC lattice core — VENDORED from fides/Tutorial.ipynb cell 21 (ground truth). Do not edit
# semantics; this is the paper's reference label algebra.
# ============================================================================================


class Lattice(ABC):
    """Abstract base class for (bounded) IFC lattices."""

    @abstractmethod
    def leq(self, other: Any) -> bool:
        """Returns True if self <= other in the lattice."""

    @abstractmethod
    def join(self, other: Any) -> Any:
        """Returns the least upper bound of self and other."""

    @abstractmethod
    def meet(self, other: Any) -> Any:
        """Returns the greatest lower bound of self and other."""

    @abstractmethod
    def __repr__(self) -> str: ...

    def __le__(self, other: "Lattice") -> bool:
        return self.leq(other)


class ConfidentialityLabel(Lattice):
    class Level(Enum):
        LOW = 0
        HIGH = 1

    def __init__(self, level: "ConfidentialityLabel.Level"):
        self.level = level

    def leq(self, other: "ConfidentialityLabel") -> bool:
        return self.level.value <= other.level.value

    def join(self, other: "ConfidentialityLabel") -> "ConfidentialityLabel":
        return other if self.leq(other) else self

    def meet(self, other: "ConfidentialityLabel") -> "ConfidentialityLabel":
        return self if self.leq(other) else other

    def __repr__(self):
        return f"{self.level.name}"

    @classmethod
    def low(cls):
        return cls(cls.Level.LOW)

    @classmethod
    def high(cls):
        return cls(cls.Level.HIGH)


class IntegrityLabel(Lattice):
    class Level(Enum):
        TRUSTED = 0
        UNTRUSTED = 1

    def __init__(self, level: "IntegrityLabel.Level"):
        self.level = level

    def leq(self, other: "IntegrityLabel") -> bool:
        return self.level.value <= other.level.value

    def join(self, other: "IntegrityLabel") -> "IntegrityLabel":
        return other if self.leq(other) else self

    def meet(self, other: "IntegrityLabel") -> "IntegrityLabel":
        return self if self.leq(other) else other

    def __repr__(self):
        return f"{self.level.name}"

    @classmethod
    def trusted(cls):
        return cls(cls.Level.TRUSTED)

    @classmethod
    def untrusted(cls):
        return cls(cls.Level.UNTRUSTED)

    @property
    def is_trusted(self) -> bool:
        return self.level == IntegrityLabel.Level.TRUSTED


T = TypeVar("T")


class PowersetLattice(Lattice, Generic[T]):
    def __init__(self, subset: FrozenSet[T], universe: FrozenSet[T]):
        if not subset.issubset(universe):
            raise ValueError("Subset must be within the universe.")
        self.subset = subset
        self.universe = universe

    def leq(self, other: "PowersetLattice[T]") -> bool:
        return self.subset.issubset(other.subset)

    def join(self, other: "PowersetLattice[T]") -> "PowersetLattice[T]":
        return PowersetLattice(self.subset.union(other.subset), self.universe)

    def meet(self, other: "PowersetLattice[T]") -> "PowersetLattice[T]":
        return PowersetLattice(self.subset.intersection(other.subset), self.universe)

    def __repr__(self):
        return f"Powerset({{{', '.join(map(str, self.subset))}}})"

    @classmethod
    def bottom(cls, universe: FrozenSet[T]) -> "PowersetLattice[T]":
        return cls(frozenset(), universe)

    @classmethod
    def top(cls, universe: FrozenSet[T]) -> "PowersetLattice[T]":
        return cls(universe, universe)


L1 = TypeVar("L1", bound=Lattice)
L2 = TypeVar("L2", bound=Lattice)


class ProductLabel(Lattice, Generic[L1, L2]):
    def __init__(self, left: L1, right: L2):
        self.left = left
        self.right = right

    def leq(self, other: "ProductLabel[L1, L2]") -> bool:
        return self.left <= other.left and self.right <= other.right

    def join(self, other: "ProductLabel[L1, L2]") -> "ProductLabel[L1, L2]":
        return ProductLabel(self.left.join(other.left), self.right.join(other.right))

    def meet(self, other: "ProductLabel[L1, L2]") -> "ProductLabel[L1, L2]":
        return ProductLabel(self.left.meet(other.left), self.right.meet(other.right))

    def __repr__(self):
        return f"({self.left}, {self.right})"


L = TypeVar("L", bound="Lattice")


class InverseLattice(Lattice, Generic[L]):
    def __init__(self, inner: L):
        self.inner = inner

    def leq(self, other: "InverseLattice[L]") -> bool:
        return other.inner.leq(self.inner)

    def join(self, other: "InverseLattice[L]") -> "InverseLattice[L]":
        return InverseLattice(self.inner.meet(other.inner))

    def meet(self, other: "InverseLattice[L]") -> "InverseLattice[L]":
        return InverseLattice(self.inner.join(other.inner))

    def __repr__(self):
        return f"Inverse({repr(self.inner)})"


# ============================================================================================
# === FIDES defense === (new glue: Table 3 policies + P-T / P-F enforcement)
# ============================================================================================

# Policy classes (paper §4.3, Table 3).
PT = "P-T"  # trusted action: may proceed only from an all-trusted context
PF_OR_PT = "P-F or P-T"  # egress: permitted-flow OR (robust declassification) trusted context

# Per-tool policy registry — AgentDojo tools, transcribed verbatim from Table 3 of the paper.
# Any tool NOT listed here is "neutral" (no policy: reads/gets) and always proceeds.
DEFAULT_TOOL_POLICIES: dict[str, str] = {
    # --- egress tools: P-F or P-T ---
    "send_email": PF_OR_PT,
    "create_calendar_event": PF_OR_PT,
    "append_to_file": PF_OR_PT,
    "send_direct_message": PF_OR_PT,
    "send_channel_message": PF_OR_PT,
    # --- consequential tools: P-T ---
    "delete_email": PT,
    "reschedule_calendar_event": PT,
    "cancel_calendar_event": PT,
    "create_file": PT,
    "delete_file": PT,
    "share_file": PT,
    "reserve_hotel": PT,
    "reserve_restaurant": PT,
    "reserve_car_rental": PT,
    "send_money": PT,
    "schedule_transaction": PT,
    "update_scheduled_transaction": PT,
    "update_password": PT,
    "update_user_info": PT,
    "add_user_to_channel": PT,
    "invite_user_to_slack": PT,
    "remove_user_from_slack": PT,
    "get_webpage": PT,
    "post_webpage": PT,
    "download_file": PT,
    "add_calendar_event_participants": PT,
}


class Defense:
    """FIDES taint + policy core.

    `ctx` keys read by `allow_call`:
      - ``context_integrity``: an ``IntegrityLabel`` = join of the labels of all messages in the
        context that produced this tool call (system/user trusted, tool outputs untrusted).

    Args:
        tool_policies: tool name -> policy class (``PT`` / ``PF_OR_PT``). Defaults to Table 3.
            Tools absent from this map are neutral (no check). Override to add custom tools.
        untrusted_roles: message roles whose content is treated as low-integrity (external
            world data). Paper default: tool outputs.
        tool_output_untrusted: if True (default, conservative — matches the paper's "label
            untrusted all fields an injection task targets"), every tool output taints the
            context. Set False to treat tool outputs as trusted (disables integrity tainting).
        enforce_pf_as_pt: in this lightweight monitor we do not track per-argument
            confidentiality readers, so the P-F arm of ``PF_OR_PT`` cannot be evaluated; we
            conservatively require P-T (trusted context) for egress tools too.
            Set False to let egress tools through in untrusted contexts (not advised).
    """

    name = "fides"

    def __init__(
        self,
        tool_policies: dict[str, str] | None = None,
        untrusted_roles: tuple[str, ...] = ("tool",),
        tool_output_untrusted: bool = True,
        enforce_pf_as_pt: bool = True,
    ):
        self.tool_policies = dict(DEFAULT_TOOL_POLICIES if tool_policies is None else tool_policies)
        self.untrusted_roles = untrusted_roles
        self.tool_output_untrusted = tool_output_untrusted
        self.enforce_pf_as_pt = enforce_pf_as_pt

    # --- taint tracking -------------------------------------------------------------------
    def label_of_message(self, msg: dict) -> IntegrityLabel:
        """Integrity label of a single message. System/user/assistant = trusted; tool outputs =
        untrusted (when ``tool_output_untrusted``). Assistant messages inherit context integrity
        via the running join, so they carry no new taint of their own here."""
        role = msg.get("role")
        if role in self.untrusted_roles and self.tool_output_untrusted:
            return IntegrityLabel.untrusted()
        return IntegrityLabel.trusted()

    def context_integrity(self, messages) -> IntegrityLabel:
        """Join the integrity labels of an iterable of messages (default trusted = bottom)."""
        label = IntegrityLabel.trusted()
        for m in messages:
            label = label.join(self.label_of_message(m))
        return label

    # --- policy enforcement (planning/control-flow archetype) -----------------------------
    def policy_for(self, tool_name: str) -> str | None:
        """Return the policy class for a tool, or None if the tool is neutral (no check)."""
        return self.tool_policies.get(tool_name)

    def is_consequential(self, tool_name: str) -> bool:
        return self.policy_for(tool_name) is not None

    def allow_call(self, tool_name: str, args: dict, ctx: dict) -> tuple[bool, str]:
        """Check the per-tool policy against the labels (paper §4.3 ``policy(MakeCall ...)``).

        Returns (allowed, reason). A consequential/egress tool call generated in an untrusted
        context is a P-T violation and is blocked — this is what stops a prompt injection from
        triggering the action."""
        policy = self.policy_for(tool_name)
        if policy is None:
            return True, f"neutral tool '{tool_name}' (no policy)"

        ctx_integrity: IntegrityLabel = ctx.get("context_integrity", IntegrityLabel.trusted())
        if ctx_integrity.is_trusted:
            return True, f"'{tool_name}' [{policy}]: trusted context (P-T satisfied)"

        # Untrusted context. P-T fails. For PF_OR_PT we would need to check P-F (recipient
        # readers) — not tracked in this lightweight monitor, so conservatively block.
        if policy == PT or (policy == PF_OR_PT and self.enforce_pf_as_pt):
            return False, f"'{tool_name}' [{policy}]: P-T violation — call generated in an UNTRUSTED context"
        return True, f"'{tool_name}' [{policy}]: untrusted context allowed (enforce_pf_as_pt=False)"


if __name__ == "__main__":
    # Smoke: neutral tool always allowed; consequential tool allowed in trusted ctx, blocked in
    # untrusted ctx (the core security behavior).
    d = Defense()
    trusted = {"context_integrity": IntegrityLabel.trusted()}
    untrusted = {"context_integrity": IntegrityLabel.untrusted()}

    assert d.allow_call("read_emails", {}, untrusted)[0] is True, "neutral tool must always pass"
    assert d.allow_call("send_money", {}, trusted)[0] is True, "P-T must pass in trusted ctx"
    assert d.allow_call("send_money", {}, untrusted)[0] is False, "P-T must block in untrusted ctx"
    assert d.allow_call("send_email", {}, untrusted)[0] is False, "egress must block in untrusted ctx"

    # taint propagation: one tool output makes the whole context untrusted.
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
    assert not d.context_integrity(msgs).is_trusted, "tool output must taint context"
    assert d.context_integrity(msgs[:3]).is_trusted, "pre-tool context must be trusted"
    print("defense.py smoke OK — taint + P-T behave as specified")
