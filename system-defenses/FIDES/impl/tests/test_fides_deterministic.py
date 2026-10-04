"""Deterministic tests for the FIDES taint + per-tool-policy core (defense.py).

FIDES's allow/deny decision is a *pure deterministic function* of message role
(taint) and the Table-3 tool policy — no LLM. Asserting that decision is therefore
testing CODE correctness, not defense efficacy. These tests cover:
  - label_of_message  (role -> IntegrityLabel; respects untrusted_roles / tool_output_untrusted)
  - context_integrity (join over a message list; all-trusted -> trusted, any tool -> untrusted)
  - policy_for        (Table-3 keys/values; neutral tools -> None)
  - allow_call        (P-T blocks in untrusted ctx, allows in trusted ctx; neutral always allows;
                       enforce_pf_as_pt toggle for P-F|P-T tools)
  - the IntegrityLabel lattice algebra (leq / join / meet; trusted = bottom)
"""
from __future__ import annotations

import pytest

from testlib import pathsetup

impl = pathsetup.load_impl("FIDES")
Defense = impl.Defense
IntegrityLabel = impl.IntegrityLabel
PT = impl.PT
PF_OR_PT = impl.PF_OR_PT
DEFAULT_TOOL_POLICIES = impl.DEFAULT_TOOL_POLICIES

TRUSTED = IntegrityLabel.trusted()
UNTRUSTED = IntegrityLabel.untrusted()


# --------------------------------------------------------------------------- lattice algebra ---
class TestIntegrityLattice:
    def test_trusted_is_bottom(self):
        # trusted <= untrusted, untrusted !<= trusted (trusted is the bottom element).
        assert TRUSTED.leq(UNTRUSTED) is True
        assert UNTRUSTED.leq(TRUSTED) is False
        assert TRUSTED.leq(TRUSTED) is True
        assert UNTRUSTED.leq(UNTRUSTED) is True

    def test_dunder_le_matches_leq(self):
        assert (TRUSTED <= UNTRUSTED) is True
        assert (UNTRUSTED <= TRUSTED) is False

    def test_join_is_least_upper_bound(self):
        # join = max integrity-rank; any untrusted poisons.
        assert TRUSTED.join(TRUSTED).is_trusted is True
        assert TRUSTED.join(UNTRUSTED).is_trusted is False
        assert UNTRUSTED.join(TRUSTED).is_trusted is False
        assert UNTRUSTED.join(UNTRUSTED).is_trusted is False

    def test_meet_is_greatest_lower_bound(self):
        assert TRUSTED.meet(TRUSTED).is_trusted is True
        assert TRUSTED.meet(UNTRUSTED).is_trusted is True
        assert UNTRUSTED.meet(TRUSTED).is_trusted is True
        assert UNTRUSTED.meet(UNTRUSTED).is_trusted is False

    def test_is_trusted_property(self):
        assert TRUSTED.is_trusted is True
        assert UNTRUSTED.is_trusted is False


# ---------------------------------------------------------------------- label_of_message ---
class TestLabelOfMessage:
    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_trusted_roles(self, role):
        d = Defense()
        assert d.label_of_message({"role": role}).is_trusted is True

    def test_tool_role_is_untrusted_by_default(self):
        d = Defense()
        assert d.label_of_message({"role": "tool"}).is_trusted is False

    def test_tool_output_untrusted_false_disables_tainting(self):
        d = Defense(tool_output_untrusted=False)
        # With tainting disabled, even a tool output is treated as trusted.
        assert d.label_of_message({"role": "tool"}).is_trusted is True

    def test_custom_untrusted_roles(self):
        # Treat a custom role as low-integrity; tool then becomes trusted.
        d = Defense(untrusted_roles=("function",))
        assert d.label_of_message({"role": "function"}).is_trusted is False
        assert d.label_of_message({"role": "tool"}).is_trusted is True

    def test_unknown_role_defaults_trusted(self):
        d = Defense()
        assert d.label_of_message({"role": "developer"}).is_trusted is True
        assert d.label_of_message({}).is_trusted is True  # no role key


# ---------------------------------------------------------------------- context_integrity ---
class TestContextIntegrity:
    def test_empty_context_is_trusted(self):
        assert Defense().context_integrity([]).is_trusted is True

    def test_all_trusted_roles_stay_trusted(self):
        d = Defense()
        msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
        assert d.context_integrity(msgs).is_trusted is True

    def test_any_tool_output_taints_whole_context(self):
        d = Defense()
        msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
        assert d.context_integrity(msgs).is_trusted is False

    def test_taint_is_monotonic_no_recovery(self):
        # Once untrusted, a later trusted message must NOT clear the taint (join is monotone up).
        d = Defense()
        msgs = [{"role": "user"}, {"role": "tool"}, {"role": "assistant"}, {"role": "user"}]
        assert d.context_integrity(msgs).is_trusted is False

    def test_pre_tool_prefix_is_trusted(self):
        d = Defense()
        msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
        assert d.context_integrity(msgs[:3]).is_trusted is True

    def test_disabled_tainting_keeps_context_trusted(self):
        d = Defense(tool_output_untrusted=False)
        msgs = [{"role": "user"}, {"role": "tool"}, {"role": "tool"}]
        assert d.context_integrity(msgs).is_trusted is True


# ---------------------------------------------------------------------------- policy_for ---
class TestPolicyFor:
    def test_neutral_tools_return_none(self):
        d = Defense()
        # IPI-fixture tools: none are in Table 3 -> all neutral.
        for name in ["run_sql_query", "read_file", "write_file", "list_directory",
                     "execute_bash", "think", "finish"]:
            assert d.policy_for(name) is None, name

    def test_pt_tools(self):
        d = Defense()
        assert d.policy_for("send_money") == PT
        assert d.policy_for("delete_file") == PT
        assert d.policy_for("update_password") == PT

    def test_pf_or_pt_tools(self):
        d = Defense()
        assert d.policy_for("send_direct_message") == PF_OR_PT
        assert d.policy_for("send_email") == PF_OR_PT
        assert d.policy_for("create_calendar_event") == PF_OR_PT

    def test_default_policy_constants(self):
        # Guard the exact policy-class string values transcribed from Table 3.
        assert PT == "P-T"
        assert PF_OR_PT == "P-F or P-T"
        # Every registered value must be one of the two policy classes.
        assert set(DEFAULT_TOOL_POLICIES.values()) == {PT, PF_OR_PT}

    def test_is_consequential(self):
        d = Defense()
        assert d.is_consequential("send_money") is True
        assert d.is_consequential("send_direct_message") is True
        assert d.is_consequential("run_sql_query") is False
        assert d.is_consequential("read_file") is False

    def test_custom_tool_policies_override(self):
        # The IPI decisive tool is neutral by default; a deployer can register it as P-T.
        d = Defense(tool_policies={"run_sql_query": PT})
        assert d.policy_for("run_sql_query") == PT
        assert d.policy_for("send_money") is None  # default table replaced wholesale


# ----------------------------------------------------------------------------- allow_call ---
class TestAllowCall:
    def test_neutral_tool_always_allowed(self):
        d = Defense()
        for ctx in ({"context_integrity": TRUSTED}, {"context_integrity": UNTRUSTED}):
            allow, reason = d.allow_call("run_sql_query", {}, ctx)
            assert allow is True
            assert "neutral" in reason

    def test_neutral_tool_allowed_with_empty_ctx(self):
        # Missing context_integrity key defaults to trusted; neutral tool still passes.
        allow, _ = Defense().allow_call("read_file", {}, {})
        assert allow is True

    def test_pt_tool_allowed_in_trusted_ctx(self):
        d = Defense()
        allow, reason = d.allow_call("send_money", {"amount": 1}, {"context_integrity": TRUSTED})
        assert allow is True
        assert "trusted context" in reason

    def test_pt_tool_blocked_in_untrusted_ctx(self):
        d = Defense()
        allow, reason = d.allow_call("send_money", {"amount": 1}, {"context_integrity": UNTRUSTED})
        assert allow is False
        assert "P-T violation" in reason

    def test_pt_tool_defaults_trusted_when_ctx_missing(self):
        # No context_integrity key -> defaults to trusted -> P-T satisfied.
        allow, _ = Defense().allow_call("delete_file", {}, {})
        assert allow is True

    def test_pf_or_pt_blocked_when_enforce_pf_as_pt_true(self):
        d = Defense(enforce_pf_as_pt=True)  # default
        allow, reason = d.allow_call(
            "send_direct_message", {"recipient": "x"}, {"context_integrity": UNTRUSTED})
        assert allow is False
        assert "P-T violation" in reason

    def test_pf_or_pt_allowed_when_enforce_pf_as_pt_false(self):
        d = Defense(enforce_pf_as_pt=False)
        allow, reason = d.allow_call(
            "send_direct_message", {"recipient": "x"}, {"context_integrity": UNTRUSTED})
        assert allow is True
        assert "enforce_pf_as_pt=False" in reason

    def test_pt_tool_blocked_even_with_enforce_pf_as_pt_false(self):
        # The toggle only relaxes P-F|P-T egress tools; strict P-T tools still block.
        d = Defense(enforce_pf_as_pt=False)
        allow, _ = d.allow_call("send_money", {}, {"context_integrity": UNTRUSTED})
        assert allow is False

    def test_pf_or_pt_allowed_in_trusted_ctx_regardless_of_toggle(self):
        for toggle in (True, False):
            d = Defense(enforce_pf_as_pt=toggle)
            allow, _ = d.allow_call("send_email", {}, {"context_integrity": TRUSTED})
            assert allow is True

    def test_returns_bool_and_str(self):
        allow, reason = Defense().allow_call("send_money", {}, {"context_integrity": UNTRUSTED})
        assert isinstance(allow, bool)
        assert isinstance(reason, str)


# --------------------------------------------------- end-to-end deterministic taint -> block ---
class TestTaintToBlock:
    def test_pt_call_after_tool_output_is_blocked(self):
        """The headline path: a tool output taints the context, so a subsequent P-T call is
        a P-T violation. Deterministic — no model involved."""
        d = Defense()
        msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
        ctx = {"context_integrity": d.context_integrity(msgs)}
        allow, _ = d.allow_call("send_money", {"amount": 9999}, ctx)
        assert allow is False

    def test_pt_call_before_any_tool_output_is_allowed(self):
        d = Defense()
        msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
        ctx = {"context_integrity": d.context_integrity(msgs)}
        allow, _ = d.allow_call("send_money", {"amount": 1}, ctx)
        assert allow is True
