"""Adapter wiring tests for the CaMeL ipi_eval adapter (CamelDefense).

Construction / wiring only — NO rollout, NO efficacy. We assert the adapter advertises
the full-loop interface (drives_loop True, name "camel", run_loop present) and that
building it ran the path-insert wiring the real adapter depends on (impl/ and
camel-prompt-injection/src on sys.path).

If constructing the adapter raises under the master .venv, the whole module is skipped
with the exact import error (the deterministic suite does not depend on this).
"""
import sys

import pytest

from testlib import pathsetup

# Try to build the adapter once. If it fails under the master .venv, skip the module
# with the real error so the failure is visible without breaking the deterministic suite.
try:
    _ADAPTER = pathsetup.get_ipi_adapter("camel")
    _BUILD_ERROR = None
except Exception as e:  # pragma: no cover - env dependent
    _ADAPTER = None
    _BUILD_ERROR = e

pytestmark = pytest.mark.skipif(
    _ADAPTER is None,
    reason=f"CamelDefense adapter unavailable under master .venv: {_BUILD_ERROR!r}",
)


class TestCamelAdapter:
    def test_is_camel_defense(self):
        # Compare by class name (each defense's adapter is loaded under a distinct
        # module, so isinstance across the registry is unreliable).
        assert type(_ADAPTER).__name__ == "CamelDefense"

    def test_name_is_camel(self):
        assert _ADAPTER.name == "camel"

    def test_drives_loop_true(self):
        # CaMeL is a full-loop defense (it owns the agent loop, not a per-message seam).
        assert _ADAPTER.drives_loop() is True

    def test_run_loop_present(self):
        assert hasattr(_ADAPTER, "run_loop")
        assert callable(_ADAPTER.run_loop)

    def test_holds_impl_defense_for_config(self):
        # The adapter reuses the impl Defense for variant validation/config.
        assert type(_ADAPTER.impl).__name__ == "Defense"
        assert _ADAPTER.impl.variant in ("undefended", "camel", "camel+secpol")

    def test_path_inserts_happened(self):
        # Constructing the adapter must have put impl/ and camel-prompt-injection/src on
        # sys.path (the adapter's import-time wiring). This verifies the wiring code ran.
        paths = [p.replace("\\", "/") for p in sys.path]   # OS-agnostic separators
        assert any(p.endswith("CaMeL/impl") for p in paths), \
            "CaMeL/impl not on sys.path"
        assert any("camel-prompt-injection/src" in p for p in paths), \
            "camel-prompt-injection/src not on sys.path"
