"""Live smoke test for the CaMeL full pipeline against the REAL Qwen3-32B (:8000).

CaMeL's pipeline normally runs inside the reference repo's own uv env and applies global
monkeypatches. Under the master .venv some reference deps may be missing, so this test
DEGRADES GRACEFULLY: it TRIES to build the composed AgentPipeline and, if the reference
pipeline can't be constructed here, skips with the exact error.

We assert ONLY that a composed AgentPipeline object is returned (by class name) — NEVER
ASR / blocking / utility. Skipped when the Qwen server is down via `live_qwen`.
"""
import pytest

from testlib import pathsetup

impl = pathsetup.load_impl("CaMeL")

pytestmark = pytest.mark.live


def test_build_agentdojo_pipeline(live_qwen):
    # Trigger the adapter's path inserts (camel-prompt-injection/src) so the impl's
    # `from agentdojo_adapter import build_pipeline` and `import camel...` resolve.
    pathsetup.get_ipi_adapter("camel")
    d = impl.Defense(variant="camel")
    try:
        pipeline = d.build_agentdojo_pipeline("banking")
    except Exception as e:  # reference pipeline not constructible in this env
        pytest.skip(f"CaMeL reference pipeline unavailable here: {e!r}")

    # A composed AgentDojo pipeline (do not run it). Identify by class name to avoid
    # importing/relying on the exact agentdojo class object across envs.
    assert pipeline is not None
    assert "Pipeline" in type(pipeline).__name__
