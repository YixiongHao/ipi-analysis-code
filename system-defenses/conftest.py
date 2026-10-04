"""Top-level conftest for the defense test suites.

pytest prepends this directory (system-defenses/) to sys.path, so `import testlib`
resolves. Shared fixtures live here so per-defense suites stay thin.
"""
import sys
from pathlib import Path

_SD_ROOT = Path(__file__).resolve().parent
if str(_SD_ROOT) not in sys.path:
    sys.path.insert(0, str(_SD_ROOT))

import pytest  # noqa: E402

from testlib import fixtures, stubs  # noqa: E402


@pytest.fixture
def ipi_record():
    """Frozen IPI-transcript fixture (DROP-TABLE injection)."""
    return fixtures.ipi_transcript()


@pytest.fixture
def adojo():
    """Frozen AgentDojo slack fixture."""
    return fixtures.agentdojo_slack()


@pytest.fixture
def live_qwen():
    """Skip the test unless the local Qwen server (:8000) is reachable."""
    if not stubs.qwen_up():
        pytest.skip("local Qwen server (:8000) not reachable")


@pytest.fixture
def live_bge():
    """Skip the test unless the local BGE embedding server (:8001) is reachable."""
    if not stubs.bge_up():
        pytest.skip("local BGE server (:8001) not reachable")
