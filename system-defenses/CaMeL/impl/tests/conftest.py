"""CaMeL-suite bootstrap.

The reference `camel` package and the pinned `agentdojo` in the CaMeL uv env have
an import-order-sensitive circular import in agentdojo's suite registration. The
repo's own entrypoints (main.py / run_agentdojo.py) import agentdojo *fully*
(`from agentdojo.task_suite import get_suite`) before importing any `camel`
submodule. We mirror that order here so that — when this suite is run under the
CaMeL uv env — the live pipeline build and the reference-dependent patch-helper
tests can actually execute instead of hitting the circular import.

Harmless under the master .venv: agentdojo imports cleanly there too; the `camel`
package simply isn't importable, so those tests still skip as before.
"""
from testlib import pathsetup  # noqa: F401  (ensures sys.path is wired)

try:
    # Supported order: full agentdojo registration before any camel submodule.
    from agentdojo import attacks, benchmark, logging  # noqa: F401
    from agentdojo.task_suite import get_suite  # noqa: F401
except Exception:
    # Odd env: let individual tests skip on their own import guards.
    pass
