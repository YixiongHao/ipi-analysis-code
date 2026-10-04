"""sys.path wiring for the defense test suites.

Importing ``testlib`` runs ``ensure_paths()`` once. Suites that need a defense's
own ``impl/defense.py`` call ``load_impl("<Name>")`` which loads it under a unique
module name (mirroring how ipi_eval/adapters/<name>.py loads it by file path), so
the six ``defense.py`` files never collide in ``sys.modules``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SD_ROOT = Path(__file__).resolve().parents[1]          # system-defenses/
REPO_ROOT = SD_ROOT.parent                             # IPI/

# Fallback vendored source roots, keyed by the module they provide. We add these
# ONLY if the module isn't already importable, so we never shadow an environment
# that ships its own copy (e.g. the CaMeL reference uv env has its own pinned
# `agentdojo`; inserting our vendored src in front of it would break CaMeL).
_FALLBACK_SRC = [
    ("agentdojo", REPO_ROOT / "agentdojo" / "src"),
    ("ipi_arena_bench", REPO_ROOT / "ipi_arena_os" / "src"),
]

_DONE = False


def ensure_paths() -> None:
    global _DONE
    if _DONE:
        return
    # Always needed so `import ipi_eval` / `import testlib` resolve.
    if str(SD_ROOT) not in sys.path:
        sys.path.insert(0, str(SD_ROOT))
    for mod, p in _FALLBACK_SRC:
        if importlib.util.find_spec(mod) is None and p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    _DONE = True


ensure_paths()


def impl_dir(name: str) -> Path:
    """impl/ directory for a defense, e.g. impl_dir('FIDES')."""
    return SD_ROOT / name / "impl"


def add_impl(name: str) -> Path:
    """Put a defense's impl/ on sys.path. For IPIGuard, import its fork env first
    (it rewrites sys.path so `agentdojo` resolves from the fork). Returns the dir."""
    d = impl_dir(name)
    sp = str(d)
    if name == "IPIGuard":
        # _fork_env MUST be imported before agentdojo for IPIGuard's defense.py.
        if sp not in sys.path:
            sys.path.insert(0, sp)
        importlib.import_module("_fork_env")  # noqa: F401
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return d


def load_impl(name: str, filename: str = "defense.py",
              modname: str | None = None) -> ModuleType:
    """Load a defense's impl module under a unique name to avoid collisions."""
    add_impl(name)
    path = impl_dir(name) / filename
    modname = modname or f"_{name.lower()}_impl_{Path(filename).stem}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def get_ipi_adapter(name: str, **kwargs):
    """Instantiate the ipi_eval DefenseAdapter for `name` via the real registry."""
    ensure_paths()
    from ipi_eval.adapters import get_defense  # type: ignore
    return get_defense(name, **kwargs)
