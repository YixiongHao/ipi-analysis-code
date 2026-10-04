"""Make `import agentdojo` resolve to the IPIGuard reference fork (the only copy that
has the `ipiguard` defense), without editing either agentdojo tree.

Two problems this solves:
  1. The master venv installs the *main* agentdojo clone (no ipiguard). We prepend the
     fork's `src/` to sys.path so the whole `agentdojo` package resolves from the fork.
  2. The fork imports `vertexai` / `proto.marshal...` at module top (Google provider,
     which we never use) and those aren't installed. We inject lazy stub modules into
     sys.modules so the import succeeds.

Import this module BEFORE importing anything from `agentdojo`.
"""
import os
import sys
import types

_FORK_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ipiguard", "agentdojo", "src",
)
assert os.path.isdir(_FORK_SRC), f"fork src not found: {_FORK_SRC}"
if "agentdojo" in sys.modules:
    raise RuntimeError("import _fork_env before agentdojo is imported")
sys.path.insert(0, _FORK_SRC)


class _Any:
    """A recursive dummy that tolerates any attribute access, call, or subscript —
    enough to survive module-load-time references like `genai.FunctionDeclaration`
    used in annotations/base classes of code paths we never execute."""

    def __getattr__(self, name):
        return _ANY

    def __call__(self, *a, **k):
        return _ANY

    def __getitem__(self, k):
        return _ANY

    # tolerate use in `X | Y` type-union annotations (real_type | dummy, etc.)
    def __or__(self, other):
        return _ANY

    def __ror__(self, other):
        return _ANY


_ANY = _Any()


class _Stub(types.ModuleType):
    """A stand-in module whose every missing attribute resolves to `_ANY`, so that
    `from <stub> import Anything` and `<stub>.Anything` both succeed at import time."""

    def __getattr__(self, name):  # only called for missing attrs
        return _ANY


def _install_stub(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = _Stub(name)
    # bind as a real attribute on the parent so `import a.b as c` finds the submodule
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, sys.modules[name])


# Only the trees that are actually missing (google.protobuf is present in the venv).
for _m in (
    "vertexai",
    "vertexai.generative_models",
    "proto",
    "proto.marshal",
    "proto.marshal.collections",
    "proto.marshal.collections.maps",
    "proto.marshal.collections.repeated",
):
    _install_stub(_m)
