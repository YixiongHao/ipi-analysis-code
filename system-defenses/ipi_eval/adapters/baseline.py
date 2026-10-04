"""No-defense baseline: identity hooks. Establishes the un-defended ASR."""
from __future__ import annotations

from ..defense_base import DefenseAdapter


class BaselineDefense(DefenseAdapter):
    name = "baseline"
