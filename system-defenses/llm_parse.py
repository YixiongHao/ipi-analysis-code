"""Shared robust-parse toolkit for LLM structured output across the IPI defenses.

Reasoning models (e.g. GLM 5.2) break naive `json.loads`: they (a) write a prose PREAMBLE before
the JSON ("I need to read the feed first…\n\n{…}") so `json.loads` fails at char 0; (b) wrap the
JSON in ```json fences; (c) return EMPTY content; or (d) refuse with plain text instead of JSON.
These helpers tolerate all four and FAIL OPEN (return a caller-supplied fallback) rather than crash.
Mirrors the pattern already proven in `Firewalls/impl/defense.py::_extract_json`.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json|python|JSON)?\s*(.*?)\s*```$", re.DOTALL)


def strip_fences(text: str) -> str:
    """Drop a single surrounding ```json / ```python / ``` fence if present."""
    text = (text or "").strip()
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text


def extract_json(text):
    """Parse the first JSON value out of a (possibly prose-prefixed / fenced) LLM response.

    Tries strict parse; else strips a fence; else extracts the first balanced {…} or […] block
    (handles GLM's preamble-then-JSON). Returns the parsed object/list, or None if nothing parses.
    """
    text = strip_fences(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                continue
    return None


def loads(text, fallback):
    """Robust `json.loads` replacement: return the extracted JSON, else `fallback`."""
    obj = extract_json(text)
    return obj if obj is not None else fallback


def loads_dict(text) -> dict:
    """Like `loads` but guarantees a dict (callers that do `.get(...)`). Non-dict / unparseable -> {}."""
    obj = extract_json(text)
    return obj if isinstance(obj, dict) else {}


def safe_args(raw) -> dict:
    """Tool-call arguments -> dict. Tolerates None, dict pass-through, and fenced/preamble/malformed
    JSON strings. Never raises; returns {} when nothing usable."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        obj = extract_json(raw)
        return obj if isinstance(obj, dict) else {}
    return {}


def extract_code_block(text, langs=("python",)) -> str:
    """Extract a fenced code block, tolerating a prose preamble and a missing/variant lang tag.
    Falls back to the stripped raw text if no fence is found."""
    text = text or ""
    m = re.search(r"```(?:" + "|".join(langs) + r")?\s*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()
