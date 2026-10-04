"""Gemini-native target client with the FLEX service tier (50% off, best-effort, slower off-peak).

Used ONLY when the attacked TARGET is a Gemini model pointed at Google's native generateContent
endpoint. Flex (`serviceTier:"flex"`) is NOT honored on the OpenAI-compat endpoint — that path
silently ignores `service_tier` and bills at standard rate — so the discount requires this native
call. Presents the same `.chat(messages, tools) -> ChatResponse` contract as ipi_arena_bench's
LLMClient, so it is a drop-in target. The GLM / vLLM / world-sim / judge paths are unchanged.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from ipi_arena_compat import ChatResponse, ToolCall

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Optional per-call latency capture: set GEMINI_FLEX_LATENCY_LOG=<path> to append one JSON line per
# chat() call (elapsed seconds, #503/5xx retries, status). Off (no overhead) when the env is unset.
_LAT_LOG = os.environ.get("GEMINI_FLEX_LATENCY_LOG")
_lat_lock = threading.Lock()


def _log_latency(elapsed: float, retries: int, status: str) -> None:
    if not _LAT_LOG:
        return
    try:
        with _lat_lock, open(_LAT_LOG, "a") as f:
            f.write(json.dumps({"elapsed": round(elapsed, 3), "retries": retries, "status": status}) + "\n")
    except Exception:
        pass


def _clean_schema(s):
    """Reduce an OpenAI/JSON-schema node to the subset Gemini's functionDeclarations accept
    (strips additionalProperties / $schema / default / format / anyOf etc.; normalizes a
    list `type` like ["string","null"] to a single type + `nullable`)."""
    if not isinstance(s, dict):
        return s
    out: dict = {}
    t = s.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if non_null:
            out["type"] = non_null[0]
        if "null" in t:
            out["nullable"] = True
    elif t:
        out["type"] = t
    for k in ("description", "enum"):
        if k in s:
            out[k] = s[k]
    if isinstance(s.get("properties"), dict):
        out["properties"] = {pk: _clean_schema(pv) for pk, pv in s["properties"].items()}
        if "required" in s:
            out["required"] = s["required"]
    if "items" in s:
        out["items"] = _clean_schema(s["items"])
    if "type" not in out:
        out["type"] = "object" if "properties" in out else "string"
    return out


def _sanitize_fn_name(name: str) -> str:
    """Gemini function names must match [A-Za-z_][A-Za-z0-9_-]* — some IPI tools use `.`/`:` etc.
    Replace illegal chars with `_` and ensure a valid leading char."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name or "tool")
    if not re.match(r"[A-Za-z_]", s):
        s = "_" + s
    return s


def _to_gemini(messages: list[dict], tools: list[dict] | None) -> tuple[dict, dict]:
    """Translate OpenAI chat messages + tools into a Gemini generateContent request body.
    Returns (body, name_map) where name_map maps the sanitized function name sent to Gemini back
    to the original tool name, so the response's functionCall can be restored for the engine."""
    # tool_call_id -> function name, so a later role=="tool" turn can name its functionResponse.
    id2name: dict[str, str] = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fid, fn = tc.get("id"), (tc.get("function") or {}).get("name")
            if fid and fn:
                id2name[fid] = fn

    system_txt: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # multimodal-style content -> join text parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            if content:
                system_txt.append(content)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content or ""}]})
        elif role == "assistant":
            # Render prior tool calls as TEXT, not `functionCall` parts. Gemini 3.x requires a
            # `thought_signature` on every history functionCall (it has no REST field to inject a
            # placeholder, and our replayed prefill comes from a different model with no signature),
            # so structured history calls 400. Text avoids it entirely; the model still emits real
            # structured calls in its RESPONSE (tools are declared), and the engine's defenses see
            # the structured OpenAI messages upstream of this client.
            text = content or ""
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                text += f"\n[Called tool {fn.get('name')} with arguments: {fn.get('arguments') or '{}'}]"
            contents.append({"role": "model", "parts": [{"text": text or "(no content)"}]})
        elif role == "tool":
            name = id2name.get(m.get("tool_call_id"), m.get("name") or "tool")
            resp = content if isinstance(content, str) else json.dumps(content)
            contents.append({"role": "user", "parts": [{"text": f"[Result of tool {name}]: {resp}"}]})

    body: dict = {"contents": contents}
    if system_txt:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_txt)}]}
    name_map: dict = {}
    if tools:
        decls = []
        for t in tools:
            f = t.get("function", t)
            orig = f.get("name")
            san = _sanitize_fn_name(orig)
            base, k = san, 1
            while san in name_map and name_map[san] != orig:  # keep distinct originals distinct
                san, k = f"{base}_{k}", k + 1
            name_map[san] = orig
            decl: dict = {"name": san, "description": f.get("description", "")}
            params = _clean_schema(f.get("parameters")) if f.get("parameters") else None
            # Gemini rejects an object schema with no properties; omit parameters for no-arg tools.
            if params and params.get("properties"):
                decl["parameters"] = params
            decls.append(decl)
        body["tools"] = [{"functionDeclarations": decls}]
    return body, name_map


class GeminiFlexClient:
    """Native-Gemini target client (flex tier). Duck-compatible with LLMClient for the target leg."""

    provider = "gemini-flex"

    def __init__(self, model: str, api_key: str, temperature: float = 0.0,
                 service_tier: str = "flex", thinking: bool = False,
                 max_retries: int = 8, backoff: float = 2.0):
        self.model = model.replace("models/", "")
        self.api_key = api_key
        self.default_temperature = temperature
        self.service_tier = service_tier
        # Thinking OFF by default: Gemini 3.x attaches a `thoughtSignature` to function calls that
        # MUST be echoed back in history, which our OpenAI-format round-trip drops (-> HTTP 400).
        # Disabling thinking removes the signature requirement. (Thinking-ON would need the signature
        # threaded through the engine's tool_call ids — out of scope for this target leg.)
        self.thinking = thinking
        self.max_retries = max_retries
        self.backoff = backoff

    def chat(self, messages, tools=None, temperature: float | None = None,
             max_tokens: int = 4096) -> ChatResponse:
        if temperature is None:
            temperature = self.default_temperature
        body, name_map = _to_gemini(messages, tools)
        gen_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if not self.thinking:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}  # no thoughtSignature round-trip
        body["generationConfig"] = gen_cfg
        if self.service_tier:
            body["serviceTier"] = self.service_tier  # camelCase top-level (snake_case is ignored)
        url = f"{_BASE}/{self.model}:generateContent"
        data = json.dumps(body).encode()
        last_err, retries, t0 = None, 0, time.monotonic()
        for i in range(self.max_retries):
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json", "x-goog-api-key": self.api_key})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    out = self._parse(json.loads(resp.read()), name_map)
                _log_latency(time.monotonic() - t0, retries, "ok")
                return out
            except urllib.error.HTTPError as e:
                last_err = e
                # flex is best-effort (503, no auto-fallback); 429/5xx are transient -> backoff retry.
                if e.code in (429, 500, 502, 503, 504):
                    retries += 1
                    time.sleep(self.backoff * (i + 1))
                    continue
                _log_latency(time.monotonic() - t0, retries, f"http_{e.code}")
                raise RuntimeError(f"Gemini flex HTTP {e.code}: {e.read().decode()[:300]}") from e
        _log_latency(time.monotonic() - t0, retries, "exhausted")
        raise RuntimeError(f"Gemini flex exhausted {self.max_retries} retries: {last_err}")

    def _parse(self, j: dict, name_map: dict | None = None) -> ChatResponse:
        name_map = name_map or {}
        content, tool_calls = "", []
        cands = j.get("candidates") or []
        if cands:
            for k, p in enumerate((cands[0].get("content") or {}).get("parts") or []):
                if "text" in p:
                    content += p["text"]
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append(ToolCall(name=name_map.get(fc.get("name"), fc.get("name")),
                                               arguments=fc.get("args") or {},
                                               id=fc.get("id") or f"gem_{k}"))
        um = j.get("usageMetadata") or {}
        usage = {"prompt_tokens": um.get("promptTokenCount", 0),
                 "completion_tokens": um.get("candidatesTokenCount", 0),
                 "total_tokens": um.get("totalTokenCount", 0)} if um else {}
        return ChatResponse(content=content or None, tool_calls=tool_calls,
                            raw_response=j, model=self.model, usage=usage)
