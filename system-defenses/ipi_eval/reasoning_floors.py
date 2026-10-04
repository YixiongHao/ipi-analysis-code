"""Lowest reasoning setting that actually WORKS, per model.

Audit + evidence grades: `planning/reasoning_floor_audit.md` (2026-08-05). A uniform
"reasoning off" arm is impossible across the quarterly lineage panel — 8 of 21 models have a
floor above zero, and two of them (`deepseek-r1`, `qwen3-235b-a22b-thinking-2507`) IGNORE the
disable flag SILENTLY, which would mislabel a reasoning-ON arm as reasoning-off.

Levels, lowest first:
  "none"       no reasoning knob exists (pre-reasoning model) -> send NO reasoning field at all.
               Distinct from "off": OpenAI 400s if you hand a reasoning param to gpt-4/gpt-4o.
  "off"        reasoning can genuinely be disabled -> `reasoning.enabled=false`.
  "minimal"/"low"/"medium"/"high"/"max"
               lowest effort level the model actually accepts (reasoning ON, max_tokens 8192).
  "mandatory"  reasoning cannot be controlled at all -> run reasoning ON and SAY SO. Never send
               a disable flag here: it either 400s or is silently ignored.

Keys are normalized slugs (provider prefix and `:tag` suffix stripped) — see `_norm`.
"""
from __future__ import annotations

# level -> (sends a reasoning field?, reasoning is actually generated?)
LEVELS = ("none", "off", "minimal", "low", "medium", "high", "max", "mandatory")

# model -> (floor, evidence, note). Evidence: MEASURED = we observed it on real contexts in
# system-defenses/hard_sample/eval_union381/reasoning_effort/ or in production runs;
# SOURCED = vendor docs / reproducible third-party report (web audit 2026-08-05).
FLOORS: dict[str, tuple[str, str, str]] = {
    # --- added 2026-08-06 from the semiannual transfer panel run (planning/open_to_frontier_transfer)
    # gpt-oss-20b is the one that BIT: no entry -> defaulted to "off" -> the endpoint 400s with
    # "Reasoning is mandatory for this endpoint and cannot be disabled" on ALL 200 records. Exactly
    # the silent-default failure this module's docstring warns about. 'mandatory' verified working.
    "gpt-oss-20b":           ("mandatory", "MEASURED", "endpoint 400s on any disable flag"),
    "gpt-oss-120b":          ("mandatory", "SOURCED", "same family; untested here"),
    # These ran a full 200-record arm at "off" with no reasoning-related error, which upgrades them
    # from unverified-default to measured.
    "llama-3.1-70b-instruct": ("none", "MEASURED", "pre-reasoning; 200/200 clean at off"),
    "llama-3.1-8b-instruct":  ("none", "MEASURED", "pre-reasoning; 200/200 clean at off"),
    "gemma-3-27b-it":        ("off", "MEASURED", "clean at off; its errors were 429s, not reasoning"),
    "o3":                    ("off", "MEASURED", "200-record arm clean at off"),
    "gpt-5.5":               ("off", "MEASURED", "200-record arm clean at off"),
    "claude-opus-4-5-20251101": ("off", "MEASURED", "200-record arm clean at off"),
    "claude-sonnet-4":       ("off", "MEASURED", "clean at off via OpenRouter"),
    "claude-3-haiku":        ("none", "SOURCED", "pre-3.7 Claude has NO extended-thinking knob; send no reasoning field (default 'off' would push a disable flag)"),
    # --- pre-reasoning: no knob exists ---------------------------------------------------
    "gpt-4":                 ("none", "SOURCED", "passing a reasoning param 400s"),
    "gpt-4-turbo":           ("none", "SOURCED", ""),
    "gpt-4o-2024-05-13":     ("none", "SOURCED", ""),
    "gpt-4o-2024-08-06":     ("none", "SOURCED", ""),
    "gpt-4o-2024-11-20":     ("none", "SOURCED", ""),
    "mixtral-8x7b-instruct-v0.1": ("none", "SOURCED", "local"),
    "mixtral-8x22b-instruct": ("none", "SOURCED", ""),
    "qwen2-72b-instruct":    ("none", "SOURCED", "local"),
    "qwen2-72b":             ("none", "SOURCED", "local served-name alias of qwen2-72b-instruct"),
    "qwen-2.5-72b-instruct": ("none", "SOURCED", "model card: 'Reasoning: No'"),
    "qwen2.5-72b-instruct":  ("none", "SOURCED", "local alias of the above"),
    "deepseek-chat":         ("none", "SOURCED", "original V3, no reasoning mode"),
    "qwen3-235b-a22b-instruct-2507": ("none", "SOURCED",
                                      "non-thinking half of the 2507 split; the swap for -thinking-2507"),

    # --- genuinely disable-able ----------------------------------------------------------
    "gpt-5.6-sol":     ("off", "MEASURED", "preflight_ext: 0 reasoning chars at off"),
    "claude-opus-5":   ("off", "MEASURED", "preflight_ext: 0 reasoning chars at off; NOT true of fable-5"),
    "glm-5.2":         ("off", "MEASURED", "our standing target runs this way"),
    "gemini-2.5-flash": ("off", "MEASURED", "first grid anchored it at off; 2.5 PRO cannot"),
    # Pre-2507 hybrid Qwen3 + the 3.5/3.6 lines: same weights BOTH ways, so a within-model
    # off-vs-low arm is possible (unlike the 2507 thinking/instruct split). qwen_off_probe.py.
    "qwen3-32b":        ("off", "MEASURED", "0 reasoning chars at off, 2,018 at low; hybrid"),
    "qwen3.5-9b":       ("off", "MEASURED", "0 reasoning chars at off, 6,185 at low"),
    "qwen3.5-35b-a3b":  ("off", "MEASURED", "0 reasoning chars at off, 640 at low"),
    "qwen3.6-35b-a3b":  ("off", "MEASURED", "0 reasoning chars at off, 2,748 at low"),
    "gemini-3-flash-preview": ("off", "MEASURED",
                               "we run it as worldsim with reasoning off; note Gemini 3 exposes "
                               "thinkingLevel with no true zero, so assert reasoning_tokens==0"),
    "gpt-5.4":         ("off", "MEASURED",
                        "off/minimal/high -> 0/13/47 reasoning tokens, so the report is live and "
                        "'off' is real. NB the PRO tier gpt-5.4-pro floors at medium; separate entry"),
    "deepseek-v3.2":   ("off", "SOURCED", "hybrid; tool-calling validated in BOTH modes"),
    "deepseek-v4-pro": ("off", "MEASURED", "union381 arms ran with reasoning off"),
    "kimi-k2.5":       ("off", "SOURCED",
                        "native thinking:{type:disabled}; the generic OR flag is reported not to "
                        "propagate on some gateways (null content) -> verify reasoning_tokens==0"),

    # --- floors above zero ---------------------------------------------------------------
    "gpt-5":           ("minimal", "SOURCED",
                        "minimal exists but DISABLES PARALLEL TOOL CALLS; use 'low' if the agent "
                        "loop depends on multi-call turns"),
    "o1":              ("low", "SOURCED", "no off state; empty-content risk even at low -> needs 8192"),
    "gemini-2.5-pro":  ("low", "SOURCED", "thinkingBudget=0 REJECTED; floor is 128 tokens"),
    "gemini-3.1-pro-preview": ("low", "MEASURED", "OR probe 08-24: reasoning.enabled=false -> 400 "
                        "'Reasoning is mandatory'; effort=low OK (182 reasoning tokens, bounded)"),
    "gemini-3.5-flash": ("low", "MEASURED", "OR probe 08-24 (ai-studio): reasoning.enabled=false -> 400 "
                        "'Reasoning is mandatory'; effort=low OK (120 reasoning tokens). NB Gemini 3.x "
                        "flash is reasoning-mandatory unlike 2.5-flash(off)/3-flash-preview(off)"),
    "claude-fable-5":  ("low", "SOURCED", "fable-5/mythos-5 always think; disable unsupported"),
    "gpt-5-mini":      ("low", "MEASURED", "preflight: 400 'Reasoning is mandatory' on off"),
    "grok-4.20":       ("low", "MEASURED", "first grid ran low/med/high"),
    "gpt-5.4-pro":     ("medium", "SOURCED", "Pro tier has no none/minimal/low; non-Pro gpt-5.4 does"),
    "kimi-k3":         ("max", "SOURCED", "only max is live at launch; low/high 'planned'"),

    # --- uncontrollable: reasoning always on ---------------------------------------------
    "deepseek-r1":      ("mandatory", "SOURCED", "reasoning-only; disable flag reported IGNORED"),
    "deepseek-r1-0528": ("mandatory", "SOURCED", "no official no-think; vendor says 'use V3 instead'"),
    "qwen3-235b-a22b-thinking-2507": ("mandatory", "SOURCED",
                                      "2507 split removed /no_think; flag SILENTLY ignored -> swap "
                                      "to qwen3-235b-a22b-instruct-2507 for a true-zero arm"),
    "qwen3-next-80b-a3b-thinking": ("mandatory", "MEASURED",
                                    "preflight: 400 'Reasoning is mandatory' on off"),
    "qwen3-30b-a3b": ("mandatory", "MEASURED",
                      "SILENT ignore: reasoning.enabled=false still emitted 2,226 reasoning "
                      "chars (more than at low). No error -> would corrupt an off arm unnoticed"),
    "qwen3.8-2.4t-a95b": ("mandatory", "MEASURED",
                          "per-vendor 2026Q3 Qwen; EVERY OR provider (siliconflow/together/"
                          "digitalocean/deepinfra/venice) 400s 'Reasoning is mandatory' on off; "
                          "runs reasoning-ON on siliconflow/fp8 (verified 08-17)"),

    # --- per-vendor lineage additions (08-17), keyed to the OR-slug tail (dots, not dashes) -----
    "qwen3.5-397b-a17b": ("off", "MEASURED", "2026Q1/Q2 Qwen; off SUPPRESSES on deepinfra/fp8 "
                          "(0 reasoning chars, tool call clean, 08-17)"),
    "qwen3-235b-a22b":   ("off", "SOURCED", "2025Q2 Qwen, LOCAL only; hybrid, off via vLLM /no_think"),
    "qwen3-235b-a22b-fp8": ("off", "MEASURED", "2025Q2 Qwen served locally as fp8 (node3 vLLM tunnel :8001); "
                            "hybrid, off via /no_think, model_finished no-loop in smoke 08-27"),
    "claude-sonnet-4.5": ("off", "SOURCED", "OR-slug key; hybrid, off = reasoning.enabled=false (confirm 0 chars in smoke)"),
    "claude-opus-4.5":   ("off", "SOURCED", "OR-slug alias of claude-opus-4-5-20251101 (=off MEASURED)"),
    "claude-opus-4.6":   ("off", "SOURCED", "OR-slug key; hybrid, off = reasoning.enabled=false (confirm 0 chars in smoke)"),
}


def _norm(model: str) -> str:
    """Strip the OpenRouter provider prefix and any `:tag` suffix (`:batch`, `:free`, `:nitro`)."""
    m = (model or "").strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[1]
    return m.split(":", 1)[0]


def floor_for(model: str, default: str = "off") -> str:
    """Lowest working reasoning level for `model`.

    Unknown models fall back to `default` ("off" = today's behaviour). That is the right default
    for the models we run, but it is exactly the case that fails silently on a reasoning-mandatory
    endpoint — add an entry rather than relying on it. `describe()` reports whether a lookup hit.
    """
    hit = FLOORS.get(_norm(model))
    return hit[0] if hit else default


def describe(model: str) -> str:
    """One-line provenance for logs, so a run records WHY it used the level it used."""
    hit = FLOORS.get(_norm(model))
    if not hit:
        return f"{model}: no floor entry -> defaulting to 'off' (UNVERIFIED for this model)"
    level, evidence, note = hit
    return f"{model}: floor={level} [{evidence}]" + (f" — {note}" if note else "")
