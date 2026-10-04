#!/usr/bin/env python3
"""Python re-host of @stackone/defender Tier 2 (ONNX MiniLM injection classifier).

Upstream: github.com/stackoneHQ/defender @ 0d56b92 (v0.7.2, Apache-2.0). The bundled
`minilm-multihead-v5` model dir is vendored verbatim under `defender/models/` (see
defender/README.md for provenance / re-fetch). This module ports ONLY the Tier 2
scoring path of `defendToolResult()` for a BARE-STRING input (our per-site protocol
feeds one isolated tool-output string per call, so extractStrings -> [text] and the
cross-string density damping never fires: it requires >2 strings). Tier 1 (regex
sanitizer), Tier 3 (BYO-LLM) and the SFE preprocessor are NOT ported.

Mirrored TS sources (file:line refs into the upstream repo):
  - chunk prep:      src/classifiers/tier2-classifier.ts prepareChunks/splitIntoSentences/packSentences
  - inference:       src/classifiers/onnx-classifier.ts  classifyBatchChunkPair (pad id 0, batch<=32,
                     multi-head logits [batch,2] row-major, sigmoid(logit/T))
  - decision:        src/core/prompt-defense.ts:751-853   per-string max + multi-head rule
  - boundary strip:  src/utils/boundary.ts stripBoundaryPatterns
Calibration (models/.../classifier_config.json): temperatureT=2.41, highRiskThreshold=0.64.
Multi-head rule thresholds {main>=0.5, aux<0.64} = README's FP-validated operating point
for the bundled model (NOT a library default upstream — must be opted into there).

Fidelity: verified chunk-identical + score-parity against the real Node package
(dist build) on store attack strings + benign texts — per-chunk max diff 3e-5,
string-level raw-main max diff 5e-5, 0 multihead-decision flips.

REQUIRES onnxruntime==1.21.0 to match onnxruntime-node 1.21.0. The int8-quantized
MiniLM is padding-sensitive (a chunk's score shifts with the batch's pad length),
so the ORT kernel version is load-bearing: at ORT 1.27 per-chunk scores drift up to
0.15 and flip near-threshold multihead decisions; at 1.21 the match is exact.

NOTE the shipped tokenizer.json has pad-to-256 baked in; upstream tokenizes with
padding:false, so we call no_padding() and pad batches manually with 0 ([PAD]).
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(HERE, "defender", "models", "minilm-multihead-v5")

_BOUNDARY_RES = [
    re.compile(r"\[UD-[A-Za-z0-9_-]+\]"),
    re.compile(r"\[/UD-[A-Za-z0-9_-]+\]"),
    re.compile(r"<user-data-[A-Za-z0-9_-]+>"),
    re.compile(r"</user-data-[A-Za-z0-9_-]+>"),
]
# tier2-classifier.ts splitIntoSentences: split on sentence enders, blank lines,
# newline-before-structural-char, or colon-newline; then sub-split >200-char chunks on \n.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\n+|\n(?=[A-Z0-9#\-*])|(?<=:)\s*\n")


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def strip_boundary_patterns(text):
    for rx in _BOUNDARY_RES:
        text = rx.sub("", text)
    return text


class DefenderTier2:
    MODEL_MAX_LEN = 256   # onnx-classifier.ts maxLength (private, incl. specials)
    MAX_BATCH_CHUNK = 32  # onnx-classifier.ts MAX_BATCH_CHUNK

    def __init__(self, model_dir=None, min_text_length=10, max_text_length=10000,
                 mh_main_threshold=0.5, mh_aux_threshold=0.64,
                 high_risk_threshold=None, temperature=None, intra_op_threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.mh_main_threshold = mh_main_threshold
        self.mh_aux_threshold = mh_aux_threshold

        cal = (json.load(open(os.path.join(self.model_dir, "classifier_config.json")))
               .get("calibration") or {})
        self.temperature = temperature if temperature is not None else cal.get("temperatureT", 1.0)
        self.high_risk_threshold = (high_risk_threshold if high_risk_threshold is not None
                                    else cal.get("highRiskThreshold", 0.8))

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads  # good neighbor on the shared box
        self.session = ort.InferenceSession(os.path.join(self.model_dir, "model_quantized.onnx"),
                                            so, providers=["CPUExecutionProvider"])
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        # upstream tokenizes with padding:false; the shipped tokenizer.json bakes pad-to-256 in
        self._tok_score = Tokenizer.from_file(tok_path)
        self._tok_score.no_padding()
        self._tok_score.enable_truncation(max_length=self.MODEL_MAX_LEN)
        self._tok_count = Tokenizer.from_file(tok_path)
        self._tok_count.no_padding()
        self._tok_count.no_truncation()

    # ------------------------- chunk preparation -------------------------

    def count_tokens(self, text):
        """Token count WITHOUT truncation, including [CLS]/[SEP] (countTokens)."""
        return len(self._tok_count.encode(text).ids)

    def _split_sentences(self, text):
        sentences = []
        for chunk in _SENT_SPLIT_RE.split(text):
            trimmed = chunk.strip()
            if not trimmed:
                continue
            if len(trimmed) > 200 and "\n" in trimmed:
                sentences.extend(s.strip() for s in trimmed.split("\n") if s.strip())
            else:
                sentences.append(trimmed)
        return sentences

    def _pack_sentences(self, sentences, max_content_tokens):
        chunks, current, current_tokens = [], [], 0
        for s in sentences:
            s_content = max(0, self.count_tokens(s) - 2)  # minus [CLS]+[SEP]
            if s_content > max_content_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.append(s)  # own chunk; tokenizer truncates at inference
                continue
            # WordPiece emits no inter-word whitespace token: counts add directly
            if current_tokens + s_content > max_content_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [s], s_content
            else:
                current.append(s)
                current_tokens += s_content
        if current:
            chunks.append(" ".join(current))
        return chunks

    def prepare_chunks(self, text):
        """tier2-classifier.ts prepareChunks: -> (chunks, skip_reason)."""
        text = strip_boundary_patterns(text)
        if len(text) < self.min_text_length:
            return [], "Text below minTextLength"
        bounded = text[: self.max_text_length]
        # fast path: char count + specials is an upper bound on WordPiece tokens
        if len(bounded) + 2 <= self.MODEL_MAX_LEN:
            return [bounded], None
        if self.count_tokens(bounded) <= self.MODEL_MAX_LEN:
            return [bounded], None
        sentences = [s for s in self._split_sentences(bounded) if len(s) >= self.min_text_length]
        if not sentences:
            return [], "No classifiable sentences"
        return self._pack_sentences(sentences, self.MODEL_MAX_LEN - 2), None

    # ----------------------------- inference -----------------------------

    def classify_chunks(self, chunks):
        """classifyBatchChunkPair: -> [(main, aux), ...], sigmoid(logit/T) both heads."""
        import numpy as np
        pairs = []
        T = self.temperature
        for off in range(0, len(chunks), self.MAX_BATCH_CHUNK):
            batch = chunks[off: off + self.MAX_BATCH_CHUNK]
            encs = [self._tok_score.encode(t) for t in batch]
            max_len = max(len(e.ids) for e in encs)
            ii = np.zeros((len(batch), max_len), dtype=np.int64)   # pad id 0 = [PAD]
            aa = np.zeros((len(batch), max_len), dtype=np.int64)
            for i, e in enumerate(encs):
                ii[i, : len(e.ids)] = e.ids
                aa[i, : len(e.ids)] = e.attention_mask
            logits = self.session.run(["logits"], {"input_ids": ii, "attention_mask": aa})[0]
            for row in logits:
                pairs.append((_sigmoid(float(row[0]) / T), _sigmoid(float(row[1]) / T)))
        return pairs

    # ------------------------------ scoring ------------------------------

    def score_text(self, text):
        """Score ONE isolated string exactly as defendToolResult() scores a bare-string
        payload (single extracted string => density damping structurally inert).
        Returns the full diagnostics dict; `main`/`aux` carry every chunk's scores so
        downstream analysis can re-aggregate without re-running inference."""
        chunks, skip_reason = self.prepare_chunks(text)
        if not chunks:
            return {"skip_reason": skip_reason, "n_chunks": 0, "main": [], "aux": [],
                    "raw_max_main": 0.0, "aux_of_max": None, "effective_score": 0.0,
                    "blocked_singlehead": False, "blocked_multihead": False,
                    "max_chunk": None}
        pairs = self.classify_chunks(chunks)
        # prompt-defense.ts:751-790 per-string max (sMax starts 0, non-finite -> 0)
        s_max, s_max_aux, s_max_chunk = 0.0, None, ""
        mh_any, mh_top_main, mh_top_aux, mh_top_chunk = False, -1.0, None, ""
        for (main, aux), chunk in zip(pairs, chunks):
            main = main if math.isfinite(main) else 0.0
            if main > s_max:
                s_max, s_max_aux, s_max_chunk = main, aux, chunk
            if main >= self.mh_main_threshold and aux < self.mh_aux_threshold:
                mh_any = True
                if main > mh_top_main:
                    mh_top_main, mh_top_aux, mh_top_chunk = main, aux, chunk
        effective = s_max  # single string: no density adjustment (needs >2 strings)
        report_chunk = mh_top_chunk if mh_any else s_max_chunk
        return {
            "skip_reason": None,
            "n_chunks": len(chunks),
            "main": [round(m, 4) for m, _ in pairs],
            "aux": [round(a, 4) for _, a in pairs],
            "raw_max_main": round(s_max, 4),
            "aux_of_max": None if s_max_aux is None else round(s_max_aux, 4),
            "effective_score": round(effective, 4),
            "blocked_singlehead": effective >= self.high_risk_threshold,
            "blocked_multihead": mh_any,
            "mh_block_main": round(mh_top_main, 4) if mh_any else None,
            "mh_block_aux": round(mh_top_aux, 4) if mh_any else None,
            "max_chunk": report_chunk[:300],
        }
