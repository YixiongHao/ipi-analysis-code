# StackOne Defender, Tier 2 classifier

A Python port of the Tier 2 classifier in `@stackone/defender`, registered as
`stackone-defender-tier2` in `../defenses.json`.

- **Upstream:** `github.com/stackoneHQ/defender` at `0d56b92` (npm `@stackone/defender` v0.7.2,
  Apache-2.0; license in `LICENSE.upstream`).
- **Model:** a fine-tuned multi-head MiniLM-L6-v2, int8 ONNX, about 22 MB, CPU only. It scores
  one untrusted tool-output string and returns `main` (injection score) and `aux` (a veto for
  text addressed to a human reader).
- **Scoring:** like upstream `defendToolResult()`, the port splits the string into sentences,
  packs them into chunks of at most 256 tokens, scores each chunk as `sigmoid(logit / T)` and takes
  the maximum. `T = 2.41` and the high-risk threshold `0.64` come from the model's
  `classifier_config.json`.
- **Not used:** Tier 1 (regex sanitizer), Tier 3 (bring-your-own LLM) and the field filter. We pass
  one tool-output string per injection site, so upstream's multi-string damping never applies.

## Files

- `models/minilm-multihead-v5/`: the model, tokenizer and configs, byte-identical to the npm package.
- `../defender_tier2.py`: the port.
- `../detectors.py`: the loader (`load_defender`, `defender_envelope`).

## Decisions

Both are saved for every site, with each chunk's `main` and `aux`, so other thresholds can be
computed without re-running the model.

- **`blocked` (primary):** the highest chunk `main` is at least 0.64. This is upstream's behavior
  with `blockHighRisk: true`, and matches how the other detectors are run at their default threshold.
- **`blocked_multihead`:** some chunk has `main >= 0.5` and `aux < 0.64`. Upstream documents this
  operating point but does not enable it by default.

## Fidelity

We checked the port against the Node package on 18 strings: identical chunks and token ids, chunk
scores within 3e-5, and no decision changes. **It needs `onnxruntime==1.21.0`**, the version
`onnxruntime-node` uses: the int8 model is sensitive to padding, and with onnxruntime 1.27 chunk
scores move by up to 0.15 and some decisions flip.

## Run

```bash
cd classifier-defenses
python run_detector.py --defense stackone-defender-tier2 --store <records.jsonl> --results <out_dir>
```

`run_defender_parallel.py` does the same over many processes. Note that the model lists AgentDojo
among its training data, so its detection rate on AgentDojo-derived attacks is optimistic.
