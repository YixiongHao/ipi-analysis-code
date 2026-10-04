# G-means at default untuned thresholds

G-means = sqrt(TPR x (1 - FPR)) (https://yongzx.substack.com/p/cot-monitorability-why-g-means-and): prevalence-free, so it compares monitors at their shipped default configs without threshold tuning. Per-event decisions from `scores.json` (same convention as op_*/roc_* figures).

## ipi head (injection detection)

| defense | TPR | FPR SWE | FPR Tool | G-means SWE | G-means Tool |
|---|---|---|---|---|---|
| LLM-judge | 87.8% | 3.5% | 2.5% | 0.920 | 0.925 |
| Gemini-judge | 83.2% | 0.4% | 2.9% | 0.910 | 0.899 |
| Cygnal | 70.7% | 0.0% | 0.2% | 0.841 | 0.840 |
| Qwen3.5-9B | 66.9% | 0.0% | 0.1% | 0.818 | 0.818 |
| StackOne | 66.2% | 5.8% | 8.4% | 0.790 | 0.779 |
| Qwen3.5-2B | 38.5% | 0.2% | 6.5% | 0.620 | 0.600 |
| Granite | 33.2% | 0.4% | 3.9% | 0.575 | 0.565 |
| Granite-think | 35.6% | 2.1% | 11.9% | 0.590 | 0.560 |
| PromptGuard | 24.7% | 0.3% | 3.1% | 0.497 | 0.489 |
| DataSentinel | 20.9% | 8.8% | 8.7% | 0.437 | 0.437 |
| ProtectAI | 50.9% | 23.3% | 73.2% | 0.625 | 0.369 |

## policy head (harmful-action detection; only the LLM monitors have one)

| defense | TPR | FPR SWE | FPR Tool | G-means SWE | G-means Tool |
|---|---|---|---|---|---|
| Cygnal | 69.5% | 0.7% | 1.3% | 0.831 | 0.829 |
| Granite | 24.8% | 5.8% | 20.2% | 0.483 | 0.445 |
| Granite-think | 31.1% | 17.4% | 39.6% | 0.507 | 0.433 |
| LLM-judge | 87.6% | 0.0% | 0.8% | 0.936 | 0.932 |
| Gemini-judge | 90.7% | 0.0% | 0.0% | 0.952 | 0.952 |
| Qwen3.5-2B | 55.9% | 2.3% | 57.8% | 0.739 | 0.485 |
| Qwen3.5-9B | 83.8% | 1.8% | 23.6% | 0.907 | 0.800 |
