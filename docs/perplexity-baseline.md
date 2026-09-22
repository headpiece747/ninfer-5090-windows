# Perplexity baseline (Windows port)

The protocol, so a number is comparable: `ninfer-perplexity.exe <artifact> --corpus
eval/corpora/perplexity-1m/manifest.json --kv-dtype <dtype>`. That is upstream's fixed corpus
(`ninfer-ppl-1m-v1`: 1,044,876 scored tokens, 496 windows, 4096/2048 context and stride). Measured
2026-09-22 on this machine, which is a different box and clock from any published figure.

| artifact | KV | PPL | notes |
|---|---|---|---|
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | fp8 | **4.90295** | full corpus, 4.21k tok/s |
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | bf16 | **4.89838** | full corpus, 4.20k tok/s |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT) | fp8 | **4.89741** | `--quick` (4 streams, 261,223 tokens) |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT) | fp8 | **5.88829** | custom corpus, 177,400 tokens |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (NVFP4-full) | fp8 | **5.92007** | custom corpus, same text |

The custom corpus is this repo's `docs/` and `tests/` markdown, concatenated in sorted order
(177,400 tokens). On it the two shipped artifacts sit 0.5% apart with QUASAR marginally better, which
is the like-for-like check: same text, same protocol, two artifacts. The full-corpus rows are not
comparable with the `--quick` row, since the corpus subset differs.

## The one open discrepancy

A third-party conversion recipe publishes PPL **4.617** for the official stock artifact on this same
corpus. Measured here it is **4.898** with bf16 KV and **4.903** with fp8, so the 6.1% gap is not the
KV dtype -- and the bf16-versus-fp8 agreement of 0.1% says the KV path is sound in general.

Two candidates remain and neither is checked: the **corpus revision**, since the recipe may have been
run against an earlier `perplexity-1m`; and the **artifact revision**, since theirs may be a newer
export of the stock model than the one this port carries. The scoring path itself is upstream's, not
this port's, so a port-specific defect is not the leading explanation -- but it is not excluded either,
and the protocol above is what would settle it.
