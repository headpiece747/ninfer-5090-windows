# ADR-0003: The draft loader binds three logical parameters, not a fused one

**Status:** accepted (fixed in `9898ecb1`)

## Context

Upstream's converter groups `attention/query`, `attention/key` and `attention/value` and never
emits a fused `query_key_value`. `parameters.cpp` assembles the fused parent itself with
`ops::prepare_attn_input_proj_weights`.

Our loader nevertheless required a fused binding. Because the binder throws when a declared
logical parameter is absent, this rejected **every** upstream-shaped artifact at startup --
including upstream's own published image and cometkim's -- while our own artifacts happened to
work because our upgrader emitted the fused name.

## Decision

The loader binds the three logical parameters. No fused binding is required, and the runtime
projection path is unchanged.

## Why this needs recording

Reasoning from our own tree suggested upstream was inconsistent. It is not: reading their loader
settled it in thirty seconds. The wrong conclusion was reachable twice in one session.

## Amendment (2026-09-20): both forms run and measure alike in throughput, but not in output

The decision above assumes the two forms share one runtime projection path ("the runtime projection
path is unchanged"). Measured by interleaving the two artifacts, that holds for **throughput** but
not for **exact numerics**:

- throughput: 316.8 against 315.2 tok/s, three measurements each, alternating between files. The 0.5%
  is noise. An earlier time-ordered comparison read 341.7 against 314.3 and was **wrong**: every
  sample of one file predated every sample of the other, and this machine's decode varies by up to
  ~8% between windows. Interleave an A/B here, always.
- output: deterministic and different. The upgrader's fused form drafts 1585 tokens at 61.8%
  acceptance (digest `e3b804e3…`); the repository's split form drafts 1570 at 62.5% (digest
  `d9952413…`). **Resolved: the model itself is the same in both files.** `ninfer-perplexity` over
  the 1M-token corpus scores them to the same six decimals (overall 1.606336, English reference
  1.901747, code 0.520469) and derives the same artifact-content hash for both, so neither the
  binding nor the 187-byte vision tensor changes the model. The token difference is the speculative
  path, which is not bit-identical by design (ADR-0002).

So a binding is a compatibility and a numerics property, not a performance one. The two files are
4,283 bytes apart, which no size check can see -- `tools/release/compare_artifacts.py` names the
exact delta by diffing the canonical JSON index.

Consequence for this fork: `tools/release/profiles.py` measures the **published** artifact, the
split form, because that is what `download_model.py` fetches.
