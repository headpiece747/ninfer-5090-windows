# ADR-0002: Speculative decoding is not bit-identical to plain decoding

**Status:** accepted

## Context

Greedy output differs between no-spec, every MTP depth, and DFlash2. This was measured across
both artifacts with the server in `--greedy` and no request-level sampling, which is the only
configuration where token equality is a meaningful claim. The difference is deterministic and
reproducible.

## Decision

Treat differing output as expected engine behaviour. Do not treat it as a porting defect, and do
not chase bit-exactness.

## Evidence

The maintainer notes state that speculation "does not impose token or logits equality between
different quantization, prefill or kernel paths". Acceptance compares a proposal token against
the target argmax for its verify column, and the batched verify kernel is not the single-token
decode path, so a near-tie can flip and the continuation diverges.

### Quantified on 2026-09-27, and the traps in measuring it

The decision above was re-derived from scratch before this note was found, which is the third
occurrence. The divergence was reproduced as deterministic and 4/4 identical run to run on both
arms, and 18 of 25 draft-width-by-prompt cells diverged somewhere, with the first differing
position erratic in width rather than monotonic -- consistent with per-width rounding rather than
accumulating drift. Both outputs were fluent, on-topic and free of the repeated-fragment
signature that state-corruption bugs produce, and this engine's greedy path is invariant to
prefill-chunk shape (1024, 2048 and 512 agree exactly), which rules out that explanation here.

The magnitude was then measured with `bench/models/qwen3_5/score_token_logprobs.cpp`, which scores
a supplied id file directly. The two routes' sequences share a prefix up to the first difference,
so that index is one context with two candidates:

| kv | first differing token | logprob A | logprob B | gap (nats) |
|---|---:|---:|---:|---:|
| fp8 | 1 | -6.540 | -9.681 | 3.141 |
| fp8 | 26 | -8.206 | -6.331 | 1.875 |
| fp8 | 29 | -2.204 | -1.204 | 1.000 |
| bf16 | 58 | -0.499 | -0.999 | 0.500 |
| bf16 | 24 | -1.385 | -1.760 | 0.375 |
| bf16 | 29 | -1.135 | -2.260 | 1.125 |

Two limits on reading that table, both of which cut toward the decision above rather than against
it. The scorer resolves log probabilities to 1/16 nat, and the reference is a teacher-forced
forward over the whole sequence, which is *neither* decode shape -- so it bounds how far a third
computation sits from the candidates, not how far the two decode forwards sit from each other. The
one-nat figures are therefore not a claim that the verify path's own logits disagree by a nat, only
that all three computations differ by that order. fp8 KV roughly triples the gap and can move the
first divergence from token 58 to token 1, so KV quantisation is an amplifier, not the cause.

The masked-draft accept that produced those sequences was read to confirm it is lossless
(`include/ninfer/ops/speculative_accept.h`): the accept index is the first column whose target argmax
differs from the draft, and the committed token at that column is the target's own argmax there.
There is no cross-column leakage, which is why both routes agree on the first generated token.

### Measuring this again: four traps, each of which produced a wrong answer first

- `ninfer-perplexity --text` re-tokenises, and re-tokenising decoded text does not reproduce the
  ids that produced it (153 generated ids re-tokenised to 265 scored rows), so an index-aligned read
  through it silently inspects a different position.
- `ninfer --print-token-ids` writes ids to **stderr**, on a line reading
  `tokens  <pad>  generated ids  <pad>  9419 0 2500 ...`; the ids follow the label. Discarding stderr
  yields the answer text instead, and every index is then a word index rather than a token index.
- `Start-Process -ArgumentList` splits on whitespace unless each element is quoted, so a prompt with
  spaces arrives as several argv entries.
- Reading a CSV by column position rather than header name turns real data into clean zeros.

Assert the shape before reporting any of it: N ids scored from index 1 must yield exactly N-1 rows.
A control that runs the same arm twice, and a parse check on the id stream, each caught a real
error that would otherwise have become a number.

## Consequences

- Speculation measured 3-4x faster, so it stays on everywhere.
- Depth is chosen on measured decode rate, not on agreement with a no-spec baseline.
- A user comparing output across profiles will see differences, which is correct.

## Why this needs recording

This was investigated twice before the maintainer note was found. The next explorer will have
the same instinct.
