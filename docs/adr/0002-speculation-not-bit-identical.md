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

## Consequences

- Speculation measured 3-4x faster, so it stays on everywhere.
- Depth is chosen on measured decode rate, not on agreement with a no-spec baseline.
- A user comparing output across profiles will see differences, which is correct.

## Why this needs recording

This was investigated twice before the maintainer note was found. The next explorer will have
the same instinct.
