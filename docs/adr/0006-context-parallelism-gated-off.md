# ADR-0006: Context parallelism is gated off until the exact boundary correction exists

**Status:** accepted

## Context

The chunked Gated DeltaNet pipeline can split a sequence's chunk range into independently-seeded
segments so `state_passing` runs with more than one wave of work, mirroring QwenLM/FlashQLA's
sm120 context parallelism. The split, the gate warmup scan, the segmented kernel and the
boundary-replay chain are implemented and were measured at T=8192: the pipeline drops from
1378.98 us to 1235.97 us, a 143 us (10.4%) win.

The scheme is only correct together with its boundary correction, and the correction has two forms:

- **Exact**: replay a segment from its true predecessor state. Implemented, but serialized - one
  launch per segment, in order. Measured ~632 us of replays at T=8192, which is 4.4x the 143 us
  win, so enabling it makes CP a net loss of roughly 489 us.
- **Approximate** (FlashQLA's): when the gate has decayed below a warmup threshold, drop the
  incoming state and keep the segment's zero-seeded state. This is what the implemented warmup
  scan selects, and it is what the port currently does when the replay is skipped.

The approximate form is not admissible at this Op's tolerance. Dropping the incoming state leaves
a residual of `exp(threshold) * |state|`; at chunk 64 the threshold of -10.0 that FlashQLA uses
(chunk 32 there) leaves ~4.5e-5 against this Op's 1.0e-5 gross-absolute criterion, and the state
magnitude reaches O(1). FlashQLA itself validates CP only to `RTOL = 0.02`, roughly 5x looser than
this Op's qualification.

Measured evidence: with CP enabled, a 35B geometry (H_v=32, T=4096, 64 chunks, 8 segments) fails
the FP64 oracle at token 1352 (chunk 21, segment 2) with `actual=0.00136566` against
`reference=0.0194563` - the output is 14x too small. Forcing the replay to always run makes the
same case pass, which isolates the defect to the skipped-correction path rather than to the
segmented kernel, the segment fill, the state slot chain or the replay indexing.

## Decision

Gate context parallelism off. `cp_enabled` returns false unconditionally, so the chunked Op always
takes the single-segment path and reproduces the exact recurrence.

The segmented kernel, the planner, the warmup scan, the segment fill and the replay chain stay in
the tree as the foundation the exact correction builds on. They are not dead code: they are a
placed seam with no valid adapter yet.

## Consequences

- The measured 143 us CP win is not shipped. The chunked pipeline runs at its non-CP cost.
- The failing case is retained as the regression guard and as the specification for unit 3:
  `35b context-parallel gated off` in `tests/ops/test_gated_delta_net.cpp`. It passes with the gate
  off and goes red the moment CP is re-enabled without the correction. The failing index `12600648`
  is quoted from the gating commit `ac469924`'s own message; it appears nowhere in the tree and is
  not reproducible without re-enabling the gate, so the case's name is the durable reference and the
  index is provenance.
- The predicate the gate removed is recoverable from `ac469924`, and re-enabling CP needs it back:
  `cp_enabled` returned `be_h <= 40 || (be_h <= 56 && total_chunks >= 128)`.
- Re-enabling CP requires the exact affine correction `h = ht + M @ h`, where `M` is the segment's
  128x128 transition matrix. The transition is a matrix, not a scalar, because the per-token delta
  depends on the state; a parallel prefix scan over scalars is therefore not available.
- NInfer materializes `h_chunk`, which the research note identifies as the advantage over FlashQLA
  for deriving `M` chunk-wise rather than only over a warmup window. That remains the intended
  route.

## Why this needs recording

The natural instinct is to keep a measured 10.4% win and accept a small numerical difference, which
is exactly what upstream does at `RTOL = 0.02`. This Op is a public semantic boundary qualified
against an FP64 oracle at a much tighter tolerance, so the same approximation is a silent
correctness defect here. The second instinct - always replaying, since it is exact - is measurably
slower than not using CP at all. Recording both measurements is what stops the next attempt from
re-deriving them.
