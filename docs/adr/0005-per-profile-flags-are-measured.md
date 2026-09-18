# ADR-0005: Spec route, draft depth and the proposal head are measured per profile

**Status:** accepted

## Context

`--lm-head-draft` is not uniformly good, and depth optima differ by artifact:

- QUASAR: the flag is worth +9% (DFlash2) and +18% (MTP d4).
- NVFP4-full MTP: worth +34% at d5.
- NVFP4-full DFlash2: worth about +2%, while costing 0.33 GiB of headroom.
- The retired NVFP4 image: the flag cost about 13% on DFlash2, plus 16,384 of context.

Depth optima: d4 on QUASAR, d5 on NVFP4-full.

## Decision

Set the spec route, draft depth and proposal head per profile from a record. Never from
convention, and never uniformly across profiles. The three cache-bound flags likewise.

## Consequences

- A profile's flags cannot be derived from another profile's, even on the same artifact.
- Acceptance rate is not the selection criterion; measured decode rate is.
- Re-measuring means re-generating the launchers from the profile table, not editing them.

## Why this needs recording

The natural instinct is to apply one setting to every profile, which is measurably worse on at
least three of them. The earlier uniform d5 was wrong on QUASAR.
