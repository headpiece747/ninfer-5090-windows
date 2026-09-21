# ADR-0004: Four vision-only profiles, and one artifact we do not control

**Status:** accepted

## Context

Vision measured free on both artifacts (QUASAR 331.3 against 333.0 tok/s; NVFP4-full 326.6
against 325.0), at the same context either way. Text-only variants existed only because the
retired NVFP4 image charged 16,384-27,008 tokens of context for Vision.

The NVFP4-full artifact is published by cometkim, not by us or upstream. It is accepted because
it reaches the native context where our own NVFP4 image could not, and it is Apache-2.0.

## Decision

Four profiles, all vision-only, all at the native context: two artifacts times two spec routes.
The NVFP4-full artifact is used knowingly, with **no in-house fallback on that lane**.

## Consequences

- A yank or a breaking republish of that artifact breaks the ninfer profiles. Producing our own
  is the only hedge, and it is real work.
- The retired NVFP4 image is not a fallback: it cannot hold a full-context pool.
- Text-only variants are not to be re-added. There is no speed to recover and context to lose.

## Why this needs recording

Both of these are the kind of decision that looks like an oversight from the outside: "why only
four?" and "why depend on someone else's artifact?"

## Amendment (2026-09-20): the flag set those numbers describe

The with-vs-without-vision figures in Context were measured on the 2026-09-17 flag set. The
launchers gained per-profile `--device-state-slots` the next day (`4908bdfc`, to move the #251
reuse cliff), which raises a profile's runtime by 0.4-1.3 GiB and was therefore absent from both
sides of that comparison. The conclusion -- Vision measured free -- stands as measured; the
absolute figures are superseded by `tools/release/profiles.py`, whose values are measured through
the shipped launcher flag sets by the `profile` mode of `v3_profile_matrix.py`. That raise turned
out to be temporary: the reclaim landed on 2026-09-19, every profile ships
`--device-state-slots 1` again, and the shipped flag set matches this comparison's once more.
