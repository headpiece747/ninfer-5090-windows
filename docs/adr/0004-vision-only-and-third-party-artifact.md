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
