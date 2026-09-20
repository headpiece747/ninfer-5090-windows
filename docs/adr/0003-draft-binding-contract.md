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

## Amendment (2026-09-20): both forms run, but they do not measure the same

The decision above assumes the two forms share one runtime projection path ("the runtime projection
path is unchanged"). They start and serve alike, but they are not performance-equivalent. On the
shipped QUASAR DFlash2 profile at `--device-state-slots 1`, the fused form our upgrader emits
measures **341.7 tok/s at 61.8% acceptance** (digest `e3b804e3…`) against the repository's
upstream-shaped v3 at **314.3 tok/s at 62.5%** (digest `d9952413…`); on QUASAR MTP4, 239.2 at 65.3%
against 215.0 at 58.3%. Same engine, flags, prompt and seed, with the artifact as the only variable,
three to six samples each.

So an artifact's *bindings* are a performance property, not only a compatibility one. The two files
are 4,283 bytes apart, so a size check cannot tell them apart --
`tools/release/compare_artifacts.py` diffs the canonical JSON index and hashes the payload, and it
is what identified the changed binding.

Consequence for this fork: `tools/release/profiles.py` measures the **published** artifact, the
unfused one, because that is what `download_model.py` fetches. The fused v3 measures faster and is
currently published by nobody.
