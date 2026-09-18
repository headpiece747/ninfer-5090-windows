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
