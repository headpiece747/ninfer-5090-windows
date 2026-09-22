# ADR-0009: A shared prefix is served only where a boundary is declared

**Status:** accepted

Supersedes the shared-prefix paragraph of
[ADR-0007](0007-turn-closure-retention-under-host-pressure.md).

## Context

A shared prefix is an immutable reuse source that several private branches may fork
(`docs/maintainer/resource-scheduling-and-context-cache.md` §4.3, §7.2). The shipped launchers enable
it, the request log counts it, and a live 51-request agent log showed every cache hit as
`private endpoint` with none shared. The question was whether the feature is broken or unused.

The first measurement was wrong because the harness was wrong. `loop_shared_reuse.py`'s `--identical`
flag was meant to send two byte-identical prompts; it replaced the marker in the first user turn only,
so the turns appended after it still carried `alpha` versus `bravo`. Two "identical" requests were
therefore two prompts differing in their tail, and every conclusion drawn from that pair was about the
wrong input. Fixed, the pair separates cleanly.

The permanent instrumentation is three counters in `RuntimeStats`, logged in the
`context_cache.selections` group (schema 23). The runtime layer cannot log, so the decisions are
counted where they are made:

| counter | meaning |
|---|---|
| `shared_reuse_candidates` | a shared index entry reached the reuse plan |
| `shared_reuse_declined` | the Program refused it (`inspect_admission` returned null) |
| `shared_reuse_key_mismatch` | it was skipped before the plan: no shortlist key at that frontier |

Two sessions that share a system message and one tool definition, differing in their first user turn,
measured with the shipped configuration (`profiles.launcher_args`):

- **No declaration.** `shared_stable_prefix 0`, `shared_reuse_candidates 0`,
  `shared_reuse_key_mismatch 1`, `root 2`. No shared prefix was offered at all.
- **A declared boundary** (`prompt_cache_breakpoint: {mode:"explicit"}` on the system content part).
  `shared_stable_prefix 1`, `shared_reuse_candidates 1`, `shared_reuse_declined 0`,
  `reused_prompt_tokens 301` of 344, served as `shared prefix`.
- **Byte-identical prompts, no declaration.** The shared candidate *is* offered
  (`shared_reuse_candidates 1`) and *is* accepted by the Program (`shared_reuse_declined 0`), and the
  planner selects `private_response_replay` instead (338 of 343 tokens). The shared path is
  available; it loses.

Two mechanisms explain all three results, and both are intended.

**The OpenAI protocol places its own automatic boundary.** `apply_openai_prompt_cache_policy`
(`src/serve/openai_common.cpp`) clears `allow_engine_automatic_shared_prefixes` for every OpenAI
request, because the protocol defines its own write policy (`include/ninfer/types.h`; the Anthropic
path does the same). The frontend's three automatic shared opportunities — the end of the first tool
definition, the end of the leading system message, and the full prompt — are therefore never
declared. The serve layer's own automatic target is the end of the last message, which is the whole
prompt. So an undeclared OpenAI request can only share a prefix with a request whose prompt matches it
at the very end, and a request whose prompt differs anywhere before that is skipped at the shortlist
key — which is what `shared_reuse_key_mismatch 1` reports.

**Shared publication is an investment, not a guarantee.** A shared candidate is admitted to the
projection when it is *declared* (`ExplicitBoundary` or `RequestedAutomatic`), or *repeated* (at least
two reuse domains demand that exact prefix), or a *surplus* candidate (`DefaultAutomatic` or
`EngineStructural` with spare shared capacity) — the shared-candidate projection in
`resource_manager.h`. The maintainer doc states the rule: shared publication must be strictly better
than the private-only baseline at the same frontier (§7.2), and a candidate that is not is skipped. An
undeclared automatic candidate therefore rides on surplus capacity, and then competes with the private
candidate at the same frontier — where, for an identical prompt, response replay already serves the
same tokens more cheaply.

Neither is a defect. The first is a protocol boundary the port inherited deliberately; the second is
the documented valuation rule.

## Decision

Accepted: keep both behaviours.

- A shared prefix on the OpenAI protocol is a **declared** resource.
  `prompt_cache_breakpoint: {mode: "explicit"}` on a tool, a message, or a message content part is the
  supported way to place one, and it is served end to end.
- The engine's structural boundaries stay disabled for OpenAI requests. Enabling them would publish
  shared prefixes that the valuation must then either refuse or pay for against a private path that
  already wins.
- An undeclared automatic candidate keeps its surplus-only admission. It is not promoted above a
  private candidate at the same frontier.
- The three counters are permanent. `shared_stable_prefix_selections` alone cannot distinguish "the
  runtime never offered a shared prefix" from "it offered one and the planner chose otherwise", and
  that distinction is what this record cost the most to establish.

## Consequences

- `tests/test_openai_schema.cpp::test_prompt_cache_boundaries` pins the policy: an OpenAI request
  clears `allow_engine_automatic_shared_prefixes` and places its automatic boundary at the end of the
  last message with `DefaultAutomatic` evidence; a declared breakpoint places a `SharedStablePrefix`
  boundary with `ExplicitBoundary` evidence. That function had no coverage before.
- The shared prefix's automatic publication on the OpenAI protocol is mostly spent on a boundary that
  a private path already covers. That is a valuation consequence rather than a correctness one, and it
  is now visible in the log instead of inferred.
- ADR-0007's private turn-closure finding is unaffected. Its shared-prefix paragraph is superseded:
  the degradation it measured is real but happens after the reuse has already been lost.
- The measured note in `tools/release/profiles.py` that recorded the shared prefixes as never
  retained is corrected to name this mechanism.

## Why this needs recording

Two sessions went into the shared path: one on a harness that could not produce its own input, and one
adding the counters that split the space. Without this record the next reader repeats both, and the
four mechanisms that look plausible — the degradation, the pools, the retention weight, the structural
credit — are already measured out.
