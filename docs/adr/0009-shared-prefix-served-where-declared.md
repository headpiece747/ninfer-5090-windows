# ADR-0009: A shared prefix is served only where a boundary is declared

**Status:** superseded 2026-10-04 (was: accepted and implemented).

Supersedes the shared-prefix paragraph of
[ADR-0007](0007-turn-closure-retention-under-host-pressure.md).

**Status: superseded 2026-10-04 by upstream `b9114396` ("replace context cache and add preemptive
scheduling"), merged into this port.** The context cache this ADR reasons about no longer exists:
`context_portfolio_value.h` and `materialization_planner.h` were deleted upstream, and
`resource_manager.h` was rewritten, so the line numbers cited below no longer resolve. They are left
as written rather than repointed, because a citation moved to a line that does not say what it used
to is worse than one that visibly fails.

**What survives:** the failure modes. Both ADRs describe *silent* degradation -- a turn closure that
loses its identity without error, and a shared prefix served where no boundary was declared. Neither
is detectable by a green build or a passing suite, which is why they were written down at all. The
new cache has its own accounting (`host_capacity_bytes`, preemptions, `snapshot_restores`,
`replay_restores`) and a different set of ways to lose state, so **whether these invariants hold
across the rewrite is unverified.** Re-deriving them is open work, and the scenarios in
`tests/models/qwen3_5/test_engine_prefix_real.cpp` are where it should start, because that is the
file both ADRs name as the case that would have caught each one.

## Context

A shared prefix is an immutable reuse source that several private branches may fork
(`docs/maintainer/resource-scheduling-and-context-cache.md` §4.3, §7.2). The shipped launchers enable
it, the request log counts it, and a live 51-request agent log showed every cache hit as
`private endpoint` with none shared. That log is recorded in
[`docs/research/agent-session-field-log-2026-09-22.md`](../research/agent-session-field-log-2026-09-22.md):
two of its requests are conversation switches, both reporting `cache 0 (0.0%)`, one of them a 59.6 s
cold prefill at 171,953 tokens. The question was whether the feature is broken or unused.

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
- An undeclared automatic candidate keeps its surplus-only admission **in the shared-candidate
  projection**. It is not promoted above a private candidate at the same frontier. This is the
  projection's rule, and it is worth being exact about the boundary, because the *capture-time*
  standing is decided separately and this ADR originally read as though it covered both: the shipped
  launchers set `--context-cache-policy rolling` (`profiles.py:260`), under which
  `pressure_evidence` — and therefore standing to replace a resident — is granted to every
  candidate regardless of evidence (`resource_manager.h` (pre-rewrite lines 659-678)), and only the fold then decides.
  `rolling` was adopted for the private frontier, where a conversation's newest checkpoint would
  otherwise pin (`profiles.py:348-356`); it was never evaluated against the shared rule, so **the
  shared path's behaviour under `rolling` is a consequence nobody has measured.** Read the triad as
  describing the projection, and treat capture-time standing as an open question rather than a
  settled one.
  **Correction (2026-10-06): the flag this paragraph names no longer exists.** `--context-cache-policy`
  went with the rewrite the parenthetical above already points at — the new `ContextCacheOptions` has
  no policy field and the launchers pass no such flag — so every `rolling` clause here describes the
  pre-`b9114396` cache. What the new cache does with an undeclared automatic candidate is a *different*
  open question, and one probe has since bounded it: a capture with no demonstrated reuse cannot
  displace a reused endpoint, refused at the value gate in
  `src/runtime/engine/context_cache/resource_manager.h:1282`
  (`docs/research/prefix-state-eviction.md`). The triad's projection rule is unaffected.
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

## Sharpened 2026-10-03, from the code rather than from a run

Three things in the rule above are less firm than the prose implies, and each was found by reading
`resource_manager.h` while chasing an unreproduced version of this ADR's own measurement.

**`shared_reuse_declined` is not a cost verdict.** It is incremented only when
`program.inspect_admission(..., shared_source, ...)` returns `nullopt`
(`resource_manager.h` (pre-rewrite line 384)), which is an *exactness* failure — missing state, KV or identity,
`!base.allow_prefix_reuse`, `!prompt.identity.reusable`, or `prefix_matches` failing on token
identity. The cost comparison this ADR calls "shared publication must be strictly better than the
private-only baseline" happens later, in `FoldedCost::key()`
(`materialization_planner.h` (pre-rewrite lines 798-813)), and **is not counted at all**. So this ADR's own evidence
(`shared_reuse_candidates 1`, `shared_reuse_declined 0`) proves the *publication-eligibility* and
*offer* stages succeeded; it does not measure the "response replay already serves the same tokens more
cheaply" claim, which is a statement about the selection stage. Reading those two counters as a cost
verdict is the inference this paragraph exists to prevent.

**`repeated` counts reuse domains, not requests, and the two cases invert.**
`matching_reuse_domains` (`resource_manager.h` (pre-rewrite lines 1515-1534)) deduplicates by `ReuseDomainId`. A client
that repeats one prompt fifty times **under a session key** contributes a single domain forever, so
`>= 2` never fires and the candidate rides on surplus or declared credit only. The same client
**without** a session key gets a domain derived from `publication_order`
(`resource_manager.h` (pre-rewrite lines 1461-1465)), which is fresh per request, so the *second* request satisfies
`repeated`. Both `demand_mask` and `repeated` are read over the **last 32 committed admissions**
(`demand_window_`, capacity 32), not over the current request. This is the largest single source of
run-to-run difference in whether a shared candidate is projected at all, and the most likely reason
this ADR's 2026-09-22 `shared_stable_prefix 1` is not reproduced on demand — **unconfirmed**; the
counters needed to test it ride on a throughput record this product's lane does not write (see
[ADR-0012](0012-native-render-for-the-registered-template.md)).

**A second, configuration-level candidate is open upstream and worth reading before attributing any
missing shared hit to valuation.** `gh issue view 270 --repo Neroued/ninfer`: `max_shared_prefixes`
defaults to `max(max_concurrency, 4)` while a single request may present up to **7** candidates (4
explicit + 3 engine-automatic), so at low concurrency the catalog can fill from one or two requests'
own candidates; an entry carrying `DefaultAutomatic` evidence "can never evict to make room", and once
the catalog is full, requests *"permanently stop getting shared-prefix cache hits and silently
re-prefill their entire prompt on every call, with no error or degradation signal."* That is a
mechanism this ADR's counters cannot distinguish from the valuation rule above, because it suppresses
the candidate before any of them is incremented.

**It does not apply to this measurement, and the code says why rather than the run merely failing to
reproduce it.** `max_shared_prefixes` defaults to `max(concurrency, 4)`
(`model_instance.cpp:115-116`, `kMaximumExplicitPromptCacheMarkers` = 4), while the 7-candidate
ceiling is `4 explicit + 3 engine-automatic` (`frontend.cpp:495`). Those 3 are gated on
`allow_engine_automatic_shared_prefixes`, and **both serving protocols force it false** —
`openai_common.cpp:177` and `anthropic_messages_request.cpp:980`. So on either protocol a request
presents at most 4 candidates against a catalog of at least 4, and the catalog cannot be undersized
relative to one request's own candidates. The bug is real and unfixed upstream; it needs a route that
leaves the flag true.

**`prompt_cache_key` is accepted and discarded, and that is a plausible contributor where replay wins.**
OpenAI documents it as a *routing and matching* input: "Requests are routed based on the initial
prompt prefix. When you provide `prompt_cache_key`, it is combined with the prefix hash, allowing you
to influence routing", and for its current model families it is required "to use the more reliable
matching for both implicit and explicit caching". This port validates it as a string hint
(`openai_common.cpp:68`) and then states in `docs/serving.md:358` that it "is not an Engine session key
or prefix identity". Two sessions sharing a byte-identical declared prefix therefore match only by
token digest, and — per `repeated` above — a session-keyed client never accumulates a second reuse
domain, so nothing is ever `repeated`. That is a coherent mechanism for a shared candidate losing
without the counters moving. **Untested here**: it predicts that an explicit boundary plus a stable
`prompt_cache_key` changes the OpenAI arm's outcome, which is a run this ADR has not done.

**`EngineObserved` is in neither bucket.** The frontend's full-prompt automatic boundary
(`frontend.cpp:541-543`) carries `EngineObserved`, and the admission test at
`resource_manager.h` (pre-rewrite lines 1885-1897) builds `declared` from `ExplicitBoundary`/`RequestedAutomatic` and
`surplus` from `DefaultAutomatic`/`EngineStructural` — omitting it. So "automatic candidates ride on
surplus" is true for `DefaultAutomatic` and `EngineStructural` and **false** for `EngineObserved`,
which is admissible only via `repeated`.
