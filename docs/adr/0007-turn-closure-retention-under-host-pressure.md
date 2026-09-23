# ADR-0007: A pressure action must not invalidate a turn closure's identity

**Status:** accepted and implemented. Layers 3 and 4 are landed and `exercise_host_restore` passes;
the case's baseline entry was removed with the fix.

## Context

`exercise_host_restore` in `tests/models/qwen3_5/test_engine_prefix_real.cpp` (scenario
`NINFER_PREFIX_REAL_SCENARIO=host-restore`) runs one conversation, pressures the cache with a second
request, then reuses the conversation. It asserts that the complete MTP checkpoint is demoted to host
under pressure and then materialized back from host.

The first half passes. The second half does not:

```
Complete MTP checkpoint was not materialized from Host: path=1 reused=316 outputs=2
state=0 main=3 backend=2 degraded=1 evicted=0
captures[offered=0 no_vacancy=0 plan_refused=0 infeasible=0 completed=2 aborted=0]
```

`path=1` is `PrefixReusePath::PrivateEndpoint`; the case requires `PrivateTurnClosure` (2). The KV
prefix is reused (316 tokens), no State is restored (`state_h2d=0`), and one private owner is degraded.

Upstream's design intent, in `docs/maintainer/resource-scheduling-and-context-cache.md` §4.3, says the
last part cannot happen: after a checkpoint is published its identity, frontier and required coverage
are unchanged, and only Device/Host placement may move.

## The failure is layered

This record originally asserted a single mechanism -- that the closure is dropped because its
`demand_mask` is 0, so its loss prices at nothing -- and therefore that one fix would close the case.
Both are wrong. Instrumenting the decision instead of reasoning about it produced this:

| # | hypothesis | status | evidence |
|---|---|---|---|
| 1 | the valuation underprices a required checkpoint, so the planner prefers a target that drops one | **refuted** | 153 assessed targets; preserving targets cost an order of magnitude *less* (ordinal 2: `total=103,030,219 dropped=0` against ordinal 1: `total=1,720,454,858 dropped=2`) |
| 2 | the sealed target drops the checkpoint | **refuted** | both sealed targets report `dropped=0`, and the closure is present in `before` and in `after` |
| 3 | a content-identical degrade advances the owner's revision, so the next request's handle stops matching | **confirmed; fixed** | `[probe-apply] dropped=0 before[endpoint=1 rewrite=1 anchors=0] after[endpoint=1 rewrite=1 anchors=0] revision_before=4`; preserving the revision when the checkpoint set is unchanged takes `degraded` from 1 to 0 |
| 4 | the closure is offered but loses the candidate ranking, because the endpoint reuses more tokens | **fixed; case green** | one request is offered both -- `kind=0 (SessionEndpoint) frontier=316` and `kind=1 (TurnClosure) frontier=305` -- and the endpoint was selected: 11 more tokens of KV reuse, on a device-resident checkpoint needing no restore, while the closure's state and KV are the ones on host. Forcing the closure to be the only private candidate turned the case green; preferring a demoted closure over its owner's endpoint does too, and a device-resident closure is left alone |

Layers 1 and 2 were eliminated by measurement; layers 3 and 4 were found, fixed, and each verified by
the case going green only once both are present. Layer 3 is a defect in its own right -- a pressure
action that changes nothing about an owner must not move its identity -- and layer 4 is an ordering
rule within one owner rather than a price. The case needs both.

The case's assertion is sharper than this record previously described it. `exercise_host_restore`
(`test_engine_prefix_real.cpp:497`) builds a `LiveSession` owner with an explicit session key, demotes
its complete MTP checkpoint with a second request, then re-asserts the *same* prompt with reuse enabled
and requires three things (`:563-568`): `prefix_reuse_path == PrivateTurnClosure`, `state_h2d` above the
pressure count, and `main_kv_h2d_pages` / `backend_kv_h2d_pages` above it. So it requires the state
**and the KV** to come back from host. Selecting a device-resident endpoint satisfies the KV
requirement trivially and leaves every host-restore counter unchanged, which is why the failure reads
`state=0 main=3 backend=2`: those are the demote counts, unchanged on the way back.

One further reading from the same probe is worth keeping: `base_public=0` and `tgt_public=0` on all
153 targets, so no checkpoint in this scenario carries any live demand and the whole decision cost is
`now_ns + private_loss`. The closure is not uniquely demand-less; nothing is.

## Candidate causes measured out

Six, each by rebuilding and re-running the reproduction rather than by argument:

- the pools;
- the materialization search grant (ceiling 250 ms -> 5 s, unchanged);
- the shared retention weight (`RetentionClass::SharedStable` 0 -> 16, unchanged) -- which was never a
  candidate: the case sets `max_shared_prefixes = 0`, so no owner carries that class and the private
  owner's weight (`RecentPrivate` 4 / `LiveSession` 16, `resource_manager.h:1499-1511`) was untouched,
  so "unchanged" was guaranteed by construction;
- crediting the engine's own structural boundary in the projection's credit test (unchanged);
- making a target that degrades a checkpoint with a non-zero `demand_mask` inadmissible (identical
  failure: the dropped checkpoint carries `demand_mask == 0`);
- pricing a non-surviving checkpoint at the whole `baseline_saving` rather than the difference the
  target makes (byte-identical failure).

Two claims this record used to make are also withdrawn, both refuted by reading the code they cited.
The loss is **not** zero at `demand_mask == 0`: `MaterializationCheckpointPolicy` is built from
`cost_model_.prefill_ns` and `price_checkpoint_recovery_work` (`resource_manager.h:2070`, `:2128`),
neither of which reads the mask, and only the *public* accumulation at `context_portfolio_value.h:82`
is gated. And the runtime **can** tell a closure from an endpoint: `context_cache` reads
`CheckpointRef::kind` and already distinguishes `SessionEndpoint` (`resource_manager.h:1537`,
`1542-1544`, `1550`, `1569`). What it does not do is consult the kind in the *valuation*.

## Attribution

Each part measured rather than argued:

- **Not this port's retention work.** With `src/runtime/engine/context_cache` and
  `src/runtime/engine/engine_core.h` reverted to `feee7122^` the build compiles and the case fails with
  byte-identical counters.
- **Not a lost upstream fix.** Upstream's `fix(runtime): preserve reusable checkpoints under pressure`
  (`6b94b8c5`) is an ancestor of HEAD.
- **Not a capture refusal.** `active_captures_offered` is 0 and every refusal branch is 0, with
  `completed=2`. `offered` counts shared captures, and this case sets `max_shared_prefixes = 0`.
- **Not the cache policy.** Identical under `ContextCachePolicy::Rolling` and under the default.
- **Not the chat template.** Identical with the BOM strip disabled, which is a separate fix
  (`ea1c0d9e`).
- **Upstream's code, not this port's.** `pressure_committed` and `pressure_private_owners_degraded` are
  the same branch upstream (`upstream/master:2512-2569`), so this is shared behaviour.

## The shared path

The shared path has the same shape and was measured separately: a live 51-request agent log shows
every cache hit as `private endpoint` and none shared, with a conversation switch costing a 59.6 s
cold prefill at 171,953 tokens. The log is recorded in
[`docs/research/agent-session-field-log-2026-09-22.md`](../research/agent-session-field-log-2026-09-22.md),
which is also this ADR's field evidence -- of its 49 cache hits, every one is an endpoint.

**Superseded for the shared case by [ADR-0009](0009-shared-prefix-served-where-declared.md).** The
degradation measured there is real, but it is not why a shared prefix serves no reuse: on the OpenAI
protocol an undeclared request places its only automatic boundary at the end of the last message, so no
shared prefix is offered to a request whose prompt differs before that point, and where one is offered
it loses the selection to `private_response_replay`. A declared boundary is served today.

One reading from that investigation is suspect on its own: `shared_active_references` was read from a
`runtime_stats()` snapshot taken after the request completed, and until commit `5076f445` the
post-release snapshot was published *after* `complete_success` woke the caller, so an immediate read
could return the previous decode boundary. The occupancy figures remain the record of what that log
reported; they are not evidence that no live reference existed.

## Prior art

A required checkpoint is one no live request is asking for yet, so the question is how a system prices
an anticipated reuse. SGLang's `priority` plus `retention_seconds` (the name `retention_priority` is
TensorRT-LLM's) is a decaying bias, after its hard-pinning PR was reverted because "priority controls
eviction order, not exemption"; T-LRU's protected class is defined by the *next turn's* uncached token
count rather than by current demand. Both are sourced from primary references in
[`docs/research/anticipated-reuse-pricing.md`](../research/anticipated-reuse-pricing.md), which also
covers vLLM, TensorRT-LLM, LMCache, Mooncake, Dynamo, the classical replacement policies, and
Marconi's hybrid-state taxonomy.

## Decision

Accepted: a pressure action that leaves an owner's checkpoint set unchanged must not advance that
owner's revision or count a degrade. Placement may move; identity may not. That is upstream's §4.3
contract expressed where it was being violated, and it is implemented in `apply_private_action`
(`resource_manager.h`) by preserving the revision when `dropped == 0` and the checkpoint count is
unchanged. Degradation stays as the fallback for when the checkpoint set genuinely shrinks.

Layer 4 is the second half of the same decision, and it is an ordering rule rather than a price: for
one private owner, its turn closure is the reuse base and its session endpoint is not a competitor.
The closure carries the complete continuation state, which is exactly the state and KV a pressure
action demotes; the endpoint reuses a few more tokens while bypassing that demoted state. The rule is
bounded by construction -- across owners nothing changes, so a different owner that reuses more still
wins the ordinary comparison -- and it exempts nothing.

It fires only when the closure's State is actually on host. A device-resident closure has nothing to
restore, and preferring it there is what an over-broad first version did. The gate caught that: with
every closure preferred, `exercise_shared_replacement_and_full_capacity_reuse` went from
`active_captures_completed` 1 -> 2 to 1 -> 1, because a `Disposable` owner's device-resident closure
was chosen over the endpoint that would have captured again. The `ReplicaResidency` check is the
narrowing that keeps both scenarios true.

A derived price was tried first and refuted by measurement. The case requests two output tokens, so a
draft round never fits and `drafted_tokens` is 0 in both configurations: there is no speculative saving
to price, and no derived term can decide this selection. The justification for the rule is therefore
the contract, not arithmetic dressed up as a price.

The evidence that layer 3 is a defect rather than a trade-off:

- **It contradicts upstream's design intent and upstream's own test.** The case is upstream's
  (`tests/models/qwen3_5/test_engine_prefix_real.cpp`, introduced by Neroued in `04350ba9`), upstream
  carries `exercise_host_restore`, and this port is 0 commits behind `upstream/master`.
- **The port never chose it.** `git diff upstream/master -- src/runtime/engine/context_cache/` is 212
  insertions across 5 files, and none of them touch `pressure_committed`,
  `pressure_private_owners_degraded`, `dropped_checkpoint` or the pressure target.
- **It is the class of defect this release exists to remove.** 1.2.0 is about silent cache shortfalls
  -- the frozen reusable frontier, the 3-of-5 conversation hits -- and this is another: a reuse that
  silently loses speculative decoding.
- **The capacity is already provisioned.** The shipped launchers pass `--host-state-slots 16`, sized to
  hold every configuration the other bounds permit, so the State to preserve is budgeted for.
- **The cheap option taxes a headline capability.** Losing the MTP state loses speculative decoding
  for that turn, and speculation is what the shipped launchers are named for.

AGENTS.md settles the tie: "Do not sacrifice these goals to reduce the diff or implementation effort.
Evaluate complexity, maintenance cost, and verification risk as engineering tradeoffs, not reasons to
retain a known inferior design."

## Consequences

- Implementation: layer 3 is a revision-preservation rule in `apply_private_action`; layer 4 is an
  ordering rule where the private reuse candidates are built, in `inspect` (`resource_manager.h`),
  keyed on `CheckpointRef::kind`.
- Verification: the case passes and is the acceptance criterion for both layers. Layer 3's evidence is
  the degrade counter moving 1 -> 0 with the case otherwise unchanged; layer 4's is the same case green
  with the rule and red without it, plus the fourteen sibling scenarios that caught an over-broad
  version of it. `tests/test_resource_manager.cpp` covers pressure and degradation at unit level and
  is in the ASan subset of `tools/scripts/test_v3_asan.cmd`; the release gate is
  `tools/release/check_test_baseline.py`.
- Measurement: the trade-off is host bytes against the reuse the demoted State buys, so it is measured
  at the shipped topology rather than asserted. Speculation is *not* the stake: the case requests two
  output tokens, so `drafted_tokens` is 0 and no draft round runs either way.
- The case passes, so its entry was removed from `tools/release/test_baseline.json`; the gate treats a
  passing test that is still baselined as a stale baseline, which is why the fix and the baseline
  removal travel together. The two remaining entries are unattainable by construction.
- This is a narrow, recorded divergence from upstream, as ADR-0006 is for context parallelism, so a
  later upstream merge can read the intent instead of fighting it.

## Why this needs recording

Upstream's maintainer docs are upstream's to edit and this port does not edit them, so the port needs
its own record of where its behaviour and that design intent disagree. This case has now cost more
measurements than any other in the tree, and every one of them eliminated a hypothesis rather than
confirming one -- which is the reason the record is a table of layers with statuses rather than a
narrative: the next reader needs to know which explanations are already dead, and that this is a
defect with more than one cause.
