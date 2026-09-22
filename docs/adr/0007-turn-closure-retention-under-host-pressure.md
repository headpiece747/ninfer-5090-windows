# ADR-0007: Host pressure degrades a private owner instead of demoting its turn closure

**Status:** proposed

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
prefix is reused (316 tokens), but no State is restored (`state_h2d=0`) and one private owner is
degraded. `pressure_private_owners_degraded` is incremented in `apply_private_action`
(`resource_manager.h`) when `result.pressure_committed` is set: the owner survives, its summary
advances, and older checkpoints are dropped. So a pressure action commits on the owner whose turn
closure the next request needs, and the closure is gone by the time it asks.

Upstream's design intent, in `docs/maintainer/resource-scheduling-and-context-cache.md` §4.3, says
this cannot happen: after a checkpoint is published its identity, frontier and required coverage are
unchanged, and only Device/Host placement may move.

Attribution, each part measured rather than argued:

- **Not this port's retention work.** With `src/runtime/engine/context_cache` and
  `src/runtime/engine/engine_core.h` reverted to `feee7122^` the build compiles and the case fails
  with byte-identical counters.
- **Not a lost upstream fix.** Upstream's `fix(runtime): preserve reusable checkpoints under pressure`
  (`6b94b8c5`) is an ancestor of HEAD.
- **Not a capture refusal.** `active_captures_offered` is 0 and every refusal branch is 0, with
  `completed=2`. `offered` counts shared captures, and this case sets `max_shared_prefixes = 0`.
- **Not the cache policy.** Identical under `ContextCachePolicy::Rolling` and under the default.
- **Not the chat template.** Identical with the BOM strip disabled, which is a separate fix
  (`ea1c0d9e`).

Upstream's code carries the same branch (`pressure_committed`, `pressure_private_owners_degraded` at
`upstream/master:2512-2569`), so this is shared behaviour, not a port invention.

## Decision

Not yet taken. The two options differ in what they spend, and the choice is a product one:

1. **Preserve the contract.** Make the pressure action demote the owner's checkpoint instead of
   dropping it, so the closure is restorable. Costs host State: the case's pool is 256 MiB
   (`host_kv_capacity_bytes = 256ULL << 20`), the shipped launchers use `--host-state-slots 16`.
   Keeps speculative decoding on the following turn.
2. **Keep the behaviour.** Reuse the KV, lose the MTP state for that turn. Cheaper on host State.
   Then §4.3's "required coverage" is intent rather than invariant, and this record is what says so.

Recommended: option 1 where host capacity allows, degrading only when it does not. The reuse already
keeps the KV, so the cost of the current behaviour is a lost speculation opportunity, not incorrect
output; but the contract is stated as an invariant and the host pool is sized as if it held.

## Consequences

- The case stays baselined in `tools/release/test_baseline.json` with a pointer to this record, so the
  gate states it rather than hiding it behind a skip, and the release is not blocked.
- The resource-manager unit test already covers pressure and degradation
  (`tests/test_resource_manager.cpp`: `FakePressureTargetHandle`, `degradation_units`) and is in the
  ASan subset of `tools/scripts/test_v3_asan.cmd`, so a change here has both a unit-level guard and
  sanitizer coverage. The integration case above is the end-to-end guard.
- If option 1 is taken, the regression guard already exists: the case fails today and will pass
  unchanged.

## Why this needs recording

Upstream's maintainer docs are upstream's to edit and this port does not edit them, so the port needs
its own record of where its behaviour and that design intent disagree. Without this, the next reader
finds a baselined test and has to re-derive the attribution from scratch, including the two
eliminations that took the longest: that the port's own retention work is not the cause, and that no
capture is being refused.
