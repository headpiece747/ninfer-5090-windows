# ADR-0007: Host pressure degrades a private owner instead of demoting its turn closure

**Status:** accepted

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

Accepted: preserve the contract. When host State capacity allows, a pressure action demotes the
owner's checkpoint instead of degrading it, so a published turn closure survives as Device/Host
placement and can be materialized again. Degradation stays as the fallback for when host State is
also full.

The evidence that this is a defect rather than a trade-off:

- **It contradicts upstream's design intent and upstream's own test.** The case is upstream's
  (`tests/models/qwen3_5/test_engine_prefix_real.cpp`, introduced by Neroued in `04350ba9`), upstream
  carries `exercise_host_restore`, and this port is 0 commits behind `upstream/master`.
- **The port never chose it.** `git diff upstream/master -- src/runtime/engine/context_cache/` is 212
  insertions across 5 files, and none of them touch `pressure_committed`,
  `pressure_private_owners_degraded`, `dropped_checkpoint` or the pressure target. The decision is
  upstream's code, untouched.
- **It is the class of defect this release exists to remove.** 1.2.0 is about silent cache
  shortfalls -- the frozen reusable frontier, the 3-of-5 conversation hits -- and this is another: a
  reuse that silently loses speculative decoding, visible only as `degraded=1`.
- **The capacity is already provisioned.** The shipped launchers pass `--host-state-slots 16`, sized
  to hold every configuration the other bounds permit, so the State to preserve is budgeted for. The
  current behaviour spends that budget on nothing.
- **The cheap option taxes a headline capability.** Losing the MTP state loses speculative decoding
  for that turn, and speculation is what the shipped launchers are named for.

AGENTS.md settles the tie: "Do not sacrifice these goals to reduce the diff or implementation effort.
Evaluate complexity, maintenance cost, and verification risk as engineering tradeoffs, not reasons to
retain a known inferior design."

## Consequences

- Implementation: the pressure target selection in `materialization_planner.h`, and the action in
  `resource_manager.h`, must prefer demotion over degradation while host State has room, keeping
  degradation as the fallback.
- Verification: upstream's case fails today and passes unchanged once the fix lands;
  `tests/test_resource_manager.cpp` covers pressure and degradation at unit level and is in the ASan
  subset of `tools/scripts/test_v3_asan.cmd`; the release gate is
  `tools/release/check_test_baseline.py`.
- Measurement: the trade-off is host bytes against restored speculation, so it is measured at the
  shipped topology rather than asserted.
- Until the fix lands the case stays baselined in `tools/release/test_baseline.json` with a pointer to
  this record, so the gate states it rather than hiding it behind a skip and the release is not
  blocked.
- This is a narrow, recorded divergence from upstream, as ADR-0006 is for context parallelism, so a
  later upstream merge can read the intent instead of fighting it.

## Why this needs recording

Upstream's maintainer docs are upstream's to edit and this port does not edit them, so the port needs
its own record of where its behaviour and that design intent disagree. Without this, the next reader
finds a baselined test and has to re-derive the attribution from scratch -- including the two
eliminations that took the longest (that the port's own retention work is not the cause, and that no
capture is being refused) and the one that decides it (that the port never touched the decision, so
the defect is upstream's own).
