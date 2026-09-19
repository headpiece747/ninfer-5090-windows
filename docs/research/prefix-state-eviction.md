# Prefix state eviction: what production engines do, and what this engine lacks

Status: research note, written before changing `src/models/qwen3_5/program/planning/pressure.cpp`.
Supersedes nothing. The measurement it responds to is in `tools/release/repro_251.py`.

## The defect this addresses

This engine caches device states (the attention KV plus the GDN linear-attention state) so a later
request whose prompt extends a cached one can reuse instead of re-prefilling. When the cache is
full it does not reclaim anything: a new state simply cannot be admitted.

Measured with `repro_251.py` on the shipped QUASAR DFlash2 profile, six conversations each asked
twice with the second request extending the first verbatim:

```
conv 1-3  second request cache 19,828 (99.8% reuse)   TTFT 0.4-0.5 s
conv 4-6  second request cache 0 (0.0% reuse)         TTFT 2.1 s
```

From the fourth conversation onward every request re-prefills from the root, permanently, until the
engine restarts. Raising `--device-state-slots` moves the cliff (1 -> 3 conversations, 4 -> 4,
8 -> 6/6) but cannot remove it, because the cache has no eviction. Upstream tracks this as issue
#251, "no LRU eviction observed".

## What production engines do

Both mainstream LLM servers solved this the same way, and the constraint they both name is the one
that makes eviction safe under concurrency: **never evict an entry that something is still using**.

**vLLM** (https://docs.vllm.ai/en/latest/design/prefix_caching) -- KV blocks are hashed by their
tokens plus the prefix before them and held in a global table, with a free queue:

> "When there are no free blocks left, we will evict a KV block with reference count (i.e., number
> of current requests using the block) equals 0. If there are multiple blocks with reference count
> equals to 0, we prioritize to evict the least recently used block (LRU)."

Allocation additionally "touches" computed blocks, which "increases the reference count of the
computed block by one, and removes the block from the free queue if the block wasn't used by other
requests. This is to avoid these computed blocks being evicted."

**SGLang / RadixAttention** (https://arxiv.org/html/2312.07104v1) -- a radix tree keyed by token
sequences, values are KV tensors:

> "We implement an LRU eviction policy that recursively evicts leaf nodes. ... each node maintains a
> reference counter indicating how many running requests are using it. A node is evictable if its
> reference counter is zero."

Leaf-first matters for a different reason than safety: evicting an internal node would discard a
shared prefix that many sequences depend on, so leaves are evicted before their ancestors.

## The case that matches this engine exactly

Both engines above assume pure attention. This model is hybrid -- attention plus GDN linear
attention -- so its cached state is a *pair* (KV and an SSM-style recurrence state), and the
linear-attention state is the part that is large and awkward to checkpoint.

**Marconi: Prefix Caching for the Era of Hybrid LLMs** (MLSys 2025,
https://mlsys.org/media/mlsys-2025/Slides/3260.pdf) is about precisely this:
* "Reuses model states (KVs, SSM states) of common prefixes across requests"
* On naive periodic checkpointing: "Catch 1: cache entries are sparsely-hit. Catch 2: cache entries
  are huge. Frequent cache thrashing & low hit rate"
* "Existing systems: admit all states of most recent request. Marconi: admit states with high reuse
  likelihood only."
* "Existing systems: recency-focused (i.e. evict using LRU). Marconi: also considers the potential
  compute savings" -- a FLOP-aware utility rather than pure recency.

## Design conclusion for this engine

1. **Eviction is the fix, not a bigger cache.** Any finite capacity without reclamation ends in the
   same permanent cliff; that is what the measurements show.
2. **Gate every eviction on reference count.** An entry with a live reference (in flight, or
   referenced by a running request) must never be a candidate. This is the single safety property
   that makes concurrent eviction sound, and both vLLM and SGLang state it explicitly.
3. **Leaf-first / dependency-aware.** Our checkpoints form a lineage (continuations, shared
   prefixes, long anchors already have bounds: `--max-shared-prefixes`,
   `--max-private-continuations`, `--max-long-anchors-per-continuation`). Evicting a parent while a
   child survives is wrong even when both are unreferenced, so eviction should follow the same
   dependency direction the existing bounds do.
4. **Start from LRU.** vLLM and SGLang both default to it and it is validated in production;
   the arXiv study "Which Eviction Policy Should an LLM Cache Use?" (2608.20280) found no policy
   beating LFU by more than 0.041 percentage points across eighteen settings, so a sophisticated
   policy is not where the value is. Recency is enough to remove the cliff.
5. **FLOP-aware utility is the follow-up, not the first move.** Marconi's contribution is worth
   having for hybrid states because our entries are large and expensive, but it is a refinement on
   a working eviction path, not a substitute for having one.

## How to verify the cure

`tools/release/repro_251.py` prints the cliff position. Acceptance is that reuse holds for at least
as many conversations as the cache can hold and never permanently stops -- with the number of
conversations pushed past capacity (8+), which the current build fails. Then the existing suite
(`tools/scripts/test_v3.cmd`), the release gate (`check_test_baseline.py`) and a soak
(`soak.py`) to confirm no concurrency regression, since the change touches state lifetime.

## Pre-change baseline (the "old value" to beat)

Captured 2026-09-18 on the shipped QUASAR DFlash2 profile (`--device-state-slots 8`,
`--host-state-slots 8`, `--max-concurrency 1`), 12 conversations of ~26,000 tokens each, each
asked twice with the second request extending the first verbatim:

```
reuse per conversation: 99.8% 99.8% 99.8% 99.8% 99.8% 99.8% 0.0% 0.0% 0.0% 0.0% 0.0% 0.0%
```

Six conversations reuse, from the seventh every request re-prefills from the root, permanently,
until restart. Smaller `--device-state-slots` moves the cliff earlier (1 -> 3 conversations,
4 -> 4); none removes it.

Acceptance for the cure: **all twelve conversations reuse**, at any count, because each second
request immediately follows its own first and is therefore the most recently used state -- it must
never be the entry eviction chooses.

## Where to look first

`src/models/qwen3_5/program/planning/pressure.cpp` already contains machinery that removes state
slots to resolve a deficit -- `option.effect.removed.device.state_slots` and
`option.effect.removed.host.state_slots` around lines 559-583 and 780-800. So the first hypothesis
is not "there is no eviction path" but "the existing removal path does not fire when admission
needs a slot", which is a much smaller question than adding eviction from scratch.

## CORRECTION: private-continuation reclamation does exist

The paragraph that stood here was wrong and is retracted. `PressureStateDecision` already carries
private-continuation actions -- `DropEndpointDeviceDuplicate`, `DemoteEndpointToHost`,
`DropEndpointHostDuplicate` (and the `Rewrite` trio) at `program_impl.h:114-125` -- and
`pressure.cpp` builds them at `L760-767` under the same refcount gates as the shared case:

```cpp
checkpoint_was_dropped || already_changed || !state_store->valid(state) ||
state_store->role(state) != StateImageRole::CheckpointImmutable ||
state_store->source_pins(state) != 0 || ...          // pressure.cpp:781-783
```

Repeated at `L1078-1080` and `L1183-1186`, and dispatched at `L1319-1329`.

So this is **not** a missing subsystem, and extending the shared block to private states is **not**
the cure -- that entry already exists. The open question is narrower and sharper:

> Why does an admitted endpoint state not get reclaimed when `device_state_slots` is full?

Candidates, in the order worth testing:

1. **Role.** The gates require `StateImageRole::CheckpointImmutable`. If the state our
   conversations leave behind is a different role, every path above skips it silently, which would
   look exactly like "no eviction".
2. **Pins.** `source_pins(state) != 0` excludes it; something may hold a pin longer than expected.
3. **Scope.** The planner builds pressure options from the sequences and owners of the request
   being admitted (`L437` also consults `state_exclusive_to_sequence`). If an older conversation's
   state is outside that scope, no option can ever name it -- and the fix is then about which
   states are *offerable*, not about the removal itself.

Note `L437` excludes a state that is `state_exclusive_to_sequence(source, state)` for the sequence
being considered, which is the mechanism that protects a request's own state -- the recency
protection the predicate depends on.

The next step is therefore to instrument which of those three rejects a full pool, not to write
new eviction code.

## MEASURED: the reclamation code is never reached

The instrumentation was run (`pressure.cpp`, temporary `std::fprintf` at both rejection points in
the endpoint `add_state` lambda, rebuilt, driven by `repro_251.py` at 12 conversations against the
shipped QUASAR DFlash2 profile).

**Result: zero rejections. Neither path executed during the entire run.**

So the three candidates above are all wrong, and so is the framing: the planner does not *attempt*
to reclaim an endpoint state image and get refused. It never attempts it. The reclamation machinery
found in `pressure.cpp` is admission-time *pressure relief* -- it runs when the planner is
resolving a deficit for a specific request's materialization. The `device_state_slots` pool that
multi-conversation workloads exhaust is evidently managed outside that path, and simply refuses
when full.

What follows for the fix: **this is not a missing eviction policy inside the planner; it is a cache
whose admission is not wired to any eviction at all.** The next investigation is where
`device_state_slots` is actually allocated and refused -- the counter and pool behind
`usage.device.state_slots` (`pressure.cpp:2639`) -- and whether that cache can consult the
reclamation machinery that already exists, or needs its own LRU.

Four hypotheses have now been proposed from reading and four disproved by measurement
(`Uint128` as the planner cause, #229's applicability, #178's applicability, and this "role/pins/
scope" gate analysis). The lesson worth keeping: **instrument before asserting a mechanism.**

## Next lead, untested: the state-slot credit is gated on splitting a private state

Reclamation also lives in `checkpoint_recovery.cpp:634-638`, and state slots are accounted as a
reservation plus a credit in `request_plan.cpp`. The gate that matters is at `request_plan.cpp:1040`:

```cpp
if (splits_private_state) {
    credit.device.state_slots  = 1;
    removed.device.state_slots = 1;
    if (physical_peak.device.state_slots == 0) {
        throw std::logic_error("StateImage identity split has no active destination");
    }
    --physical_peak.device.state_slots;
}
```

The credit -- and the matching removal -- exist **only when a plan splits a private state**, i.e.
when a new state branches off an existing private one. A brand-new independent conversation does
not split anything, yet it still consumes a slot.

If that reading is right, it explains both observations at once: why the pool fills with no
reclamation, and why the instrumented removal paths were never reached (none of the twelve
conversations was a split).

**This is a hypothesis, not a finding** -- the previous four were all wrong. Test it by
instrumenting `splits_private_state` and the state-slot reservation/credit for each request in the
twelve-conversation run, and confirm that a non-splitting admission takes a slot with no credit and
no removal attempt. Only then is the fix "allow a non-splitting admission to reclaim an unpinned
LRU state slot" worth writing.
