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

## MEASURED: that block is not reached either

Instrumented `request_plan.cpp` around the `splits_private_state` computation and the state-slot
credit (`~L1021-1050`), rebuilt, and drove the same twelve-conversation reproduction. The run
reproduced the cliff exactly (99.8% x6 then 0.0% x6), and the trace printed **zero** lines --
confirmed with a literal search after a first check using a `-like` pattern was found to be wrong
(`[plan]` is a character class, not text).

So the `splits_private_state` hypothesis is disproved as stated, and something more useful is now
established:

> **Neither of the two state-slot accounting paths found in the planning code is executed for this
> workload.** The `device_state_slots` pool that multi-conversation traffic exhausts is not the
> state-slot reservation/credit tracked in `request_plan.cpp`, nor the reclamation in
> `pressure.cpp` (or `checkpoint_recovery.cpp`).

The cache under test therefore lives outside the model program's planning layer. The next
investigation should start from the pool itself rather than from planning: find what owns the
device state slots (`usage.device.state_slots` at `pressure.cpp:2639`), where its capacity
(`max_concurrency + device_state_slots`, `model_instance.cpp:123`) is enforced, and which component
refuses a new state when it is full.

Process note worth keeping: two of the negatives above were nearly reported falsely because of a
bad filter (`-like '[plan] ...'`, `Select-String '[state] ...'` without `-SimpleMatch`). **Check the
observation method before concluding the system did nothing.**

## FOUND: the refusal point

The pool and the allocator are:

- **`StateImageDevicePool`** -- `src/models/qwen3_5/state/state_image.h:125`. A fixed-slot device
  memory pool (`slot_count()`, `copy_slot`, `copy_to_host`), wrapping `LinearAttentionStatePool`;
  this is the GDN/SSM state Marconi describes as large and sparsely hit. It is memory only: **no
  eviction, no LRU, no policy of any kind.**
- **`StateImageStore`** -- `src/models/qwen3_5/program/storage/state_store.h` (36 KB), the allocator
  over that pool. It owns free lists (`free_objects_`, `free_device_slots_`,
  `free_object_count_`, `free_device_count_`), an object status enum (`Free`,
  `ReservedDestination`, ...), and:

```cpp
[[nodiscard]] std::optional<StateImageHandle> reserve_destination() noexcept;   // L144
```

**That optional is the refusal.** When the free lists are empty it returns `nullopt`, and the
request proceeds without a cached state image -- which is why the failure is silent and why reuse
stops rather than erroring.

The free lists are replenished by `release()`, so nothing is missing: **slots are freed only when
an object is released, and nothing releases a stale conversation.** The list empties at capacity
and stays empty, so `reserve_destination()` refuses forever. That is the cliff, and it is in the
allocator, outside the planning layer -- which is exactly what the two instrumented negatives
predicted.

## The fix, now precisely located

At `reserve_destination()` (and the corresponding `reserve` for the object, `L144` onward), when the
free lists cannot satisfy a request, release an existing object -- chosen by the two gates the
research note already established:

1. **unpinned** -- nothing in flight references it (`source_pins == 0`, and whatever the
   store-level equivalent of `checkpoint_references`/writer references is), which is vLLM's
   "reference count equals 0";
2. **least recently used** among the unpinned candidates, so the state a request is about to use is
   never the one released.

The store already knows both facts: it tracks pins and references, and objects carry a status. So
this is a release-on-demand path in the allocator, not a new subsystem -- and it is the same
discipline the planning code already applies when it can, just reached from the place that
actually refuses.

## Correction and the strongest conclusion: the safety machinery is already correct

Read `release()` and `can_release()` before writing anything, and both the design and the diagnosis
change for the better. `state_store.h:643-666`:

```cpp
[[nodiscard]] bool can_release(StateImageHandle handle) const noexcept {
    ...
    return object.checkpoint_references == 0 && object.source_pins == 0 &&
           !object.destination_pinned && !has_pending_replica(object);
}
```

`release()` **refuses** to free a state that any checkpoint references, and there is a
`can_release_after_checkpoint_references()` for the case where the checkpoint itself is being
dropped. So:

- **there is no dangling-reference risk** to design around -- `release()` is safe by construction,
  gated on exactly the reference conditions vLLM and SGLang use;
- the allocator therefore does **not** need new eviction logic. It needs to be **asked**.

Which closes the loop with both instrumented negatives, coherently:

| Observation | Explanation |
|---|---|
| `pressure.cpp` removal gates never reached | no state-slot deficit was ever presented to the planner, so no removal option was built |
| `request_plan.cpp` split-credit never reached | these requests are not splits, so that path never accounts for their state slot |
| the cliff is silent and restart-only | `checkpoint_references` never reaches 0, so `can_release` always returns false and `allocate` keeps returning `nullopt` |

So the defect is not a missing policy, and not a missing safety gate. It is that **a new
conversation's state-image reservation does not appear as a resource deficit in admission
planning**, so the machinery that could relieve it -- drop the least valuable old checkpoint and
its state -- is never invoked. Restarting the engine drops every checkpoint at once, which is why
only a restart restores reuse.

**The fix, restated:** make the state-image requirement part of the admission demand, so the
existing pressure planning can resolve it, and let the planner choose the victim by the value
ordering it already computes (fewest affected hits, fewest evictions, then recency). No new
eviction policy, no new reference tracking, no changes to `release()`.

This is a hypothesis again, and the last six were wrong, so it gets instrumented the same way: log
the physical demand a brand-new conversation presents at admission and whether it contains a
state-slot component.

## MEASURED: the demand has the slot but never a credit

Instrumented the `plan->demand` construction in `request_plan.cpp` (`~L1068`) and ran the same
twelve conversations. The reproduction held (100% reuse x6, then 0.0% x6), and the trace is
decisive:

```
matches: 6
[demand] active_d=1 added_d=1 credit_d=0 peak_d=1 removed_d=0     (x6, identical)
```

Three things follow, and one of them corrects the hypothesis above:

1. **The demand does contain a state slot** (`added_d=1`), so "the requirement is missing from the
   demand" was wrong. It is present.
2. **No credit and no removal is ever offered** (`credit_d=0`, `removed_d=0`) for these requests --
   consistent with the split-only gate at `request_plan.cpp:1040`. So the plan demands a slot it
   can only satisfy by *consuming* one, never by *reclaiming* one.
3. **The trace fires exactly six times and then stops**, matching the cliff at conversation 7
   exactly. Once the pool is full, this admission path is no longer taken at all -- the request
   proceeds down a different path and simply runs without a cached state, which is why the failure
   is silent.

So the fix is narrower than "add the requirement to the demand". It is:

> **When a plan demands a state slot (`added_d = 1`) and the store cannot supply one, offer a
> credit/removal for it** -- let the planner name an existing state to release, exactly as the
> split case already does at `request_plan.cpp:1040-1047`, and exactly as `release()`/`can_release`
> already support safely (`state_store.h:661`, gated on `checkpoint_references == 0` and
> `source_pins == 0`).

The victim choice can reuse the value ordering the planner already applies (fewest affected hits,
fewest evictions, then recency), so no new policy is needed -- only a removal option where today
there is none. The remaining question, which is where the next instrumentation should start, is
*why the path stops being taken* once the pool is full: whether admission takes an early
"no state available" branch before reaching the demand construction, or fails later.

## CORRECTION: the path does not stop, and credit-bearing candidates do exist

Instrumented **both** demand constructions -- `request_plan.cpp:1068` and `:1216` -- with distinct
tags, and re-ran the twelve conversations. The reproduction held (99.8% x6 then 0.0% x6).

```
[demand]  (builder 1) matches: 31
[demand2] (builder 2) matches: 31
```

Both builders fire for every request, so **the admission path is not stopped once the pool is
full**, and the previous section's "fires exactly six times" was wrong. That count came from a
stderr file overwritten by a duplicate server start (the PowerShell launcher reported
`ChildProcess.kill` while the process survived, so the server was started twice into the same
path). The cliff-at-7 correlation drawn from it was luck, not evidence.

What the fuller trace shows is more useful:

```
[demand2] active_d=1 added_d=1 credit_d=0 removed_d=0     (most candidates)
[demand2] active_d=1 added_d=0 credit_d=1 removed_d=2     (some candidates)
[demand2] active_d=1 added_d=1 credit_d=0 removed_d=1     (one)
```

**State-slot credit-bearing candidates exist** -- `added_d=0 credit_d=1 removed_d=2` frees two
slots and adds none. So the planner can already express the reclamation the fix needs; it is not
missing from the vocabulary.

The question therefore narrows again, and is now about *selection and effect* rather than
existence:

> When the store cannot supply a state slot, why does a plan that consumes one without reclaiming
> still win over a candidate that credits and removes slots -- or, if such a plan wins, why does
> the pool still end up full?

That is where the next instrumentation starts, and it is a smaller question than the last: log,
per admission, which candidate is selected and its `state_slots` credit/removal, and correlate with
`StateImageStore` occupancy (`device_occupied()` vs `device_capacity()`, `state_store.h:132-137`).

## ANSWERED by the design doc: it is "effect", not "selection"

`docs/maintainer/resource-scheduling-and-context-cache.md`, section 6.1 "Admission reservation",
lines 440-444 (translated):

> Materialization progress only transitions between allocation and reserved-but-unmapped; **it does
> not hand capacity to another request.** Active truncate or speculative rollback can release
> mappings, **but the corresponding capacity still belongs to that active reservation.** Only a
> **terminal release**, or a resource transition that explicitly shrinks the active entitlement,
> can **return capacity to the global pool.**

That settles the question, and it settles it in favour of the second possibility:

- A plan's `credit` releases resources **inside that request's own accounting**. It does **not**
  return capacity to the global pool, so choosing a `credit_d=1` candidate cannot help a *different*
  conversation -- which is exactly why the trace can show slot-crediting candidates while the pool
  stays full.
- Capacity returns to the global pool only on **terminal release** or an **explicit shrink of the
  active entitlement**.
- A **retained cache checkpoint is neither**. It is deliberately held so a later turn can reuse it,
  so the capacity it occupies is not returned.

**So a pool full of retained conversations is a design consequence with no eviction path at all** --
which is precisely what #251 reports as "no LRU eviction observed", and why only an engine restart
(which drops every checkpoint at once) restores reuse.

The doc also states the matching accounting rule at 446-450: a borrowed immutable StateImage/KV
source must not be charged twice through the primary binding, or a legitimate fork is misjudged as
exceeding the active guarantee. Any fix must preserve that.

### The fix this implies

Eviction must be expressed as **dropping a retained checkpoint** -- the least valuable one by the
planner's existing ordering -- so that its state image's `checkpoint_references` falls to zero, its
`can_release` becomes true, and its capacity returns to the global pool. Then `allocate()` can
succeed for the new conversation.

Every piece of that exists: the planner can drop checkpoints and already values them (fewest
affected hits, fewest evictions, then recency); `release()`/`can_release()` are safe
(`state_store.h:661`); and the credit/removal vocabulary is in `PhysicalResources`. What is missing
is a trigger: a state-slot shortfall for a *new* conversation is not currently a deficit the
planner will resolve by dropping a checkpoint, because the reclamation it knows about is scoped to
the request's own materialization (measured: the removal gates in `pressure.cpp` and the split
credit in `request_plan.cpp` are both unreached for this workload).

**So the fix is a trigger, not a policy**: present "a state slot is needed and the pool is full" to
the planner as a deficit it may resolve by dropping a checkpoint, and let the machinery that
already exists do the rest.

## The trigger's home, and the exact implementation shape

`capture.cpp:653-656` (`.../program/transactions/capture.cpp`) is where a conversation's state is
actually captured, and it settles where the trigger belongs:

```cpp
std::optional<StateImageHandle> destination = state_store->reserve_destination();
if (!destination) {
    throw std::logic_error("selected capture has no prepared Device State capacity");
}
```

The word **prepared** is decisive: capture treats a missing slot as a *precondition violation*, not
as a condition to handle. The planning layer is contractually required to have prepared the
capacity before capture runs. So the fix is not here, and not in `allocate()` -- it is in planning:
**when a plan will capture a state and the store cannot supply a slot, the planner must make one
available by dropping a retained checkpoint.**

Concretely, the change is:

1. At the point a plan decides to capture a state image (`capture`'s transaction preparation, or
   the plan construction that precedes it), ask `StateImageStore` whether a device slot is
   available (`device_occupied()` vs `device_capacity()`, `state_store.h:132-137`).
2. If not, include a checkpoint drop in the plan's options -- the victim chosen by the ordering the
   planner already applies (fewest affected selected hits, fewest explicit shared losses, fewest
   owner evictions, then recency). Dropping it drives that checkpoint's state image to
   `checkpoint_references == 0`.
3. The existing, safe `release()` path then returns its device slot
   (`can_release`, `state_store.h:661`), so the capture's `reserve_destination()` succeeds.

No new policy, no new reference tracking, no change to `release()`, and no change to the
`logic_error` contract above -- the contract is correct and should stay; it is the planner that
must stop violating it.

### Acceptance

`tools/release/repro_251.py --conversations 12` must show **all twelve conversations reusing**,
where the current build reuses six and then stops permanently. Then the suite
(`tools/scripts/test_v3.cmd`), the release gate (`check_test_baseline.py`), and a soak
(`soak.py`), because the change touches checkpoint lifetime.
