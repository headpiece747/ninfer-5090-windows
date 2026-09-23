# Anticipated reuse pricing: how systems keep an entry for a turn that has not arrived

The question: an engine caches checkpoints -- a KV prefix plus a complete continuation state at a
prompt frontier -- and under memory pressure a planner picks which cached owner to degrade (drop its
checkpoints, keep it catalogued) or evict. The defect is that the planner drops a checkpoint **no
live request demands yet** but a **future** request will use, specifically the next turn of the same
conversation. At decision time its live-demand mask is 0, its priced transition loss is 0, and it
loses to anything with a live demand; a weight multiplier cannot repair this, because
`weight x 0 = 0`. This note asks how production inference/serving systems, and caching theory, price
an anticipated rather than a currently-demanded reuse: what signal makes an entry worth keeping for
the future.

## What this note does not re-derive

These are taken as given, per the ADR that recorded them. This note confirms the parts a source
could confirm and does not reopen the measurement.

- **The defect and its measured-out candidates.** ADR-0007
  (`docs/adr/0007-turn-closure-retention-under-host-pressure.md`) records that a pressure action
  commits on the owner whose turn closure the next request needs, that the dropped checkpoint
  carries `demand_mask == 0`, and that the pressure target, the search grant and the retention weight
  were each tried and changed nothing (lines 31-101). Confirmed by reading the local fold: the
  per-demand term is gated on the live mask at
  `src/runtime/engine/context_cache/context_portfolio_value.h:82`
  (`if ((checkpoint.demand_mask & (1U << bit)) == 0) { continue; }`), and the private term is a
  retention weight applied to a per-checkpoint saving
  (`context_portfolio_value.h:61-102`).
- **The engine's existing vocabulary.** `CheckpointKind` (`SessionEndpoint`, `TurnClosure`,
  `ResponseReplay`, `SharedStablePrefix`, `LongAnchor`) at `src/runtime/contract/resources.h:127`;
  `RetentionClass` (`SharedStable`, `LiveSession`, `RecentPrivate`, `Disposable`) at
  `resources.h:146`; the 32-entry demand window and `demand_mask` at
  `resource_manager.h:1443-1456`; the retention weights `Disposable=1, RecentPrivate=4,
  LiveSession=16` and the private/one-unobserved-reuse prior in
  `docs/maintainer/resource-scheduling-and-context-cache.md` section 8.3 (lines 634-705).
- **SGLang's `retention_priority` reasoning and T-LRU's "protected" class.** The task supplies
  these; below they are quoted from the primary sources rather than re-argued.

One correction up front: `retention_priority` is TensorRT-LLM's name; SGLang's field is `priority`
plus `retention_seconds`, mapping onto it (quoted below). The ADR's point stands.

## Sources

### vLLM -- free-queue LRU, with an admission-count filter on the offload tier

vLLM's v1 prefix cache holds blocks in a global hash table keyed by the block's tokens chained with
its prefix, and keeps free blocks in a doubly linked list ordered for eviction. The order is
explicitly framed as LRU in the code and the design doc:

> The queue is ordered by block ID in the beginning. When a block is allocated and then freed, it
> will be appended back with the eviction order: 1. The least recent used block is at the front
> (LRU). 2. If two blocks have the same last accessed time ... the one with more hash tokens ... is
> at the front.
> -- `FreeKVCacheBlockQueue` docstring, `vllm/v1/core/kv_cache_utils.py`
> (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)
Two mechanisms sit on top of recency. A prefix-cache hit "touches" the block -- increments its
reference count and removes it from the free queue "to avoid these computed blocks being evicted"
(`BlockPool.touch`, `vllm/v1/core/block_pool.py`). And a released-but-still-era `unpin_blocks` path
makes blocks "last-resort eviction candidates regardless of prefix caching", with a callback fired
when the pool reuses them (`block_pool.py`). Neither uses session identity: vLLM stores a
`session_id` on the request and emits it in KV events, but no session term was found in the block
pool's eviction decision. The signal is **reference count zero, then access history (recency)**.

vLLM's older `vllm/core/evictor.py` exposes only one policy (`EvictionPolicy.LRU`), but the v1 CPU
**offload** tier is genuinely pluggable and frequency-aware:

> An OffloadingManager with a pluggable CachePolicy, resolved by name via CachePolicyFactory (built
> in: "lru", "arc"; external policies can either register their own or be loaded out-of-tree via
> cache_policy_module_path).
> -- `vllm/v1/kv_offload/cpu/manager.py`
> (https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_offload/cpu/manager.py)

That manager also carries an admission filter: a new key is stored only after its access count
reaches `store_threshold` (`prepare_store`: `self.counts.get(key, 0) >= self.store_threshold`), and
it keeps an `OrderedDict` of per-key counts so the LRU entry can be evicted in O(1). This is a
**past-access-frequency** estimate of future reuse -- the same idea as an admission policy, applied
one tier down. It is not exposed for the GPU block pool.

### SGLang -- priority, decay, T-LRU, and a hybrid component tree

SGLang's radix cache ranks evictable leaves through an `EvictionStrategy` returning a priority key;
the built-ins are `LRUStrategy` (`last_access_time`), `LFUStrategy` (`hit_count, last_access_time`),
`FIFO`, `MRU`, `FILO`, `PriorityStrategy`, `SLRUStrategy`, `TLRUStrategy`
(`python/sglang/srt/mem_cache/evict_policy.py`,
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/evict_policy.py).
`evict()` pops the smallest priority and re-queues a parent when its last child is removed
(`radix_cache.py`).

**`priority` is a bias, not an exemption -- the claim to verify is true.** The hard pin was merged
(SGLang PR #18941, "feat: TTL-based prefix pinning with refresh-on-hit for HiRadixCache") and then
reverted by a merged PR titled "revert: remove TTL-based hard pin from HiRadixCache" (#21884, state
`MERGED`). A companion PR, "Revert TTL pin in L2 cache with softer duration" (#21045), states the
reason verbatim:

> All blocks are evictable under memory pressure. Priority controls eviction *order*, not exemption
> -- lower priority blocks are evicted first (LRU within same tier).
> -- SGLang PR #21045 (https://github.com/sgl-project/sglang/pull/21045)

The follow-up that carried the design forward repeats it and names the reverted chain:

> Under enough pressure it is evicted and re-prefilled -- protection is a bias, not a lock (contrast
> with the reverted hard TTL pin #18941/#21884 ...).
> -- SGLang PR #27970 (https://github.com/sgl-project/sglang/pull/27970)

#21045's behaviour table is explicit that the ceiling is still evictable: `priority=99` is "evicted
last ... but still evictable under memory pressure. Priority never decays", whereas
`priority=50, retention_seconds=300` "decays to 0" after 5 minutes idle. The per-token-range RFC
that generalises this says the same: "This is a soft ranking, not a hard pin. ... evicts
lowest-priority first. A priority-100 block is still evicted if the cache fills and no
lower-priority free block remains -- think priority queue, not lock." (SGLang issue #36208,
https://github.com/sgl-project/sglang/issues/36208)

**T-LRU is the future-aware signal, and it is defined by the next turn's uncached token count.**
SGLang implements the policy from "Tail-Optimized Caching for LLM Inference" (Zhang, Li, Moallemi,
Peng, arXiv:2510.15152):

> A conversation with history length L whose next prompt is expected to add Q_hat tokens only has to
> keep L + Q_hat - threshold tokens cached to hold its next prefill under the TTFT budget; tokens
> past that budget cannot improve tail latency and are "TEL-safe", i.e. free to evict.
> -- `TLRUStrategy` docstring, `evict_policy.py`

The code:

```python
budget = max(node._tlru_history_len + self.next_prompt_estimate - self.threshold, 0)
cached_without_this_node = node._tlru_cached_prefix_len - len(node.key)
tel_safe = cached_without_this_node >= budget
return (-1 if tel_safe else 0, node.last_access_time)
```

A node is **protected** (not TEL-safe) when evicting it would leave fewer than `budget` cached
tokens, i.e. when it would push the next turn's uncached token count above the threshold. The
signal is per conversation: its high-water history length `L` and a predicted next-prompt size
`Q_hat` (SGLang CLI `--tlru-qhat-tokens`, default 200). No element of it is the current live demand.
TEL-safe nodes are reported as infinitely old, so the driver drains them first and falls back to
recency order (SGLang PR #21708).

**Hybrid/recurrent state is a first-class component with its own priority.** SGLang's unified radix
cache holds a tree with `FULL`, `SWA` and `MAMBA` components, evicts greedily per component, and
cascades: "When a component evicts a node, all other components with equal or lower eviction_priority
on the same node are also evicted." A `FULL` leaf cascade can satisfy a `MAMBA` target by acting as a
"donor" (`unified_radix_cache.py`). EAGLE draft keys are cached as bigrams in the same tree
(`is_bigram=self.tree_core.is_eagle`, `RadixKey(..., is_bigram=True)`) -- there is no separate
draft-state retention policy; the draft state rides the same eviction order.

SGLang also folds priority and retention into one tuple on shared prefixes: "higher priority wins;
on equal priority, longer retention wins, with 0/None = permanent dominating" (#27970).

### TensorRT-LLM -- declared per-token-range priority

TensorRT-LLM assigns every block a priority 0-100 and evicts by priority, then LRU:

> The core eviction scheme is prioritized LRU. All blocks are assigned a priority between 0 and 100
> (100 being most important). All blocks of the lowest priority must be evicted before any blocks of
> the next priority can be evicted. If all blocks have the same priority, the least recently used
> block is evicted.
> -- `docs/source/features/kvcache.md`
> (https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md)

The priority is **declared by the request**, not inferred: `KvCacheRetentionConfig` carries a list of
`TokenRangeRetentionConfig(start, end, priority, duration)` plus `decode_retention_priority`.
"Blocks with lower priority scores will be freed preferentially"; priority reverts to the default of
35 after `duration_ms` from the first reuse, and `None` means it never expires. NVIDIA states the
intent plainly: the API "enables an LLM deployer to use knowledge about their workload to improve
reuse opportunities by persisting blocks that are likely to be reused"
(https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/).
The C++ manager keeps `mRefCount` for the safety gate and `mPriority` for the ranking, evicts only
when a new block is needed, and even detaching a block from an interior trie node keeps descendants
eligible under their own priorities (`kvCacheManager.cpp`). Offload preserves priority, and
`secondary_offload_min_priority` (default 3) gates what may leave the GPU.

This is the clearest production answer to the defect: the value of a not-yet-demanded entry is
**declared ahead of time**, adjacent to the request's semantics (a token range), and decays on
idleness. It sidesteps the zero-multiplier problem by giving the entry a priority that exists even
when its reference count is zero -- but it is still ordinary evictable memory ("the lowest priority
must be evicted before any blocks of the next priority"; nothing is exempt).

### LMCache -- pluggable LRU/LFU/FIFO/MRU over a lock filter

LMCache's distributed eviction module separates the policy from the tier:

> Eviction module to determine what to evict from L1 and L2 caches. ... Pure abstract base class for
> eviction policies. Subclasses implement the LRU (or other) tracking logic via the `on_keys_*`
> methods and expose eviction decisions via `get_eviction_actions`.
> -- `lmcache/v1/distributed/eviction.py`
> (https://github.com/LMCache/LMCache/blob/main/lmcache/v1/distributed/eviction.py)

The storage-backend policy set is `{"LRU", "LFU", "FIFO", "MRU"}`
(`lmcache/v1/storage_backend/cache_policy/__init__.py`); `LRUEvictionPolicy` "evicts the least
recently used keys first". Eviction is gated on eligibility, not just recency: a `key_eligible_filter`
skips locked keys ("if `target_count = 10` but 8 out of the first 10 LRU keys are currently
read-locked or write-locked, the subsequent delete call can only successfully remove 2 keys"; LMCache
PR #2978, https://github.com/LMCache/LMCache/pull/2978). There is an `IsolatedLRUEvictionPolicy`
with per-`cache_salt` quotas (`storage_controllers/eviction_controller.py`). No session-or-turn term
was found; the signal is **access history plus a lock/eligibility filter**.

### Mooncake -- LRU plus soft pin (TTL) and hard pin, both declared

Mooncake Store's memory eviction is "an approximate LRU policy", gated so that "objects that have
leases or have not been marked as complete by `PutEnd` requests will be ignored by the eviction
task" (`docs/source/design/mooncake-store.md`,
https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/mooncake-store.md). On top of
that it offers exactly the two retention strengths the ADR contrasts:

> **Soft Pin** ... During eviction, objects that are not soft pinned are prioritized for eviction.
> Soft pinned objects are only evicted when memory is insufficient and no other objects are eligible
> for eviction. ... `default_kv_soft_pin_ttl`: the duration ... after which a soft pinned object will
> have its soft pin status removed if not accessed. The default value is 30 minutes.

> **Hard Pin** ... objects that must never be evicted under any circumstances ... will never be
> selected as eviction candidates regardless of memory pressure.

Soft pin is a **declared, idle-decaying** retention: it protects a system prompt or a session's
prefix, and refreshes on access. Hard pin is an outright exemption (added in PR #1728,
https://github.com/kvcache-ai/Mooncake/pull/1728), reserved for weights and critical metadata, and
explicitly described as replacing a timeout-based approach. Mooncake is the clearest statement that
"keep this for a future reuse" is a **client-declared** attribute, and that the safe form of it
decays.

### NVIDIA Dynamo / KVBM -- declared prefix priority, LFU admission, and a proactive warm

Dynamo's KV Block Manager offloads GPU -> host -> disk and filters what leaves each tier by declared
rank and by estimated frequency:

- `DYN_KVBM_HOST_MIN_PREFIX_PRIORITY` is the "minimum prefix priority a block must have to be
  offloaded to the host tier" -- a declared floor.
- offload policies are `pass_all` / `presence` / `presence_lfu` (AND-combined), with
  `presence_lfu.min_lfu_count` defaulting to 8
  (https://docs.nvidia.com/dynamo/v1.4.1/reference/components/kvbm-configuration).
- the disk offload filter (on by default, "to extend SSD lifespan") offloads only when frequency
  >= 2, where "Frequency doubles on cache hit (initialized at 1) and decrements by 1 on each time
  decay step" (https://docs.nvidia.com/dynamo/dev/user-guides/kv-cache-offloading).

The most directly relevant single mechanism in Dynamo is on the SGLang backend: an agent hint
`speculative_prefill` means "After response completes, sends a `max_tokens=1` prefill to warm the KV
cache for the predicted next turn" (`docs/backends/sglang/agents.md`,
https://github.com/ai-dynamo/dynamo/blob/main/docs/backends/sglang/agents.md). That is an active
purchase of anticipated reuse: rather than raise the entry's price, it makes the future request
exist as a present one. (The same doc notes `X-Dynamo-Session-Final` is normalised into an internal
eviction hint the SGLang backend does not act on in that release.)

### Caching theory -- all of it estimates future reuse from the past, except one offline bound

Every classical policy infers "will be reused" from **access history**; none reads future demand.

| Policy | Signal for "will be reused" | Stated guarantee / note |
|---|---|---|
| **GreedyDual-Size (GDS)** | `H = L + cost/size` set on access; `L` is the running eviction floor, so H ages | online-optimal: miss cost `<= (s_cache/s_min) x` offline optimal (Cao & Irani, USITS '97, https://www.usenix.org/conference/usits-97/cost-aware-www-proxy-caching-algorithms) |
| **GreedyDual-Size-Frequency (GDSF)** | `H = L + frequency x cost / size`, frequency incremented on access | extends GDS with frequency, "incorporates ... file size, file access frequency and recentness of the last access" (Cherkasova, HPL-98-69R1, https://eclass.uoa.gr/modules/document/file.php/D245/2015/HPL-98-69R1_GDS.pdf) |
| **LRU-K** | time of the K-th last reference (backward K-distance); LRU is K=1 | optimal under the independent reference model among algorithms using the last K references (O'Neil/O'Neil/Weikum SIGMOD '93, https://dl.acm.org/doi/10.1145/170035.170081; proof https://dl.acm.org/doi/10.1145/300515.300518) |
| **2Q** | probationary FIFO A1 + main LRU Am + ghost A1out; a re-reference from the ghost promotes to Am | constant-time approximation of LRU-2; frequency enters via the ghost hit (Johnson & Shasha, VLDB '94) |
| **ARC** | recency list T1 and frequency list T2, with ghosts B1/B2; learning rule sets the split `p` | self-tuning, "empirically universal", scan-resistant, constant per request (Megiddo & Modha, FAST '03, https://www.usenix.org/conference/fast-03/adaptive-replacement-cache) |
| **LIRS** | Inter-Reference Recency / reuse distance, not recency directly; LRR vs HIR sets | "significantly outperforms LRU"; constant average-case overhead (Jiang & Zhang, SIGMETRICS '02, https://dl.acm.org/doi/10.1145/511399.511340) |
| **TinyLFU / W-TinyLFU** | approximate frequency of the candidate vs the victim (Count-Min Sketch + doorkeeper), used as an **admission** test | "equal or better hit-ratios than other state of the art replacement policies on these traces. It is the only scheme to obtain such good results on all traces" (Einziger/Friedman/Manes, arXiv:1512.00727, https://arxiv.org/abs/1512.00727) |

Two structural points matter. The theory separates **admission** from **eviction**, and TinyLFU's
result is that a frequency *admission* filter makes the eviction choice much less consequential. The
only future-aware optimum is *offline*: Belady's MIN needs the actual future request sequence, so it
is an upper bound, not a signal. LRU-K's and LIRS's prediction is a statistical estimate from past
inter-reference times.

### Recurrent/hybrid state -- Marconi's reuse-likelihood taxonomy

Marconi is the system closest to a "continuation-state checkpoint": hybrid attention + SSM models,
where the SSM state admits only exact-match prefix reuse and "a deluge of (large) cache entries per
sequence, most of which yield minimal reuse opportunities" (arXiv:2411.19379,
https://arxiv.org/abs/2411.19379; MLSys 2025). Its answer is **admission by predicted reuse
likelihood**, not eviction weight:

> Marconi judiciously admits SSM states, only accepting states with a high reuse likelihood based on
> a taxonomy of potential prefix reuse scenarios.
> ... our novel admission and eviction policies that more judiciously assess potential cache entries
> based not only on recency, but also on (1) forecasts of their reuse likelihood across a taxonomy of
> different hit scenarios, and (2) the compute savings that hits deliver relative to memory
> footprints.

Two consequences transfer. A shared prefix is a *different* admission case from a session's own tail
-- the same distinction `CheckpointKind::SharedStablePrefix` vs `TurnClosure` draws. And "size fails
as a proxy in Hybrid LLM inference, where longer sequences (with greater compute savings) are
represented by equally sized SSM states": price the *compute saved*, not the bytes held. SGLang's
`FULL`/`SWA`/`MAMBA` cascade makes the same point.

### Speculative draft state -- no dedicated retention policy found

No production system was found that prices **draft-state** retention separately from its KV. vLLM's
speculative-decoding pages show EAGLE/MTP setup but no retention surface; SGLang stores EAGLE keys as
bigrams in the same radix tree and evicts them by the same policy; TensorRT-LLM's draft KV uses the
same block manager. Where a signal exists it is a **lookahead predictor**, not a retention weight:
OasisKV uses MTP/EAGLE drafted tokens as "a training-free signal for future KV-cache access patterns"
to drive sparse prefetch and eviction (arXiv:2608.08097, preprint). This is the closest thing found
to pricing a continuation state for a turn that has not happened, and the least productised. Mark the
absence of a dedicated draft-state retention policy as **unverified** beyond these observations.

## Signals production systems actually use

Category is the origin of the signal: **demand** = a live request's reference count/mask;
**access-history** = past accesses (recency/frequency); **session** = conversation/turn structure;
**declared** = an out-of-band hint from the client or engine.

| System | Signal | Category | Note |
|---|---|---|---|
| vLLM v1 block pool | `ref_cnt == 0`, then LRU order in the free queue | demand + access-history | prefix hits `touch` and leave the queue; `unpin_blocks` is last-resort |
| vLLM CPU offload tier | pluggable `lru`/`arc`; `store_threshold` access-count admission | access-history | `arc`; counts kept for O(1) LRU eviction |
| SGLang radix cache | `EvictionStrategy` key: recency, `hit_count`, `cumulative_tokens`, or `priority` | access-history | `lock_ref > 0` blocks are never candidates |
| SGLang `priority` + `retention_seconds` | declared 0-99 rank, decaying to 0 on idle | declared | "eviction order, not exemption" |
| SGLang T-LRU | next turn's uncached token count `L + Q_hat - threshold` | session | no live demand used; needs `Q_hat` |
| SGLang unified tree | per-component `eviction_priority` (FULL/SWA/MAMBA), cascading | declared (structure) | hybrid/recurrent state freed together |
| TensorRT-LLM | declared `TokenRangeRetentionConfig` priority 0-100 + duration, `decode_priority` | declared | prioritized LRU; default 35; decays |
| LMCache | LRU / LFU / FIFO / MRU + lock-eligibility filter; per-salt quotas | access-history | pluggable; no session term |
| Mooncake | approximate LRU; soft pin (TTL, refreshed on access); hard pin | declared + access-history | soft pin yields to pressure; hard pin is exemption |
| Dynamo KVBM | declared prefix priority floor; `presence_lfu` count >= 8; disk frequency >= 2 with decay | declared + access-history | tier-aware |
| Dynamo router | host/disk cache-hit weights 0.75/0.25 | access-history | values a hit before it is on GPU |
| Dynamo + SGLang `speculative_prefill` | warm the cache for the predicted next turn | session (active) | turns a future request into a present one |
| GDSF / GDS | `frequency x cost / size` (GDS: `cost/size`), age floor | access-history | online-optimal within a size factor |
| LRU-K | K-th last reference time | access-history | optimal under IRM; self-tuning |
| 2Q | probation FIFO + main LRU + ghost re-reference | access-history | constant-time LRU-2 |
| ARC | recency/frequency lists + adaptive split via ghosts | access-history | self-tuning, scan-resistant |
| LIRS | inter-reference recency / reuse distance | access-history | recency estimates IRR; low overhead |
| W-TinyLFU | approximate frequency as an **admission** test | access-history | frequency filter, not eviction weight |
| Marconi | reuse-likelihood taxonomy + compute saving vs memory footprint | session + declared (taxonomy) | hybrid SSM state; admission, not eviction weight |
| OasisKV | MTP/EAGLE drafted tokens as a lookahead for future KV access | session (predicted) | preprint; no retention weight |

No system in the table prices a not-yet-demanded entry from **current demand alone**. They use a
declaration, a past-access estimate, a session/turn model, or a lookahead predictor -- and, where
they protect, they decay or cap the protection.

## Design implication for this engine

Given the vocabulary already present (`CheckpointKind`, `RetentionClass`, `demand_mask`), the
literature and the systems above imply the following. These are implications, not a patch.

1. **The signal must not be `demand_mask`.** `demand_mask` is a correctness/safety notion: it is the
   engine's analogue of vLLM's `ref_cnt` and SGLang's `lock_ref`, and both of those *only* gate
   evictability; neither prices the future. The engine currently also lets it gate the per-demand
   value term (`context_portfolio_value.h:82`), which is why a zero mask yields a zero price. Every
   production system separates the two. The fix is not a better mask; it is a second signal.

2. **`CheckpointKind` is the engine's declaration, and it should feed an additive term, not a
   multiplier.** TensorRT-LLM's `TokenRangeRetentionConfig` and SGLang's `priority` both give a
   not-yet-demanded entry a rank that exists independent of any demand, and both use it as an
   *additive/categorical* ordering. The engine already knows the semantic kind at the model layer
   (`TurnClosure`, `ResponseReplay`, `SharedStablePrefix`, `LongAnchor`, `SessionEndpoint`); this is
   exactly the "taxonomy of potential prefix reuse scenarios" Marconi uses for admission. A
   multiplier on a saving that is zero when no demand matches cannot express it; an additive floor
   (or a separate kind-selected `Saving` recipe) can. Concretely: `TurnClosure` and `ResponseReplay`
   exist *for a later turn*; `SessionEndpoint` does not. That difference is a value, and it should
   not depend on whether a live request has matched yet.

3. **`RetentionClass` is the right ladder, but it must decay and must not exempt.**
   `Disposable/RecentPrivate/LiveSession` mirrors SGLang's priority tiers and Mooncake's
   soft-pin-plus-TTL. The two hard-won lessons are: priority controls *order*, not *exemption* (the
   SGLang pin was reverted), and a declared retention should expire on idle (Mooncake
   `default_kv_soft_pin_ttl`; SGLang `retention_seconds`; TensorRT-LLM `duration_ms`). An
   un-evictable closure would reproduce the reverted design and could wedge the pool; a decaying
   bias cannot. Note the engine already demotes before degrading when host State allows
   (ADR-0007's decision) -- decay fits that, because the degraded checkpoint is exactly the one that
   could have been demoted.

4. **The "one unobserved subsequent reuse" prior needs a signal behind it.** The maintainer doc's
   private term already states the intent ("each owner still has one unobserved subsequent reuse");
   the classical policies show what that prior must be conditioned on to be useful. LRU-K conditions
   on the time of the K-th last reference; ARC/LIRS condition on whether the entry has been re-seen;
   TinyLFU conditions on approximate frequency; GDSF conditions on frequency *and* the compute saved
   by a hit. A single constant weight per owner is the crudest form and cannot distinguish a
   `TurnClosure` (created every turn, reused next turn) from a superseded `SessionEndpoint`. The
   signal the engine already holds for this is the owner's own demand/catalog history -- the same
   32-entry records that feed `EmpiricalValue` -- read as *history*, not as *live demand*.

5. **Price recovery, not bytes, and derive it from the checkpoint.** Marconi's "size fails as a
   proxy ... price the compute savings" and GDSF's `cost/size` both say the value of keeping an entry
   is the recompute it avoids. The engine already computes `Rebuild - Recovery`, which is meaningful
   for an un-demanded checkpoint too, so it should be derivable from the checkpoint and the portfolio
   state as the maintainer doc's formula defines it, not only from the demand records.

6. **If prediction is wanted, prefer a session/turn model over a demand predictor.** T-LRU needs only
   a conversation's history length and a next-prompt estimate -- close to the engine's
   `LiveSession`/`RecentPrivate` classes. Dynamo's `speculative_prefill` makes the next turn real;
   OasisKV uses the model's own MTP drafts. Both are heavier and should not be the first move.

## What this does NOT settle

- **Declare or infer.** TRT-LLM, SGLang, Mooncake and vLLM's explicit eviction let the *caller*
  declare retention; only the classical policies infer it. The engine has no client-facing retention
  hint, and adding one is an external-contract decision, not a planner tweak.
- **Magnitude.** No source calibrates a `TurnClosure` against a live demand. T-LRU uses a token
  threshold against an SLA; TRT-LLM defaults to 35/100; SGLang uses 0-99 tiers; the engine uses
  1/4/16 -- not comparable units, and the ADR already showed a weight change alone moves nothing.
- **Bias vs exemption for a `TurnClosure`.** SGLang reverted hard pinning, but the engine's case
  differs: it holds the only copy of the closure, whereas SGLang's entries are re-derivable from
  surviving ancestors. Whether the answer is "bias, never exempt" or "demote-to-host first"
  (ADR-0007's direction) is not decided by these sources.
- **Whether the runtime can see the kind at all.** Resolved while this note was being written, and in
  the engine's favour: the grep in ADR-0007 is accurate -- `TurnClosure` and `ResponseReplay` are
  never named under `context_cache/` -- but that layer already reads `CheckpointRef::kind` and already
  distinguishes `SessionEndpoint` (`resource_manager.h:1537, 1542-1544, 1550, 1569`). So this is
  plumbing, not a new contract: `MaterializationCheckpointPolicy` holds `checkpoint.kind`,
  `retention_class`, `selected_hit_count` and `last_hit_epoch` (`materialization_planner.h:23-32`),
  and `ContextPortfolioCheckpointValue` (`context_portfolio_value.h:21-27`) keeps none of them. Only
  the magnitude question above remains open.
- **Admission vs eviction.** TinyLFU's and Marconi's results are about *admission*; the defect is on
  the eviction/pressure side. The ADR's "2 of 16 state slots occupied" favours pricing over admission
  filtering here, but that is one reproduction.
- **Multi-turn hit rate vs memory, and absence.** No source measures the trade-off at the engine's
  topology. And "no system prices not-yet-demanded reuse from current demand alone" is a finding
  across the systems read, not a proof; the draft-state retention question has no authoritative
  product source, and OasisKV is an unrefereed preprint.
