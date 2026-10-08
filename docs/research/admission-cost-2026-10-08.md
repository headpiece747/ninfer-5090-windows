# The per-request admission cost, and why it is evidence for #334 rather than a patch

MEASURED HERE, 2026-10-08, on this port's tree. This note exists because the cost was measured, then
bounded by three controlled arms, and then placed against upstream's own record — which changed what it is
for.

## The measurement

`cold-short` (a 30-token prompt, so the engine's per-request floor is essentially all of the prefill
phase) at 262,144 KV capacity, with the lane's own flags, one flag or setting changed per arm:

| arm | prefill floor |
|---|---|
| the lane as it ships | 42.85 ms |
| no speculative backend | 43.51 ms |
| host context off | 44.89 ms |
| **`--no-prefix-reuse`** | **29.86 ms** |

So **13–15 ms per request is charged by the cross-request history machinery**, and the flag's own
documentation says what that is: `docs/serving.md:939`, *"`--no-prefix-reuse` disables cross-request
history reads and writes"*. Three candidate explanations were refuted by controlled arms rather than by
argument:

- **KV capacity**: flat within 0.89 ms across a 32× range (8,192 → 262,144), so it is not pool-proportional
  page or mapping work. My earlier two-point claim that ~19.5 ms was capacity-linked was wrong, because the
  arm I compared against also differed in its spec route — two variables, one conclusion.
- **Host context**: −1.38 ms on against off at the same capacity.
- **The draft backend**: −0.66 ms at 30 tokens, where no speculative round runs.

And it is **per-request, not once-per-process**: the `anonymous-hot-continuation` case's second request —
a continuation with 7,695 of 7,718 tokens reused — still reports 43.8 ms of prefill for its 23 new tokens.
For an agent loop that is 13–15 ms on every turn.

The tree this was measured on **contains** the context-cache rewrite: `HEAD..upstream/dev` and
`HEAD..upstream/master` are both 0 commits, and the two tests that rewrite deleted
(`ninfer_admission_policy_test`, `ninfer_materialization_budget_test`) are absent. So this is current
behaviour, not a superseded implementation.

## Why it is evidence and not a patch

Upstream already knows about this class and has proposed replacing the machinery:
[#334](https://github.com/Neroued/ninfer/issues/334), *"replace the checkpoint-catalog prefix cache with a
content-addressed KV-block and state-snapshot cache"*, whose problem statement names the mechanism
directly — *"Planning sits on the critical path. Admission runs a bounded heuristic search over pressure
targets (up to a 250 ms planning allowance per admission)"*. This port's 13–15 ms is that admission work
charged on a request that cannot benefit from it, and it is the smallest instance of the same cost.

Hand-optimising it here would aim at a moving target that someone is already replacing. The useful form of
this finding is as a number attached to that direction, with its controls, in the place a redesign can use
it.

Three neighbours on the tracker say the same subsystem is the live area, and none of them is this cost:
[#251](https://github.com/Neroued/ninfer/issues/251) (prefix reuse stops once the checkpoint budget is
exercised; only a restart restores it), [#236](https://github.com/Neroued/ninfer/issues/236) (mass
shared-prefix eviction, 38.5 % repeat reuse), [#144](https://github.com/Neroued/ninfer/issues/144) and
[#135](https://github.com/Neroued/ninfer/issues/135) (both closed: concurrent streams re-triggering root
re-prefill, and a host-KV defect that terminated the engine).

## The corroboration worth recording beside it

The same session attributed the long-context prefill decay to attention by fitting the engine's own
timings — `t(n)/n = c/n + a + b·n`, three points fitted, the 260,081-token point predicted within 2.8 % —
with a kernel-level confirmation left open because the nsys trace came back void. Upstream's
[#352](https://github.com/Neroued/ninfer/issues/352) supplies that confirmation from its own bench:

```
| *_kv_tiled_mma_kernel | 46.10 s | 70.4 s (1.53x) | 87.7 s (1.90x) |
```

That is the kernel family the void trace had appeared to point at, measured properly by the people who own
it: the NVFP4-KV attention QK product on 16-bit tensor ops, 1.53× to 1.90×. The curve said attention
dominates at depth; the tracker says which kernel and by how much. **The void trace is not evidence and is
not cited as any** — this note records that the identification it failed to make is available from a
primary source, which is the reason not to re-run it.

## And the instrument for the quality side

[#171](https://github.com/Neroued/ninfer/issues/171) reports a Needle-in-a-Haystack harness whose
`judge_strategy: rule` scores answer wording rather than retrieval, with retrieval reading 72.7 % at
131,072 tokens and recovering to 95.5 % at 260,000 — a non-monotonicity the issue itself says has no
mechanism behind it. Adopting that instrument is cheaper than inventing one, and its documented pitfall is
the first thing to avoid.

## Pointers

| what | where |
|---|---|
| the measurement runs | `profiles/bench/floor-vs-capacity-2026-10-08/` |
| the validation that the flag, not capacity, is the difference | same directory, `no-reuse/` |
| the prefill curve and its out-of-sample check | `.audit/models-rebuild.tsv`, rows of 2026-10-08 |
| the void trace and its recipe | `profiles/nsys/prefill-decay-2026-10-08/VOID.md` |
| the lane figures this cost sits inside | `tools/release/profiles.py` |
