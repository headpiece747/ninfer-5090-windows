# The stream-ordered TMA descriptor buffer is not safe under CUDA Graph capture

**Target:** `Neroued/ninfer`, and specifically the open Windows/TMA-descriptor PRs
(`#233` "perf(ops,artifact): stream-ordered TMA descriptor allocation and unbuffered
Windows reads" and `#82` "feat(platform): native Windows (MSVC + CUDA) build for
ninfer-serve"), whose descriptor handling this analysis is against. Reproduced on
`master`-derived `dev` with the W4A4 linear route.
**Severity:** device fault (illegal instruction) in any captured graph that contains the
route; silent-wrong-answer risk if the descriptor bytes are stale rather than invalid.

## Summary

When the kernel parameter cannot carry the descriptor by value, the workaround is to copy it
into a device buffer and pass a pointer. That is correct for eager launches and for
single-stream use, which is what the existing comments cover — but it is **not** correct
under stream capture. During capture the `cudaMemcpyAsync` becomes a *node*: it does not
execute. Its source is a caller-scoped local, so by the time the graph replays, the memcpy
node reads a dead stack frame and the TMA unit is handed a garbage tensor map.

The by-value `__grid_constant__` path that upstream uses on other platforms has no such
problem, because the descriptor bytes are part of the kernel node itself.

## Reproduction

`tests/ops/linear/test_nvfp4_a4.cpp` captures the launch into a CUDA graph and replays it.
At the first shape whose token count selects the TMA route (`tokens >= 1024`, per
`src/ops/linear/nvfp4/shapes/n14336_k5120.cu`), it fails:

```
NVFP4_A4 [14336,5120] T=1024: unexpected exception: synchronize linear:
an illegal instruction was encountered
```

`T=1023` passes; `T=1024` fails. Those two take different routes, which is why the boundary
is so sharp.

## Evidence

`compute-sanitizer --tool memcheck` names the fault precisely:

```
Illegal instruction
  at ninfer::ops::detail::nvfp4_tma_load_2d(void*, const CUtensorMap_st*, int, int,
     unsigned long long*)+0x660 in nvfp4_w4a4_tma.cuh:176
  by thread (0,0,0) in block (0,0,0)
     Device Frame: nvfp4_w4a4_tma_kernel<...>(const Nvfp4W4a4TmaDescriptors*, ...) at line 258
```

and the host backtrace shows `cuGraphLaunch` — i.e. the faulting launch is a replay, not an
eager launch. Consistent with that, an eager prefill of 30,160 tokens through the server
completes normally, which is why this does not show up in serving today: graphs are used for
decode, decode runs a few tokens, and a few tokens do not select the TMA route.

## Suggested fix

Prefer passing the descriptor by value, as the non-Windows path already does.

The C2719 that motivates the pointer form comes from an **over-aligned wrapper struct**, not
from the descriptor's size. `cuda.h` already declares `CUtensorMap` as
`alignas(TENSOR_MAP_ALIGN)` with `TENSOR_MAP_ALIGN` = **64 under `_MSC_VER`** and 128
elsewhere, so a struct of four `CUtensorMap` (512 bytes) inherits an alignment MSVC accepts
once a local `alignas(128)` override is removed. With that removed, the by-value
`__grid_constant__` parameter compiles on MSVC 14.51 + CUDA 13.3, the kernel node carries the
bytes, and the replay works.

If the pointer form must be kept for some configuration, it needs a lifetime that outlives
the graph and a source that is stable across replays — not a per-call stack local. Note that
`#82`'s comment ("Safe for single-stream use (the current deployment); a multi-stream caller
must supply per-stream buffers") covers streams but not capture, and a persistent buffer
does not fix this on its own: the memcpy node's **source** is still the caller's frame.

## Related: the formerly failing planner test

Unrelated to the above, `tests/test_resource_manager.cpp`'s
`test_candidate_search_prefers_deep_reuse_without_eviction` used to fail and is now green. It was
not timing-dependent: the optional materialization search's initial grant was pinned flat at 5 ms,
which the incumbent's `/20` term never bound, and it now scales up to a 250 ms ceiling
(`src/runtime/engine/context_cache/materialization_budget.h`, `kMaximumGrantNs`) so the search
reaches the preserving reuse closure. See `docs/research/prefix-state-eviction.md`.
