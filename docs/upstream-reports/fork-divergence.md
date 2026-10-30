# Fork divergence from upstream

What this tree changes relative to `Neroued/ninfer`, and which of it will need reconciling when
upstream publishes. Read this before pulling an upstream release.

## Position

`dev` is a **strict superset** of upstream. Both `upstream/master` and `upstream/dev` are
ancestors of `dev`:

```
git merge-base dev upstream/master   ->  1882c6adb   (== upstream/master's tip)
git rev-list --count <base>..upstream/master  ->  0
git rev-list --count <base>..dev              ->  913
```

**Last integrated: `458863be2`, 2026-10-29** (`chore(merge): integrate upstream/dev (35 commits)
into the port`), 35 commits from `996d92146`. Code conflicts were five: `http_transport.cpp` (upstream
added Linux `SO_REUSEADDR`; the port kept its Windows `SO_EXCLUSIVEADDRUSE` branch and moved
`httplib::default_socket_options` into it), `anthropic_messages_http.cpp` (upstream restructured the
handler around `wait_for_admission`; the port's Cancelled check moved inside the non-stream branch),
`test_engine_prefix_real.cpp` (include rename plus upstream's new `terminal-publication` scenario),
`test_sparse_gqa.cpp` (upstream's FP8 scale branch, port `half_bits` helper), and
`paged_kv_fixture.h` (upstream's `cudaStreamSynchronize` supersedes the port's
`cudaDeviceSynchronize`). Everything else was documentation or a delete/modify.

Two structural reconciliations are worth knowing:

* **KV format header.** Upstream's `824bd3a41` created `include/ninfer/kv_cache_format.h` with its own
  `KvCacheStorage` and `KvCacheProfileInfo`, superseding both port headers (`kv_types.h`'s narrow enum
  and `kv_storage.h`'s name/parse table). Both port headers are deleted; the three call sites use
  `kv_cache_format.h`, and the enum spelling is upstream's (`Fp8E4m3fnGroup128Ue8m0`, not
  `Fp8E4M3Row256`). The device-tree include-narrowing purpose is preserved by upstream's header.
* **`__int128` in upstream's new code.** `src/runtime/engine/context_capacity.cpp` gained a 128-bit
  budget product in `e5e18b974`; it is computed through `ninfer::Uint128` here.

**Documentation language.** Upstream's `19c3a2ae4` translated `build-system.md`, `op-development.md`
and five support READMEs (`bench/`, `tests/`, `tools/`, `eval/`, and three maintainer pages) into
Chinese and condensed them heavily (bench 1786 -> 108 lines, tests 598 -> 77). This port keeps its
**English** `build-system.md` and the five support READMEs, which are current and carry the Windows
recipes; `op-development.md` follows upstream's Chinese original with the port's §6.4 appended, because
upstream's other Chinese pages link into its Chinese headings. `docs/maintainer/README.md` indexes the
twelve Chinese originals.

## Scale

`git diff --stat upstream/master dev` reports **588 files, +58,641 / -6,084** after the
`458863be2` merge (was 384 files, +27,572 / -5,342 at `594930e7`; the per-area table below is the
older `9e163eee` snapshot and is kept as a shape reference, not as a count).

| Area | Files | Kind |
| --- | --- | --- |
| `src/ops` | 45 | **shared** — the reconciliation surface |
| root (`*.bat`, `CMakeLists.txt`, `AGENTS.md`, `README.md`) | 17 | ours only — Windows port surface |
| `tools/release`, `tools/scripts`, `tools/artifact`, `tools/convert` | 24 | ours only |
| `tests/*` | 26 | mostly ours (MSVC portability) |
| `src/models` | 9 | **shared** |
| `docs/*` | 16 | ours only |
| `src/runtime`, `src/product`, `src/artifact`, `src/serve`, `src/core`, `include/ninfer` | 19 | mixed |
| `third_party/llama-jinja` | 2 | **shared**, not ours only as an earlier pass recorded: the files are upstream's, tracked at `upstream/master`. Corrected here because it changes the reconciliation surface |

## The reconciliation surface

Only files that **exist upstream** can conflict. Everything else is permanent, intentional
divergence and needs no attention.

### Windows-permanent (upstream will likely never fix these)

These exist because this is the Windows/MSVC port. Upstream targets Linux; the conditions do not
arise for them.

| File | Patch | Commit |
| --- | --- | --- |
| `src/artifact/file_io.cpp` | bound a Win32 direct read to one `ReadFile` (+100 lines of Windows path) | `da40626b`, `1d4c108c` |
| `src/serve/http_transport.cpp` | `winsock2.h`/`mstcpip.h` include order and the `_WIN32` keepalive branch | `369e5be1`, `556b6e02` |
| `CMakeLists.txt`, `src/CMakeLists.txt` | `WIN32_LEAN_AND_MEAN`, FFmpeg target decoupling, libcurl conditionality | `8d9a26f9`, `590ccfc8`, `b77d64bf` |
| `tests/*` | MSVC portability: `<array>` for CTAD, `constexpr` on `std::sqrt`, fixture share mode | `24e8850d`, `7effcafb`, `d8117ed4`, `1d4c108c` |
| `bench/context_cost/model_context_fixture.cpp` | `attention_pairs` divided through `ninfer::Uint128` instead of `__int128`, which MSVC x64 does not support - with it, every target in the benchmark tree failed to compile, so `NINFER_BUILD_BENCHMARKS=ON` could not build on Windows at all | `e9328d03` |
| `src/runtime/engine/context_capacity.cpp` | the context-capacity budget product (added by upstream `e5e18b974`) computed through `ninfer::Uint128` instead of `unsigned __int128` | this merge |
| `src/product/config/schema.cpp` | quantity parsing through `Uint128`; the multiply is done in 64 bits after the quotient is cast, which is the same value modulo 2^64 as upstream's 128-bit multiply-then-cast | `e16f2aef6` |
| `apps/perplexity/report.cpp` | `gmtime_s` under `_WIN32` in `timestamp()`, which upstream moved out of `main.cpp`; MSVC has no `gmtime_r` | this merge |
| `bench/models/qwen3_5/benchmarks.cmake`, `bench/models/qwen3_5/chat_render_bench.cpp`, `bench/README.md` | host-only chat-template render cost bench: reads a template, synthesizes its conversation, needs no artifact and no GPU | `c81e5bcf` |
| `.codex/hooks.json`, `.codex/hooks/clang_format.py` | removed: upstream's Codex hook runs clang-format from a Linux path and is not usable on this port. A merge resolves the modify/delete silently as "deleted by us", so it stays removed by design rather than by accident — worth knowing, because an incoming change to it will not conflict and will not appear in the merge stat | `c49eed3b` |

`src/core/uint128.h` is **ours only** — it does not exist upstream, so the `constexpr` work in
`4df4d2e8` cannot conflict. It was mis-classified as upstream-owned in an earlier pass.

### Shared logic — reconcile on upstream contact

These touch mathematics, binding or planning that upstream also owns. **When upstream modifies one
of these files, read our diff before merging.**

| File | Our change | Commit | Risk |
| --- | --- | --- | --- |
| `src/models/qwen3_5/load/dflash2.cpp` | bind selector codebooks and QKV format-agnostically (+10/-4) | `45387ec5`, `9898ecb1` | **highest** — upstream may tighten the binding we relaxed |
| `src/models/qwen3_5/execution/parameters.cpp` | derive draft context row offsets from the draft geometry (+11/-2) | `985907e0` | high — a geometry fix upstream may make differently |
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` | `Nvfp4Geometry<34816,5120>` for the A16Only gate/up workspace; A16 beyond T=16 (+24/-4) | `9e0ce8d7`, `fb141405` | medium — workspace shape, upstream may retune |
| `src/runtime/engine/context_cache/resource_manager.h` | reclaim retained continuations at active-capture admission (+116/-2) | `d05ee90a` | medium — upstream issue #251; they may fix it their own way |
| `src/runtime/engine/context_cache/materialization_budget.h` | scale the search grant off the 5 ms pin (+9/-1) | `2fcffaa9`, `e92cd9d7` | medium — upstream issue #229; same caveat |
| `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh` | pass the W4A4 TMA descriptor by value so it survives graph capture | `1218d574` | medium — see `ninfer-tma-descriptor-graph-capture.md` |
| `third_party/llama-jinja/jinja/runtime.cpp` | take `get_builtins()` by reference in `try_builtin_func`; the vendored copy took the type's static filter map by value and copied it on every filter application, ~1374 of them for a 229-message prompt | *(this worktree)* | low — an efficiency fix with byte-identical output, measured at 77.5 ms to 71.4 ms on the render bench. Upstream's own copy of the vendored file carries the same line, so it is a candidate to send them rather than a divergence to defend |

### Already reconciled

`src/models/qwen3_5/load/dflash.cpp`, `weights.h` and the fused-binding removal in `9898ecb1` are
settled: upstream was correct and our earlier fused binding was the divergence. Nothing to watch.
See the README for the full account.

The accurate silu in the NVFP4 SwiGLU is settled the same way, from the other direction. This port
landed it before upstream did, on contract and oracle grounds rather than scores; upstream restored
it in `c4ae8a9c` and measured the same conclusion (4.617111902 with the approximation against
4.615642116 with silu on the full ninfer-ppl-1m-v1 corpus). `src/ops/common/math.cuh` is therefore
identical between `dev` and upstream, and the only remaining difference in
`nvfp4_linear_swiglu_w4a4_tma.cuh` is the port's own descriptor-by-value fix (`1218d574`) plus the
comment recording why the accurate activation is kept.

## Two upstream issues we had fixed, now closed upstream

`#251` (prefix reuse stops after the checkpoint budget) and `#229` (5 ms materialization budget
forces root re-prefill) were fixed in this tree by `d05ee90a` and `e92cd9d7`. **Upstream closed both
on 2026-10-29** with the context-cache rewrite (`366`, `bea5a04ec` and `2cf044529`), and the
`458863be2` merge takes that rewrite:

- `src/runtime/engine/context_cache/materialization_budget.h` is **gone** from upstream and from this
  tree. The port's scaled-grant fix went with the file it lived in; the grant policy is upstream's.
  The port has not re-applied it, because its subject (a flat 5 ms cap) no longer exists. Nothing in
  the merged tree references it.
- `src/runtime/engine/context_cache/resource_manager.h` still carries the port's one semantic line:
  `release_unreferenced` returns instead of throwing when Native refuses a release, because the throw
  reaches a `noexcept` engine path and becomes `std::terminate`. Upstream's rewrite did not change
  that call site, so the guard is kept as a safety net; it is not a behaviour claim about #251.

The port's `resource_manager.h` reclaim-at-admission work (`d05ee90a`) is also gone with upstream's
rewritten manager. Nothing in the merged tree references `can_release_continuation` or the old
reclaim, so the port's replacement is upstream's, by contract precedence.

## Procedure when pulling an upstream release

1. `git fetch upstream && git merge-base --is-ancestor upstream/master dev` — confirm the new tip
   is not already contained.
2. `git diff --stat upstream/master dev` — the file list above should be the only overlap.
3. For each **shared logic** file in the new upstream range, read our diff before merging:
   `git diff upstream/master dev -- <file>`.
4. Re-run the suite and the release gate. The gate is the authority, not the merge result.
5. If upstream fixed `#229`/`#251`, decide explicitly: keep ours, take theirs, or combine — and
   record which in the commit message.
