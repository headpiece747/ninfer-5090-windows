# Windows / Blackwell ports: does any run the FP8 A8 TMA route, and does anyone use rank 3 for it?

Survey date: 2026-10-01. Author: subagent. Scope: `Neroued/ninfer` upstream, its 293 listed forks,
and one upstream-PR Windows port by a non-fork contributor.

Question under test (from `docs/active-work.md` item 12): our FP8 A8 TMA route
(`src/ops/linear/fp8/fp8_a8_tma_mma.cuh`) does not compile on MSVC (`C2719`) and, with the
by-pointer descriptor workaround, faults at execution with `cudaErrorIllegalInstruction`. Untested
hypothesis: `cp.async.bulk.tensor.2d` is broken on sm_120a while `.3d` works, so `fp8_tma_map`
should take the rank-3 sector-factored shape `bf16_tma_map` already uses.

**Answer in one line: three Windows ports build and run the FP8 A8 TMA route on sm_120a hardware, all
three keep the rank-2 descriptor and `.2d`, and none has converted it to rank 3. The 2d-versus-3d
hypothesis is not supported by any port in existence; the ports that succeed differ from our tree in
the descriptor *transport* (proxy acquire fence, persistent staging) and in the parameter *alignment*
(`alignas(64)` by value instead of pointer-passing), not in the descriptor rank.**

## Verdict table

`win32` = does the route/gate mention `_WIN32`. Read at the default branch unless a branch is named.

| repo | has `fp8_a8_tma_mma.cuh` | fp8 rank / instruction | bf16 rank / instruction | `_WIN32` in fp8 file | how it dispatches on Windows | relevant commits |
|---|---|---|---|---|---|---|
| `Neroued/ninfer` (upstream, `master`) | yes | **2 / `.2d`** | 3 / `.3d` | no | upstream has no Windows support | `909fb087` (adds the file, 2026-09-28) |
| `Wallawalla47/ninfer-custom` (`master`) | yes, modified | **2 / `.2d`** | 3 / `.3d` | yes (6 sites) | **TMA route stays selected on Windows** | `d0abd0cb` (2026-09-29, Ian Ranson) |
| `igorls/ninfer` (`workstation`) | yes, modified | **2 / `.2d`** | 3 / `.3d` | **no** | **TMA route stays selected on Windows** | `183cdca6`, `24e04e99`, `0f25f279` |
| `cometkim/ninfer` (`cometkim/dev`) | yes, modified | **2 / `.2d`** | 3 / `.3d` | no (macro in `ops/common/`) | **TMA route stays selected on Windows** | `ef61aa08` (2026-10-01, Hyeseong Kim) |
| `natpate/ninfer-windows` (`master`) | **no** (bf16 TMA only) | — | 3 / `.3d` | yes, bf16 file | bf16 TMA pointer-passed | see per-repo |
| `natpate/ninfer-windows` (`dev`) | yes, **byte-identical to upstream** | 2 / `.2d` | 3 / `.3d` | no | would not compile under MSVC as written | — |
| `MirkoCovizzi/ninfer-rtx5090-mobile` | **no** (bf16 TMA only) | — | 3 / `.3d` | no | — | — |
| `ValerioDolci/ninfer-tp2` (`main`) | yes, byte-identical to upstream | 2 / `.2d` | 3 / `.3d` | no | — | — |
| 19 further forks (see list below) | yes, byte-identical to upstream | 2 / `.2d` | 3 / `.3d` | no | — | — |
| `devan-carlin/ninfer` (PR #84, upstream PR) | no (NVFP4-era tree) | — | — | nvfp4 only | NVFP4 rank-2 `.2d` pointer-passed | PR `Neroued/ninfer#84` |

Tier-1 and Tier-2 repositories named in the brief that do **not** have the file at any branch, and so
cannot bear on the question: `Don-Chad/ninfer-3090`, `SV8ARJ/ninfer-windows-5090`,
`teo-mateo/ninfer-sm120`, `MushroomLover200/ninfer-rtx-6000-blackwell`, `gdevsnack-ai-labs/ninfer-gb10`,
`lch08/ninfer`, `plugmind-dev/ninfer-windows-wsl2-bridge`, `chillaheal/ninfer-win`,
`troubadour-hell/ninfer-win`, `2dameneko/ninfer-xx90-win`, `narsimhaReddyJuspay/ninfer-5090`,
`toddballinger/ninfer-5080`, `ivanov84/ninfer-windows-tp2`, `wamansou/ninfer-tp2-1m`,
`HelloaZelda/ninfer-5000ada`, `YurunTao/ninfer-4080S-32G`, `eniga/ninfer-pro4000`, `oosawak/ninfer3090`,
`songlinxin/ninfer3080-20G`, `juninron/ninfer-4070`, `SuperArilo/ninfer-4070Ti`, `Qm1n/ninfer-4070tisuper`,
`5p00kyy/ninfer-5060ti`, `ruwwww/ninfer-5060ti`, `lyux-gt/ninfer-tp2-5060ti`, `harryslimes/ninfer-fast`,
`agorevski/ninfer-rtx8000-48g-nvlink`, `MN-mansour/ninfer-4070Ti-26GB-8g-VRAM-shrink-120`.
Three repos in the brief returned no default branch and could not be read: `juninron/ninfer-4070`,
`lyux-gt/ninfer-tp2-5060ti`, `MN-mansour/ninfer-4070Ti-26GB-8g-VRAM-shrink-120` (`repos/<r>` returns no
`default_branch`; `branches` returns 404). Their contents were not examined.

The 19 forks whose `fp8_a8_tma_mma.cuh` blob SHA equals upstream's `a2470ca8…` byte for byte:
`KimDaeHwan99/ninfer`, `kossum/ninfer`, `qzshch/ninfer-upstream`, `BoysGameStudio/ninfer`,
`CoryPolicht/ninfer`, `770120799/ninfer-RTX3070-adapt`, `monochrome-muncher/ninfer`,
`kido5217/ninfer-yarn`, `pkochubey/ninfer`, `iflyshadow/ninfer`, `hamawang/ninfer`, `iodeh/ninfer`,
`Doelfke/ninfer-yarn`, `andypeng2015/ninfer`, `Little-Star888/ninfer`, `feiyunwill/ninfer`,
`knoopx/ninfer`, `giveen/ninfer`, and `ValerioDolci/ninfer-tp2` (counted separately in the table).
None of the 19 has `_WIN32` anywhere in its fp8 or bf16 TMA file.

---

## Per-repo detail

### 1. `Wallawalla47/ninfer-custom` — runs FP8 A8 TMA on Windows, rank 2, `.2d`

**Read in source.** `src/ops/linear/fp8/fp8_a8_tma_mma.cuh` at `master`:

- L20-23 `struct alignas(128) Fp8TmaDescriptors { CUtensorMap activation; CUtensorMap weight; };`
- L64-86 `fp8_tma_map` encodes **`CU_TENSOR_MAP_DATA_TYPE_UINT8, 2`** (rank 2) with
  `dimensions[]{k, rows}`, a single `strides[]{k}`, and 128B/64B swizzle by `block_k`.
- L88-95 `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes`.
- L113-126 the descriptor parameter is pointer-passed under `_WIN32`:

```
__global__ ... void fp8_a8_tma_mma_kernel(
#ifdef _WIN32
    // MSVC cannot pass the over-aligned (alignas(128)) CUtensorMap struct by value as a
    // __grid_constant__ parameter (C2719), so on Windows the descriptors are pointer-passed from
    // the staging buffer.
    const Fp8TmaDescriptors* descriptors_pointer,
#else
    const __grid_constant__ Fp8TmaDescriptors descriptors,
#endif
```

- L170-175, immediately before the first copy, per 128-byte map:

```
    // A staged tensor map was written through the generic proxy; each 128-byte map needs
    // its own acquire for the TMA (tensormap) proxy before its first use.
    acquire_staged_tensor_map(&descriptors.activation);
    acquire_staged_tensor_map(&descriptors.weight);
```

- L317-334 the launch site stages once through a singleton and passes the pointer:
  `fp8_tma_descriptor_staging().stage(descriptors, stream)`, comment *"Staged once: every token-slice
  launch below reads the same copy, in stream order."*

`src/ops/linear/bf16/bf16_a16_tma_mma.cuh` is the same file upstream has — `alignas(128)` at L15,
**rank 3** `cuTensorMapEncodeTiled` at L40, **`cp.async.bulk.tensor.3d`** at L64, plus the same
pointer/`_WIN32` split and the same `acquire_staged_tensor_map` calls at L118-121. So the fork has
exactly our 2d-fp8 / 3d-bf16 pairing.

`src/core/tma_descriptor_staging.cuh` (new file, 140 lines) is the mechanism, and its header comment
states the invariants:

```
// Stages an over-aligned (alignas(128)) CUtensorMap descriptor struct into device global memory
// for a kernel launch on Windows, where MSVC cannot pass the struct by value as a
// __grid_constant__ parameter (C2719). The kernel reads the descriptors from the device pointer
// returned by stage() and makes each tensor map visible to the TMA (tensormap) proxy with a
// fence.proxy.tensormap acquire before its first cp.async.bulk.tensor.
```
L53-60 the staging kernel ends with `fence.proxy.tensormap::generic.release.gpu;` and L67-72
`acquire_staged_tensor_map` is `fence.proxy.tensormap::generic.acquire.gpu [%0], 128;`.

**Dispatch is not gated.** `src/ops/linear/fp8/shapes/n5120_k6144.cu` L32-44 calls
`launch_fp8_a8_tma<Geometry, Tma32x64>` at ≤64 tokens, `Tma64x128` at ≤192, `MidBulk` at ≤768,
`Bulk` above — with **no `_WIN32` anywhere** in any of the five fp8 shape files (scanned L-by-L). The
same is true of the bf16 shapes. That is the concrete difference from our tree, which under `_WIN32`
dispatches the pre-merge `launch_fp8_a8` bodies instead
(`src/ops/linear/fp8/shapes/n5120_k6144.cu:34-45`).

**Read in a commit message.** `d0abd0cbf6112601fcbbe3e1bfc94d6261b67af4`, Ian Ranson, 2026-09-29,
*"build: native Windows build and run"* (the whole port is one commit; branch `upstream-Windows-Port`
points at it):

> The NVFP4 A4, BF16 and FP8 A8 TMA GEMMs take their CUtensorMap descriptors as `__grid_constant__`
> parameters, which MSVC cannot pass (C2719). On Windows an eager launch has a staging kernel store
> its descriptors into a device buffer, and a launch captured into a CUDA Graph reads a device copy
> of its own, written once while capturing and keyed by the descriptor bytes … Staging per replay
> cost about 65 us of each 18-23 ms DFlash2 decode round at 4-8 concurrent requests on an RTX 5090
> (72 FP8 TMA launches per round).

The same commit adds `tests/ops/linear/test_nvfp4_tma_staging_race.cpp` (222 lines) — a regression test
for the staging race, NVFP4 only; there is no FP8-specific staging test.

**Read in the README.** L491-497:

> **Native build and run** with MSVC and CUDA …: static CUDA runtime, FFmpeg/curl from vcpkg, non-RDC
> NVFP4 kernels, **TMA descriptors staged into device memory by a kernel** (launches captured into a
> decode graph read a copy written once, so decode rounds replay no staging kernels) …

and L108: *"Everything ran on an RTX 5090 under Windows with the official Qwen3.8-27B NVFP4 artifact."*
Environment: MSVC + CUDA, RTX 5090, sm_120a.

**Bearing on the hypothesis.** This port issues `cp.async.bulk.tensor.2d` from an FP8 A8 GEMM on
sm_120a under Windows and is not reported as disabled. So `.2d` is not shown to be the failure cause.
What it has that our tree does not is the proxy acquire and a persistent staging buffer.

### 2. `igorls/ninfer` (`workstation`) — runs it, and passes the descriptor **by value**

This is the most interesting result, because it solves C2719 in the *opposite* direction from us: no
pointer at all, and no `_WIN32` in the fp8 or bf16 directory.

**Read in source.** `src/ops/linear/fp8/fp8_a8_tma_mma.cuh` L19-27:

```
// Passed by value as a __grid_constant__ kernel parameter. TMA requires 64-byte descriptor
// alignment. It is stated here because cuda.h keys CUtensorMap's alignas on __cplusplus, which
// MSVC reports as 199711L, and MSVC cannot pass a parameter aligned beyond 64 bytes by value.
struct alignas(64) Fp8TmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
};
// Elsewhere CUtensorMap keeps its own 128-byte alignas, which the struct inherits.
static_assert(alignof(Fp8TmaDescriptors) % 64 == 0);
```

- L69 `cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, …)` — **still rank 2**.
- L84 `cp.async.bulk.tensor.2d…`.
- L108 `const __grid_constant__ Fp8TmaDescriptors descriptors, …` — **by value, unconditionally**.
- `fp8/shapes/n5120_k6144.cu` L32-44 routes ≤64 / ≤192 / ≤768 / above to
  `Tma32x64` / `Tma64x128` / `MidBulk` / `Bulk`, all `launch_fp8_a8_tma`, with **no `_WIN32`** in any
  of the eight fp8 shape files or in `bf16/shapes/`.
- `bf16_a16_tma_mma.cuh` L14-21 makes the same change (`alignas(64)`), keeps **rank 3** and
  `.3d` (L59), by value at L77.
- `nvfp4/nvfp4_a4_tma.cuh` L25-27 `alignas(64)`, `.2d` at L139 — NVFP4 stays rank 2 here too.

**Read in commit messages** (all Igor Lins e Silva):

- `183cdca6ebeb691f9c35dca838999e1fd91feabf`, 2026-09-29, *"build(windows): align the FP8 TMA
  descriptors to 64 bytes for MSVC"*:
  > Upstream 909fb087 added FP8 A8 TMA MMA routes whose Fp8TmaDescriptors is alignas(128) and passed
  > by value as a __grid_constant__ kernel parameter. MSVC rejects a by-value parameter aligned
  > beyond 64 bytes (C2719 in every FP8 Linear, LinearAdd, SwiGLU and attention-input unit), and
  > cuda.h drops CUtensorMap's own alignas under MSVC's __cplusplus. **TMA needs 64-byte descriptor
  > alignment**, so the struct states alignas(64) with the same static_assert as the BF16 and NVFP4
  > descriptors. Elsewhere the struct still inherits CUtensorMap's 128-byte alignment, so Linux code
  > is unchanged.
- `24e04e99375ca8d9d55982551b4704904a50bb3e`, 2026-09-28, same change for BF16.
- `0f25f2798b72754ce27b7554e39ecac5551b63bf`, 2026-09-28, *"accept CUtensorMap's 128-byte alignment in
  TMA descriptor asserts"* — the `== 64` assert broke the Linux build, so it became `% 64 == 0`.

**Reported by the port's author** (`docs/maintainer/upstream-ports.md`, not independently verified by
me) L133-135 and L151:

> Follow-ups: `183cdca6` gives upstream's new FP8 TMA descriptors the fork's 64-byte alignment
> (MSVC error C2719 in every FP8 Linear, LinearAdd, SwiGLU and attention-input unit; Linux code
> unchanged)

> `909fb087` … | Integrated with `183cdca6`: **the production FP8 projections and late MLP**; FP8
> A16/A8 Linear oracle tests …

and the qualification block L163-171:

> **Windows** (MSVC 19.51, CUDA 13.3, `sm_120a`, RTX PRO 6000, driver 616.92, beside the running
> production service): full CTest, 138 tests: 128 passed, 10 skipped … 0 failed … as do every FP8
> and NVFP4 Linear, LinearAdd and LinearSwiGLU test …

The `test_fp8_a8.cpp` invocation list (L109-136 for `[5120, 6144]`) includes widths 65, 129, 193, 257,
513, 769, 1025 — which by that shape file's dispatch land on the TMA schedules — so the A8 oracle test
does reach the TMA route at those widths.

**Bearing.** A Windows sm_120a port compiles *and runs* the rank-2 `.2d` FP8 A8 route with a
**by-value** `__grid_constant__` parameter, by relaxing the alignment from 128 to 64 rather than
pointer-passing. Its speed numbers are Linux/Colab G4 figures, not Windows
(`docs/performance/rtx-pro-6000.md` L142-150 says so explicitly).

### 3. `cometkim/ninfer` (`cometkim/dev`) — runs it, with a generic transport header

**Read in source.** `src/ops/common/tma_descriptors.cuh` (new, 129 lines) is the port-wide mechanism.
L34-38:

```
#ifdef _WIN32
#    define NINFER_TMA_DESCRIPTORS_PARAM(Type) const Type* __restrict__
#else
#    define NINFER_TMA_DESCRIPTORS_PARAM(Type) const __grid_constant__ Type
#endif
```

Its header comment (L3-24) states the design and, importantly, that a kernel-side write to the
descriptor needs the proxy acquire:

> - The block is written through the generic proxy and read by the tensormap proxy. The stream pool
>   hands back recently used addresses, so the TMA-issuing thread acquires every map with
>   `fence.proxy.tensormap::generic.acquire` before its first use.
> - The parameter spelling depends only on _WIN32, never on __CUDA_ARCH__, so the host stub and the
>   device code always agree on the kernel signature.

L44-61 `tma_descriptor_block(const Descriptors*)` emits, once per 128-byte `CUtensorMap` in the block,
`fence.proxy.tensormap::generic.acquire.gpu [%0], 128;`. L90-107 the host side stages with a one-warp
`uint4`-word kernel into a `cudaMallocAsync` block freed on the same stream.

`fp8_a8_tma_mma.cuh` is upstream's with the swap applied: L54 `fp8_tma_map` still encodes **rank 2**
(`CU_TENSOR_MAP_DATA_TYPE_UINT8, 2`), L80 still `cp.async.bulk.tensor.2d`, L102 the signature becomes
`NINFER_TMA_DESCRIPTORS_PARAM(Fp8TmaDescriptors) descriptors`, L291 the launcher wraps the block in a
`TmaDescriptorArgument` and passes `descriptors.get()`. `bf16_a16_tma_mma.cuh` is the same treatment
on the **rank-3** `.3d` route. No `_WIN32` in either file; the fp8 shapes dispatch to
`launch_fp8_a8_tma` with no gate.

`docs/features/windows-port.md` describes only platform IO and says *"The port does not change the
supported GPU architecture or introduce another artifact format."* — it does not mention TMA at all.

**Read in a commit message.** `ef61aa0865`, Hyeseong Kim, 2026-10-01,
*"squash(feat/windows-port): adjacent delta through fe6f477…"*. The README credits natpate:
*"The Windows port and WebUI originate from natpate/ninfer-windows."*

### 4. `natpate/ninfer-windows` — has the C2719 macro, but only on the BF16 file

**Read in source.** `src/ops/linear/bf16/bf16_a16_tma_mma.cuh` L20-27 — this is the closest spelling to
ours anywhere in the fork network:

```
#ifdef _WIN32
// MSVC cannot pass an alignas(128) struct by value as a kernel parameter (C2719). Keep the
// descriptor block in a device buffer and hand the kernel a pointer; the TMA unit reads the
// tensor map from that address. POSIX keeps the by-value __grid_constant__ spelling.
#define NINFER_BF16_TMA_DESCRIPTOR_PARAM const Bf16TmaDescriptors* __restrict__
#else
#define NINFER_BF16_TMA_DESCRIPTOR_PARAM const __grid_constant__ Bf16TmaDescriptors
#endif
```

L142-161 `Bf16TmaDescriptorBlock` allocates with `cudaMallocAsync` and frees with
`cudaFreeAsync(device, nullptr)`; L168-172 `cudaMemcpyAsync` H2D on the consuming stream. **No
`fence.proxy.tensormap` anywhere in the file** — unlike Wallawalla's and cometkim's equivalents.

**The FP8 file is absent on `master`.** `src/ops/linear/fp8/` on `master` lists
`fp8_a8_mma.cuh`, `fp8_a8_plan.h`, `fp8_schedule.cuh` … and **no** `fp8_a8_tma_mma.cuh`; the
directory also lacks `fp8_a8_mma_common.cuh`, i.e. `master` predates upstream `909fb087`. On the `dev`
branch the file exists and is **byte-identical to upstream's** (both SHA-256
`DBDF3CE3E1D7CDCAFAF6EA81A274E2E7F0B744B164531B0248A0F402BD275D53`): `alignas(128)`, rank 2, `.2d`,
by-value `__grid_constant__`, no `_WIN32`. So on `dev` this port would hit C2719 on the FP8 route; it
has not been addressed.

README L65-67 (their own words): *"**MSVC/TMA kernel compatibility** — fixes that let the upstream
Blackwell kernels compile under MSVC: device-pointer NVFP4 TMA descriptors, the pair-row SwiGLU TMA
epilogue, and MSVC move-construction details in the target runtime."* Note the scope: **NVFP4**.

### 5. `MirkoCovizzi/ninfer-rtx5090-mobile` — BF16 TMA only, no Windows

`src/ops/linear/bf16/bf16_a16_tma_mma.cuh` is upstream's verbatim (rank 3, `.3d`, by-value
`__grid_constant__`, no `_WIN32`); no fp8 TMA file; README states 64-bit Linux, RTX 5090. Carries 22
branches, including `hip-gfx1151-port` and several `integration/upstream-master*`; **none** of the 22
carries the FP8 TMA file.

### 6. `devan-carlin/ninfer` — the one place the *cause* is named

Not a fork of the current tree (its `master` predates the TMA routes; the Windows work is on branch
`windows-native-port`, upstream PR `Neroued/ninfer#84`, opened 2026-08-22, still open, not merged).
Its tree has `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh`, which is **rank 2** with
`cp.async.bulk.tensor.2d` (L43, L202) — the same 2d shape as our FP8 route.

**Read in source**, `nvfp4_w4a4_tma.cuh` L220-226, inside the kernel:

```
    // MSVC cannot pass the 128-aligned TMA descriptor by value (C2719), so it is
    // passed as a pointer to a device buffer in global memory. That buffer is
    // written by the host (cudaMemcpyAsync H2D in nvfp4_stage_tma_descriptor), so
    // it is already visible to the TMA unit's tensormap proxy — no in-kernel
    // staging and no tensormap fence are required. (Staging the descriptor into
    // local/shared memory inside the kernel makes it invisible to the TMA unit
    // without a fence.proxy.tensormap, which surfaces as "Illegal instruction".)
```

and L50-76 `nvfp4_stage_tma_descriptor` uses a **persistent `cudaMalloc`'d** buffer (function-local
`static`), 256-byte-aligned per the PR body, `cudaMemcpyAsync` H2D on the given stream.

**Read in the PR body** (`Neroued/ninfer#84`, identical to the earlier closed #82):

> The TMA unit reads tensor maps through a separate tensormap proxy: kernel-side (generic-proxy)
> writes to a descriptor staged in local memory are invisible to it without `fence.proxy.tensormap`,
> which surfaced as `Illegal instruction` at the first `cp.async.bulk.tensor`.

Verification claimed: 1024-token repro, WSL↔Windows byte-identical parity (seed 42),
compute-sanitizer clean, prefill 6378 tok/s, decode 172.5 tok/s, on Windows 11 / MSVC 19.44.35228 /
CUDA 13.3.73 / **RTX 5090 (sm_120a)**.

This is the only primary-source statement anywhere in the corpus that attributes
`cudaErrorIllegalInstruction` in a TMA kernel to descriptor visibility across the tensormap proxy, and
it is a **rank-2 `.2d` route on sm_120a** — which is direct evidence against the 2d-is-broken reading.
Note the scope difference: it is the NVFP4 route, and the fix is "write it from the host, no fence",
whereas Wallawalla and cometkim write it from a kernel and therefore do need the acquire fence.

### 7. The 19 unmodified forks

Their `fp8_a8_tma_mma.cuh` blob is byte-identical to upstream's, so by construction: rank 2, `.2d`,
by-value `__grid_constant__`, no `_WIN32`, no gating at the shape files. They are not Windows builds;
each carries a different feature (TP2, YaRN, a router endpoint, RTX 3070 / sm_86 adaptation, nix
packaging). Nothing in them bears on the question.

---

## Central answer

**Does any port run the FP8 A8 TMA route on Windows or on sm_120a? Yes — three, and a fourth runs a
rank-2 `.2d` TMA route on Windows/sm_120a.**

1. `Wallawalla47/ninfer-custom` `master`: FP8 A8 TMA **rank 2 / `.2d`** dispatched on Windows,
   descriptors pointer-passed from a persistent device staging buffer, with
   `fence.proxy.tensormap::generic.acquire.gpu` per map. Commit `d0abd0cb`, 2026-09-29. Reports runs on
   an RTX 5090 under Windows.
2. `igorls/ninfer` `workstation`: FP8 A8 TMA **rank 2 / `.2d`** dispatched on Windows with the
   descriptor passed **by value** as `__grid_constant__` at `alignas(64)` — no pointer, no staging, no
   `_WIN32` in the file at all. Commits `183cdca6` / `24e04e99` / `0f25f279`, 2026-09-28/29. Reports
   the FP8 A8 route as production on an RTX PRO 6000 Blackwell (sm_120a) under MSVC 19.51.
3. `cometkim/ninfer` `cometkim/dev`: FP8 A8 TMA **rank 2 / `.2d`** dispatched on Windows via a shared
   `ops/common/tma_descriptors.cuh` (`NINFER_TMA_DESCRIPTORS_PARAM` + per-map proxy acquire).
   Commit `ef61aa08`, 2026-10-01.
4. `devan-carlin/ninfer` (upstream PR #84, unmerged): NVFP4 W4A4 **rank 2 / `.2d`** on Windows/sm_120a,
   host-written persistent device buffer, with the causal explanation for the illegal instruction.

**Has any port converted its FP8 TMA descriptor to rank 3? No. Not one.** Every fork that has the file
keeps rank 2 for FP8 and rank 3 for BF16 — the same pairing our tree has, including all three
successful Windows ports. The rank-3 sector-factored shape appears only in the BF16 route, everywhere,
in every repo examined. `Wallawalla47`'s BF16 file is upstream's byte-for-byte apart from the
`_WIN32` transport; `igorls`' keeps rank 3; `cometkim`'s keeps rank 3.

**What the successful ports have that our tree does not**, in the order the sources support:

- a `fence.proxy.tensormap::generic.acquire.gpu [ptr], 128` per 128-byte map before the first copy
  (Wallawalla, cometkim) — ours emits no `fence.proxy.*` at all; `git grep 'fence.proxy' -- src/`
  returns nothing;
- a **persistent** device buffer for the descriptor block rather than a `cudaMallocAsync`/
  `cudaFreeAsync` pair per launch (all four);
- or no device buffer at all, by lowering the alignment to 64 and keeping the by-value parameter
  (igorls) — which is the variant our own `tools/scripts/probe_tma_align.cmd` measurements
  (`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:24-28`) already point at, since that sweep records
  8/16/32/64 as ACCEPTED.

None of those three is the descriptor rank.

## Ranked by how directly they bear on the 2d-versus-3d hypothesis

1. **`devan-carlin/ninfer` PR #84** — a rank-2 `.2d` TMA route running on Windows/sm_120a, with the
   illegal instruction attributed to proxy visibility rather than to the instruction form. Directly
   refutes "`.2d` is broken on sm_120a".
2. **`Wallawalla47/ninfer-custom` `d0abd0cb`** — the same file, the same rank-2 descriptor, the same
   `.2d`, dispatched on Windows and not gated off; only the transport and the proxy acquire differ.
3. **`igorls/ninfer` `183cdca6`** — compiles and runs the rank-2 `.2d` FP8 route on Windows by value at
   `alignas(64)`, with no `_WIN32` gate at all. If rank 2 were the fault, this fork could not ship it.
4. **`cometkim/ninfer` `ef61aa08`** — third independent confirmation of rank-2 `.2d` on Windows.
5. **`natpate/ninfer-windows`** — has the C2719 macro for BF16, none for FP8; `dev` still carries
   upstream's `alignas(128)` FP8 file, i.e. that route is unresolved there. Negative.
6. The 19 byte-identical forks and the 31 repos without the file — negative, listed above.

## What I searched and did not find

- **No port anywhere uses rank 3 for an FP8 A8 descriptor.** Searched: the blob SHA of
  `src/ops/linear/fp8/fp8_a8_tma_mma.cuh` at the default branch of all 293 forks `Neroued/ninfer`
  reports (`repos/Neroued/ninfer/forks`, 3 pages × 100, `sort=newest`), fetched the file for the 22
  that have it, and read `cuTensorMapEncodeTiled`'s rank argument and the `cp.async.bulk.tensor.Nd`
  mnemonic in each. All 22 are rank 2 with `.2d`.
- **No fork modified the BF16 route's rank either.** Same sweep on
  `src/ops/linear/bf16/bf16_a16_tma_mma.cuh`: 34 forks have it, all rank 3 with `.3d`; the three that
  changed it changed only the transport or the alignment.
- **No repo other than upstream's ever touched `fp8_tma_map`'s shape.** 19 of the 22 blobs are
  upstream's `a2470ca827bf911ca6736f1a2a8e47384f020930` byte for byte (verified by SHA-256 of the
  downloaded file against the API blob SHA for `ValerioDolci` and `natpate@dev`).
- **Upstream tracker searches**, all via `gh api search/issues` on `repo:Neroued/ninfer`:
  `C2719` → 2 hits, both the same PR (#84 and its closed duplicate #82);
  `tensormap` → the same 2;
  `IllegalInstruction` → **0**;
  `illegal memory access` → 1 (#208, an NVFP4 + MTP `cudaErrorIllegalAddress` report on RTX 5090, not
  a TMA compile issue);
  `bulk.tensor` → **0**;
  `rank 2 descriptor` → **0**; `tensorRank` → **0**;
  `fp8 tma` → 10, of which the relevant ones are #167 (MichaelDementii, the FP8 A8 TMA route, still
  open, whose body reports prefill +2.4 %/+4.9 % and explicitly says *"Decode is not on this path"*),
  #324 (FP8 TMA pilot for the attn Prefill route), #233 and #59 (Windows MSVC builds);
  `TMA windows` → 9; `TMA descriptor` → 14. **No upstream issue or PR mentions the 2d/3d split, a
  rank conversion, or an illegal instruction in a TMA kernel.**
- **No repo issue or commit message anywhere in the four port repositories mentions
  `Illegal instruction`**: `search/issues q='Illegal instruction in:title,body'` returns
  `total_count = 0` for `igorls/ninfer`, `Wallawalla47/ninfer-custom`, `cometkim/ninfer` and
  `natpate/ninfer-windows`. The only statement of that failure anywhere I found is devan-carlin's
  PR body and its in-source comment.
- **No port documents a Blackwell TMA limitation.** No README, `docs/` page or NOTICE in
  `Wallawalla47`, `igorls`, `cometkim`, `natpate` or `MirkoCovizzi` says TMA, `.2d`, or the 5090 is
  held back or unsupported. Wallawalla's README lists TMA staging as a *feature* it adds.
- **Repos that could not be read**, recorded rather than skipped: `juninron/ninfer-4070`,
  `lyux-gt/ninfer-tp2-5060ti` and `MN-mansour/ninfer-4070Ti-26GB-8g-VRAM-shrink-120` return no
  `default_branch` from `repos/<r>` and 404 from `repos/<r>/branches`; their trees were not examined.
  `oosawak/ninfer3090` and `5p00kyy/ninfer-5060ti` have no `src/ops/linear/fp8` directory at all.
- **Not verified by me:** every claim that a port's route *runs* is the port author's own report —
  Wallawalla's RTX 5090 Windows runs, igorls' 138-test Windows CTest, devan-carlin's byte-identical
  parity. I read the code and the messages; I did not reproduce any of them, and none of the three
  Windows ports publishes a compute-sanitizer or fault-attribution record for the **FP8** route
  specifically (Wallawalla's one regression test is NVFP4-only).
- **Not searched:** non-GitHub mirrors or gists of these repositories; private forks; the
  `headpiece747/ninfer-5090-windows` origin beyond what is already in this tree.

## Method

`gh api` as `headpiece747`. Blob SHAs from `repos/<r>/contents/<path>?ref=<default_branch> --jq .sha`;
file contents with `-H 'Accept: application/vnd.github.raw'`; commit metadata and per-commit patches
via `repos/<r>/commits/<sha>`; PR/issue bodies via `repos/<r>/issues/<n>` and `repos/<r>/pulls/<n>`;
fork enumeration via `repos/Neroued/ninfer/forks?per_page=100&sort=newest` (293 returned of 512
reported — GitHub's fork list is not exhaustive for deleted or unlisted forks). Every quoted line
number is from a file fetched at the branch named; where I quote a commit message I say so, and where
a claim is the port author's own assertion in a document I say that instead of presenting it as
measured.