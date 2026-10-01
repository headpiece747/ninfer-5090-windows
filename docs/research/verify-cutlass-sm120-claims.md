# Adversarial verification of four CUTLASS / sm_120 claims

Scope: read CUTLASS source at `main` (`0b55a2f`, "update 4.8 release", PR #3663) and at tags
v3.8.0, v3.9.0, v4.0.0, v4.4.0, v4.8.0; NVIDIA CUDA C++ Programming Guide v13.4.2 and PTX ISA
v9.4; the third-party repository's actual files. Local clone:
`C:\Users\tobia\AppData\Local\Temp\opencode\cutlass` (`--filter=blob:none`, full history, all
remote branches fetched).

Every figure below is read from source. Where I did not measure, I say so.

## Verdict table

| Claim | Verdict | Core finding |
|---|---|---|
| 1. All 8 SM120 collectives are `*_tma.hpp`, all 7 builders TMA-only, `sm120_mma_builder.inl` hardwires `SM90_TMA_LOAD`; no non-TMA sm_120 mainloop exists and no TMA-vs-non-TMA sm_120 FP8 comparison can be produced from CUTLASS | **VERIFIED** (with two corrections to the wording, neither of which changes the conclusion) | 8 collectives and 7 builder files confirmed exactly. But `sm120_mma_builder.inl` hardwires `detail::sm90_cluster_shape_to_tma_atom(...)`, **not** the literal symbol `SM90_TMA_LOAD`; and one of the 7 files (`sm120_common.inl`) is **not a builder** — it defines no `CollectiveBuilder` specialization. The conclusion (no non-TMA sm_120 mainloop) holds at three independent levels: file naming, builder internals, and dispatch-policy taxonomy. |
| 2. CUTLASS's sm_120 shared-memory cap is 101,376 B (99 KiB) vs sm_100's 232,448 | **VERIFIED** | `include/cutlass/arch/arch.h:45,47`. And the CUTLASS number is not a CUTLASS policy — it is exactly CUDA's hardware per-CTA opt-in maximum for cc 12.0. |
| 3. `gau-nernst/learn-cuda` exercise "02c" measured TMA 4-9% ahead of `cp.async` on RTX 5090, controlled on tile/stages/warps, with 6.9% run-to-run swing | **REFUTED** | No exercise named "02c" exists (135 files enumerated; the sm_120 exercise is `02_matmul_sm120`). The TMA-ahead finding is real but is **BF16 + INT8**, not FP8. The 6.9% swing figure appears nowhere in the repository. The comparison is controlled only for BF16; the INT8 pair differs in `BLOCK_K`. |
| 4. sm_120 supports neither wgmma nor tcgen05, and uses warp-level `mma.sync` extended by sm_120a-only `.kind`/`.block_scale` | **VERIFIED** | CUTLASS's SM120 collectives use `mma.sync.aligned` exclusively — 78 mnemonics, all `mma.sync`. Zero `wgmma` and zero `tcgen05` occurrences in any SM120 file. `.kind::f8f6f4` and `.kind::mxf8f6f4.block_scale` confirmed in the atoms. |

---

## Claim 1 — VERIFIED (two wording corrections)

### The counts are exactly right

`include/cutlass/gemm/collective/` — 8 files matching `sm120`, all with `_tma` in the name:

```
sm120_blockscaled_mma_array_tma.hpp
sm120_blockscaled_mma_tma.hpp
sm120_blockscaled_sparse_mma_tma.hpp
sm120_mma_array_tma.hpp
sm120_mma_array_tma_blockwise_scaling.hpp
sm120_mma_tma.hpp
sm120_mma_tma_blockwise_scaling.hpp
sm120_sparse_mma_tma.hpp
```

`include/cutlass/gemm/collective/builders/` — 7 files matching `sm120`, as claimed.

### Correction 1: `sm120_mma_builder.inl` does not hardwire the literal `SM90_TMA_LOAD`

`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:121-122`:

```cpp
using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));
```

This resolves to a TMA atom, so the *substance* is right, but the claim names the wrong symbol for
this file. `detail::sm90_cluster_shape_to_tma_atom`
(`include/cutlass/gemm/collective/builders/sm90_common.inl:114-125`) returns
`cute::SM90_TMA_LOAD{}` when the cluster extent is 1 and `SM90_TMA_LOAD_MULTICAST{}` otherwise:

```cpp
constexpr auto
sm90_cluster_shape_to_tma_atom(UnimodalClusterShape) {
  if constexpr (cute::size(UnimodalClusterShape{}) == 1) {
    return cute::SM90_TMA_LOAD{};
  }
  else {
    return cute::SM90_TMA_LOAD_MULTICAST{};
  }
}
```

Since `sm120_mma_builder.inl:84` static-asserts
`cute::size(ClusterShape_MNK{}) == Int<1>{}` ("no programmatic multicast on this arch"), the
resolved type is `SM90_TMA_LOAD` in practice. But the claim should say "a TMA atom via
`sm90_cluster_shape_to_tma_atom`", not "hardwires `SM90_TMA_LOAD`".

The literal `SM90_TMA_LOAD` spelling is real — just in *two other* builders:

- `sm120_blockscaled_mma_builder.inl:154-155` — `using GmemTiledCopyA = SM90_TMA_LOAD;`
- `sm120_blockscaled_sparse_mma_builder.inl:231-232` — same

Full table of all six builders' gmem copies (line numbers are the `using` lines):

| Builder | GmemTiledCopyA/B |
|---|---|
| `sm120_mma_builder.inl` | `:121-122` `sm90_cluster_shape_to_tma_atom(...)` |
| `sm120_array_mma_builder.inl` | `:122-123` `sm90_cluster_shape_to_tma_atom(...)` |
| `sm120_blockwise_mma_builder.inl` | `:184-185` `sm90_cluster_shape_to_tma_atom(...)` |
| `sm120_sparse_mma_builder.inl` | `:206-207` `sm90_cluster_shape_to_tma_atom(...)` |
| `sm120_blockscaled_mma_builder.inl` | `:154-155` `SM90_TMA_LOAD` |
| `sm120_blockscaled_sparse_mma_builder.inl` | `:231-232` `SM90_TMA_LOAD` |

Six builders, every one TMA. No exceptions.

### Correction 2: one of the 7 files is not a builder

`sm120_common.inl` contains **zero** `struct CollectiveBuilder` specializations — it is a shared
detail header (the `sm120_rr_smem_copy_selector_A/B` helpers, the
`sm120_smem_capacity_bytes` alias at `:46`). The real builder count is **6**:

```
sm120_array_mma_builder.inl                  : 1 CollectiveBuilder
sm120_blockscaled_mma_builder.inl            : 1
sm120_blockscaled_sparse_mma_builder.inl     : 1
sm120_blockwise_mma_builder.inl              : 1
sm120_mma_builder.inl                        : 1
sm120_sparse_mma_builder.inl                 : 1
sm120_common.inl                             : 0   <-- not a builder
```

"7 builders" is wrong as a count of builders; "7 files" is right. This does not weaken the claim,
since all six real builders are TMA.

### The TMA-only conclusion holds at three further independent levels

**Level 1 — dispatch-policy taxonomy.** All 8 `MainloopSm120*` policies in
`include/cutlass/gemm/dispatch_policy.hpp` carry `Tma` in the name, and all eight declare
`using ArchTag = arch::Sm120`:

```
MainloopSm120TmaWarpSpecialized                        :1565
MainloopSm120ArrayTmaWarpSpecialized                   :1579
MainloopSm120TmaWarpSpecializedBlockScaled             :1598
MainloopSm120ArrayTmaWarpSpecializedBlockScaled        :1613
MainloopSm120TmaWarpSpecializedSparse                  :1634
MainloopSm120TmaWarpSpecializedSparseBlockScaled       :1651
MainloopSm120TmaWarpSpecializedBlockwiseScaling        :1667
MainloopSm120ArrayTmaWarpSpecializedBlockwiseScaling   :1683
```

The contrast with SM100 is decisive. SM100 has three genuinely non-TMA policies:

```
MainloopSm100UmmaCpAsyncWarpSpecialized                    (gemm/dispatch_policy.hpp)
MainloopSm100UmmaMixedTmaCpAsyncWarpSpecialized
MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled
```

and correspondingly ships `sm100_cpasync_umma_builder.inl`, `sm100_simt_builder.inl`,
`sm100_mixed_tma_cpasync_umma_builder.inl`. **SM120 has no counterpart to any of these four
files.** That asymmetry is the substantive content of the claim, and it holds.

**Level 2 — no builder knob exists to select a non-TMA load.** The `CollectiveBuilder` primary
template (`include/cutlass/gemm/collective/collective_builder_decl.hpp:77-95`) takes exactly
13 template parameters — `ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA, ElementB,
GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType,
KernelScheduleType` — plus `Enable`. **There is no `GmemTiledCopy` parameter and no
`EnableGmemTma` flag.** The gmem copy is not a builder input at all; it is computed inside the
builder body. So the specific mechanism the claim asks about does not exist in CUTLASS 4.x's
builder API.

**Level 3 — bypassing the builder does not work either.** A user *can* instantiate
`CollectiveMma<MainloopSm120TmaWarpSpecialized<...>, ..., GmemTiledCopyA_, ...>` directly, since
`GmemTiledCopyA_`/`GmemTiledCopyB_` are template parameters of the collective
(`include/cutlass/gemm/collective/sm120_mma_tma.hpp:56-90`). But the specialization is
structurally TMA-only and cannot be made to use a `cp.async` atom:

- `:118` — `using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;` — the
  pipeline type is not a parameter.
- `:217-229` — `Params::TMA_A`/`TMA_B` are built by `make_tma_copy(GmemTiledCopyA{}, ...)` and
  held as `tma_load_a`/`tma_load_b`. A non-TMA `GmemTiledCopyA_` makes `make_tma_copy` ill-formed.
- `:364-365` — `mainloop_params.tma_load_a.get_slice(...)`, then `partition_S`/`partition_D` into
  `(TMA, TMA_M, TMA_K, k)` shaped tensors.
- `:408-413` — the load loop calls `pipeline.producer_get_barrier(...)` to fetch a TMA mbarrier and
  issues `copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), ...)`. There is no
  branch for a non-TMA producer.

Passing a `cp.async` tiled copy would be a compile error inside `make_tma_copy`, not a working
non-TMA mainloop. **So the "cannot be produced" half of the claim is correct.**

### Exhaustive negative search

Searched the whole `include/` tree for every escape hatch the claim asks about:

| Searched for | Result |
|---|---|
| `SM90_TMA_LOAD` in any `sm120*` file | present only as the resolved atom and in the 8 collectives' own bodies — never as an *alternative* |
| `SM80_CP_ASYNC` / `cp_async` in `gemm/collective/sm120_*.hpp` | 2 hits, both **scale-factor side-loads only** (see below) |
| `Cpasync`/`Simt`/`SIMT` in any `sm120` builder or collective | none |
| `wgmma`, `tcgen05` in any sm120 file | 0 occurrences |
| `Sm120`/`sm120` referenced from any `sm90_*.hpp` or `sm90_*.inl` | 0 occurrences — no SM90 collective is enabled for Sm120 |
| any `sm120_*` file outside the `_tma` set | none in `gemm/` |
| `gemm/kernel/` sm120 files | exactly 1: `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` |

**One nuance worth recording, which does not refute the claim.** Two blockwise-scaling collectives
*do* contain a real `cp.async` global→shared copy — but only for the scale-factor tensors, not for
A or B. `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:174-175`:

```cpp
using SmemBlockScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementSF>, ElementSF>;
using SmemBlockScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementSF>, ElementSF>;
```

used at `:456-459` and issued at `:520-522`, with
`pipeline.producer_commit(..., cutlass::arch::cpasync_barrier_arrive_noinc)` because `cp.async`
does not self-signal the barrier the way TMA does. Same at
`sm120_mma_array_tma_blockwise_scaling.hpp:180-181,590`. The **A/B operand loads remain TMA** in
both files (`:515-516`). So "no non-TMA sm_120 mainloop" is accurate; the precise statement is
"no non-TMA *operand* load — the mainloop's A and B path is TMA in every SM120 collective."

### Other branches do not differ

Checked every remote branch with SM120-relevant content. All are TMA-only:

| Branch | sm120 files in `gemm/collective/` | builder files |
|---|---|---|
| `origin/main` (= `release/4.8`) | 8 collectives + 7 builders | 7 |
| `origin/v4` | 7 collectives + 6 builders | 6 |
| `origin/4.6_update` | 8 + 7 | 7 |
| `origin/feature/enable-mxfp-group-gemm-sm120` | 7 + 6 | 6 |

`release/4.8` and `main` are identical here. The feature branch adds no non-TMA path.

Historical note, and a limit on the "v3.8.0 onward" framing: **v3.8.0 has zero SM120 GEMM
files**; `sm120_smem_capacity_bytes` and `struct Sm120` do not exist in its `arch.h`. SM120 GEMM
support first appears at **v3.9.0** with 5 collectives + 5 builders, all TMA-only
(`v3.9.0:include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` already hardwires
`sm90_cluster_shape_to_tma_atom`). So the claim is true of every version that has SM120 support at
all, but v3.8.0 is a version with no SM120 mainloop to be TMA-only about.

### Corrected statement for Claim 1

> All 8 SM120 collectives in CUTLASS `gemm/collective/` are `*_tma.hpp`, and all 6 SM120
> `CollectiveBuilder` specializations (across 7 `sm120_*.inl` files, one of which is a shared
> detail header) hardwire a TMA gmem copy — four via `sm90_cluster_shape_to_tma_atom(...)` and two
> via the literal `SM90_TMA_LOAD`. Every `MainloopSm120*` dispatch policy is `Tma`-named and
> declares `ArchTag = arch::Sm120`; SM100's three `CpAsync` policies have no SM120 counterpart.
> The `CollectiveBuilder` API has no `GmemTiledCopy` parameter, and instantiating `CollectiveMma`
> directly with a non-TMA copy fails in `make_tma_copy` because `Params::TMA_A` is hardwired.
> Therefore CUTLASS ships no non-TMA sm_120 operand mainloop, and no TMA-vs-non-TMA sm_120 FP8
> comparison can be produced from CUTLASS's SM120 collectives.

---

## Claim 2 — VERIFIED

### The CUTLASS constants

`include/cutlass/arch/arch.h:45-47`:

```cpp
constexpr int sm100_smem_capacity_bytes = 232448;
constexpr int sm107_smem_capacity_bytes = 334848;
constexpr int sm120_smem_capacity_bytes = 101376;
```

Wired into the arch tags at `include/cutlass/arch/arch.h:106-121`:

```cpp
struct Sm100 {
  static int const kMinComputeCapability = 100;
  static int const kSharedMemoryCapacityBytes = sm100_smem_capacity_bytes;
  static int const kTmemCapacityColumns = sm100_tmem_capacity_columns;
};
...
struct Sm120 {
  static int const kMinComputeCapability = 120;
  static int const kSharedMemoryCapacityBytes = sm120_smem_capacity_bytes;
};
```

101376 = 99 × 1024 exactly. 232448 = 227 × 1024 exactly. The claim's "99 KiB" is right.

This is the constant the SM120 builders actually consume:
`include/cutlass/gemm/collective/builders/sm120_common.inl:46`

```cpp
constexpr int sm120_smem_capacity_bytes = cutlass::arch::sm120_smem_capacity_bytes;
```

fed to stage-count computation at `sm120_mma_builder.inl:130-132` via
`detail::sm100_compute_stage_count_or_override<detail::sm120_smem_capacity_bytes, ...>`.

### Which number is CUTLASS's policy and which is the hardware's

**Neither is a CUTLASS policy. 101,376 is exactly the hardware limit, and CUTLASS has adopted it
verbatim.** CUDA C++ Programming Guide v13.4.2, §5.1.3, Table 31 "Memory Information per Compute
Capability", column `12.x`:

| Row | 8.9 | 9.0 | 10.0 | 10.7 | 11.0 | 12.x |
|---|---|---|---|---|---|---|
| Maximum amount of shared memory per SM | 100 KB | 228 KB | 228 KB | 328 KB | 228 KB | **100 KB** |
| Maximum amount of shared memory per thread block | 99 KB | 227 KB | 227 KB | 327 KB | 227 KB | **99 KB** |

99 KB = 99 × 1024 = **101,376 bytes** — CUTLASS's constant, to the byte. And 227 KB =
227 × 1024 = **232,448 bytes**, also to the byte. So on both sides CUTLASS is reporting the
hardware's per-CTA ceiling, not a self-imposed discount.

The semantics of the runtime attribute are documented locally at
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\driver_types.h:2667`:

```c
size_t sharedMemPerBlockOptin; /**< Per device maximum shared memory per block usable by special opt in */
```

That is the field `cudaDeviceProp` exposes; it is the same 99 KB figure on sm_120. Note the guide's
footnote [3] to Table 31: anything over 48 KB per block requires dynamic shared memory and an
explicit opt-in — which is why this is the *opt-in* maximum rather than the static-allocation
maximum (the guide's Table 32 lists 12.x SMEM capacity sizes as `0, 8, 16, 32, 64, 100` KB).

One caveat on the Blackwell Tuning Guide, which is *not* consistent with the Programming Guide and
should not be cited for this: it states "For devices of compute capability 12.0, shared memory
capacity per SM is 128KB" while also giving 99 KB per thread block. The Programming Guide's Table
31 (100 KB per SM, 99 KB per block) is the authoritative table for the per-CTA limit, and
101,376 matches the per-CTA row. **The per-SM figure differs between the two documents; the
per-CTA figure does not.** Since CUTLASS's constant is the per-CTA one, the claim is unaffected.

Stable across tags: `sm120_smem_capacity_bytes = 101376;` is identical at v4.0.0, v4.4.0, v4.8.0,
and `main`.

---

## Claim 3 — REFUTED

The repository is real; the cited exercise, the FP8 attribution, and the 6.9% figure are not.

### What exists

`gau-nernst/learn-cuda` exists — 490 stars, last push 2026-09-25, default branch `main` (GitHub API
`/repos/gau-nernst/learn-cuda`). Hardware is an **RTX 5090**: `02_matmul_sm120/main.py:26-29`

```python
SOL_LOOKUP = {
    "NVIDIA GeForce RTX 5090": dict(bf16=209.5, int8=838),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": dict(bf16=503.8, int8=1007.6),
}
```

and the README header, "**5090 @ 400W**: Max 209.5 BF16 TFLOPS, 838 INT8 TFLOPS."

### There is no exercise "02c"

Full recursive tree enumeration returns 135 entries, 110 of them blobs. There is **no path
matching `02c`**, and no file whose content matches `02c`. The SM120 exercise is
`02_matmul_sm120/` (`matmul_v0.cu`, `matmul_v1.cu`, `matmul_v2.cu`, `matmul_v3.cu`,
`main.py`, `common.h`, `README.md`, `matmul.cpp`). The README's own index lists it as
"02. Matrix multiplication SM120" with no letter suffix. Other `02_*` directories are
`02_matmul_simt`, `02_matmul_sm80`, `02_matmul_sm100`, `02_matmul_cdna3`.

The claim appears to have been constructed from a plausible-looking but non-existent identifier.
**"02c" is not a real citation.**

### The TMA-ahead finding is real — but it is BF16 and INT8, not FP8

`02_matmul_sm120/README.md`, verbatim table:

```
BF16

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.11) | 164.42 (78.48%) | 172.61 (82.39%) | 166.78 (79.61%)
Inductor Triton (PyTorch 2.11) | 141.45 (67.52%) | 174.40 (83.25%) | 184.88 (88.25%)
v0 (`cp.async`)                | 158.24 (75.53%) | 164.01 (78.29%) | 192.29 (91.78%)
v1 (TMA)                       | 164.27 (78.41%) | 171.24 (81.74%) | 200.29 (95.60%)
v2 (warp specialization)       | 164.41 (78.48%) | 173.07 (82.61%) | 201.03 (95.95%)

INT8

Kernel name                    | 2048            | 4096            | 8192
-------------------------------|-----------------|-----------------|----------------
CuBLAS 13.0 (via PyTorch 2.11) | 396.99 (47.37%) | 423.61 (50.55%) | 424.18 (50.62%)
Inductor Triton (PyTorch 2.11) | 423.78 (50.57%) | 447.90 (53.45%) | 474.87 (56.67%)
v0 (`cp.async`)                | 401.87 (47.96%) | 425.00 (50.72%) | 448.42 (53.51%)
v1 (TMA)                       | 440.57 (52.57%) | 463.30 (55.29%) | 485.84 (57.98%)
v2 (warp specialization)       | 439.72 (52.47%) | 464.89 (55.48%) | 485.83 (57.97%)
```

v1 (TMA) is ahead of v0 (`cp.async`) in all six cells. The magnitude is 3.8-4.4% for BF16 and
8.3-9.6% for INT8 — so "4-9%" is a reasonable summary of the BF16/INT8 range, though the BF16
figure at 2048 is 3.81%, slightly below 4%.

**But neither v0 nor v1 supports FP8 at all.** Both kernels are dispatched on `nv_bfloat16` and
`int8_t` only (`matmul_v0.cu:178,192`; `matmul_v1.cu:180,199`), and the epilogue has exactly two
`if constexpr` branches, `nv_bfloat16` and `int8_t` (`matmul_v0.cu:157,164`;
`matmul_v1.cu:159,166`). `main.py:29-35` likewise generates only `bf16` and `int8` inputs. There
is no FP8 measurement anywhere in `02_matmul_sm120`.

The FP8-ish content in the repo is elsewhere and is **not** a TMA-vs-`cp.async` comparison:
`09_block_scaled_mm_sm120/` (MXFP8/NVFP4) has no results table in its README at all — its entire
README is two lines — and its newest kernel `mxfp8_mm_v3.cu:97-100,145-152` is itself
`cp.async`-based. So no FP8 TMA-vs-non-TMA comparison exists in this repository.

### The comparison is controlled only for BF16

The claim asserts "same tile, stages and warps, only the load function differing." For BF16, that
holds exactly:

| | `matmul_v0.cu` (`cp.async`) | `matmul_v1.cu` (TMA) |
|---|---|---|
| `BLOCK_M` | 256 (`:174`) | 256 (`:176`) |
| `BLOCK_N` | 128 (`:174`) | 128 (`:176`) |
| `BLOCK_K` | 64 (`:174`) | 64 (`:176`) |
| `NUM_WARP_M` × `NUM_WARP_N` | 4 × 2 (`:175`) | 4 × 2 (`:177`) |
| `NUM_STAGES` | 2 (`:176`) | 2 (`:178`) |

Identical. The two kernels also share the same MMA and the same `ldmatrix`/swizzle addressing
(`v0:106-127` vs `v1:92-111`), differing only in the `load_AB` lambda — `cp.async.cg.shared.global`
at `v0:19` versus `tma_2d_g2s` with a `CUtensorMap` at `v1:86-87`. That is a clean control for BF16.

**For INT8 the control breaks: `BLOCK_K` differs.** `matmul_v0.cu:188` uses `BLOCK_K = 64` while
`matmul_v1.cu:195` uses `BLOCK_K = 128`. So the 8-10% INT8 advantage is *not* attributable to TMA
alone — the tile depth changed too. Any use of the INT8 numbers as a TMA effect is unsound.

Also note the v1 INT8 instantiation is what makes the BF16/INT8 comparison asymmetric: v0 has no
`BLOCK_K=128` INT8 configuration at all.

### The 6.9% run-to-run swing is not in the repository

I searched every `.md`, `.cu`, `.cpp`, `.py`, and `.h` file in the repo for `6.9`, `variance`,
`run-to-run`, `stdev`, `stddev`, and `repeat`. Findings:

- `6.9` matches only unrelated numbers (`423.61`, `168.09`, `69.08`).
- `variance` matches only code comments about loop invariance
  (`matmul_v0.cu:133,146`, `matmul_sm100/matmul_v0.cu:140,153`).
- No variance figure of any kind appears in `02_matmul_sm120/README.md`.

The benchmark *does* have a noise metric available — `main.py:150,155` formats a `Noise` column
from nvbench — but the published README table **omits the Noise column entirely**. It is not
reported, so the 6.9% cannot have been read off this source.

Related honesty caveat the README itself raises: "Some combinations can be better if we
autotune" and "For CuBLAS, the result is very different at 600W" — i.e. the author flags
thermal/power sensitivity, which is the same class of drift a swing figure would quantify but
does not quantify here.

### Character of the source

The code is real, runnable, and reproducible (it is a proper nvbench harness with correctness
checks against an FP32/INT reference at `main.py:105-113`), and the BF16 TMA-ahead result is
legitimately obtainable from it. But the **published** form is a two-table README with single
point estimates, no error bars, no Noise column, no commit-pinned numbers, and no record of tile
or stage configuration next to the figures. Per the claim's own framing, the "4-9% ahead" and
"6.9% swing" numbers are blog-style assertions about a code base, and one of the two is
fabricated.

### Corrected statement for Claim 3

> `gau-nernst/learn-cuda` exists and its `02_matmul_sm120` exercise benchmarks an RTX 5090
> (209.5 BF16 / 838 INT8 TFLOPS peak). Its `v1` (TMA) kernel beats its `v0` (`cp.async`) kernel in
> all six published cells: 3.8-4.4% for BF16 and 8.3-9.6% for INT8. The BF16 pair is a genuine
> controlled comparison — same 256×128×64 tile, same 4×2 warp grid, same 2 stages, same MMA and
> `ldmatrix` addressing, differing only in the global→shared load. **The comparison is BF16 and
> INT8; there is no FP8 measurement in that exercise at all.** The INT8 pair is not controlled:
> `BLOCK_K` is 64 in `v0` and 128 in `v1`. No exercise named "02c" exists, and the repository
> reports no run-to-run variance — the README publishes single point estimates with the nvbench
> `Noise` column omitted.

---

## Claim 4 — VERIFIED

### wgmma: absent from every SM120 path

`wgmma` occurs 0 times across `gemm/collective/sm120_*.hpp`,
`gemm/collective/builders/sm120_*.inl`, and `cute/arch/mma_sm120*.hpp`.

Every SM120 collective additionally *asserts against* GMMA descriptor operands.
`include/cutlass/gemm/collective/sm120_mma_tma.hpp:153-154`:

```cpp
static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
              not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
```

The same static_assert appears in all 8 SM120 collectives. CUTLASS is explicitly refusing the
wgmma/GMMA-descriptor operand path on this arch.

### tcgen05: absent from every SM120 path

`tcgen05` occurs 0 times in the same file set. Consistent with this, `cute/arch/config.hpp:179-186`
gates the TensorCore-5th-gen macros behind SM100-family tags only:

```cpp
#if defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) ||\
    defined(CUTLASS_ARCH_MMA_SM107F_ENABLED)
#  define CUTE_ARCH_LDSM_SM100A_ENABLED
#  define CUTE_ARCH_STSM_SM100A_ENABLED
#  define CUTE_ARCH_TCGEN05_TMEM_ENABLED
#  define CUTE_ARCH_TMA_SM100_ENABLED
#  define CUTE_ARCH_FLOAT2_MATH_ENABLED
#endif
```

Note also that CUTLASS's Sm120 arch tag declares **no** `kTmemCapacityColumns`
(`arch.h:118-121`), unlike `Sm100`/`Sm101`/`Sm103` — sm_120 has no Tensor Memory.

### The instruction family actually used: `mma.sync`, exclusively

`include/cute/arch/mma_sm120.hpp` contains 78 distinct `mma.sync.aligned` mnemonics and nothing
else in the MMA family. The selector every SM120 builder routes through is
`cute::rr_op_selector_sm120` (`mma_sm120.hpp:3251-3256`):

```cpp
CUTE_HOST_DEVICE constexpr
auto
rr_op_selector_sm120()
{
  return SM120_16x8x32_TN<ElementA, ElementB, ElementC>{};
}
```

whose `fma` body emits inline PTX (`mma_sm120.hpp:68-78`, the FP4 × FP4 case):

```cpp
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      ...
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
```

The warp-level shape is `m16n8k32` — a warp-synchronous instruction, not a warpgroup- or
CTA-synchronous one.

### `.kind` and `.block_scale` are real and are the sm_120a extension

Two distinct forms are present, matching the claim's two-part description:

**`.kind::f8f6f4`** (mixed FP8/FP6/FP4, no block scaling) — 78 combinations at `m16n8k32`,
`mma_sm120.hpp:70` onward:

```
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
```

**`.kind::mxf8f6f4.block_scale.scale_vec::1X`** (MX block scaling) —
`mma_sm120.hpp:1809` onward:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0
```

The blockscaled selector is `cute::rr_blockscaled_op_selector_sm120` (`mma_sm120.hpp:3266-3276`),
returning `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS` or `SM120_16x8x64_TN_VS`.

These are gated on architecture-specific compilation. `include/cute/arch/config.hpp:165-177`:

```cpp
#if (defined(CUTLASS_ARCH_MMA_SM120_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM121_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED)\
    )
#  if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#    define CUTE_ARCH_F8F6F4_MMA_ENABLED
#    define CUTE_ARCH_MXF8F6F4_MMA_ENABLED
...
```

— i.e. only under an `SM120A`-class macro, corroborating the "sm_120a-only" part of the claim
from CUTLASS's side. CUDA C++ Programming Guide v13.4.2 §5.1.2.3 states the same rule: the
architecture-specific target "allows the use of the complete set of architecture-specific features
in Compute Capability 10.0 devices... will only be compatible with devices of Compute Capability
10.0 and no others," and gives `compute_120a` as the sm_120 example. PTX ISA v9.4 §9.7.16.3 is the
section titled "Block Scaling for mma.sync", and §9.7.17 (wgmma) and §9.7.18 (tcgen05 / TensorCore
5th Generation) are the separate instruction families the claim says sm_120 does not use.

### Corrected statement for Claim 4

CUTLASS's SM120 mainloops use warp-level `mma.sync.aligned` exclusively — 78 distinct mnemonics in
`include/cute/arch/mma_sm120.hpp`, all `m16n8k32` or `m16n8k64`, with zero `wgmma` and zero
`tcgen05` occurrences in any SM120 file. All 8 SM120 collectives static_assert that
`TiledMma::FrgTypeA/B` are *not* `cute::GMMA::DescriptorIterator`, actively excluding the wgmma
operand path, and `arch::Sm120` declares no `kTmemCapacityColumns`, so there is no Tensor Memory.
The sm_120a-only extensions are `.kind::f8f6f4` (78 combinations) and
`.kind::mxf8f6f4.block_scale.scale_vec::1X`, gated by `CUTE_ARCH_F8F6F4_MMA_ENABLED` /
`CUTE_ARCH_MXF8F6F4_MMA_ENABLED`, which `cute/arch/config.hpp:165-177` defines only under
`CUTLASS_ARCH_MMA_SM120A_ENABLED` (or the 121 variants).

---

## What I could not check

- **No compilation or execution was performed.** Every claim here is a reading of source text, not
  a build or a run. The Claim 1 conclusion that substituting a `cp.async` tiled copy "would be a
  compile error inside `make_tma_copy`" is an inference from the hardwired `make_tma_copy` call at
  `sm120_mma_tma.hpp:217-229`, not an observed compiler diagnostic.
- **Claim 3's benchmark numbers were not re-run.** They are quoted from the repository's README;
  the GPU here is the same class of part (RTX 5090, sm_120) but I did not reproduce the
  measurement, and no independent run-to-run variance exists in the source to check against.
- **GitHub commit history for `gau-nernst/learn-cuda`** was not retrieved — the unauthenticated API
  rate-limited. The 6.9% figure's absence is established from the full current-tree content
  search across all 110 files, which is sufficient to show it is not in the repository as it
  stands; it cannot be excluded from some deleted or unmerged commit.
