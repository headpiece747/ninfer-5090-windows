# sm_120 TMA vs non-TMA FP8 GEMM throughput: evidence

Research date **2026-10-01**. Scope: is there published measured evidence that a TMA FP8/FP4 GEMM is
faster or slower than a non-TMA mainloop on consumer Blackwell (sm_120/sm_120a), what published tile
shapes and bandwidths exist, whether the sm_120 shared-memory budget plausibly explains a large gap,
whether anything documented makes consumer TMA slower, and what a controlled transport-only
comparison would require.

Primary sources: CUTLASS, vLLM, TensorRT-LLM and FlashInfer source and first-party issue/PR trackers;
NVIDIA PTX ISA and CuTe headers. Third-party measurements are labelled as such and are never presented
as NVIDIA's.

Every claim below carries one of: **[read in source]**, **[published by NVIDIA]**, **[reported by a
third party]**, **[measured by the project]**, **[computed from the project's own source]**, **[inferred]**,
**[not found]**. Nothing here recommends an action for the port; §5 states what a controlled comparison
would require and stops there.

Machine the project figures describe: RTX 5090, `compute_cap = 12.0`, CUDA 13.3, WDDM, 170 SMs,
`sm_120a` build target. The benchmark in question is N=5120, K=6144, FP8 A8, weight matrix
31,457,280 B (30.00 MiB), effective GB/s computed by the project's own bench as
`model_bytes / seconds / 1e9` with `kRtx5090DramGBs = 1792.0` and `kRtx5090SustainedReadGBs = 1674.5`
(`bench/ops/linear_bench.cu:43-44,610-612`) **[read in source]**.

---

## Verdict table

| # | Question | Answer | Confidence | Source |
|---|---|---|---|---|
| 1 | Are published TMA FP8/FP4 GEMMs actually faster than non-TMA on sm_120? | **No NVIDIA/CUTLASS-published comparison exists, and none can: CUTLASS ships no non-TMA sm_120 mainloop.** All 8 SM120 collectives in `include/cutlass/gemm/collective/` are `*_tma.hpp`; all 7 SM120 builders are TMA-only; `sm120_mma_builder.inl` hardwires `GmemTiledCopyA/B = SM90_TMA_LOAD`. The nearest controlled measurement (third party, BF16/INT8, not FP8, M=N=K≥2048) puts TMA **4-9 % ahead** of `cp.async` at identical tile/stages/warps — and the BF16 margin is inside that source's own run-to-run spread. **A ~10× TMA deficit is not reproduced by any published sm_120 measurement.** | **High** for "no CUTLASS comparison exists"; **Medium** for the 4-9 % magnitude | §1.1-1.3 |
| 2 | Do small token tiles explain ~100 GB/s? | **No, for the T≤64 band — on the project's own schedule arithmetic.** At T≤64 both arms use a **32-token** tile; the TMA arm's row tile is *larger* (64 vs 32), stages are equal (3), consumer threads are equal (64) and threads/SM are equal (192). The TMA arm is 2× slower in CTA count, not smaller. A thin-M GEMM being weight-bandwidth-bound is a correct general statement, but it cannot explain a 10× gap between two arms at the same M that differ in the *opposite* direction on tile width. **Published sm_120 FP8 throughput-vs-M tables at M≤256: [not found].** | **High** on the arithmetic; **High** that the published small-M FP8 table does not exist | §2.1-2.3 |
| 3 | Does the TMA route cost occupancy here? | **Materially, and asymmetrically — and the sm_120 smem budget is the binding constraint, confirmed by NVIDIA's own tracker.** sm_120 cap is **101,376 B (99 KiB)** vs sm_90/sm_100's 232,448 B. Computed from the project's own templates: the TMA arms sit at **1 CTA/SM** for the three widest tiles (73.8 KB, 98.4 KB, 98.3 KB) while the MMA arms sit at 2-3 CTA/SM. At T≥129 the two arms use the **same 64×128×128 tile** and differ in stage count (3 vs 2), which alone moves smem 73,776→49,152 B and occupancy 288→512 threads/SM. TensorRT-LLM PR #12141 is an NVIDIA-authored instance of SM100 tiles silently overflowing the 99 KiB SM12x budget. | **High** on smem/occupancy arithmetic; **share of the gap: [not found]** | §3.1-3.3 |
| 4 | Documented reason TMA is slower on consumer Blackwell? | **Three documented architectural differences, none of which is a per-byte TMA throughput reduction.** (a) TMA **multicast is absent**, cluster forced 1×1×1 — CUTLASS `static_assert(..., "no programmatic multicast on this arch")` plus the same note in examples 79a-79d/80a. (b) The sm_120 TMA encoding is `.shared::cta`, not `.shared::cluster` — CuTe `copy_sm90_tma.hpp` under `CUTE_ARCH_TMA_SM120_ENABLED`. (c) 99 KiB smem vs 228/232 KiB, which caps stage count. Plus one **third-party measured** case on sm_121a FP8 where adding TMA *regressed* 20→18 TFLOPS until warp specialization gave it something to overlap (18→30). **An NVIDIA statement that consumer TMA has lower throughput than datacenter TMA: [not found].** | **Medium-High** on the carve-outs; **Low** on any per-byte TMA throughput claim | §4.1-4.3 |
| 5 | Right way to A/B transport alone? | **Yes — and CUTLASS cannot do it on sm_120.** The controlled design already exists in the literature as a 3-kernel pattern (identical template params, only the global→shared load function changes). CUTLASS offers **no** toggle: the SM120 builder has no non-TMA counterpart, no non-TMA schedule tag, and no non-TMA `GmemTiledCopy` path. A transport-only A/B on sm_120 must therefore be built as one template with two load paths, not selected from a CUTLASS parameter. | **High** | §5 |

---

## 1. Question 1 — are published TMA FP8/FP4 GEMMs faster than non-TMA on sm_120?

### 1.1 The decisive structural fact: CUTLASS has no non-TMA sm_120 mainloop

`include/cutlass/gemm/collective/` at `main` contains exactly eight SM120 files **[read in source]**:

```
sm120_blockscaled_mma_array_tma.hpp   sm120_blockscaled_sparse_mma_tma.hpp
sm120_blockscaled_mma_tma.hpp         sm120_mma_array_tma.hpp
sm120_blockscaled_mma_array_tma...    sm120_mma_array_tma_blockwise_scaling.hpp
sm120_mma_tma.hpp                    sm120_mma_tma_blockwise_scaling.hpp
sm120_sparse_mma_tma.hpp
```

Every one is `_tma`. There is no `sm120_mma_cpasync.hpp` or equivalent, and
`include/cutlass/gemm/collective/builders/` likewise has seven SM120 builders, all feeding TMA
collectives **[read in source]**.

The dense builder makes it explicit. `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl`
**[read in source]**:

```cpp
using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));
...
using DispatchPolicy = MainloopSm120TmaWarpSpecialized<PipelineStages,
    SchedulerPipelineStageCount, ClusterShape_MNK, KernelSchedule>;
```

The accepted `BuilderScheduleTag`s are `KernelScheduleSm120DenseGemm`, `KernelTmaWarpSpecializedPingpong`,
`KernelTmaWarpSpecializedCooperative`, `KernelScheduleAuto` — every one of which resolves to
`MainloopSm120TmaWarpSpecialized`. There is no cp.async variant to select.

**Consequence:** no CUTLASS-published TMA-vs-non-TMA sm_120 FP8 number exists, and none can be produced
from CUTLASS. The comparison the project wants is not a CUTLASS configuration question.

### 1.2 What the production engines actually ship on sm_120 — all TMA

vLLM's sm_120 FP8 W8A8 path was added by NVIDIA in vllm-project/vllm PR #17280, adding
`scaled_mm_sm120_fp8.cu` and `scaled_mm_sm120_fp8_dispatch.cuh` **[read in a first-party PR file list]**.
The config, quoted verbatim in the `vllm.cpp` 1:1 lift of vLLM at commit `e24d1b24`
**[read in source, third-party repository quoting vLLM]**:

| M | Schedule | Tile |
|---|---|---|
| `M > 256` | `KernelScheduleAuto` / `EpilogueScheduleAuto` | **128×128×128** |
| `M ≤ 256` | `KernelTmaWarpSpecializedPingpong` | **64×64×128** |

carrying CUTLASS's own comment *"SM120 Cooperative kernel requires Tile M >= 128; for smaller tiles use
Pingpong"*. The dispatch is a pure function of M, and **every** branch is TMA.

Two things follow that matter here:

- The smallest FP8 TMA tile any production sm_120 path selects is **64×64×128**. The project's slow
  band uses **32×64×128** **[computed from §2.1]** — half the token extent of anything vLLM ships.
- At M≤256, i.e. exactly the project's slow regime, the production engine is still on TMA. There is no
  published evidence that production abandoned TMA at small M.

TensorRT-LLM is the same story: on SM12x its FP8/NVFP4 GEMM and MoE paths are
`cutlass::arch::Sm120` CUTLASS TMA collectives (`MainloopSm120ArrayTmaWarpSpecializedBlockScaled`,
"cutlass TMA WS grouped gemm") **[read in first-party PR text]**. Its SM120 block-scaled builder exposes
a **`Stages` template parameter, default 4**, added explicitly "for flexible pipeline depth tuning" and
"better SM120 occupancy" **[read in first-party PR text]** — i.e. NVIDIA's own sm_120 tuning axis is
stage count, which is the smem-limited axis.

### 1.3 The nearest controlled measurement: TMA beats cp.async by 4-9 %, not 10×

`gau-nernst/learn-cuda`, module `02c_matmul_sm120`, is the only controlled sm_120 transport
comparison found. Three kernels, **identical** `BLOCK_M/N/K`, `NUM_WARP_M/N` and `NUM_STAGES`; only the
global→shared path differs **[reported by a third party, via DeepWiki's index of the repo]**:

- `matmul_v0` — `cp.async.cg`, all warps load and compute
- `matmul_v1` — `cp.async.bulk.tensor.2d` via `CUtensorMap` + `mbarrier`, one elected thread
- `matmul_v2` — same TMA, plus a dedicated TMA warp (`(WM*WN+1)*32` threads)

RTX 5090, CUDA 13.0, PyTorch 2.11, `-gencode=arch=compute_120a,code=sm_120a`. Tile **128×64×64**,
warps 2×2, **stages 2** for BF16 — identical across v0/v1/v2.

**BF16** (M=N=K):

| Kernel | 2048 | 4096 | 8192 |
|---|---|---|---|
| cuBLAS 13.0 | 164.42 (78.5 %) | 172.61 (82.4 %) | 166.78 (79.6 %) |
| Triton | 141.45 (67.5 %) | 174.40 (83.3 %) | 184.88 (88.3 %) |
| **v0 `cp.async`** | **158.24 (75.5 %)** | **164.01 (78.3 %)** | **192.29 (91.8 %)** |
| **v1 TMA** | **164.27 (78.4 %)** | **171.24 (81.7 %)** | **200.29 (95.6 %)** |
| v2 TMA + warp spec | 164.41 (78.5 %) | 173.07 (82.6 %) | 201.03 (96.0 %) |

**INT8** (tile 128×128×64, stages 3; v2 uses BLOCK_K=128):

| Kernel | 2048 | 4096 | 8192 |
|---|---|---|---|
| cuBLAS 13.0 | 396.99 (47.4 %) | 423.61 (50.6 %) | 424.18 (50.6 %) |
| **v0 `cp.async`** | **401.87 (48.0 %)** | **425.00 (50.7 %)** | **448.42 (53.5 %)** |
| **v1 TMA** | **440.57 (52.6 %)** | **463.30 (55.3 %)** | **485.84 (58.0 %)** |
| v2 TMA + warp spec | 439.72 (52.5 %) | 464.89 (55.5 %) | 485.83 (58.0 %) |

Three readings, in order of how much they should change a belief:

1. **TMA's advantage over `cp.async` at matched tile is single-digit percent, never 10×.** +3.8/4.4/4.2 %
   BF16, +9.6/9.0/8.3 % INT8. **[reported by a third party]**
2. **Warp specialization on top of TMA adds essentially nothing** in this configuration (+0.4/1.1/0.4 %
   BF16, ~0 % INT8) — worth noting because the project's TMA schedule *already* has a dedicated producer
   warp (`kProducerThreads = 32`), so the remaining delta cannot be attributed to a missing producer
   warp on the strength of this table.
3. **The BF16 margin is not separable from run-to-run variation in its own source.** The same repo's
   earlier commit `6143cb4` reports, at M=N=K=4096, v0 = 175.30 and v1 = 180.40 TFLOPS, against 164.01 /
   171.24 in the later table. That is a **6.9 % swing in v0 alone** between two runs of the same binary
   on the same shape — larger than the 4.4 % BF16 TMA gain being claimed. This is the same
   warm-up/clock-drift effect this project's own rules anticipate; the INT8 gain is larger and is the
   more defensible of the two.

**Substitutions stated:** BF16 and INT8, not FP8; M=N=K ∈ {2048, 4096, 8192}, i.e. **no small-M point at
all**; one GPU, one session each, no interleaving or repetition reported.

Corroborating datapoint that the **non-TMA path is the vendor default** on sm_120: an nsys trace of
PyTorch FP16 matmul on a 5090 names `cutlass_80_tensorop_f16_s16816gemm_...128x64_64x3_nn_align8` — an
Ampere-generation `cp.async` kernel, no TMA **[reported by a third party]**. cuBLAS on sm_120 is largely
forward-porting sm_80 mainloops, so "cuBLAS is slower than my TMA kernel" is not evidence about TMA.

**Nearest published sm_120 FP8/FP4 TFLOP/s** (all TMA, all large-shape, third-party-reported):

| Source | Precision | Shape | TFLOP/s |
|---|---|---|---|
| CUTLASS example 87a, FP8 dense | FP8 | 1024³ | 206 |
| " | FP8 | 2048³ | 522 |
| " | FP8 | 4096³ | 746 |
| " | FP8 | 4096×4096×8192 | **771** |
| CUTLASS PR #3280 / TRT-LLM #11368 | NVFP4 | — | 41.6 (GB10) |
| CUTLASS PR #3187 (unmerged) | FP16 | RTX 5090, large | 223.7 vs cuBLAS 211.0 |

> **Do not use 771 TFLOP/s as an FP8 ceiling without resolving the peak convention.** The 5090's FP8
> dense peak is reported across at least three mutually inconsistent values in the sources read here:
> **419 TFLOPS with FP32 accumulate** (arXiv 2606.10493, hardware description), **~838 dense**
> (gigagpu spec table), and **838** used as the INT8 SOL denominator by `learn-cuda` — which is what
> makes its 485.84 INT8 TFLOPS print as "58 % of SOL" rather than exceeding 100 %. 771 is coherent only
> against 838, i.e. ~184 % of the 419 figure. **[inferred]** I could not find an NVIDIA document stating
> the sm_120 FP8 dense peak with sparsity *and* accumulation convention attached **[not found]**, so any
> "% of FP8 peak" number on this part is fragile. This is an argument for the project's existing choice
> to score this Op in **GB/s against a DRAM constant** rather than TFLOP/s against a tensor constant.

---

## 2. Question 2 — do small token tiles explain ~100 GB/s?

### 2.1 The project's own schedule geometry, computed from its own templates

Formulas transcribed from `src/ops/linear/fp8/fp8_schedule.cuh:130-193`; shapes and band edges from
`src/ops/linear/fp8/shapes/n5120_k6144.cu:41-56`; MMA instances from
`src/ops/linear/fp8/fp8_instances.cuh:5-27` **[all read in source]**. smem cap 99 KiB = CUTLASS's
`sm120_smem_capacity_bytes / 1024` (§3.1). CTA counts are for N=5120.

| Arm | Band (TMA build) | Tile | Stages | Consumer thr | Producer thr | Total thr | smem B | CTAs @T=64 | CTA/SM (smem ∩ min) | thr/SM |
|---|---|---|---|---|---|---|---|---|---|---|
| **TMA** `Tma32x64` | T ≤ 64 | **32×64**×128 | 3 | 64 | 32 | 96 | 36,912 | **160** | 2 | 192 |
| **MMA** `Fp8A8T32R32K128` | (Windows T ≤ 64) | **32×32**×128 | 3 | 64 | 0 | 64 | 24,576 | **320** | 3 | 192 |
| **TMA** `Tma64x128` | 129 ≤ T ≤ 192 | **64×128**×128 | 3 | 256 | 32 | 288 | 73,776 | 80 | **1** | 288 |
| **MMA** `Fp8A8T64R128K128` | (Windows T > 128) | **64×128**×128 | **2** | 256 | 0 | 256 | 49,152 | 80 | **2** | **512** |
| **TMA** `MidBulk` | 193 ≤ T ≤ 768 | 128×128×128 | 3 | 256 | 32 | 288 | 98,352 | 40 | **1** | 288 |
| **TMA** `Bulk` | T > 768 | 128×256×128 | 2 | 256 | 32 | 288 | 98,336 | 20 | **1** | 288 |

### 2.2 What that rules in and out, band by band

**Band T ≤ 64 — the confound exists but points the wrong way to explain the gap.** The two arms use
*different* row tiles (64 vs 32), so this band is genuinely confounded. But the confound is not
"the TMA schedule has a too-small tile":

- Both arms have the same **32-token** tile. Small-M is shared, so it cannot be the discriminator.
- The TMA arm's row tile is **twice as large** (64 vs 32) — better L2 reuse per byte of weight, half the
  CTA count, and the same 3 stages, the same 64 MMA threads and the same **192 threads/SM**.
- Per k-tile the TMA CTA moves (32+64)×128 = 12,288 B for 2× the MMA work; the MMA CTA moves
  (32+32)×128 = 8,192 B. Weight traffic from DRAM is the same 30.00 MiB either way (160 row-tiles ×
  32 rows vs 80 × 64, both × 6144 K).

So the honest statement is: **a thin-M GEMM being weight-bandwidth-bound is true, and it does not
explain this band**, because the MMA arm is thin-M too and reaches 1110-1316 GB/s **[measured by the
project]** — 62-73 % of the 1792 GB/s spec constant and 66-79 % of the 1674.5 GB/s sustained-read
constant. At 30.00 MiB that is 28.3-23.9 µs; at 100-125 GB/s it is 314.6-251.7 µs **[computed]**.

**Band T ≥ 129 — the arms already share a tile shape.** `Tma64x128` and `Fp8A8T64R128K128` are both
**64×128×128**. What differs is stage count (3 vs 2), which alone moves smem 73,776 → 49,152 B and
occupancy 288 → 512 threads/SM, plus the producer warp and the transport. This is a materially narrower
confound than "a TMA schedule and an MMA schedule are different tile shapes" implies for this band
**[computed from the project's own source]**. The T>768 arms (`MidBulk`, `Bulk`) have **no MMA
counterpart in the dispatch at all** — the MMA build falls through to `Fp8A8T64R128K128` — so those
bands are not a transport comparison under any reading.

### 2.3 Published FP8 throughput-vs-M on a 5090

**[not found]** No published table of sm_120 FP8 GEMM TFLOP/s or GB/s versus M, at any M, from CUTLASS,
vLLM, TensorRT-LLM or FlashInfer. The nearest substitutes, all stated as substitutions:

- **Tile selection as a proxy for what production considers adequate at small M**: vLLM uses
  64×64×128 TMA pingpong at M ≤ 256 (§1.2) **[read in source]**. That is the only published statement
  about small-M sm_120 FP8 tiling, and it is a choice, not a measurement.
- **FlashInfer's sm_120 MoE note** lists only 128×128×128 cooperative as implemented, with 64×128×128
  ping-pong as a TODO, and motivates it by tokens-per-expert being small **[read in third-party doc]**.
- **The general thin-M bandwidth statement** is well supported and not in dispute: 5090 sustained read
  is quoted at 1674 GB/s (93 % of 1792 spec) and a hand-written bf16 single-token decode kernel reaches
  1192 GB/s effective, 71 % of that **[reported by a third party]**. This says a *good* thin-M kernel
  on this part reaches 1000-1700 GB/s. It does not say anything about a 100 GB/s one.

---

## 3. Question 3 — does the TMA route cost occupancy here?

### 3.1 The sm_120 shared-memory budget is 101,376 B, and it is the binding constraint

`include/cutlass/arch/arch.h` at `main` **[read in source]**:

```cpp
constexpr int sm100_smem_capacity_bytes = 232448;
constexpr int sm107_smem_capacity_bytes = 334848;
constexpr int sm120_smem_capacity_bytes = 101376;
```

sm_120 has **43.6 %** of sm_100's shared memory. NVIDIA's own tracker confirms this is a real device
limit and a recurring source of failure. TensorRT-LLM PR #12141, "[#11368][fix] FP4 CUTLASS GEMM shared
memory overflow on GB10 (SM121)" **[read in first-party PR text]**:

> All SM12x devices (SM120=RTX 5090, SM121=GB10) report `cudaDevAttrMaxSharedMemoryPerBlockOptin` =
> 101,376 bytes (99 KiB), confirmed by the `static_assert(smemSize <= 99 * 1024)` in `mla_sm120.cu`
>
> SM100 (B200/B100) has ~227 KiB per block — the large SM120 tile configurations were likely ported
> from SM100 without accounting for the smaller SM12x SMEM
>
> CUTLASS `StageCountAutoCarveout` sizes pipeline stages at compile time using the arch
> `SharedMemoryCapacity`, so the resulting `SharedStorage` struct exceeds SM12x limits

The same overflow recurs on the MoE grouped path, where TRT-LLM's `cutlass TMA WS grouped gemm` kernels
`gemm_grouped_sm120_M128_BS_group2` (128×256×128) and `..._M256_BS_group0` (256×128×128) fail to
initialize because they exceed the 99 KiB budget **[read in first-party issue text]**. NVIDIA's fix was
to make 128×128×128 the SM120 default and select the larger tile only when the device reports enough
smem.

This is directly load-bearing for the project: its own `kSharedBytes <= 99 * 1024` assertion
(`fp8_schedule.cuh:124,167,192`) is the same 101,376 B number, so the project is already correctly
bounded — but bounded *at 1 CTA/SM* for three of its four TMA arms (§2.1).

### 3.2 CUTLASS's own stage counts at that budget

`sm100_compute_stage_count_or_override<StageCountAutoCarveout<carveout>>` in
`include/cutlass/gemm/collective/builders/sm100_umma_builder.inl` **[read in source]**:

```cpp
constexpr int stage_bytes =
    bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);
return (CapacityBytes - carveout_bytes) / stage_bytes;
```

Applied to FP8 (`uint8_t`, 8 bits) at `CapacityBytes = 101376`, carveout 0, pipeline storage taken as
the two 8-byte mbarriers of `PipelineTmaUmmaAsync<1>` (the ±16 B is immaterial at these sizes)
**[computed from the published formula]**:

| FP8 tile | stage B | stages @99 KiB |
|---|---|---|
| 128×128×128 (vLLM/TensorRT-LLM default) | 32,784 | **3** |
| 64×64×128 (vLLM M≤256) | 16,400 | **6** |
| 32×64×128 (project's `Tma32x64`) | 12,304 | **8** |

So CUTLASS's *budget* would allow more stages than the project uses at 32×64×128 (3 used, 8 affordable)
and 64×128×128 (3 used, 3 affordable). At 128×128×128 and 128×256×128 the project is at or near the
ceiling, which is why those two arms are pinned to 1 CTA/SM. **[inferred]** — this is arithmetic on the
published formula, not a measurement of occupancy.

### 3.3 Could smem alone explain a large gap?

**It could explain a large *fraction* of a 2× gap; it does not obviously explain 10×.** Reasoning
explicitly, so the limit is visible:

- At T ≥ 129 the two arms share a 64×128×128 tile. Stage count 3 vs 2 takes occupancy from 288 to 512
  threads/SM — a **1.78×** difference in resident threads, in the MMA arm's favour. A memory-bound
  kernel's achievable bandwidth typically scales with memory-level parallelism, so this is a credible
  mechanism for a gap of that order. **[inferred]**
- At T ≤ 64 the two arms have **equal** threads/SM (192) and equal stage count. Occupancy is therefore
  *not* a candidate explanation for that band. **[computed]**
- A third-party sm_121a teardown reaches the same structural conclusion from the other side: on SM121
  "every flagship runs at about one block per SM" — 45 KB and 82 KB tiles against 99 KB, and a GEMM
  pinned at 168 registers/thread to 1 block/SM at 15 % occupancy — and attributes the gap to datacenter
  Blackwell's tensor memory holding accumulators off both the register file and the smem budget
  **[reported by a third party]**. sm_120 has no TMEM and no `tcgen05` **[read in source / third-party
  ptxas analysis]**, so this wall is architectural, not a tuning miss.

**No published measurement attributes any specific share of an sm_120 TMA-vs-`cp.async` gap to shared
memory or occupancy: [not found].**

---

## 4. Question 4 — is there a documented reason a TMA GEMM would be slower on consumer Blackwell?

### 4.1 TMA multicast does not exist on sm_120

This is stated by NVIDIA, repeatedly, in the kernel sources themselves. `sm120_mma_builder.inl`
**[read in source]**:

```cpp
static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");
```

and the header comment of CUTLASS examples 79a/79b/79c/79d and 80a, verbatim **[read in source]**:

> Note that GeForce RTX 50 series GPUs do not support:
> 1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
> 2. Dynamic datatypes.

Every CTA therefore loads its own A and B tiles. On sm_90/sm_100 a tile can be fetched once and
broadcast into several CTAs' smem, so the same arithmetic is served by strictly less DRAM traffic there.
This is a genuine capability loss **[published by NVIDIA]**, and it works against TMA on consumer parts
in a way that has nothing to do with TMA's per-byte rate. It is also **not** a candidate explanation for
a *round-trip* regression against a `cp.async` kernel that also has no multicast — the MMA arm loses the
same broadcast.

### 4.2 The sm_120 TMA instruction is encoded `.shared::cta`, not `.shared::cluster`

`include/cute/arch/copy_sm90_tma.hpp` **[read in source]**:

```cpp
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
 asm volatile (
   "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
   " [%0], [%1, {%3}], [%2], %4;" ...
#else
 asm volatile (
   "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
   ...
#endif
```

The same `SM120_ENABLED` split is present for the 2d/3d/4d/5d forms. So sm_120's TMA is a genuinely
distinct encoding from Hopper's, confirming the sm_120 TMA unit is a cut-down one. **No NVIDIA document
was found that quantifies a throughput penalty from this** — it is consistent with the 1×1×1 cluster
limit, and with the multicast removal, rather than being an independent documented slowdown.
**[inferred]**

### 4.3 The one measured case where TMA lost to `cp.async` on consumer Blackwell

A third-party writeup of hand-written FP8 flash attention on **sm_121a** (D=128, S=8192) **[reported by
a third party]**:

| Version | Mechanism | TFLOP/s |
|---|---|---|
| v1 | every thread loads, no overlap | 10 |
| v3 | bigger tiles + `cp.async.cg` register-bypass | 20 |
| **v7** | **TMA added, no warp specialization** | **18 (regression)** |
| v11 | TMA + warp specialization (1 DMA warp, 4 MMA warps, mbarrier handoff) | ~30 |

Its summary: *"TMA is not free. A bulk copy with no compute to overlap is just a slower load… without
warp specialization to hide it behind the MMA, the descriptor machinery costs more than it saves."*

This is the closest published precedent for a **TMA-slower-than-`cp.async` outcome on consumer
Blackwell**, and its mechanism is *lack of something to overlap the copy with*. It is a different
kernel family (attention, FP8, sm_121a) at a different scale, and the project already has a dedicated
producer warp, so it does not transfer directly. It does establish that the direction "TMA slower" is
reachable on this silicon and is a scheduling/overlap phenomenon, not a bandwidth phenomenon.

**Consistent with the same reading:** §1.3's v1→v2 result, where adding warp specialization to TMA
bought ~0-1 % on sm_120 GEMM at 8192 — the overlap that mattered in the attention case was worth far
less once the GEMM tile was already large enough to keep the pipe full.

### 4.4 What was searched for and does not exist

**An NVIDIA statement that consumer Blackwell's TMA unit has lower throughput than datacenter
Blackwell's: [not found].** No such statement in the CUTLASS source, examples, or the issue trackers
searched. The documented sm_120 differences are capability removals (§4.1, §4.2) and the smem budget
(§3.1) — not a stated throughput reduction. The same negative result holds for "narrower L2 on GeForce":
the 5090's L2 is **96 MB**, larger than the H100's 50 MB **[reported by a third party]**, so a
"narrower L2 on consumer parts" hypothesis is contradicted rather than merely unsupported.

---

## 5. Question 5 — what would a controlled transport-only A/B require?

Stated as requirements, not as a recommendation.

### 5.1 The controlled design, as published

The `learn-cuda` 02c pattern is the reference: **one kernel template, one set of tile/stage/warp
parameters, two implementations of the global→shared load.** Everything else — MMA atom, `ldmatrix`
addressing, swizzle, accumulator layout, epilogue, rasterization, grid mapping — held identical. Its
parameters are visible in the indexed source: `BLOCK_M/N/K`, `NUM_WARP_M/N`, `NUM_STAGES` are shared
constants, `TB_SIZE` is `WM*WN*32` for v0/v1 and `(WM*WN+1)*32` for v2, and only `gmem_to_smem` is
replaced **[reported by a third party]**.

For an FP8 A8 GEMM on this port's shape, the invariants that would have to be pinned:

- tile `BM × BN × BK`, stage count, and the **producer/consumer warp split** (or its absence) in both arms
- MMA atom and fragment layout, accumulator count, register budget
- rasterization order and grid mapping
- the same `kActivationCache` / `kWeightCache` cache hints
- the same epilogue and output path

Anything left varying is a second variable, and the comparison stops being about transport.

### 5.2 What CUTLASS can and cannot do here

**Cannot.** §1.1: no non-TMA sm_120 collective, no non-TMA schedule tag, `GmemTiledCopyA/B` hardwired to
`SM90_TMA_LOAD`. There is no `CollectiveBuilder` parameter, named kernel, or stage-count setting that
swaps only the transport on sm_120. `StageCountAutoCarveout` changes depth, not transport.
`KernelTmaWarpSpecializedPingpong` vs `...Cooperative` changes the warp-group layout, not the transport.

**Can, as substitutes rather than equivalents:**

- CUTLASS's **sm_80/sm_89-era** `cp.async` FP8 GEMM mainloop, compiled for `sm_120a`. sm_120 retains the
  warp-level `mma.sync` family from sm_70-sm_89, so an sm_89-style FP8 mainloop is ISA-reachable on
  sm_120 **[read in source: `rr_op_selector_sm120` wraps the same `mma.sync` lineage; ptxas analysis
  confirms no new warp-level tensor path]**. This gives a working non-TMA FP8 GEMM on sm_120, but it
  arrives with sm_89's own tile shapes and epilogue, so it is a *cross-family* comparison, not a
  transport-only one.
- `gau-nernst/learn-cuda` 02c as a **shape-matched external reference** for what a transport delta looks
  like at matched tiles on this exact GPU (§1.3) — usable as a sanity bound on the expected magnitude,
  which is single-digit percent, not an order of magnitude.

### 5.3 The measurement conditions a controlled A/B on this machine would need

Independent of which transport is being tested, and drawn from the failure modes this project has
already recorded:

- **Interleave the two arms; do not group them.** Un-pinned clocks on this card move by several percent
  between the two arms of a sequential A/B, which is the same order as the effect being looked for
  (§1.3 point 3 measures a 6.9 % swing in one arm across two sessions).
- **Vary one thing at a time across the pair set**: transport at fixed stage count, then stage count at
  fixed transport. §2.1 shows stage count alone moves occupancy 288↔512 threads/SM at a fixed tile.
- Confirm the grid actually differs only where intended, and record CTA count and smem per CTA as
  measured rather than as computed from templates.
- Record the achieved DRAM throughput, not only the Op's own effective GB/s, so a bandwidth-bound
  ceiling and a scheduling stall are distinguishable.

---

## 6. What I searched and did not find

| Searched for | Result |
|---|---|
| A CUTLASS-published TMA-vs-non-TMA FP8/FP4 GEMM comparison on sm_120 | **[not found] — and structurally impossible:** all 8 SM120 collectives are `*_tma.hpp`; no cp.async SM120 mainloop exists |
| Any CUTLASS sm_120 schedule tag or `CollectiveBuilder` parameter that toggles only the mainloop transport | **[not found]** — `GmemTiledCopyA/B` are hardwired `SM90_TMA_LOAD` in `sm120_mma_builder.inl` |
| Published sm_120 FP8 GEMM TFLOP/s or GB/s **versus M**, especially M ≤ 256 | **[not found]** in CUTLASS, vLLM, TensorRT-LLM, FlashInfer. Nearest is vLLM's *tile choice* (64×64×128 TMA at M≤256), which is a decision, not a measurement |
| Published sm_120 TMA-vs-`cp.async` measurement at **small M** | **[not found]** — the only controlled sm_120 transport comparison found (`learn-cuda` 02c) uses M=N=K ∈ {2048, 4096, 8192} |
| An NVIDIA statement that consumer Blackwell TMA has lower **throughput** than datacenter | **[not found]** — documented differences are multicast removal, `.shared::cta` encoding, and the 99 KiB smem cap |
| CUTLASS **occupancy / achieved-bandwidth tables** for sm_120 FP8 TMA kernels | **[not found]** — stage counts are computable from the published formula (§3.2) but no achieved-occupancy or DRAM-throughput table was located |
| An NVIDIA-published sm_120 FP8 dense **peak** with sparsity *and* accumulation convention attached | **[not found]** — 419 / 838 both circulate (§1.3), so %-of-peak figures for FP8 on this part are fragile |
| TensorRT-LLM published **TFLOP/s** for an sm_120 FP8/NVFP4 GEMM | **[not found]** — its sm_120 PRs and issues report correctness, tile-selection fixes and end-to-end tok/s, not isolated GEMM throughput. Its FP4 datapoint (41.6 TFLOP/s, GB10) is cited in an NVIDIA issue as a CUTLASS example run, not a TRT-LLM benchmark |
| SGLang sm_120 FP8 GEMM numbers | **[not searched]** — not attempted in this session; recorded as an open gap rather than a negative |
| A `gau-nernst/learn-cuda` `02c_matmul_sm120` tree readable at HEAD | **404 at the time of writing** (repo root README lists the module; the directory and its files do not resolve). Numbers above are **as indexed by DeepWiki**, with line references into the repo's own `README.md`/`main.py`/`matmul_v0.cu`, not as read from the files |
| vLLM `main` source for the sm_120 FP8 dispatch (paths moved; `csrc/quantization/w8a8/cutlass/c3x/…` and `csrc/quantization/cutlass_w8a8/c3x/…` both 404) | **[not read directly]** — the config in §1.2 is quoted from a third-party verbatim lift that names the vLLM commit (`e24d1b24`) and is corroborated by the file list in NVIDIA's vLLM PR #17280 |

### Confound discipline, restated

Two comparisons in this document are **not** apples-to-apples and are labelled where they appear:
`learn-cuda` v0/v1 is transport-matched but BF16/INT8 and large-M; the project's T≤64 band is
transport-*and*-row-tile confounded (§2.2), while its T≥129 band is tile-matched and differs in stage
count, occupancy and transport (§2.2). No comparison in this document mixes tile shape with transport
without saying so.
