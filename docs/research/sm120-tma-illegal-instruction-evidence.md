# sm_120 TMA: `cudaErrorIllegalInstruction` — evidence

Research date **2026-10-01**. Primary sources only: NVIDIA PTX ISA, CUDA C++ Programming Guide, CUDA
Driver/Runtime API reference, CUDA release notes and NVIDIA's own driver release notes; CUTLASS,
vLLM, TensorRT-LLM, FlashInfer and SGLang source at `main`; and the first-party issue trackers of
those projects.

Toolchain the claims were read against, and the machine they describe **[measured here]**:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, `nvidia-smi` `compute_cap = 12.0` |
| Driver | **617.14** |
| Toolkit | CUDA **13.3.73** (`nvcc --version`) |
| Host | MSVC 14.51, Windows 11, WDDM |
| Build target | `CMakeLists.txt:7` — `set(CMAKE_CUDA_ARCHITECTURES 120a ...)`, hard-required at `CMakeLists.txt:9-12` |

Every claim below carries one of: **[documented by NVIDIA]**, **[read in source]**, **[reported by a
third party]**, **[inferred]**, **[not found]**, **[measured here]**. Nothing here recommends an action.

---

## Verdict table

| # | Question | Answer | Confidence | Source |
|---|---|---|---|---|
| 1 | Does TMA work on sm_120? | **Yes.** `cp.async.bulk.tensor` "Requires sm_90 or higher"; the TMA unit is listed as supported for CC 9.0/10.x/11.0/**12.x**; `tensormap.replace` is explicitly listed for `sm_120a` from PTX ISA 8.6. Documented sm_120a restriction is narrow (see §1.3). | **High** — three independent NVIDIA documents | PTX ISA 9.4 §9.7.10.28.5.3, Table 72; C++ Programming Guide Table 29; Driver API `cuTensorMapEncodeTiled` |
| 2 | Does sm_120 support `wgmma`? | **No.** `wgmma.mma_async` — "Target ISA Notes: **Requires sm_90a**". Table 72 lists `wgmma.*` for **sm_90a only** (PTX ISA 8.0). | **High** — explicit ISA statement | PTX ISA 9.4 §9.7.17.5.2, Table 72 |
| 2b | Does sm_120 support `tcgen05.mma`? | **No.** Every `tcgen05.*` variant is "Supported on … `sm_100a`, `sm_101a`(→`sm_110a`)" plus family-specific `sm_100f`/`sm_101f`/`sm_110f`. `sm_120a`/`sm_120f` appear in **no** `tcgen05` row. | **High** — explicit ISA table | PTX ISA 9.4 Table 72 |
| 2c | So what *does* sm_120 use for tensor-core MMA? | Warp-level `mma.sync`, extended on `sm_120a` with `.kind`, `.block_scale`, `.scale_vec_size` from PTX ISA 8.7. | **High** | PTX ISA 9.4 Table 72; CUTLASS `blackwell_functionality.md`; TensorRT-LLM `warpspec_sm120/README.md` |
| 3 | Documented causes of `cudaErrorIllegalInstruction` from a TMA path | Enumerated in §3. Key discrimination: a malformed descriptor is **`CUDA_ERROR_INVALID_VALUE` at creation**, a misaligned address is **`cudaErrorMisalignedAddress`**, and **`cudaErrorIllegalInstruction` is documented only as "the device encountered an illegal instruction"** — i.e. an instruction the hardware traps on. Violating a TMA constraint is documented as *undefined behaviour*, never as illegal-instruction. | **High** for the API-level mapping; **no NVIDIA document found** that maps a TMA-constraint violation to illegal-instruction | Driver API `cuTensorMapEncodeTiled`; Runtime API error enumerators; PTX ISA §9.7.10.28.5.1 |
| 3b | The one documented TMA requirement this tree does not meet | A tensor map in `.global` space needs `fence.proxy.tensormap::generic.acquire`, because the tensormap proxy "is not acquired from generic-proxy at CUDA Kernel start". The tree's three TMA kernels read the map from a `cudaMallocAsync` device buffer and emit no such fence (`git grep` finds no `fence.proxy` in `src/` or `include/`). | **High** that the requirement is documented and unmet **[measured here] for the absence**; the causal link to the fault is **[inferred]** | PTX ISA 9.4 §8.8.4, §9.7.15.x `fence` example, `tensormap.cp_fenceproxy` example; `git grep` |
| 4 | Does CUTLASS support FP8 TMA GEMM on sm_120? | **Yes, and it is a first-class path**: `MainloopSm120TmaWarpSpecialized`, `…BlockwiseScaling` (FP8), `sm120_blockscaled_mma_tma.hpp` (NVFP4/MXFP), `sm120_mma_array_tma.hpp`, plus TMA epilogues. `cutlass::arch::Sm120` exists. MMA is `mma.sync…kind::f8f6f4`, **not** wgmma/tcgen05. Consumer Blackwell is treated separately from SM100 throughout. | **High** — read in CUTLASS `main` | `include/cutlass/arch/arch.h`, `include/cute/arch/config.hpp`, `sm120_mma_tma.hpp`, `sm120_mma_tma_blockwise_scaling.hpp`, `media/docs/cpp/blackwell_functionality.md` |
| 5 | Is this Windows-specific? | **Not established, and there is positive evidence against it**: the one CUTLASS illegal-instruction report inside a TMA copy on `sm_120a` is **Ubuntu 24.04**. A documented NVIDIA **driver** bug in TMA descriptor configuration on Blackwell exists and is still listed in the 610 driver notes, but its symptom is IMA / XID 13, **not** illegal-instruction. Two Windows-only SM120 TMA defects are filed, but both are *misalignment*, which NVIDIA documents as a distinct error code. | **Medium** — the absence of a Windows-specific illegal-instruction finding is a negative result | CUTLASS #2728, #2905, #2906; NVIDIA driver release notes 595/610 |
| 6 | Has anyone landed an FP8 TMA GEMM that runs on sm_120 / RTX 5090? | **Yes.** vLLM `main` ships a dense-FP8 CUTLASS SM120 GEMM on `KernelTmaWarpSpecialized*` schedules; TensorRT-LLM ships an sm_120 TMA warp-specialized FMHA as the *default* sm_120 prefill. Separately, several **block-scaled NVFP4 MoE** TMA paths on SM120 have been broken at various times and were fixed (FlashInfer #2708/#2716, CUTLASS 4.4). | **High** — read in source, plus first-party trackers | vLLM `scaled_mm_sm120_fp8_dispatch.cuh`; TensorRT-LLM `warpspec_sm120/README.md`; FlashInfer #2708 |

### The single most important sentence

**sm_120 supports TMA, and it supports neither `wgmma` nor `tcgen05`.** Consumer Blackwell keeps the
Hopper TMA unit and replaces Hopper's *asynchronous MMA* with the warp-level `mma.sync` path extended
by `sm_120a`-only `.kind`/`.block_scale` qualifiers. TensorRT-LLM states this in its own source:

> "Only half of the Hopper warp-specialization recipe ports to consumer Blackwell: TMA-driven async
> loads survive, but async MMA does not (sm_120 / sm_121 have no `wgmma.async` equivalent), so the
> compute warps stay on `mma.sync` while a dedicated producer warp drives the loads with TMA."
> — [read in source] `TensorRT-LLM/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md`

Consequence for this project: "TMA is not supported on sm_120" is **false**, and "the kernel needs
wgmma" is **false**. Any framing of the fault that requires either is contradicted by the ISA.

---

## Question 1 — does TMA work on sm_120 at all?

### 1.1 Three independent NVIDIA statements that it does

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.10.28.5.3 `cp.async.bulk.tensor`, *Target ISA Notes*:
> "**Requires sm_90 or higher.**"

sm_120 satisfies "sm_90 or higher". The additional qualifiers that are **not** available on sm_120a are
listed separately in the same note and are all sm_100-family: `.tile::gather4`, `.im2col::w`,
`.tile::scatter4`, `.im2col::w::128`, `.cta_group` ("Supported on following architectures: sm_100a
sm_101a … sm_100f or higher in the same family sm_101f … sm_110f"). The base form
`.dim .shared::cta .global .tile .mbarrier::complete_tx::bytes` — which is the form this tree emits — is
in the "sm_90 or higher" group.

**[documented by NVIDIA]** CUDA C++ Programming Guide, §5.1.3, **Table 29 "Feature Support per Compute
Capability"**, whose columns are `7.x | 8.x | 9.0 | 10.x | 11.0 | 12.x`:

> "Tensor Memory Accelerator (TMA) unit ( Using the Tensor Memory Accelerator (TMA) ) — No | Yes"

The `Yes` cell spans 9.0 through **12.x**. Same document, §4.12.2:

> "To offload these computations, compute capability 9.0 (Hopper) and later … have a tensor memory
> accelerator (TMA)."

**[documented by NVIDIA]** CUDA Driver API 13.4, *Tensor Map Object Management*,
[`cuTensorMapEncodeTiled`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html):

> "Tensor map objects are only supported on devices of compute capability **9.0 or higher**."

and, for the sibling entry points, "This API is only supported on devices of compute capability 10.0 or
higher" (that is `cuTensorMapEncodeIm2col`, not the tiled encoder).

### 1.2 `tensormap.replace` — explicitly on `sm_120a`

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.10.29 `tensormap.replace`, *Target ISA Notes*:

> "Supported on following architectures: sm_90a sm_100a sm_101a (Renamed to sm_110a from PTX ISA
> version 9.0) **sm_120a** And is supported on following family-specific architectures from PTX ISA
> version 8.8: … **sm_120f or higher in the same family** …"

Table 72 (*Arch-specific/Family-specific PTX Features Release History*) pins the version:

| Instruction variant | PTX ISA | Supported targets |
|---|---|---|
| `tensormap.replace` base | 8.3 | `sm_90a` |
| | **8.6** | **`sm_100a`, `sm_120a`** |
| | 8.8 | `sm_100f`, `sm_120f` |
| | 9.0 | `sm_110f` |

### 1.3 The supported subset on sm_120a — the documented restriction, verbatim

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.10.28.5.1 *Restriction on Tensor Copy instructions*, the
only sm_120a-specific restriction in the whole tensor-copy section:

> "Following are the restrictions for sm_120a: `cp.async.bulk.tensor` with the direction
> `.shared::cluster.global` doesn't support: the sub-byte types; the qualifier `.swizzle_atomicity`"

That is the complete list. It constrains **only** the `.shared::cluster` (multicast/distributed-shared)
destination. The tree emits `cp.async.bulk.tensor.2d.shared::cta.global.tile…` — destination
`.shared::cta`, source `.global` — so this restriction does not apply. **[inferred]** from the direction
qualifier.

`sm_120f`/`sm_121f` in addition enable the family-specific forms (`tensormap.replace.swizzle_atomicity`,
`mma` `.block_scale`, …) per Table 72. The tree targets `sm_120a`, not `sm_120f`
(**[measured here]**, `CMakeLists.txt:7`).

### 1.4 CUTLASS treats sm_120a as TMA-capable

**[read in source]** `include/cute/arch/config.hpp` (CUTLASS `main`) enables the TMA machinery for
SM120a:

```cpp
#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\
     …
     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED) \
     …
#  define CUTE_ARCH_TMA_SM90_ENABLED
#  define CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED
```

and separately defines `CUTE_ARCH_MMA_SM120_ENABLED` / `CUTE_ARCH_TMA_SM120_ENABLED` for
`CUTLASS_ARCH_MMA_SM120_ENABLED`/`SM120A_ENABLED`. `include/cutlass/arch/config.h` defines
`CUTLASS_ARCH_MMA_SM120A_ENABLED` only when `__CUDA_ARCH_FEAT_SM120_ALL` is defined — i.e. when
compiling `-arch=sm_120a`, which is what the tree does **[measured here]**.

---

## Question 2 — does sm_120 support `wgmma`? (the decisive question)

### 2.1 `wgmma`: no. Stated twice, unambiguously.

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.17.5.2 *Asynchronous Multiply-and-Accumulate Instruction:
`wgmma.mma_async`*, *Target ISA Notes*:

> "**Requires sm_90a.**"

The same single line — "Requires sm_90a" — is the *Target ISA Notes* text for every `wgmma` variant in
the document: `wgmma.mma_async` (f16/bf16/tf32/FP8/integer), `wgmma.mma_async.sp`,
`wgmma.commit_group`, `wgmma.wait_group`, `wgmma.fence`. All five of those rows in Table 72 carry
"Base variant | 8.0 | **sm_90a**" and nothing else.

`sm_90a` is a distinct architecture-specific compilation target from `sm_120a`; the PTX ISA
"Architecture-Specific and Family-Specific Targets" section treats them as disjoint, and
CUTLASS's own documentation states the practical consequence: "kernels compiled for Blackwell SM100
architecture with arch conditional features (using `sm100a`) are not compatible with RTX 50 series
GPUs" **[documented by NVIDIA]**, `docs.nvidia.com/cutlass/…/overview.html`.

**So: `wgmma` is not available on sm_120. A kernel containing `wgmma.mma_async` compiled for
`sm_120a` cannot be produced by a conforming compiler, and executing that instruction on sm_120 is
precisely the "illegal instruction" case.**

### 2.2 `tcgen05.mma`: also no.

**[documented by NVIDIA]** PTX ISA 9.4, Table 72 — every `tcgen05` row lists only `sm_100a` /
`sm_101a`(→`sm_110a`) plus family-specific `sm_100f` / `sm_101f` / `sm_110f`, and (for some ops)
`sm_103a` / `sm_103f` / `sm_107a` / `sm_107f`. Examples:

| Instruction variant | PTX ISA | Supported targets |
|---|---|---|
| `tcgen05.mma` base | 8.6 | `sm_100a` |
| | 8.8 | `sm_100f` |
| | 9.0 | `sm_110f` |
| `tcgen05.alloc`, `.dealloc`, `.relinquish_alloc_permit` base | 8.6 | `sm_100a` → 8.8 `sm_100f` → 9.0 `sm_110f` |
| `tcgen05.ld`, `.st`, `.wait`, `.cp`, `.fence`, `.commit` base | 8.6 | `sm_100a` → 8.8 `sm_100f` → 9.0 `sm_110f` |

`sm_120a` and `sm_120f` appear in **no** `tcgen05` row. **Neither asynchronous MMA generation exists on
sm_120.**

### 2.3 What sm_120 uses instead — and what this tree already uses

**[documented by NVIDIA]** PTX ISA 9.4, Table 72:

| Instruction variant | PTX ISA | Supported targets |
|---|---|---|
| `mma` — Types `.e3m2`,`.e2m3`,`.e2m1`; Qualifiers `.kind`,`.block_scale`,`.scale_vec_size` | **8.7** | **`sm_120a`** (8.8 → `sm_120f`) |
| `mma.sp` — Kind `.kind::mxf4nvf4`, `.kind::mxf4` | 8.7 | `sm_120a`, `sm_121a` |

**[documented by NVIDIA]** `mma` *Target ISA Notes* also record: "`.e4m3` and `.e5m2` alternate floating
point type mma operation requires sm_89 or higher", and "Support for shape `.m16n8k32` and `.f16`
dtype/ctype with `.e4m3`/`.e5m2` … requires sm_120".

**[read in source]** CUTLASS `main`, `media/docs/cpp/blackwell_functionality.md`, section *Blackwell
SM120 GEMMs*, documents the SM120 PTX instruction set as:

| Ptx Instruction | Throughput | Notes |
|---|---|---|
| `mma.sync.aligned.kind::f8f6f4` | 1x Ada FP8 (2x FP32 accum) | A={f4,f6,f8} × B={f4,f6,f8}, TN |
| `mma.sync.aligned.kind::mxf8f6f4.block_scale` | 1x Ada FP8 (2x FP32 accum) | block-scaled, TN |
| `mma.sync.aligned.kind::mxf4.block_scale` | 2x Ada FP8 (4x FP32 accum) | TN |
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::[2X\|4X]` | 2x Ada FP8 (4x FP32 accum) | TN |

**[read in source]** This tree's MMA helper, `src/ops/common/mma.cuh:62-79`, emits exactly that
family:

```cpp
asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
             "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
             "{%10}, {%11,%12}, {%13}, {%14,%15};\n" …);
```

with hard-coded unit scales (`ue8m0` = `0x7F`) and scale ids `0` — the block-scaled instruction used
without scaling. **`wgmma` appears nowhere in this tree** (`git grep wgmma` finds only file names
containing `tma_mma`, not the instruction).

**The same MMA is used by both routes.** **[read in source]** `fp8_mma_compute_stage` in
`src/ops/linear/fp8/fp8_a8_mma_common.cuh` (which contains every `mma_fp8_e4m3` call site, lines 106 and
125) is called by the TMA kernel `fp8_a8_tma_mma_kernel`
(`src/ops/linear/fp8/fp8_a8_tma_mma.cuh:214`) *and* by the shipping non-TMA route. **[inferred]** The
A/B that moved the fault with the TMA tile branch therefore isolates the TMA path — the
`sm_120a`-only MMA is exercised successfully by the route that works.

### 2.4 Other sm_120a-vs-sm_100a differences worth carrying

**[documented by NVIDIA]** PTX ISA 9.4, §5.1.7, maximum statically allocated shared memory per CTA:

| Target architecture | Max static shared memory |
|---|---|
| `sm_90a` | 228 KB |
| `sm_100a`, `sm_103a`, `sm_107a` | 228 KB |
| `sm_110a` | 228 KB |
| **`sm_120a`, `sm_121a`** | **100 KB** |

**[read in source]** CUTLASS agrees: `include/cutlass/arch/arch.h` sets
`sm120_smem_capacity_bytes = 101376` (= 99 KB) and `struct Sm120` has no `kTmemCapacityColumns`.
`Fp8A8TmaMmaSchedule` already asserts `kSharedBytes <= 99 * 1024`
(`src/ops/linear/fp8/fp8_schedule.cuh:192`).

**[documented by NVIDIA]** CUTLASS docs on SM120 GEMMs: "On Geforce series graphics card, there is no
multicast feature therefore the cluster shape is fixed to 1x1x1" and "Only TN layout is supported."
The PTX-level statement backing the first is the same Table-72 family split: `.multicast::cluster` is
an sm_100-family qualifier.

---

## Question 3 — what is documented to cause `cudaErrorIllegalInstruction` from a TMA path?

### 3.1 The symptom classes are cleanly separated by NVIDIA, and they point at different things

**[documented by NVIDIA]** CUDA Runtime API, error enumerators:

| Enumerator | Documented meaning |
|---|---|
| `cudaErrorIllegalInstruction` | "The device encountered an **illegal instruction** during kernel execution." |
| `cudaErrorMisalignedAddress` | "The device encountered a **load or store instruction on a memory address which is not aligned**." |
| `cudaErrorAssert` | "An assert triggered in device code during kernel execution." |

**[documented by NVIDIA]** Driver API `cuTensorMapEncodeTiled`, *Returns*:
`CUDA_SUCCESS`, `CUDA_ERROR_DEINITIALIZED`, `CUDA_ERROR_NOT_INITIALIZED`, `CUDA_ERROR_INVALID_CONTEXT`,
**`CUDA_ERROR_INVALID_VALUE`**. `CUDA_ERROR_ILLEGAL_INSTRUCTION` is not in that list.

So the two symptom classes are not interchangeable:

- **Malformed descriptor parameters** → `CUDA_ERROR_INVALID_VALUE` **at creation time**, on the host,
  before any kernel exists.
- **Misaligned address reached by an executing instruction** → `cudaErrorMisalignedAddress`.
- **An instruction the hardware does not implement / traps on** → `cudaErrorIllegalInstruction`.

### 3.2 Enumeration of causes, with the documented or observed symptom

| Cause | Documented symptom | Source / status |
|---|---|---|
| Instruction not supported by the target architecture | `cudaErrorIllegalInstruction` | **Inferred** from the enumerator's wording + the ISA's `Target ISA Notes` mechanism. **[not found]** any NVIDIA document stating this mapping verbatim. |
| `trap;` / `brkpt;` executed | abort + interrupt to host | **[documented by NVIDIA]** PTX ISA §9.7.21.4 `trap`: "Abort execution and generate an interrupt to the host CPU"; "Supported on all target architectures." CUTLASS #3125 shows a `trap` surfacing as `cudaErrorIllegalInstruction (715)`. |
| Invalid `globalDim` / `globalStrides` / `boxDim` / `elementStrides` (e.g. stride not a multiple of 16 B, `boxDim[0]*elementSize` not a multiple of 16 B, `boxDim > 256`, dim zero or > 2^32) | **`CUDA_ERROR_INVALID_VALUE` at `cuTensorMapEncodeTiled`** | **[documented by NVIDIA]** Driver API, full requirement list and *Returns* list. |
| Swizzle mode the shape does not permit (`boxDim[0]*elementSize` > swizzle size) | **`CUDA_ERROR_INVALID_VALUE` at creation** | **[documented by NVIDIA]** "When interleave is `CU_TENSOR_MAP_INTERLEAVE_NONE` and swizzle is not `CU_TENSOR_MAP_SWIZZLE_NONE`, the bounding box inner dimension … must be less than or equal to the swizzle size. `CU_TENSOR_MAP_SWIZZLE_64B` requires … ≤ 64. `CU_TENSOR_MAP_SWIZZLE_128B` requires … ≤ 128." |
| `tensorMap` address not 64-byte aligned | **`CUDA_ERROR_INVALID_VALUE`** ("tensorMap address must be aligned to 64 bytes") | **[documented by NVIDIA]** Driver API. (Distinct from `prefetch.tensormap`'s 64-byte requirement, which is **[reported by a third party]**: CUTLASS #2905.) |
| `globalAddress` not 16-byte aligned | **`CUDA_ERROR_INVALID_VALUE`** | **[documented by NVIDIA]** "globalAddress … must be 16 byte aligned." |
| Violating a *tensor-copy* constraint at execute time (e.g. bounding box outside tensor bounds, `.swizzle_atomicity` on sm_120a with `.shared::cluster.global`) | **undefined behaviour** — the PTX text says "the behavior is undefined", not that an error is raised | **[documented by NVIDIA]** PTX ISA §9.7.10.28.5.1. |
| `.override::global_address` address not 16-byte aligned | "a **runtime error** is raised" (unspecified which) | **[documented by NVIDIA]** PTX ISA §9.7.10.28.5.2.1. |
| `tensormap.replace` / descriptor mutation **without** the `tensormap::generic` release→acquire fence pair | **undefined behaviour** — documented as a *requirement*, not an error | **[documented by NVIDIA]** see §3.3. |
| A tensor map read from `.global` space **without** acquiring the tensormap proxy | **undefined behaviour** — documented as a *requirement* | **[documented by NVIDIA]** see §3.3. |
| Driver mis-configuring a TMA descriptor (NVIDIA bug, < 128 KB backing allocation, non-dense/non-overlapping tensor) | **sporadic illegal memory access or MMU fault / XID 13** | **[documented by NVIDIA]** driver release notes — see §5. |

**[not found]** I found **no** NVIDIA document that maps "descriptor or constraint violation observed at
kernel execution" to `cudaErrorIllegalInstruction`. Every documented mechanism for that enumerator is
about the *instruction*, and every documented descriptor mechanism lands in
`INVALID_VALUE`/`MisalignedAddress`/undefined behaviour/IMA.

### 3.3 The one documented TMA execution requirement this tree does not meet

This is the most specific actionable-by-facts item in the whole note, and it is a requirement the tree
visibly does not satisfy.

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.10.28.5.3 `cp.async.bulk.tensor` *Description*:

> "The operand tensorMap is the generic address of the opaque tensor-map object which resides in
> **`.param` space or `.const` space or `.global` space**. … The tensorMap is accessed in **tensormap
> proxy**."

**[documented by NVIDIA]** PTX ISA 9.4, §8.8.4 (CUDA API synchronization), verbatim:

> "…the CUDA API synchronization mechanisms above participate in proxy-preserved base causality order
> **except for the tensormap-proxy which is not acquired from generic-proxy at CUDA Kernel start and
> must therefore be acquired explicitly using `fence.proxy.tensormap::generic.acquire` when needed.**"

**[documented by NVIDIA]** PTX ISA 9.4, §9.7.15.16 `fence`, *Examples* — the acquire is written
immediately before the TMA that consumes the map:

```ptx
tensormap.replace.tile.global_address.shared.b1024.b64 [sMem], new_addr;
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [gbl], [sMem], 128;
fence.proxy.tensormap::generic.acquire.gpu [gbl], 128;
cp.async.bulk.tensor.1d.shared::cluster.global.tile [addr0], [gbl, {tc0}], [mbar0];
```

and §9.7.15.17 `tensormap.cp_fenceproxy` repeats the same sequence in its example.

**What this tree does.** **[measured here]** `git grep -n "fence.proxy\|tensormap" -- src include`
returns exactly three hits, all of them the copy instruction, no fence:

```
src/ops/linear/bf16/bf16_a16_tma_mma.cuh:83:  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
src/ops/linear/fp8/fp8_a8_tma_mma.cuh:107:   asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:140:   asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
```

The descriptor's *placement* differs by platform **[read in source]**:
`src/ops/linear/fp8/fp8_a8_tma_mma.cuh:44-49, 136-140, 348-362`:

```cpp
#ifdef _WIN32
#define NINFER_FP8_TMA_DESCRIPTOR_PARAM const Fp8TmaDescriptors* __restrict__
#else
#define NINFER_FP8_TMA_DESCRIPTOR_PARAM const __grid_constant__ Fp8TmaDescriptors
#endif
```

On Windows the kernel therefore reads the map through a pointer into a `cudaMallocAsync` device buffer
copied H2D with `cudaMemcpyAsync` — i.e. **`.global` space**, the case the quote above singles out. On
non-Windows the map is a `__grid_constant__` by-value parameter, i.e. **`.param` space**, which is the
placement the Programming Guide calls *recommended* and which CUTLASS uses (§4.3 below).

**Status.** That the requirement is documented and that the tree does not emit it is
**[measured here]**. That omitting it *causes* `cudaErrorIllegalInstruction` on this target is
**not documented anywhere I found** — PTX says "when needed", not "must", and a violation is undefined
behaviour, which by definition can manifest as anything. **[inferred]**, and explicitly not established.
The one thing it does explain structurally is why the fault can be **Windows-only while tracking the TMA
route**: the Windows build changes both the placement of the map *and* the instruction sequence, while
keeping the MMA identical.

### 3.4 Descriptor legality, checked arithmetically against the Driver API's own requirement list

**[read in source]** `fp8_tma_map` (`src/ops/linear/fp8/fp8_a8_tma_mma.cuh:81-103`):

```cpp
dimensions[] = { (uint64_t)k, (uint64_t)rows };
strides[]   = { (uint64_t)k };
box[]       = { (uint32_t)block_k, (uint32_t)block_rows };
steps[]     = { 1, 1 };
swizzle     = block_k == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pointer,
                       dimensions, strides, box, steps,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

Arithmetic against the **[documented by NVIDIA]** requirement list (this is arithmetic over the source,
not a measurement):

| Requirement | This descriptor | Status |
|---|---|---|
| `tensorRank` non-zero, ≤ 5 | 2 | ok |
| `globalDim[i]` non-zero, ≤ 2^32 | `k`, `rows` | ok (runtime) |
| `globalStrides[i]` multiple of 16 B, < 2^40 | `k` bytes | ok iff `k % 16 == 0` — runtime; not checked in the kernel |
| `boxDim[i]` non-zero, ≤ 256 | `block_k ∈ {64,128}`, `block_rows ≤ 256` (static_assert) | ok |
| `boxDim[0]*elementSize` multiple of 16 B | 64 or 128 bytes | ok |
| bounding box inner dim ≤ swizzle size | 128 ≤ 128 / 64 ≤ 64 | ok |
| `elementStrides[i]` non-zero, ≤ 8 | `{1,1}` | ok |
| `globalAddress` 16-byte aligned | `p.x`, `p.codes` from engine allocations | **[not verified here]** |
| `tensorMap` 64-byte aligned | `cudaMallocAsync` block, struct `alignas(128)`, `sizeof == 2*128 == 256` | **[not verified here]** (alignment of `cudaMallocAsync` results not independently confirmed) |

The kernel also checks `cuTensorMapEncodeTiled`'s status and throws on failure
(`fp8_a8_tma_mma.cuh:96-101`), so a malformed descriptor would surface as a host-side
`std::runtime_error("FP8 TMA descriptor: …")`, not as `cudaErrorIllegalInstruction`.

### 3.5 CUTLASS's `TmaDescriptor` alias and its alignment

**[read in source]** CUTLASS `main`, `include/cute/arch/copy_sm90_desc.hpp:291-297`:

```cpp
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
```

So CUTLASS **adds no alignment of its own** when the CUDA headers are present — it inherits
`CUtensorMap`'s. Its own fallback is `alignas(64)`, i.e. **not** 128.

**[measured here]** CUDA 13.3.73 `include/cuda.h:3745-3762`:

```c
/** Tensor map descriptor. Requires compiler support for aligning to 128 bytes. */
#if defined(_MSC_VER)
  #define TENSOR_MAP_ALIGN 64
#else
  #define TENSOR_MAP_ALIGN 128
#endif

typedef struct CUtensorMap_st {
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    alignas(TENSOR_MAP_ALIGN)
#elif __STDC_VERSION__ >= 201112L
    _Alignas(TENSOR_MAP_ALIGN)
#endif
    cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;
```

This corroborates **[reported by a third party]** — CUTLASS maintainer `lsyyy666`, CUTLASS #2906:

> "CUDA toolkit 13.1 is automatically aligning it to 128B which has known issues with MSVC. **CUDA
> toolkit 13.2 will revert the alignment of tensor descriptors back to 64B by default.** Before 13.2,
> you can workaround this issue with `alignas(64)`"

and it is consistent with the tree's own finding in
`docs/research/grid-constant-tma-descriptor-msvc.md`. On this machine (13.3) the MSVC default is
therefore already 64 B **[measured here]**; the tree's `struct alignas(128) Fp8TmaDescriptors`
(`fp8_a8_tma_mma.cuh:19`) raises it back to 128 for the struct, which is *more* aligned, not less.

### 3.6 What CUTLASS does that this tree does not

**[read in source]** `include/cutlass/device_kernel.h:118`:

```cpp
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
```

with `#define CUTLASS_GRID_CONSTANT __grid_constant__` when `__CUDA_ARCH__ >= 700` (lines 41-55).
`Params` contains the `TMA_A`/`TMA_B` copy atoms, so the descriptor lands in **`.param` space** — the
*recommended* placement, and no proxy acquire is needed for first use.

CUTLASS *does* own the acquire for the case where the map lives in global memory or is mutated:
`include/cute/arch/copy_sm90_desc.hpp:455-465`:

```cpp
void tma_descriptor_fence_acquire(TmaDescriptor const* desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" …);
}
```

alongside `tma_descriptor_fence_release()` (`fence.proxy.tensormap::generic.release.gpu;`) and
`tma_descriptor_cp_fence_release()` (`tensormap.cp_fenceproxy.…`). **[not found]**: any CUTLASS issue
mentioning `fence.proxy.tensormap` (`gh search issues "fence.proxy.tensormap" --repo NVIDIA/cutlass`
returns nothing).

---

## Question 4 — does CUTLASS support FP8 TMA GEMM on sm_120?

**Yes, on both `main` and in the 4.x tags, and it is a first-class path rather than a gap.** All
**[read in source]** against `NVIDIA/cutlass@main` unless stated.

### 4.1 `cutlass::arch::Sm120` and its differences from `Sm100`

`include/cutlass/arch/arch.h`:

```cpp
constexpr int sm120_smem_capacity_bytes = 101376;

struct Sm120 {
  static int const kMinComputeCapability = 120;
  static int const kSharedMemoryCapacityBytes = sm120_smem_capacity_bytes;
};
```

No `kTmemCapacityColumns` — consistent with **[documented by NVIDIA]** tcgen05 being absent.

### 4.2 The SM120 TMA collectives

`gh search code "Sm120" --repo NVIDIA/cutlass` (main) returns, among others:

```
include/cutlass/gemm/collective/sm120_mma_tma.hpp
include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp
include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
include/cutlass/gemm/collective/sm120_mma_array_tma.hpp
include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp
include/cutlass/gemm/collective/builders/sm120_mma_builder.inl
include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl
include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp
include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp
```

**FP8 specifically** is `…BlockwiseScaling`. `sm120_mma_tma_blockwise_scaling.hpp` declares
`using DispatchPolicy = MainloopSm120TmaWarpSpecializedBlockwiseScaling<…>`, uses
`cutlass::PipelineTmaAsync<DispatchPolicy::Stages>`, and requires the copy atoms
`cute::SM90_TMA_LOAD` / `cute::SM90_TMA_LOAD_MULTICAST` (lines 189-191) — the same Hopper TMA atoms.
`sm120_mma_tma.hpp` (the non-blockwise variant) is structurally identical.

**FP4** is `sm120_blockscaled_mma_tma.hpp` (`MainloopSm120TmaWarpSpecializedBlockScaled`).

### 4.3 Valid SM120 tile shapes and schedules

**[documented by NVIDIA]** CUTLASS `media/docs/cpp/blackwell_functionality.md`, Table 16:

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
|---|---|---|---|---|---|
| 64x64x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 64x128x128 | Y | N | N | N | same |
| 128x64x128 | Y | N | N | N | same |
| 128x128x128 | Y | N | N | N | same |

with the stated constraints: cluster fixed 1x1x1 (no multicast on GeForce), TN layout only,
`EpilogueScheduleAuto` required, and `KernelTmaWarpSpecializedCooperative` chosen by default under
`KernelScheduleAuto`.

Note the K extent: **every** documented SM120 tile is K=128, which is a multiple of 16 bytes × the
128-byte TMA box. This tree's TMA schedules use `BlockK ∈ {64, 128}`
(`static_assert(BlockK == 64 || BlockK == 128)`, `fp8_schedule.cuh:190`), i.e. the 64-byte-box variant
is not in CUTLASS's published SM120 table **[noted, not claimed as a defect]**.

### 4.4 Is consumer Blackwell treated differently from sm_100? Yes, pervasively.

- **[documented by NVIDIA]** CUTLASS docs: "The NVIDIA Blackwell SM100 architecture used in the
  datacenter products has a different compute capability than the one underpinning NVIDIA Blackwell
  GeForce 50 series GPUs (SM120). As a result, kernels compiled for Blackwell SM100 architecture with
  arch conditional features (using `sm100a`) are not compatible with RTX 50 series GPUs."
- Distinct dispatch policies (`MainloopSm120TmaWarpSpecialized*` vs `MainloopSm100…`), distinct
  builders (`sm120_mma_builder.inl` vs `sm100_mma_*`), distinct epilogues, distinct MMA traits
  (`include/cute/atom/mma_traits_sm120.hpp`, `include/cute/arch/mma_sm120.hpp`).
- Different shared-memory capacity (99 KB vs 232448 B).
- No TMEM, no tcgen05.

### 4.5 Version coverage

**[read in source]** The SM120 TMA collectives and `Sm120` tag are present at `main`. The SM120 FP8/NVFP4
work is documented as "CUTLASS 4.0 has added support for these newly introduced narrow precision GEMMs"
(`blackwell_functionality.md`) and the SM120 block-scaled TMA path was reported broken and fixed in the
4.2–4.4 window (CUTLASS #2906 says v4.3.4 crashes; FlashInfer #2708's resolution comment says "after the
cutlass submodule was updated to 4.4 … seems that now it is working"). **[not found]** I did not
enumerate the exact tag at which each SM120 TMA collective first appeared.

---

## Question 5 — is any of this Windows-specific?

**No positive Windows-specific finding for `cudaErrorIllegalInstruction`.** The evidence cuts the other
way, and the strongest single item is a Linux report of the identical failure mode.

### 5.1 CUTLASS #2728 — illegal instruction inside a TMA copy, on Linux, sm_120a

**[reported by a third party]** [NVIDIA/cutlass#2728](https://github.com/NVIDIA/cutlass/issues/2728),
"[BUG] [Cutlass C++] Illegal instruction on Sm120a with certain shape" — **state: closed**, filed by a
user, **no comments**:

- Arch: **RTX5090 (sm_120a)**, OS: **Ubuntu 24.04.1 LTS**, NVCC 12.8.93.
- `mx_float8_t` / `mx_float4_t` MoE-style grouped GEMM.
- `compute-sanitizer` output:
  ```
  ========= Illegal instruction
  =========     at cute::SM90_TMA_LOAD_2D::copy(...) at copy_sm90_tma.hpp:121
  =========     by thread (0,0,0) in block (0,0,0)
  ```
  …with `collect_mma` frames naming
  `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` and `KernelPtrArrayTmaWarpSpecializedCooperative…Sm120`.
- Host frames place it inside `sglang::…fp4_moe`, i.e. an SM120 TMA grouped GEMM in production use.
- "I ran the same kernel with the same test data on a B200 GPU … and there was no illegal instruction
  error."
- `GemmUniversal::run: cudaGetLastError reports success` / `kernel_launch reports success` immediately
  before the fault.

**What this establishes:** (a) `cudaErrorIllegalInstruction` originating *inside* `cp.async.bulk.tensor`
on sm_120a is a real, reported failure mode; (b) it is **not** Windows-specific; (c) a sanitizer run
**can** name the instruction, unlike the 30-minute run described in this project's brief.
**What it does not establish:** that this tree's fault has the same cause — that kernel is
block-scaled (`mx_float8_t`), a different descriptor geometry, on a different toolkit version.

### 5.2 Documented NVIDIA driver bug in TMA descriptor configuration on Blackwell

**[documented by NVIDIA]** NVIDIA Data Center GPU Driver release notes, 595.71.05 (Linux) / 596.36
(Windows) — and the identical text is still present in the **610.57.04 / 610.88** notes
(`docs.nvidia.com/datacenter/tesla/tesla-release-notes-610-57-04/index.html`), under *Known Issues*:

> "Applications using `cuTensorMapEncodeTiled()` or `cuTensorMapEncodeIm2col*()` **on Blackwell GPUs** may
> encounter **sporadic illegal memory accesses (IMA) or MMU faults (like XID 13)** due to driver
> incorrectly configuring TMA descriptors when the tensor's backing memory allocation is less than 128KB
> and it is not a dense non-overlapping tensor. To workaround this issue, add the following after
> descriptor creation for such tensors.
> ```
> CUtensorMap tensorMap;
> cuTensorMapEncode*(&tensorMap, ...);   // Create descriptor
> ((uint64_t*)&tensorMap)[1] &= ~((uint64_t)1 << 21);  // Workaround : clear bit 85
> ```
> This workaround is specific to Blackwell's tensor map descriptor layout and may not be portable to
> future GPU architectures. It should be removed once updated drivers with the fix are deployed."

Assessment **[inferred]**:

- It is a real, vendor-acknowledged TMA defect specific to Blackwell, and the workaround is a *host-side*
  post-creation mutation of the descriptor — i.e. a descriptor-level fault.
- Its documented symptom is **IMA / XID 13**, not illegal-instruction, and NVIDIA documents IMA and
  illegal-instruction as **different error codes** (§3.1). So it does not by itself explain this
  project's symptom.
- Its precondition — backing allocation < 128 KB **and** the tensor not dense non-overlapping — is not
  established for this project's operands.
- It is still listed as a known issue at driver **610**. This machine runs **617.14**
  **[measured here]**. **[not found]** any release-notes page for 617.x
  (`docs.nvidia.com/datacenter/tesla/tesla-release-notes-617-14/index.html` → 404), so whether 617.14
  carries the fix is **unknown from primary sources**.

### 5.3 WDDM / TCC: no TMA limitation found

**[documented by NVIDIA]** The same driver release notes carry the only WDDM-adjacent statement found:

> "The default TCC mode in the NVIDIA driver does not support IOMMU-based isolation (necessary for Windows
> features such as DMA protection, kernel DMA guard, virtualization-based security, etc.). The impacted
> GPUs are NVIDIA L40, L40S, L20, L4, and NVIDIA RTX PRO 6000 Blackwell Server Edition."

The RTX 5090 is **not** in that list, and the statement is about IOMMU isolation, not TMA.

**[not found]** — searched and did not find, in any NVIDIA source:
- any statement that TMA, `cp.async.bulk.tensor`, or `tensormap.replace` behaves differently under WDDM
  versus TCCM;
- any GeForce/Windows-specific TMA known issue in the CUDA Toolkit release notes. The only two
  Blackwell-GeForce-adjacent entries there are **[documented by NVIDIA]**: "FP8 matmuls may fail to launch
  on multi-device Blackwell GeForce systems. As a workaround, run a separate process per device.
  [CUB-9487]", and a cuBLAS concurrency issue explicitly scoped to "GPUs with Compute Capability **10.x
  and 11.x**" — note the scoping excludes 12.x;
- any driver version documented to *fix* a TMA problem on RTX 50-series.

### 5.4 The two Windows-only SM120 TMA defects that *are* filed — both misalignment

Both are **[reported by a third party]**, both on **RTX 5090 + Windows 11**, and both are *alignment*
problems, which NVIDIA documents as `cudaErrorMisalignedAddress`, a different code from
`cudaErrorIllegalInstruction`:

- **[NVIDIA/cutlass#2905](https://github.com/NVIDIA/cutlass/issues/2905)** (open) — "[BUG] SM120
  blockwise FP8 GEMM: TMA descriptor misalignment in Params struct". Driver 591.44, CUDA 13.1,
  CUTLASS 4.3.3, Windows 11, bare metal. Symptom: `prefetch.tensormap` → "misaligned address", traced
  to `tma_load_a`/`tma_load_b` landing at `(mod64 = 24)` and `(mod64 = 32)` inside `Params` because the
  members lack `alignas(64)`. Fix proposed: `alignas(64)` on `Params` and on each TMA member.
- **[NVIDIA/cutlass#2906](https://github.com/NVIDIA/cutlass/issues/2906)** (open) — "[BUG] SM120 NVF4
  GEMM (example 79a): misaligned address crash". CUDA 13.1, CUTLASS 4.3.4, Windows 11.
  `Misaligned shared or local address` in `MainloopSm120TmaWarpSpecializedBlockScaled`, plus
  `ldmatrix.sync.aligned` needing 16-byte alignment on the scale-factor smem. A contributor's fork
  benchmarked 174 TFLOPS (8192³) and 316 TFLOPS (16384³) NVF4 BF16 on an RTX 5090 with the fix — a
  concrete data point that **TMA GEMM does run on sm_120a under Windows** once alignment is right.

A third Windows+SM120 report, **[NVIDIA/cutlass#3673](https://github.com/NVIDIA/cutlass/issues/3673)**
(order-dependent wrong register values, RTX 5090, Windows, CUDA 12 stack), was attributed in a comment
to a **ptxas 12.9 miscompile** that the commenter reproduced on **Linux + B200**, "so this is not
Windows-specific".

---

## Question 6 — is there a worked fix anywhere?

### 6.1 Dense FP8 TMA GEMM on sm_120 that runs: **yes** (vLLM)

**[read in source]** vLLM `main`,
`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh` builds a dense
FP8 (`cutlass::float_e4m3_t` × `cutlass::float_e4m3_t`) GEMM on the SM120 CUTLASS collectives:

```cpp
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB,
        LayoutA, AlignmentA, ElementAB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        StageCountAutoCarveout<…>, KernelSchedule, void>::CollectiveOp;
```

with `AlignmentA = AlignmentB = 128 / sizeof_bits<ElementAB>::value` (**= 16 elements for FP8**, i.e. a
128-byte TMA box), `ClusterShape = Shape<_1,_1,_1>`, and these configs:

| Config | KernelSchedule | TileShape |
|---|---|---|
| `sm120_fp8_config_M16` | `KernelTmaWarpSpecializedPingpong` | 16×64×128 |
| `sm120_fp8_config_M32` | `KernelTmaWarpSpecializedPingpong` | 32×64×128 |
| `sm120_fp8_config_M64` | `KernelTmaWarpSpecializedPingpong` | 64×64×128 |
| `sm120_fp8_config_default` | `KernelScheduleAuto` | 128×128×128 |

with the comment "SM120 Cooperative kernel requires Tile M >= 128" and "CUTLASS 3.x on SM120 currently
restricts programmatic multicast (Cluster > 1) … Reverting to 1x1x1 to ensure compilation."

The entry point is `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu`
(`cutlass_scaled_mm_sm120_fp8`), and the arch guard is
`csrc/libtorch_stable/cutlass_extensions/common.hpp:130` — `enable_sm120_family`, which `asm("trap;")`s
unless `__CUDA_ARCH__` is in `[1200, 1300)`.

**This is a shipped, TMA-scheduled, dense-FP8 GEMM targeting sm_120/sm_121** — precisely the capability
this project's TMA kernel claims and cannot execute. The material differences from this tree's kernel are
worth naming **[inferred]**, as facts rather than conclusions:

| | vLLM `scaled_mm_sm120_fp8` | this tree's `fp8_a8_tma_mma_kernel` |
|---|---|---|
| MMA | CUTLASS `SM120::…` warp `mma.sync…kind::f8f6f4` | `mma.sync…kind::mxf8f6f4.block_scale.scale_vec::1X…` with unit scales |
| Descriptor placement | `__grid_constant__` by value (`.param`) | **pointer into a device buffer** (`.global`) on Windows |
| `fence.proxy.tensormap::generic.acquire` | not needed (`.param`) | **absent** (§3.3) |
| A/B layout | A row-major, B column-major (TN) | A = activations, B = weight span; rows/cols ordered as the descriptor's `dimensions[] = {k, rows}` |
| Split-K + producer/consumer warp roles | none | yes (`kSplitWaveCtas`, 32 producer threads) |
| K tile | 128 for every documented config | 64 and 128 |
| Swizzle | CUTLASS/CuTe chosen from the MMA atom | hand-chosen: 128B for `BlockK == 128`, else 64B |

### 6.2 TMA on sm_120 as a shipping default: **yes** (TensorRT-LLM)

**[read in source]** `TensorRT-LLM/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md` —
"`skip_softmax` — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121":

> "This is the sm_120 / sm_121 warp-specialized context FMHA … **This is the default sm_120 / sm_121
> context FMHA — there is no opt-in flag.**"

> "`dma_sync_mma.h` — Producer (`DMA::run`). Issues `cp.async.bulk.tensor.3d.shared::cta.global.tile` for
> Q / K / V into the granular buffers. `DMA::Host::init_params` builds the three `CUtensorMap`
> descriptors with the driver-API `cuTensorMapEncodeTiled`."

> "Blackwell's TMA engine requires the driver-API `cuTensorMapEncodeTiled` (128-byte `CUtensorMap`)
> descriptor — the same form the shipping trtllmGenKernels FMHA uses."

Note the `.3d` form, matching this tree's BF16 kernel
(`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:83`), and note that it is **BF16 attention, not FP8 GEMM**.

### 6.3 Block-scaled NVFP4/FP8 **MoE** TMA on SM120: broken at various times, fixed in specific ways

This is a distinct and better-documented failure history. All **[reported by a third party]**:

- **[vllm#35566](https://github.com/vllm-project/vllm/issues/35566)** — "CUDA illegal memory access in
  MoE layer with MiniMax-M2.5 NVFP4 on Blackwell (SM120)" (RTX PRO 6000, driver 590.48.01, CC 12.0;
  Docker ⇒ Linux). Several commenters, then a close comment: NVFP4 MoE on SM120 works on
  `cu130-nightly` on an **RTX 5090** under torch.compile + CUDA graphs, no IMA; the FlashInfer CUTLASS
  NVFP4 grouped-GEMM path (flashinfer#2708) "has since rolled into vLLM's pinned FlashInfer; probably
  the root cause." One commenter attributes a different, intermittent `illegal memory access` /
  `misaligned address` on sm_120 to a **cuDNN sm_120 kernel** (`cudnn_generated_fort_native_sm120_matMul_…`,
  faulting `STS.128`), fixed in cuDNN 9.23.1.
- **[flashinfer#2708](https://github.com/flashinfer-ai/flashinfer/issues/2708)** (closed) — "SM120 CUTLASS
  FP4 GEMM: missing GDC compile flags cause PDL race condition". Root cause: `enablePDL=true` without
  `-DCUTLASS_ENABLE_GDC_FOR_SM100=1`, so `wait_on_dependent_grids()` compiles to a no-op and dependent
  kernels read stale memory. Affected modules include `gemm_sm120` (SM120 FP8 groupwise) and
  `fp4_gemm_cutlass_sm120`. Fix: PR #2716. Resolution comment: updating the CUTLASS submodule to 4.4
  fixed it.
- **[NVIDIA/cutlass#2906](https://github.com/NVIDIA/cutlass/issues/2906)** — as in §5.4, fixed by
  `alignas(64)` on `Params`/TMA members and `alignas(16)` on the scale-factor smem; benchmarked on RTX
  5090.
- **[vLLM forums, 2026-04-11](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536)**
  — "SM120 (RTX PRO 6000) NVFP4 MoE Performance Report": "**FlashInfer CUTLASS NVFP4 MoE path: All 80 TMA
  Warp Specialized grouped GEMM tactics fail at initialization.** Falls back to slow non-TMA tactics
  producing garbage output (6-7 tok/s) …"; "The native CUTLASS NVFP4 MoE path is broken on SM120 due to
  **TMA WS grouped GEMM initialization failures**"; the working configuration was Marlin W4A16 at
  50.5 tok/s.
- **[NVIDIA/cutlass#2908](https://github.com/NVIDIA/cutlass/issues/2908)** (open) — "02_dump_reg_shmem
  faults on sm_120: Fragment is 2-byte aligned but accessed through a 16-byte-aligned pointer": a
  separate `sm_120` alignment fault, again not illegal-instruction.
- **[NVIDIA/cutlass#3263](https://github.com/NVIDIA/cutlass/issues/3263)** (closed) — "[QST] FP4 Tensor
  Core mma.sync Instruction Unsupported on SM_101 Architecture": an arch-conditional MMA reaching the
  wrong target.

**Reading of this cluster, kept separate from the above:** in every case the mechanism named was
tactic-enumeration failure, alignment, a PDL/compiler flag, a version bug, or a missing kernel — and in
several cases the resolution was that **TMA works once X is fixed**. That is meaningfully different from
"TMA is broken on SM120". The one place TMA tactics were *systematically* unavailable is
`TmaWarpSpecialized` **grouped** GEMM as used for MoE, which is a different collective from this tree's
dense per-tile FP8 GEMM.

### 6.4 Projects with no SM120 TMA story found

- **llama.cpp** — **[not found]**: `gh search code "cp.async.bulk.tensor" --repo ggml-org/llama.cpp`
  returns **zero** files. No TMA in the repository at all.
- **TensorRT-LLM FP8 GEMM on sm_120** — **[not found]**: TRT-LLM's `cp.async.bulk.tensor` occurrences are
  FMHA (`hopper/utils_tma.h`, `warpspec_sm120/`, `xqa/tma.h`), `selectiveScan`, `tinygemm2`, and a
  CuTe-DSL MoE. No FP8 GEMM TMA specialization for sm_120 was found.
- **SGLang FP8 GEMM on sm_120** — **[not found]** as a direct answer: SGLang's `cp.async.bulk.tensor`
  occurrences are `kda_prefill.cu`, `deepseek_v4/wo_a_fused.cuh`, `hicache_tma.cuh`. SGLang does reach
  SM120 TMA GEMM **indirectly** through CUTLASS/FlashInfer — that is how CUTLASS #2728's crash surfaced
  (§5.1) — but no first-party SGLang source file was found declaring an SM120 FP8 TMA GEMM.

---

## What I searched and did **not** find

Stated explicitly, because several of these would have changed conclusions if found.

1. **No NVIDIA document mapping a TMA-constraint violation to `cudaErrorIllegalInstruction`.**
   Searched: CUDA Driver API `group__CUDA__TENSOR__MEMORY.html` (*Returns* list, full requirement list);
   CUDA Runtime API `group__CUDART__TYPES.html` (all error enumerators); PTX ISA 9.4 §9.7.10.28.5
   ("Restriction on Tensor Copy instructions" — the text is "the behavior is undefined" throughout);
   §9.7.10.28.5.2.1 ("a runtime error is raised", unspecified which). PTX ISA documents "undefined
   behaviour", `CUDA_ERROR_INVALID_VALUE`, and "runtime error" for TMA; it never says
   "illegal-instruction". The mapping from `cudaErrorIllegalInstruction` to "an instruction the hardware
   traps on" is my **[inferred]** reading of the enumerator's wording, not a quotation.

2. **No documentation that memcheck (or any compute-sanitizer tool) is unable to name an illegal
   instruction.** I assumed nothing here. What the sources *do* show is that it can: CUTLASS #2728 and
   #3125 both show `compute-sanitizer` printing `Illegal instruction` with a full device frame. So the
   project's 30-minute run that "did not name an instruction" is unexplained by any documentation I
   found. **[not found]** — not explained.

3. **No `sm_120` entry in any `wgmma` or `tcgen05` ISA row.** Searched Table 72 exhaustively (all
   114 `sm_120` occurrences in PTX ISA 9.4, each inspected in context). The sm_120 rows are:
   `tensormap.replace` (+ `.swizzle_atomicity`, + `.field3`), `setmaxnreg`, `multimem.ld_reduce`/
   `.st`/`.red` (+ types, + `.acc::f16`), `cvt` families, `clusterlaunchcontrol.try_cancel`,
   `ldmatrix`, `stmatrix`, `mma` (+ `.kind`/`.block_scale`), `mma.sp`, `add`/`sub`/… packed types,
   `cp.async.bulk.tensor`'s sm_120a restriction text, `mbarrier.*`, the static shared-memory table, and
   the `.target` target list. No `wgmma`, no `tcgen05`.

4. **No Windows/WDDM TMA limitation.** Searched: CUDA Toolkit release notes (13.0 U3, 13.1 U2, 12.9 U2,
   12.8, 12.0, current 13.3 U1) for `TMA`, `tensormap`, `cp.async.bulk`, "tensor map", "Tensor Memory";
   NVIDIA Data Center driver release notes 595 and 610 for `cuTensorMap`; CUTLASS tracker for
   `TMA`/`windows`/`sm120`. Found: one Blackwell TMA **driver** bug (§5.2, symptom = IMA/XID 13, still
   listed at driver 610) and one TCC/IOMMU statement whose affected-GPU list does **not** include
   GeForce RTX 50-series (§5.3). Nothing tying TMA behaviour to WDDM.

5. **No release-notes page for driver 617.x.** `docs.nvidia.com/datacenter/tesla/tesla-release-notes-617-14/index.html`
   → **404**. So the status of the §5.2 `cuTensorMapEncode*` Blackwell bug on this machine's 617.14
   driver is **unknown from primary sources** — it could be fixed, or still present.

6. **No CUTLASS issue about `fence.proxy.tensormap`.** `gh search issues "fence.proxy.tensormap" --repo
   NVIDIA/cutlass` → nothing. CUTLASS *does* implement the acquire
   (`cute::tma_descriptor_fence_acquire`), but no one has filed that omitting it is a problem, and no
   NVIDIA document says what omitting it does on sm_120 specifically.

7. **No report of `cudaErrorIllegalInstruction` from a **Windows** sm_120 TMA kernel.** The two
   Windows SM120 TMA reports (CUTLASS #2905, #2906) are both *misaligned address*, which NVIDIA
   documents as a **distinct** error code. The only illegal-instruction-in-TMA report I found
   (CUTLASS #2728) is **Ubuntu**. This is a negative result, not evidence of absence.

8. **No working FIXP/bricked-Blackwell or engine-level reference for an FP8 TMA GEMM on sm_120 using a
   *pointer-passed, global-memory* descriptor.** Every shipped SM120 TMA path found — vLLM's dense FP8,
   CUTLASS's own SM120 collectives, TensorRT-LLM's FMHA — places the descriptor by value as a
   `__grid_constant__` parameter. I found **no** example of anyone running an sm_120 TMA kernel that
   reads the descriptor out of `.global` memory the way this project's Windows build does.

9. **No `compute-sanitizer` documentation explaining an unnamed illegal instruction**, and no NVIDIA
   statement that WDDM suppresses the device-side frame. Not searched exhaustively; the `sanitizers` /
   `cuda-debugging` skills cover the tool, not this platform behaviour.

10. **Tag-level CUTLASS history not enumerated.** "Read at `main`" is stated throughout; which CUTLASS
    tag first contained each SM120 TMA collective was not determined, beyond the reporter's statement
    that 4.3.4 crashes and 4.4 fixes it (CUTLASS #2906, FlashInfer #2708 resolution).

---

## Provenance of this note

Sources read on **2026-10-01**, all primary:

- **PTX ISA version 9.4** — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
  (§1.3 version, §5.1.7 shared-memory table, §8.8.4, §9.7.10.28.5.1–3, §9.7.10.29, §9.7.15.16–17,
  §9.7.17.5.2, §9.7.21.4, Table 72, §11.1.2 `.target`).
- **CUDA C++ Programming Guide 13.x** — <https://docs.nvidia.com/cuda/cuda-programming-guide/>
  (§4.12, §4.12.2.2, §5.1.2 Table 28, §5.1.3 Table 29).
- **CUDA Driver API 13.4** — <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
  (`cuTensorMapEncodeTiled`).
- **CUDA Runtime API** — <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html>
  (error enumerators).
- **CUDA Toolkit release notes** — <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>,
  archives 13.1.2 / 13.0.3 / 12.9.2 / 12.8.0 / 12.0.1.
- **NVIDIA Data Center GPU Driver release notes** —
  <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-610-57-04/index.html> and the 595 v2.0 PDF.
- **CUTLASS `main`** — `include/cutlass/arch/arch.h`, `include/cutlass/arch/config.h`,
  `include/cute/arch/config.hpp`, `include/cute/arch/copy_sm90_desc.hpp`,
  `include/cutlass/device_kernel.h`, `include/cutlass/gemm/collective/sm120_mma_tma.hpp`,
  `…/sm120_mma_tma_blockwise_scaling.hpp`, `media/docs/cpp/blackwell_functionality.md`,
  `docs.nvidia.com/cutlass/4.6.2/overview.html`.
- **CUTLASS tracker** — issues #2728, #2905, #2906, #2908, #3263, #3673; code search for `Sm120`,
  `TmaDescriptor`, `CUTE_ARCH_TMA_SM120_ENABLED`, `grid_constant`, `cp.async.bulk.tensor`,
  `tensormap`, `fence.proxy.tensormap`.
- **vLLM `main`** — `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu`,
  `…/scaled_mm_sm120_fp8_dispatch.cuh`, `csrc/libtorch_stable/cutlass_extensions/common.hpp`,
  `vllm/platforms/cuda.py`; issues #35566; the vLLM forum report of 2026-04-11.
- **TensorRT-LLM `main`** — `cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md` and the
  `cp.async.bulk.tensor` code search.
- **FlashInfer** — issue #2708; code search for `cp.async.bulk.tensor`.
- **SGLang** — code search for `cp.async.bulk.tensor`.
- **llama.cpp** — code search for `cp.async.bulk.tensor` (zero hits).
- **This tree** — `src/ops/linear/fp8/fp8_a8_tma_mma.cuh`, `fp8_a8_mma_common.cuh`,
  `fp8_schedule.cuh`, `src/ops/common/mma.cuh`, `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu`,
  `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`, `CMakeLists.txt`, `git grep`,
  `nvidia-smi`, `nvcc --version`, and this machine's
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\cuda.h`.
- **Prior in-tree notes** (not re-derived, cited for continuity): `docs/research/grid-constant-tma-descriptor-msvc.md`,
  `docs/research/first-request-transient-and-msvc-tma-evidence.md`.