# Why nobody else hits this: the sm_120 TMA fault is a bug in our reproducer

Research date **2026-10-01**. Primary sources: NVIDIA PTX ISA and CUDA Driver API reference; CUTLASS,
vLLM and TensorRT-LLM source at `main`; those projects' own issue trackers; and NVIDIA's driver
release notes. Local measurements were taken on the machine described below.

Machine **[measured here]**:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, `nvidia-smi` `compute_cap = 12.0`, **WDDM** |
| Driver | **617.14**, "Release 615 Driver for Windows", `KMD 617.14`, `CUDA UMD 13.4` |
| Toolkit | CUDA **13.3.73** (`nvcc --version`), MSVC 14.51, Windows 11 |
| Build target | `CMakeLists.txt:7` — `set(CMAKE_CUDA_ARCHITECTURES 120a ...)` |

Labels used below: **[documented by NVIDIA]**, **[read in source]**, **[reported by a third party]**,
**[measured here]**, **[inferred]**, **[not found]**.

---

## Answer

**The premise of the question is false, and the reason is measurable.** `cp.async.bulk.tensor` does
not fail on this machine. `tools/scripts/probe_sm120_tma_load.cu` reports "an illegal memory access"
in every arm because **its bounded wait has mis-numbered inline-asm operands**, so
`mbarrier.try_wait.parity` reads its *barrier address* out of the predicate-result register and its
*phase* out of the barrier address. The instruction then dereferences a kernel-parameter value as a
shared-memory address and faults. **No TMA instruction is involved in the fault at all** — a kernel
containing no `cp.async.bulk.tensor` and no tensor map reproduces it exactly.

**[measured here]** With the operand numbering corrected and nothing else changed, TMA runs on
sm_120a in every configuration tried: rank 2 and rank 3, swizzle 128B and NONE, descriptor by value
(`.param`) and by pointer (`.global`), delivering the expected bytes.

So the hypotheses rank as follows.

| Rank | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **1** | **H2 — our probe is wrong** | **CONFIRMED. It is not merely wrong, it is the fault.** | `compute-sanitizer` names `SYNCS.PHASECHK.TRANS64.TRYWAIT`, not `UTMALDG`; a TMA-free kernel reproduces the fault; the corrected numbering runs in 5/5 legal arms. §1, §2 |
| 2 | H1 — they compile but never run it on this hardware | **SPLIT. True for vLLM and TensorRT-LLM. False for CUTLASS.** | CUTLASS example 79a NVFP4 is benchmarked on RTX 5090; vLLM's CI has no sm_120 device. §3 |
| 3 | H4 — GeForce vs Quadro driver behaviour | **NOT SUPPORTED, but one real documented consumer restriction exists** (TMA multicast). Engines fall back on *tactic init failure*, not on illegal instruction. | §5 |
| 4 | H6 — Windows / WDDM | **NOT SUPPORTED.** TMA demonstrably runs on bare-metal Windows 11 on an RTX 5090. | §6 |
| 5 | H3 — geometry or shape constraint on sm_120 | **NOT FOUND.** The probe's descriptor was inside every documented bound; no sm_120-specific limit was found beyond the sm_90-wide swizzle cap. | §4 |
| 6 | H5 — driver 617.14 | **UNRESOLVED AND UNRESOLVABLE from the notes.** The release notes exist and say nothing about TMA, Blackwell, or compute. | §7 |

**The one thing I cannot resolve**: whether the *project's* FP8 TMA kernel, as distinct from the
reproducer, has a real fault. Its Windows dispatch never runs it (§3.3), so nothing in this session
observed it. The `cudaErrorIllegalInstruction` recorded in its comment is a **[reported]** claim from
before this measurement and is now unverified — see §8.

---

## 1. The fault is in the reproducer's wait, not in TMA

### 1.1 What the sanitizer says **[measured here]**

`tools/scripts/probe_sm120_tma_load.cu` built for `-arch=sm_120a -DBY_VALUE=1 -DSWIZZLE_MODE=3` and
run under `--tool memcheck`:

```
========= Out-of-range shared or local address
=========     at tma_probe(CUtensorMap_st, unsigned char *, int)+0x280
=========     by thread (0,0,0) in block (0,0,0)
```

Disassembling that exact binary puts the faulting instruction at `+0x280`:

```
/*01c0*/  SYNCS.ARRIVE.TRANS64 RZ, [UR5], R0 ;          <- mbarrier.arrive.expect_tx
/*01e0*/  UTMALDG.2D [UR4], [UR8] ;                     <- the TMA copy, at +0x1e0
...
/*0280*/  SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [R1+URZ], R2 ;   <- FAULTS HERE
```

`UTMALDG` is at `+0x1e0` and does **not** fault. The instruction at `+0x280` is the mbarrier wait, and
its address operand `R1` was loaded from `c[0x0][0x37c]` — a **kernel parameter**, i.e. a generic /
global address, used where a shared-window address is required.

This is consistent with the prior note's observation that a 30-minute memcheck run "did not name an
instruction" ([docs/research/sm120-tma-illegal-instruction-evidence.md](sm120-tma-illegal-instruction-evidence.md)
§"What I searched and did not find", item 2): memcheck reports the *kernel* and an offset, and the
offset has to be resolved through `cuobjdump -sass` before the instruction is identifiable.

### 1.2 The operand numbering **[read in source]**

`tools/scripts/probe_sm120_tma_load.cu:126-135`:

```cpp
asm volatile("{\n"
             ".reg .pred p;\n"
             "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], [%1], [%2];\n"
             "selp.b32 %3, 1, 0, p;\n"
             "}\n"
             : "=r"(arrived)                                       // %0
             : "r"(smem_u32(&bar)), "r"(0), "r"(kSuspendTicks)    // %1  %2  %3
             : "memory");
```

The output operand is `%0`, so the inputs are `%1`, `%2`, `%3`. The PTX text instead treats `[%0]` as
the barrier address, `%1` as the phase and `%2` as the suspend hint, and writes the predicate result
into `%3` — an *input* operand. The mapping the assembly actually gets is:

| PTX operand | intended | actually bound to |
|---|---|---|
| `[%0]` barrier address | `smem_u32(&bar)` | `arrived` — the predicate-result register |
| `%1` phase | `0` | `smem_u32(&bar)` |
| `%2` suspend hint | `kSuspendTicks` | `0` |
| `%3` selp destination | `arrived` | `kSuspendTicks` (an input) |

Confirmed in the PTX **[measured here]**:

```
mbarrier.try_wait.parity.shared::cta.b64 p, [%r2], %r9, %r10;
selp.b32 %r11, 1, 0, p;
```

### 1.3 The fault needs no TMA **[measured here]**

Two programs containing **no tensor map and no `cp.async.bulk.tensor`**:

- `mbarrier.init` + `mbarrier.arrive` + the reproducer's wait loop → **`illegal memory access`**
- same, but a single `try_wait` + `selp` not in a loop → **OK**
- same, with the reproducer's wait but a suspend hint of `64` or `0` → **`illegal memory access`**
  (so the hint value is irrelevant; the loop is not either — the operand mapping is)

Sanitizer on the failing one: `Out-of-range shared or local address at probe(unsigned char *)+0x180`,
and `+0x180` is again `SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [R1+URZ], R2`.

### 1.4 Corrected: TMA runs **[measured here]**

Same shape matrix as the reproducer, changing **only** the wait's operand numbering (barrier `%1`,
phase `%2`, hint `%3`, result `%0`). Source byte `i` is `i & 0xFF`, so an untouched tile reads all
zeros and a landed tile reads `0,1,2,3`:

| rank | swizzle | descriptor | result |
|---|---|---|---|
| 2 | 128B | by value (`.param`) | **TMA RAN** — tile `0,1,2,3`, wait satisfied |
| 2 | 128B | by pointer (`.global`, `cudaMalloc` + `cudaMemcpy`) | **TMA RAN** |
| 2 | 128B | by pointer, **with** `fence.proxy.acquire.tensormap::generic.cta` | **TMA RAN** |
| 2 | NONE | by value | **TMA RAN** |
| 3 | 128B | by value | **TMA RAN** |
| 3 | 128B | by pointer | **TMA RAN** |
| 2 | 64B | by value | **ENCODE REJECTED** — `CUDA_ERROR_INVALID_VALUE` |

The rejected arm is not a fault; it is the driver correctly enforcing the documented swizzle cap
(§4): a 1024-byte inner box dimension exceeds the 64-byte swizzle width.

### 1.5 The project's own helper is correctly numbered **[read in source]**, **[measured here]**

`src/ops/common/mbarrier.cuh:16-29`, `cta_mbarrier_wait`, has **no output operand**, so `%0` is the
barrier, `%1` the phase and `%2` the hint — the numbering the PTX text assumes. Compiled to PTX:

```
mbarrier.init.shared::cta.b64    [%r1], %r2;
mbarrier.arrive.shared::cta.b64  _, [%r4];
mbarrier.try_wait.parity.shared::cta.b64 done, [%r4], %r5, %r6;
```

`try_wait` and `arrive` share `%r4`. This is the only `try_wait` in `src/` (`git grep`), so **no
production kernel in this tree carries the defect**. The bug is confined to the reproducer.

---

## 2. What the corrected result implies for each arm the reproducer "ruled out"

The reproducer's header comment claims four causes were varied and ruled out: the by-pointer
descriptor form, the swizzle mode, the missing tensormap proxy fence, and the `expect_tx` ordering.
**[measured here]** None of those was ever under test, because every arm aborted in the wait before
the result was observable. With the wait corrected, all four variants run.

---

## 3. H1 — is it run on this hardware?

The honest answer differs per project.

### 3.1 CUTLASS: yes, on this exact part **[reported by a third party]**, **[read in source]**

CUTLASS #2906 is an RTX 5090 report — GPU "NVIDIA GeForce RTX 5090 (SM 12.0)", CUDA 13.1, **OS:
Windows 11**. It exercises example 79a, whose mainloop is
`MainloopSm120TmaWarpSpecializedBlockScaled` (`sm120_blockscaled_mma_tma.hpp`), i.e. exactly the
SM120 TMA collective. A later comment on that issue records the state after the fix:

> "I tested with `alignas(16)` on RTX 5090 (SM 120a): Build passes; NVF4 GEMM/GEMV tests pass;
> Benchmarks run correctly"

so CUTLASS's SM120 TMA path is not merely compiled — it is benchmarked on an RTX 5090. CUTLASS #2905
is the same hardware class on the same OS (RTX 5090, driver 591.44, CUDA 13.1, Windows 11,
bare-metal) and is an **FP8 blockwise** SM120 TMA GEMM (example 87a,
`MainloopSm120TmaWarpSpecializedBlockwiseScaling`) misaligned at `prefetch.tensormap`.

CUTLASS's public CI, by contrast, runs nothing. **[read in source]** `.github/workflows` contains
only `auto-label-issues.yml`, `blossom-ci.yml`, `labeler.yml`, `new-issues-to-triage-projects.yml`
and `stale.yml`. `blossom-ci.yml` is the only job definition; it `runs-on: blossom` (an NVIDIA-internal
self-hosted pool), is gated on an `issue_comment` from a fixed list of ~30 authorised NVIDIA handles,
and then dispatches to an internal CI server. There is no public GPU runner and no public test log.
The unit tests exist (`test/unit/gemm/device/sm120_tensorop_gemm/`,
`sm120_blockscaled_tensorop_gemm/`, `sm120_sparse_tensorop_gemm/`, wired in at
`test/unit/gemm/device/CMakeLists.txt:62-65`) but **[not found]**: any public record of what hardware
runs them.

### 3.2 vLLM: compiled for sm_120, no sm_120 in CI **[read in source]**

`CMakeLists.txt` builds the SM120 CUTLASS sources for arch `12.0f` when CUDA ≥ 13.0, else
`12.0a;12.1a`, intersected against `CUDA_ARCHS`:

```cmake
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()
```

`CUDA_ARCHS` derives from the build machine's gencode flags (`CMakeLists.txt:225-233`), so a wheel
built for the release arch set contains the sm_120 kernels. But **[measured here]** the complete set
of `device:` values across all 33 files in `.buildkite/test_areas/*.yaml` is:

```
a100  b200-k8s  l4  h100  h200  h200_18gb  h200_35gb  dgx-spark
mi250_1  mi250_2  mi250_4  mi300_1  mi300_2  mi300_4  mi300_8
mi355_1  mi355_2  mi355_4  mi355_8  mi355_dpx
cpu-small  cpu-medium
```

**No sm_120 device.** `dgx-spark` is GB10 = **sm_121**, not sm_120 — a distinct compute capability
with distinct shared-memory capacity (CUTLASS #3144: "breaks SM121 (99 KiB vs SM120 228 KiB)"), and
TensorRT-LLM's own source confirms the mapping ("loads correct data on GB10 (sm_121)"). So
`scaled_mm_sm120_fp8.cu` is built and shipped but **not exercised by vLLM's CI on sm_120 hardware**.

### 3.3 TensorRT-LLM: compiled, dispatched by default, no execution evidence **[read in source]**

`cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md`: "This is the **default** sm_120 / sm_121
context FMHA — there is no opt-in flag", compiled into `_context_attention_kernels_120` under
`TLLM_ENABLE_SKIP_SOFTMAX_SM120`. **[not found]**: any benchmark, issue, or CI record showing that
kernel executing on an sm_120 consumer part.

### 3.4 This tree: which routes actually execute **[read in source]**

This matters for interpreting the rest. **[measured here]** by reading the dispatch:

- **BF16 and NVFP4 TMA routes run on Windows.** `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:101` and
  `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:151` take the descriptor as a by-value `__grid_constant__`
  parameter, and neither is `#ifdef _WIN32`-gated. Commit `d3742bbd` measured the BF16 route
  completing on this machine.
- **The FP8 A8 TMA route does not run on Windows.** `src/ops/linear/fp8/shapes/n5120_k6144.cu:34`
  and its four siblings select the pre-merge non-TMA `launch_fp8_a8` under `#ifdef _WIN32`; only
  the `#else` branch calls `launch_fp8_a8_tma`. `_WIN32` appears in `src/ops/linear/` **only** in
  `fp8/` — the five shape files and `fp8_a8_tma_mma.cuh`.

So this tree has never executed its FP8 TMA kernel on Windows, and its two ungated TMA routes are the
ones that work.

---

## 4. H3 — geometry or shape constraints on sm_120

### 4.1 Documented `cuTensorMapEncodeTiled` requirements **[documented by NVIDIA]**

From the CUDA Driver API, *Tensor Map Object Management*:

- `tensorMap` address must be aligned to **64 bytes**; `globalAddress` to **16 bytes**.
- `tensorRank` non-zero, ≤ 5 (≥ 3 if `interleave != NONE`).
- `globalDim[i]` non-zero, ≤ 2^32.
- `globalStrides[i]` a **multiple of 16** and < 2^40.
- `boxDim[i]` non-zero and **≤ 256**.
- `elementStrides[i]` non-zero and ≤ 8; dim-0 entry ignored when `interleave` is `NONE`.
- "When interleave is `CU_TENSOR_MAP_INTERLEAVE_NONE` and swizzle is not
  `CU_TENSOR_MAP_SWIZZLE_NONE`, the bounding box inner dimension (computed as `boxDim[0]` multiplied
  by element size) must be **less than or equal to the swizzle size**. `CU_TENSOR_MAP_SWIZZLE_32B`
  requires ≤ 32; `64B` ≤ 64; `128B*` ≤ 128."

### 4.2 The reproducer's descriptor is inside every one of them **[measured here]**

`probe_sm120_tma_load.cu:170-182`: rank 2, `dimensions {128, 32}`, `strides {128}`, `box {128, 32}`,
`elementStrides {1, 1}`, `INTERLEAVE_NONE`, swizzle 128B, `L2_PROMOTION_NONE`, `OOB_FILL_NONE`,
`UINT8`. `boxDim[0] * 1 byte = 128 ≤ 128`. The map address is 64-byte aligned (the program asserts
it and exits 2 otherwise). The driver accepted it — the program prints "descriptor encoded OK".

The one arm that *was* rejected by the driver in my corrected matrix (rank 2, 64B swizzle,
1024-byte inner box) confirms the cap is enforced rather than merely documented.

### 4.3 The only sm_120-specific restriction in the ISA **[documented by NVIDIA]**

PTX ISA, §9.7.10.28.5.1, *Restriction on Tensor Copy instructions*, verbatim:

> Following are the restrictions for `sm_120a`: `cp.async.bulk.tensor` with the direction
> `.shared::cluster.global` doesn't support: the sub-byte types; the qualifier `.swizzle_atomicity`

It constrains only the `.shared::cluster` (multicast) destination. The tree and the reproducer both
emit `.shared::cta.global`, so it does not apply. **[not found]** any sm_120-specific box-size,
transaction-byte, tensor-extent, or shared-memory-destination-size limit. CUTLASS asserts the same
generic bounds (`copy_traits_sm90_tma.hpp:957-1024`: `boxDim ≤ 2^8`, strides multiple of 16B, extent
≤ 2^32) and adds none for sm_120.

---

## 5. H4 — GeForce vs Quadro

### 5.1 One documented consumer restriction, and our probe satisfies it **[read in source]**

CUTLASS `examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu:36-51`, NVIDIA's
own header:

> "This kernel is optimized for the GeForce RTX 50 series GPUs. […] **Note that GeForce RTX 50 series
> GPUs do not support: 1. Multicast feature of TMA load. Cluster shape has to be 1x1x1. 2. Dynamic
> datatypes.**"

The same restriction is enforced in the collective builder:
`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:84`,
`static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch")`.

**H4 does not explain our fault**: the reproducer launches `<<<1, 128>>>` with the default
single-CTA cluster, i.e. 1×1×1, and the corrected version runs (§1.4). The restriction is real,
documented, and satisfied here.

### 5.2 No capability check distinguishes GeForce from Quadro **[not found]**

Searched CUTLASS, vLLM and TensorRT-LLM for a `cudaDeviceProp` / `cudaDeviceGetAttribute` field
consulted before selecting a TMA path, and for "fallback" or "disable tma" logic keyed on consumer
parts. There is no such field in the CUDA runtime's `cudaDeviceAttr` enumeration either — TMA support
is not reported by any device attribute; it is expressed only as a compute-capability ≥ 9.0
requirement at descriptor creation. All capability gating found is on **compute-capability family**
(`is_device_capability_family(100)` / `(120)`, and `supported_major_versions=[10]` in FlashInfer's
JIT filters), not on GeForce-versus-datacenter **[read in source]**, via CUTLASS #3096's account of
those filters.

### 5.3 Engines *do* fall back on consumer Blackwell — but on tactic-init failure **[reported by a third party]**

CUTLASS #3096 (4× RTX PRO 6000 Blackwell, driver 582.16, Windows 11 + WSL2) is the most complete
third-party account of an SM120 TMA path failing on consumer silicon. Its autotuner log:

```
[TensorRT-LLM][ERROR] Failed to initialize cutlass TMA WS grouped gemm.
Error: Error Internal (cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group2.generated.cu:60)
```

with the reporter's summary: "**All TMA warp-specialized grouped GEMM tactics fail** to initialize on
SM120 with `compute_120a`. The autotuner falls back to slower, non-TMA tactics." Measured outcome:
**6-7 tok/s** non-TMA fallback, versus **46-49 tok/s** for Marlin W4A16. So a shipping engine
disabled TMA on this hardware class and lost roughly 7× throughput — but the trigger was
**tactic initialisation**, not an illegal instruction.

That issue also reports the `compute_120a` → `compute_120f` switch (with CUDA 13.0) enabling the fast
tactics, reaching 39.0 tok/s **[reported]**. **[measured here]** — and this bears on how much weight
that observation carries: in **CUDA 13.3**, `__CUDA_ARCH_FEAT_SM120_ALL` is defined for `-arch=sm_120a`
and **not** for `-arch=sm_120f`:

| `-arch` | `__CUDA_ARCH__` | `__CUDA_ARCH_SPECIFIC__` | `__CUDA_ARCH_FAMILY_SPECIFIC__` | `__CUDA_ARCH_FEAT_SM120_ALL` |
|---|---|---|---|---|
| `sm_120` | 1200 | – | – | not defined |
| `sm_120a` | 1200 | 1200 | 1200 | **defined** |
| `sm_120f` | 1200 | – | 1200 | not defined |

Since CUTLASS gates `CUTLASS_ARCH_MMA_SM120A_ENABLED` on `__CUDA_ARCH_FEAT_SM120_ALL`
(`include/cutlass/arch/config.h`), `-arch=sm_120f` does **not** enable the SM120A feature set. vLLM's
current CMake default for CUDA ≥ 13.0 is `12.0f`, so vLLM and this tree (`120a`) are compiling the
same SM120 collectives under **different CUTLASS feature macros** **[measured here]**. That is worth
knowing, but it did not cause our fault: the corrected probe runs under `-arch=sm_120a` (§1.4).

---

## 6. H6 — Windows / WDDM

**Not supported.** **[read in source]**, **[reported by a third party]**: CUTLASS #2906 records
NVFP4 GEMM tests passing and benchmarks running on an RTX 5090 on **Windows 11**; #2905 is a bare-metal
Windows 11 FP8 blockwise SM120 TMA GEMM. The SM120 TMA unit executes on bare-metal Windows on this
part. **[measured here]**: so does it in this tree — the BF16 and NVFP4 routes are ungated and BF16
was measured completing (commit `d3742bbd`), and the corrected reproducer runs on WDDM.

The prior note's Ubuntu-vs-Windows argument (CUTLASS #2728 reproduces on Ubuntu) stands unchanged,
and it is now moot: the fault was never platform-specific.

---

## 7. H5 — driver 617.14

**Release notes found.** The previous pass concluded "no release-notes page for driver 617.x"
because `docs.nvidia.com/datacenter/tesla/tesla-release-notes-617-14/` returns 404 — but 617.x is a
**GeForce Release 615** branch, not a datacenter branch, so the datacenter URL was the wrong place to
look. The GeForce notes are a PDF at
`https://us.download.nvidia.com/Windows/617.14/617.14-win11-win10-release-notes.pdf`
(`RN-08399-617.14_v01`, "Release 615 Driver for Windows, Version 617.14", dated **September 22,
2026**) **[measured here]**, cross-checked against NVIDIA's driver page
`nvidia.com/en-us/geforce/drivers/details/279803/`.

Extracted contents **[measured here]**:

- Supported products list "NVIDIA GeForce RTX 5090 … NVIDIA Blackwell architecture" — this is the
  driver this machine runs.
- §2.5 "What's New in Release 615": "**Support for CUDA 13.4**". The machine reports `CUDA UMD
  13.4`.
- §3.1 "Fixed Issues in Version 617.14 WHQL" contains **three** entries in total: two display bugs
  (HDMI no-picture on a Samsung Odyssey G95NC after R615; DisplayPort wake failure after R615) and
  one game bug (Judgement/Lost Judgement/Virtua Fighter 5 R.E.V.O. launch after 616.56). §3.2 lists
  one open issue (Assassin's Creed Shadows).
- Searched the full extracted text for `TMA`, `PTX`, `Tensor Memory`, `UTMALDG`, `cuTensorMap`,
  `Deprecated`: **zero** occurrences. `CUDA` appears twice, both times the CUDA 13.4 support line.
- §2.10 "Limitations in This Release" covers NVENC, video colour settings, OpenCL 3.0, HDR, SLI,
  DisplayPort/HDMI — no compute content.

**[not found]** any statement about TMA, Blackwell compute, or GeForce compute behaviour in the 617.x
notes. Consequently the status of the `cuTensorMapEncode*` Blackwell driver bug recorded in
`sm120-tma-illegal-instruction-evidence.md` §5.2 (symptom IMA / XID 13, listed in the 595 and 610
*datacenter* notes) **cannot be determined for this driver from primary sources**: the bug is
documented on a branch this driver is not part of, and this branch's notes carry no compute section.
Note also that 617.14 is the CUDA **13.4** driver while this tree builds with 13.3 — **[measured
here]**, a toolkit/UMD skew, though a toolkit runs fine against a newer UMD by NVIDIA's compatibility
policy.

---

## 8. What remains unresolved, and what is now false

### 8.1 Unresolved

1. **Does the project's FP8 TMA kernel have a real fault?** Not observed in this session: its Windows
   dispatch never selects it (§3.4), so the corrected-reproducer result says nothing about it. The
   `cudaErrorIllegalInstruction` in `n5120_k6144.cu:36` is a **[reported]** claim whose observation
   conditions are not recorded, and it now sits alongside a measurement showing that the standalone
   program used to support it faults for a different reason. It needs its own reproduction, not an
   inference from the reproducer.
2. **Whether `compute_120a` vs `compute_120f` matters on CUDA 13.3** for the SM120 TMA collectives.
   §5.3 measures that the two suffixes enable different CUTLASS feature sets, which is a fact about
   the macros; it does not establish a behavioural difference on this hardware. The only evidence for
   one is CUTLASS #3096, confounded by a CUDA version change (12.8 → 13.0) at the same time.
3. **Driver 617.14's TMA status** — §7. Unanswerable from the notes.

### 8.2 Now false, and previously load-bearing

- "A minimal `cp.async.bulk.tensor` kernel on sm_120a dies with an illegal memory access in **every**
  configuration tried." The configuration set was not exercised; the wait aborted each arm first.
  **[measured here]** TMA runs in every legal configuration tried.
- "Descriptor by value vs by pointer" — **[measured here]** both run.
- "Swizzle 128B vs NONE" — **[measured here]** both run.
- "`fence.proxy` present vs absent" — **[measured here]** both run; and CuTe's
  `tma_descriptor_fence_acquire` spelling and the reproducer's
  `fence.proxy.acquire.tensormap::generic.cta` compile to **identical SASS** **[measured here]**
  (`LDCU.64` / `DEPBAR` / `CCTL.E.C.LDCU.IV.DEEP` / `UTMACCTL.IV` / `UTMACMDFLUSH`).
- "`mbarrier.expect_tx` issued before vs after the copy" — **[measured here]** both orders run.
- "`.tile` is required on sm_120" (from TensorRT-LLM's comment) versus CuTe omitting it: both
  assemble to `UTMALDG.2D` **[measured here]**, so the qualifier is not the variable.
- The `.2d`-versus-`.3d` lead recorded in commit `d3742bbd`: **[measured here]** both ranks run. That
  hypothesis is dead, and it should be retired rather than pursued.
- The 2d-vs-3d and 128B-swizzle notes in `probe_sm120_tma_load.cu`'s header now describe a harness
  bug as a platform finding.

### 8.3 Also searched, and not found

1. **No `cp.async.bulk.tensor` in `NVIDIA/cuda-samples`** — zero hits for `cp.async.bulk.tensor` or
   `cuTensorMapEncodeTiled` across the whole repository, and no TMA directory under
   `cpp/3_CUDA_Features`. So there is no NVIDIA-shipped minimal TMA example to compare against; the
   closest verified references are CUTLASS `test/unit/pipeline/pipeline_tma_async.cu` (SM90,
   pipeline-only) and the collective builders.
2. **No public record of what hardware CUTLASS's CI runs.** Its only workflow is internal
   (§3.1). The SM120 unit tests exist in the tree; whether they are executed, and where, is not
   public.
3. **No `cudaDeviceAttr` for TMA**, in the CUDA runtime reference, so no engine can query TMA
   support at runtime (§5.2).
4. **No NVIDIA document mapping a TMA-constraint violation to `cudaErrorIllegalInstruction`** — this
   carries over from the prior note and is unchanged. It is what made the illegal-instruction reading
   of the reproducer's output plausible in the first place.

### 8.4 Searched and found, beyond the hypotheses

- `examples/79_blackwell_geforce_gemm/` and `87_blackwell_geforce_gemm_blockwise/` are NVIDIA's own
  **consumer-Blackwell** examples, i.e. GeForce is a first-class CUTLASS target with its own
  documented restriction list, not a gap (§5.1).
- `sm120_mma_builder.inl` and `sm120_common.inl` reveal that the SM120 mainloop's shared-memory atoms
  come from `sm120_rr_smem_selector`, and that its TMA gmem atoms are built by
  `sm90_cluster_shape_to_tma_atom` — the SM90 TMA atom, unmodified **[read in source]**. There is no
  SM120-specific TMA copy path; the Hopper one is reused with `CUTE_ARCH_TMA_SM120_ENABLED` swapping
  `.shared::cluster` for `.shared::cta` in `copy_sm90_tma.hpp:115-131`.
- `CUDA_SUPPORTED_ARCHS` in vLLM's `CMakeLists.txt:120-129` includes `12.0`, and `12.1` from CUDA 12.9.

---

## Provenance

**Measured here, 2026-10-01**, all on the machine in the header table, one process per arm (a fault
poisons the CUDA context), CUDA 13.3.73 / MSVC, `-arch=sm_120a`:

- `tools/scripts/probe_sm120_tma_load.cu` under `compute-sanitizer --tool memcheck`, and its SASS and
  PTX via `cuobjdump`.
- `fenceproxy` / `.tile` / wait-form assembly comparisons, via `nvcc -ptx -cubin` + `cuobjdump`.
- The `__CUDA_ARCH_*` macro table, via `nvcc -arch=sm_{120,120a,120f}` and a device-side print.
- The rank × swizzle × by-value/by-pointer matrix with the wait's operands corrected, and the
  TMA-free controls.
- `cuTensorMapEncodeTiled` arithmetic against the Driver API requirement list.
- Driver 617.14 release notes, downloaded and text-extracted from NVIDIA's own PDF.

**Read in source** (2026-10-01): CUTLASS `main` — `arch/config.h`, `detail/helper_macros.hpp`,
`cute/arch/config.hpp`, `cute/arch/copy_sm90_tma.hpp`, `cute/arch/copy_sm90_desc.hpp`,
`cute/atom/copy_traits_sm90_tma.hpp`, `gemm/collective/builders/sm120_mma_builder.inl`,
`sm120_common.inl`, `gemm/collective/sm120_blockscaled_mma_tma.hpp`, `sm120_mma_tma.hpp`,
`sm120_mma_array_tma.hpp`, `examples/79_blackwell_geforce_gemm/79a_…cu`,
`test/unit/pipeline/pipeline_tma_async.cu`, `.github/workflows/`. vLLM `main` — `CMakeLists.txt`,
`.buildkite/test_areas/*.yaml`, `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/`.
TensorRT-LLM `main` — `warpspec_sm120/README.md`, `dma_sync_mma.h`. This tree —
`tools/scripts/probe_sm120_tma_load.cu`, `src/ops/common/mbarrier.cuh`,
`src/ops/linear/{bf16,nvfp4,fp8}/*`, `src/ops/linear/fp8/shapes/*.cu`.

**Documentation**: NVIDIA PTX ISA (cp.async.bulk.tensor target notes, the sm_120a restriction list,
`fence.proxy` examples, `.load_mode` defaulting to `.tile`); CUDA Driver API `cuTensorMapEncodeTiled`;
CUDA runtime `cudaDeviceAttr` enumeration; driver release notes `RN-08399-617.14_v01`.

**Issue trackers**: CUTLASS #2728, #2905, #2906, #3096, #3144; vLLM issue search on
`sm120 5090`; NVIDIA `cuda-samples` code search.

**Not consulted for a conclusion**: any secondary write-up. No external benchmark number in this
document comes from anywhere but NVIDIA's own release notes or an NVIDIA-maintained issue tracker.
