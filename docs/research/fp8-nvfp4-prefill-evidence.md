# FP8 against NVFP4 prefill on one RTX 5090: what the 13-14% is, and what would settle it

**RESEARCH NOTE 2026-10-07.** Written for one decision. Importing the source checkpoint's FP8
attention and GDN words instead of re-encoding them to NVFP4 is worth **1.41-2.35% perplexity** and
**29-37% relative DFlash2 acceptance** on this port, and costs **13.4-14.2% prefill throughput** and
**3.16-3.86 GB** of weights (`attention-topology-2026-10-07.md`). Is that prefill cost inherent to
8-bit weights on this part, or an artifact of the port's own kernels? No GPU job was run for this
note; the arithmetic below is computed from stated constants and labelled as arithmetic.

| label | meaning |
|---|---|
| **[measured]** | someone ran it and published a figure; hardware, shape and workload are quoted with it |
| **[read in source]** | read in the owning project's code, config or first-party document |
| **[third party]** | an independent measurement or a vendor-run benchmark, quoted with its setup |
| **[in-tree]** | this repo's own recorded measurement, cited to the document that holds it |
| **[arithmetic]** | computed here from the constants named in the line |
| **[not found]** | searched, did not find — stated as a negative, not padded |

## 0. The short answer

1. **The bytes do not explain it.** One pp2048 pass reads ~15.9 GB of text weights in a window whose
   DRAM budget is ~333 GB; the FP8 variant's extra 3.16 GB is **0.95%** of that window
   **[arithmetic]**. Both arms' TMA routes rasterise token-fastest, which is the choice that stops a
   weight being re-read once per token tile (`nvfp4_a4_tma.cuh:120-133`;
   `fp8_schedule.cuh:181`; upstream `fp8-a8-tma-route.md` §5.3) **[read in source]**.
2. **The tensor-rate ratio does explain most of it.** On GB202 an NVFP4 block-scaled MMA is
   `m16n8k64` and an FP8 block-scaled MMA is `m16n8k32` — twice the K per issue for the 4-bit one
   **[read in source]** — and NVIDIA's own whitepaper publishes FP4 dense at 2x FP8 dense
   **[read in source]**. Moving 28.2% of a pass's FLOPs onto a half-rate path adds **17.6 ms to a
   GEMM-only pass of 70.9 ms at peak, +24.9%**; the measured whole-pass time delta of +15.4 to
   +16.5% is that same delta on a 185.8 ms pass, the ~61 ms of non-rate-scaled work diluting the
   ratio **[arithmetic]**.
3. **The port's FP8 route is not the suspect.** The one measured figure for these sites — upstream's
   own MXFP8 fused-projection throughput, 660-712 TFLOP/s at T=1024 — is **79-85% of the 838
   TFLOP/s FP8 peak**, against this port's NVFP4 A4 `linear` at 882-985 TFLOP/s, **53-59% of the
   1676 FP4 peak** **[measured, upstream / in-tree]**. The 8-bit route is the *better* utilised of
   the two relative to its own ceiling. What remains is a 2.4-2.7x gap between the two Op-level
   figures and the model-level delta, which is §1.5 and measurement 1.

## 1. The arithmetic

### 1.1 What moves, site by site

Text stack of Qwen3.8-27B (hidden 5120, 64 layers, MLP width 17408, 16 full-attention layers and 48
GDN layers; `qwen3.8-27b-artifact.md` §2, §5.2-5.4) **[read in source]**:

| site | layers | shape `[N,K]` | params |
|---|---:|---|---:|
| full-attention input (`q,k,gate,v`) | 16 | `[14336,5120]` | 1.174 B |
| full-attention output | 16 | `[5120,6144]` | 0.503 B |
| GDN input (`q,k,v,z`) | 48 | `[16384,5120]` | 4.027 B |
| GDN output | 48 | `[5120,6144]` | 1.510 B |
| MLP `gate_up` + `down`, layers 0-55 | 56 | `[34816,5120]`, `[5120,17408]` | 14.974 B |
| MLP `gate_up` + `down`, layers 56-63 | 8 | same | 2.139 B |
| output head | 1 | `[248320,5120]` | 1.271 B |

The precision swap moves exactly the 128 attention and GDN parents: **7.214 B parameters, 28.2% of
the 25.598 B parameters that carry FLOPs** (the embedding is a gather and is excluded) **[arithmetic]**.
That is the `159 NVFP4 + 128 FP8` against `287 NVFP4` object split recorded in
`attention-topology-2026-10-07.md` **[in-tree]**.

### 1.2 Bytes per pass, and what the pass has to move

Byte formulas are the port's own registered ones (`linear-benchmark.md` §4): `fp8_e4m3fn_row_bf16`
is `N*(K+2)`, i.e. 1.00039 B/element at K=5120; NVFP4 is 9 B per 16 elements, 0.5625 B/element.

| | shipped (`nvidia`) | `fp8attn` |
|---|---:|---:|
| attention + GDN weights | 4.058 GB | **7.217 GB** |
| MLP 0-55 (NVFP4) | 8.423 GB | 8.423 GB |
| MLP 56-63 + head (FP8) | 3.412 GB | 3.412 GB |
| **weights per pass** | **15.89 GB** | **19.05 GB** |

The delta is **3.159 GB = 2.94 GiB**, against **3.16 GiB measured** for the nvidia pair and
**3.86 GiB** for the unsloth pair — the latter also moves MLP 56-63 (2.139 B params, +0.87 GiB), and
2.94 + 0.87 = 3.81 GiB reconciles with it **[arithmetic against the measured artifact sizes]**.

At pp2048 the shipped line prefills 2048 tokens in **185.8 ms** (11,024 tok/s) and the variant in
**216.5 ms** (9,459 tok/s) **[in-tree]**. The DRAM budget over that window is
`185.8 ms * 1792 GB/s = 332.9 GB` at the spec rate, or **311 GB** at the machine's measured
sustained read of 1674.5 GB/s (`linear-benchmark.md` §4) **[in-tree]**. Weight traffic is therefore
**4.8%** of the pass's DRAM budget, and the FP8-vs-NVFP4 delta is **0.95%** of it — **1.76 ms of a
185.8 ms pass** at 1792 GB/s **[arithmetic]**.

### 1.3 Is the pass bandwidth-bound?

No, in both arms, by the same factor. At M=2048 a GEMM has `2M = 4096` FLOP per FP8 weight byte and
`8192` per NVFP4 weight byte. The machine balance is `838e12/1792e9 = 468` FLOP/B for FP8 and
`1676e12/1792e9 = 935` FLOP/B for NVFP4, so both arms run **8.8x above their own balance point**
**[arithmetic]**. A pass moves ~15-19 GB of weights and tens of GB of activations in a window that
could move 333 GB; DRAM is not the constraint.

The pessimistic form of the byte hypothesis is a per-token-tile weight re-read, the only version that
could matter: at the FP8 route's 256-token tile, 8 tiles at pp2048, an FP8 arm that re-read would
move `8 * 7.217 = 57.7 GB` against an NVFP4 arm's `8 * 4.058 = 32.5 GB`, a delta of **25.3 GB =
14.1 ms = 7.6% of the pass** **[arithmetic]**. Both routes in fact take the token-fastest
rasterisation whose own comment states the counterfactual ("the whole weight matrix is re-read from
memory once per token tile") and which upstream measured as worth 1.7-3.3% on the one geometry that
does not fit the 96 MB L2 **[read in source; in-tree]**. Measurement 2 below falsifies this form
directly.

### 1.4 The tensor-rate ratio, and why it is 2x

| what | value | source |
|---|---:|---|
| GB202 dense FP4, FP32 accumulate | **1676** TFLOP/s | whitepaper Appendix A Table 3 **[read in source]** |
| GB202 dense FP8, FP16 accumulate | **838** TFLOP/s | same table |
| GB202 dense FP8, FP32 accumulate | **419** TFLOP/s | same table |
| whitepaper figure caption | "FP4, double throughput of FP8" | Figure 8 |
| sm_120 NVFP4 block-scaled MMA | `kind::mxf4nvf4...m16n8k64` | `cute/arch/mma_sm120.hpp` **[read in source]** |
| sm_120 FP8 block-scaled MMA | `kind::mxf8f6f4...m16n8k32` | same file |

Same instruction family, twice the K per issue for the 4-bit one; this port uses the block-scaled FP8
form with a unit `UE8M0 0x7f` scale precisely because "on RTX 5090, block-scaled FP8 avoids the plain
form's half-rate FP32 accumulation" (`src/ops/common/mma.cuh:59-76`) **[read in source]**, and its
bench carries 838 for that path and 1676 for NVFP4 (`bench/ops/linear_bench.cu:45-50`;
`README.md:799`) **[in-tree]**.

At peak, the swap costs `2 * 7.214e9 * 2048 / 838e12 - 2 * 7.214e9 * 2048 / 1676e12 = 17.6 ms` on a
pass whose GEMM-only time is 70.9 ms — **+24.9% of the GEMM time** **[arithmetic]**. The measured
whole-pass *time* delta is +15.4 to +16.5% (the 13.4-14.2% throughput drop) **[in-tree]**: the same
17.6 ms, scaled by the achieved efficiency and diluted by the ~61 ms of the pass that is attention,
the GDN recurrence, epilogues and launch gaps rather than rate-scaled GEMM. The scaling is not a free
parameter — at 57% of each peak the delta becomes `17.6/0.57 = 30.9 ms`, against **30.7 ms measured**
for the nvidia pair **[arithmetic against the measured endpoints]**.

### 1.5 The two-term model, and where it disagrees with the Op bench

Fit each pair with `t = 2*P_fp4/n + 2*P_fp8/f + s`, with `n` and `f` the achieved rates and `s` a
fixed per-token term. Requiring `n = 2f` (the peak ratio, both at the same fraction of their own
peak) makes the system exactly determined by the two measured endpoints:

| pair | implied FP4 rate | implied FP8 rate | fixed term |
|---|---:|---:|---:|
| `nvidia` (11,024 -> 9,459 tok/s) | 962 TFLOP/s (57% of 1676) | 481 TFLOP/s (57% of 838) | 30.4 us/token |
| `unsloth` (11,617 -> 10,064 tok/s) | 1,083 TFLOP/s (65% of 1676) | 542 TFLOP/s (65% of 838) | 32.5 us/token |

**[arithmetic]** Two independent artifact pairs, one free parameter each, and the same fixed term.
The identity `1/f - 1/n = 1.04e-15 s/FLOP` also bounds the FP8 side unconditionally: **`f <= 611`
TFLOP/s (73% of 838)** however fast the FP4 side runs.

The disagreement: this port's NVFP4 A4 `linear` measured **882-985 TFLOP/s at T=1024**
(`linear-benchmark.md` §9.12) and upstream's MXFP8 fused projections measured **660-712 TFLOP/s at
T=1024** (`40bfe7dc`) **[measured]**. Those two Op-level rates predict a delta of only
`2*7.214e9*(1/686e12 - 1/934e12) = 5.58 us/token`, i.e. **+6%**, where the model measures +13.3 to
+15.0 us/token — **2.4 to 2.7x more** **[arithmetic]**. So at least one Op-level figure does not
describe in-model behaviour, and which one is measurement 1 below. Two candidate explanations are
recorded rather than resolved: the fused-op figure is a fusion number, not a pure `linear` number,
and the in-model routes at T=2048 may select different tiles than the benches at T=1024 — the FP8
ladders have a 1024-token width floor and per-geometry ceilings above which the older `cp.async`
route runs (`fp8-a8-tma-route.md` §5) **[read in source]**.

## 2. What NVIDIA and the kernel projects measure for FP8 against FP4

| measurement | hardware | shapes | FP4/FP8 |
|---|---|---|---|
| Tensor-core microbenchmark, `tcgen05.mma`, FP32 accumulate | B200 | `m64n8k16` | FP8 3850.6 vs FP4 7700.2 TFLOP/s, both 96% of peak — **2.0x** **[third party, arXiv 2512.02189 Tables VI-VII]** |
| Peak table, dense | B200 / GB300 | — | NVFP4 10 vs FP8 5 PFLOP/s on B200 (**2.0x**), 15 vs 5 on GB300 (**3.0x**) **[read in source, NVIDIA Blackwell Ultra blog]** |
| FLUX.1 FC layers, FP4 against FP8, MHA left at FP8 | RTX 5090 | pipeline, 30 steps | up to **3.1x** on FC layers; 6,680.93 ms (fp8) against 3,852.75 ms (fp4) end to end **[third party, vendor-run, NVIDIA blog 2025-05-14]** |
| vLLM serving, precision step only | B200 | MiniMax-M2.5, 8K/1K | **1.65x** at 22 tok/s/user rising to **2.77x** at 110, at iso-interactivity **[third party, InferenceX 2026-05-22]** |

**CUTLASS publishes no FP8-vs-FP4 comparison for sm_120, and cannot:** example 87 is the FP8
blockwise family, example 79 the NVFP4 family, and every SM120 collective is `_tma` with no non-TMA
counterpart **[read in source; also `sm120-tma-fp8-throughput-evidence.md` §1.1]**. The one
first-party statement in the examples is directional: "MXFP8 MMA has 2x throughput compared to Ada
Tensor Core FP8 MMA" (`79c_...cu` header) **[read in source]**. TensorRT-LLM publishes no isolated
sm_120 FP8 or FP4 GEMM TFLOP/s, and its support table lists NVFP4, MXFP4 and FP8 for
`Blackwell(sm120)` but **not** W4A8 AWQ/GPTQ **[read in source, TRT-LLM quantization docs]**.

## 3. Can an 8-bit site reach 4-bit prefill?

**Not for the same FLOPs on this part**, because the ceiling is the instruction's K extent and not a
tuning parameter: the only `m16n8k64` block-scaled MMA on sm_120 is `mxf4nvf4` (E2M1 x E2M1), and
every operand mix that includes an 8-bit side is `m16n8k32` (§1.4). What the techniques change is the
*fraction of each format's own peak*, and the FP8 route is already closer to its own than the NVFP4
route is to its:

| technique | what the sources report | is it bandwidth or occupancy? |
|---|---|---|
| larger output tile + deeper pipeline (256x128, 4 stages of K=64, 1 CTA/SM, 288 threads) | the route as shipped; costs 16 -> 9 resident warps/SM and is why a width floor exists **[read in source, `fp8-a8-tma-route.md` §2, §5.1]** | occupancy; measured cost, not a win |
| TMA transport at a matched tile | +6.8 to +17.9%, median **+9.2%**, arms not occupancy-identical (96 vs 64 threads/CTA) **[in-tree, `docs/active-work.md` item 12]** | neither; instruction count and parallelism |
| split-K, tail-wave | [14336,5120] T512 **114 us against 203**, T1024 **220 against 307**; split calls reserve 21.25 MiB and add one reduction **[measured, upstream `344d69b8`]** | occupancy (idle SMs in the tail wave) |
| token-fastest rasterisation | 1.7-3.3% on the 178 MB geometry only, ~1.000 where the matrix fits the 96 MB L2 **[measured, upstream `fp8-a8-tma-route.md` §5.3]** | bandwidth (L2 reuse) |
| TMA multicast, >1 CTA clusters | **absent** on sm_120 — cluster forced 1x1x1, `static_assert("no programmatic multicast on this arch")` **[read in source, CUTLASS]** | n/a |
| persistent kernel + configurable stages | TRT-LLM's sm_120 block-scaled builder added a persistent scheduler and a `Stages` parameter (default 4) "for better SM120 occupancy" **[read in source, PR #11502]** | occupancy |
| doing less work | the only lever that changes the 2x: fewer 8-bit FLOPs (§5), or the k64 instruction, which requires E2M1 x E2M1 | n/a |

The shared-memory ceiling is the binding constraint on the tiling axis: sm_120's opt-in limit is
**101,376 B**, against 232,448 on sm_100 and 334,848 on sm_107 **[read in source, CUTLASS
`arch.h`]**, and the FP8 route's 98,816 B sits at 97% of it. CUTLASS's own stage-count formula
allows 3 stages at 128x128x128 but 8 at 32x64x128 **[arithmetic on the published formula]**.

## 4. What a mixed precision stack costs, and what upstream measured

**Proportional, with one width threshold.** Precision is assigned per artifact parent and every site
is its own Op and launch either way, so a mixed stack does not add a launch or a second graph: the
cost tracks the FP8 *fraction of FLOPs* (§1.4). The one non-proportional effect is the FP8 route's
own width bounds — a 1024-token floor below which the older `cp.async` route runs, and a per-geometry
ceiling (4096 on the 5120-row residual shapes) above which it does **[read in source,
`fp8-a8-tma-route.md` §5]**. At the port's default prefill chunk of 1024-2048 the floor is the
relevant bound.

Upstream's own FP8 A8 TMA route, commit by commit, is the measured record for this route family:

| commit | measured |
|---|---|
| `f0c5ed88` "stage the A8 GEMM operands through TMA" | operator **+1.5 to +3.3%** on `[34816,5120]` (178 MB, does not fit L2) and **-0.2 to -0.5%** on `[5120,17408]` through `linear_add`; **end-to-end prefill +2.40% at chunk 1024, +4.95% at 4096, +4.38% at 8192**; the route launches **zero times** in a decode-only kernel census **[measured]** |
| `909fb087` "optimize fp8 34816x5120 with tma split-k" | adopts the unit-scale MXFP8 MMA and adds TMA templates plus tail-wave split-K **[read in source]** |
| `344d69b8` "tune fp8 14336x5120 for mxfp8" | T512 114 against 203 us, T1024 220 against 307 us, paired cold-graph **[measured]** |
| `40bfe7dc` "tune fp8 fused projections with tma split-k" | "Bulk fusion throughput reaches about **660-712 TFLOP/s at T1024**"; partial storage can add up to 21.25 MiB **[measured]** |
| `7f6aafed` "optimize fp8 and k8v4 prefill with split-kv" | 64k prefill **9.0 s -> 7.6 s** (FP8 KV) and 9.1 -> 7.6 (K8V4); model workspace 123.6 -> 243.3 MiB **[measured]** |

The dispatch decisions, and why: the route is selected by a **cost model**, not a flag —
`fp8_a8_tma_cheaper` compares waves needed times work-per-SM, with one empirical constant for this
part, a width floor, and a per-geometry ceiling the model cannot express; the constant is explicitly
"not portable" **[read in source, `fp8-a8-tma-route.md` §4]**. Rasterisation is token-fastest for the
same reason as the NVFP4 route (§1.3). And the topology the port measured is the producer's own:
NVIDIA's shipped AutoQuantize assignment for Qwen3.8-27B is MLP + `lm_head` NVFP4 with both
attention families at FP8 **[read in source, port's `attention-topology` note and ModelOpt recipes]**.

Adjacent, not this question: upstream's `#352`/`#354`/`#356` cover **KV** precision in the attention
kernel, where ncu attributed a 1.33-1.37x FP8-vs-NVFP4 prefill gap to K staging instructions (73%)
and exposed load stalls (27%) — the same shape of conclusion, different operand **[measured,
upstream tracker]**.

## 5. Keeping the quality without the bytes

- **8-bit activations with 4-bit weights (W4A8) does not buy back what was lost**, because the loss
  is weight precision; and on Blackwell a BF16 or FP8 activation operand falls off the FP4 tensor
  cores entirely. NVIDIA measured weight-only NVFP4 **slower than BF16 in 10 of 12 shapes** on vLLM
  (a non-FP4 activation forces the Marlin dequantise-to-BF16 fallback), while W4A4 beats BF16 in 9 of
  12 and was **1.30x** on a 32K/400 prefill shape **[measured, vendor-run, ModelOpt QAD
  announcement 2026-09-16]**. TRT-LLM does not list W4A8 for `Blackwell(sm120)` at all **[read in
  source]**; the nearest sm_120 path is a community Marlin W4A8-FP8 patch with no numbers **[not
  found]**.
- **Better 4-bit scales at 4-bit bytes, measured, and free at runtime.** Local-Hessian block-scale
  selection on Qwen3.5-9B W4A4 cuts the average accuracy drop against BF16 from **5.10 to 3.10
  points** (MSE 3.87, four-over-six 4.75), and the same page states the reason this port's own MSE
  attempt lost: weight-tensor error "does not correlate well with downstream accuracy" **[measured,
  vendor-run, ModelOpt Local-Hessian announcement 2026-09-09; in-tree `docs/perplexity-baseline.md:185`
  for the losing attempt]**. A checkpoint-side change with no runtime cost, and the one 4-bit lever
  the port has not tested with an output-error objective.
- **Selective precision is the published way to buy the quality at nearly FP4's speed.** ModelOpt's
  AutoQuantize scores per-layer sensitivity and solves for the assignment under an effective-bits
  budget; its measured MMLU-vs-budget sweeps on Qwen3.5-2B/9B show "searching over NVFP4, FP8, and
  BF16 matches or beats NVFP4 and BF16 alone" at every budget, and ModelOpt's own guidance is to
  leave attention projections unquantized for higher NVFP4 accuracy **[measured, vendor-run]**. The
  port's experiment already *is* this; what it has not produced is the marginal curve — which sites
  earn the 8 bits.
- **Training is the only measured route to 4-bit bytes with FP8-level task quality.** QAD on
  Qwen3.6-35B-A3B recovers IFBench from -2.6 pp to -0.3 pp and about 40% of MMMU-Pro's deficit, at
  **735 GPU-hours** for 500 iterations on 128 GB200s **[measured, vendor-run]**; the port's own
  QAT/QAD line does not beat importing the producer's FP8 words on its corpus (4.6849 against
  4.6208) **[in-tree]**. A storage trick that gives FP8 quality in 4-bit bytes: **[not found]** —
  every published route to that goal is fewer 8-bit FLOPs or training, not a format.
- Outside corroboration of the port's own number: the fully-W4A4 Qwen3.8-27B study reports
  **+14-19% prefill** against the other published quantized recipes, 17.53 GiB of weights, decode
  within 4% **[measured, third party, arXiv 2609.04098 Tables 1-3, via the port's
  `nvfp4-fp8-technique-survey.md`]**. That is the same 13-14%, on a different stack — the strongest
  evidence that the cost is the format and not this tree.

## 6. The measurements to take next, ranked by how much each changes the decision

**1. The Op-level A/B at the four real geometries, T=2048, both arms in one session.**
`ninfer_linear_bench --qtype fp8 --policy a8 --n {14336,16384,5120} --k {5120,6144,17408} --t 2048`
against its `--qtype nvfp4 --policy a4` twin, interleaved, plus `ninfer_fp8_linear_add_bench` and its
NVFP4 counterpart for the two residual shapes. This is the only measurement that separates "the
format's ceiling" from "this route's deficit": a ratio near 0.5 says no kernel work helps and the
decision is quality-only; a ratio near 0.3 says there is a fixable ~10 points of prefill. It also
resolves §1.5, where the two existing Op-level figures and the model-level delta disagree by 2.4-2.7x.

**2. `ncu` on both arms at `[16384,5120]`, T=2048, reporting DRAM read bytes, tensor-pipe busy,
achieved occupancy and the warp-stall breakdown.** Instrument: `ninfer_linear_bench --profile`, one
public Linear call under ncu (`linear-benchmark.md` §1.2), read with the `ncu-report` skill. It is
the falsifier for the pessimistic byte hypothesis (§1.3): if the FP8 arm reads about one weight pass
and the delta is in issue slots, bytes are dead; if it reads ~8 passes, the byte story is the whole
story after all. The attribution step if measurement 1 shows a gap.

**3. The per-site precision split.** Build three variants that keep FP8 on (a) the full-attention
projections only, (b) the GDN projections only, (c) the two output projections only, and measure
corpus perplexity, DFlash2 acceptance and pp2048 for each. Instrument: `tools.convert` with a user
recipe — the mechanism both existing variants used, so no tracked file changes — plus
`ninfer-perplexity` and `ninfer_bench --spec dflash2 -p 2048`. It produces the exchange rate the
decision needs (prefill percent per point of perplexity) and tests whether the 1.4-2.4% is
concentrated in one family, which would shrink the FP8 fraction and its cost. It changes what ships
even if measurements 1 and 2 say the 13-14% is inherent.

## 7. What was searched and not found

| searched for | result |
|---|---|
| A CUTLASS, vLLM, TensorRT-LLM or FlashInfer **measured** FP8-vs-FP4 GEMM ratio on sm_120, at any shape | **[not found]** — the sm_120 examples exist in both formats (87 FP8, 79 NVFP4) but publish no numbers; every SM120 collective is TMA-only |
| An sm_120 **FP8 GEMM TFLOP/s versus M** table from any first party | **[not found]** |
| A published **W4A8** path on sm_120 with numbers | **[not found]** — TRT-LLM lists W4A8 for sm100/103/107, Hopper and Ada only |
| Any technique that reaches 4-bit **bytes** with 8-bit **weight** quality | **[not found]** — no such format or kernel is published |
| The port's own FP8 A8 Op-level throughput at T=1024 or T=2048 | **[not found in-tree]** — `linear-benchmark.md` §9.12 records the NVFP4 A4 rows only; the FP8 rows were never recorded, which is measurement 1 |

## Sources

- **In-tree**: `docs/maintainer/qwen3.8-27b-artifact.md`, `docs/maintainer/linear-benchmark.md`;
  `docs/research/attention-topology-2026-10-07.md`, `nvfp4-fp8-technique-survey.md`,
  `sm120-tma-fp8-throughput-evidence.md`; `docs/active-work.md` item 12; `bench/ops/linear_bench.cu`;
  `src/ops/common/mma.cuh`; `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh`; `README.md`.
- **Upstream NInfer**: `docs/maintainer/fp8-a8-tma-route.md` at `f0c5ed88`; commits `f0c5ed88`,
  `909fb087`, `344d69b8`, `40bfe7dc`, `7f6aafed`; issues `#352`, `#354`, `#356`.
- **NVIDIA**: *RTX Blackwell GPU Architecture* whitepaper Appendix A Table 3 and Figure 8; CUTLASS
  examples 79b/79c, 87, `include/cute/arch/mma_sm120.hpp`, `include/cutlass/arch/arch.h`;
  TensorRT-LLM quantization and support docs and PR #11502; Model Optimizer announcements —
  AutoQuantize (2026-08-24), Local-Hessian (2026-09-09), QAD/W4A4 (2026-09-16); "NVFP4 for Image
  Generation" (2025-05-14); "Introducing NVFP4" (2025-06-24); "Inside Blackwell Ultra" (2025-08-22).
- **Others**: arXiv 2512.02189 (B200 tensor-core microbenchmark); arXiv 2609.04098 (Minima, via the
  port's survey); arXiv 2405.04532 (QServe, for 4-bit being 2x 8-bit); InferenceX / SemiAnalysis
  B200 NVFP4-vs-FP8, 2026-05-22.
