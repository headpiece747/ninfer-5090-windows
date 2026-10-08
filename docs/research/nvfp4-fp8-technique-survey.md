# NVFP4 and FP8 techniques that could improve these artifacts

Research note, 2026-10-07. Written for one question: **what technique, if adopted, would measurably
improve the shipped Qwen3.8-27B artifacts' quality at the same or lower size?** Primary sources only; no
GPU job was run for this note. Every claim carries its source and one of these labels:

| label | meaning |
|---|---|
| **[measured]** | someone ran it and published a figure; model, benchmark and context are quoted with it |
| **[read in source]** | read in the owning project's code, config or first-party document |
| **[third party]** | an independent measurement, quoted with its setup |
| **[in-tree]** | this repo's own recorded measurement, cited to the document that holds it |
| **[not found]** | searched, did not find — stated as a negative, not padded |

In-tree context: the shipped NVFP4 line (`tools/convert/official_recipes.py:431-498`, `:217-253`) encodes
the MLP and attention/GDN projections to NVFP4 with max-abs activation divisors; `gdn/a_projection` and
`gdn/b_projection` stay BF16 only because `block_scale_k16_m128x4_v1` needs the row dimension padded to
128; norms, GDN convolution, `a_log` and `dt_bias` are BF16; embedding, head and MTP/draft are Q8; vision
is Q4–Q8; KV is FP8.

---

## 1. The format and the block size

**The format.** NVFP4 is E2M1 (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6) with two-level scaling:
`x = x_e2m1 · s_block · s_global`, where `s_block` is an FP8 E4M3 factor per 16 consecutive elements and
`s_global` is one FP32 scalar per tensor; `s_block = (block_amax / 6) / s_global`. **[read in source]**
Transformer Engine docs, *NVFP4* (2.16/2.20); NVIDIA blog, *Introducing NVFP4* (2026-01-08). Training uses
2D 16×16 blocks for weights and 1D 1×16 for activations and gradients; PTQ recipes use 1D block-16 per row.
**[read in source]** TE common-API docs; ModelOpt Qwen3.8-27B recipe. The blog claims 4.5 bits/value and
~3.5× smaller than FP16 — vendor-run.

**Block size, measured.** The one direct sweep found covers six small CNNs (ResNet18, MobileNetV3/V4,
MobileViT, ShuffleNetV2, EfficientNet-Lite0; Tiny-ImageNet; NVFP4 activations) over B ∈ {1…256}. B=16 costs
4.5078 bits/input, B=32 4.2578, B=64 4.1328 — a further ~10% metadata saving for worse accuracy at several
models (ResNet18 drop 3.40 → 4.40 → 4.96; MobileNetV3 5.42 → 5.42 → 5.38; MobileViT 5.82 → 6.32 → 6.12).
B=16 is the practical point; large blocks pay only when metadata dominates. **[measured]** Sen et al.,
arXiv 2606.06527, Table I. Scope: small vision CNNs, not LLMs.

**NVFP4 vs MXFP4 (block 16 vs 32, E4M3 vs E8M0), measured.** An 8B hybrid Mamba-Transformer on 1T tokens:
NVFP4 reaches ~1.5% relative loss error vs BF16, MXFP4 ~2.5%, and MXFP4 needs **36% more tokens** (1.36T
vs 1T) to match NVFP4. **[measured]** arXiv 2509.25149 §5. This is the strongest published case that the
block-16/E4M3 combination is not interchangeable with block-32.

**Where the error sits.** Simulated quantization on Llama-3.1-8B shows NVFP4's error concentrates in values
rounding into the grid's 4–6 gap; scale-factor precision changes little, value precision recovers fully.
**[measured]** *Four Over Six*, arXiv 2512.02010, §4/Fig. 2.

---

## 2. Which sites to keep in higher precision

**NVIDIA's shipped recipe for this exact model** (`nvfp4_w4a4_mlp_fp8_attn_max.yaml` at `main`): MLP
`gate/up/down` weights `nvfp4_static` + inputs `nvfp4` (W4A4); `self_attn.*_proj` and
`linear_attn.*_proj*` weights and inputs FP8; `linear_attn.in_proj_a*`/`in_proj_b*` disabled (BF16);
`lm_head` re-enabled to NVFP4 after the base config disables it. The producer's map is *MLP = 4-bit,
attention/GDN projections = 8-bit, the two GDN gates = 16-bit*. **[read in source]** ModelOpt recipes.

**NVIDIA's training paper, measured layer sensitivity.** A 12B hybrid Mamba-Transformer on 10T tokens:
quantizing every linear layer to NVFP4 diverges; keeping the final few linear layers BF16 restores
stability, and the first blocks only help in combination with the last. The 12B run kept the first two and
last eight blocks BF16 (16% of linear layers); 1.2B ablations show the last four suffice. Embeddings, the
output head, normalization, non-linearities and attention components (softmax, QK and SV batched GEMMs)
stay BF16/FP32; the Mamba-2 output projection stays MXFP8. The measured mechanism: the last layers carry
the largest weight-gradient quantization errors. **[measured]** arXiv 2509.25149 §4.1, App. E.2.

**A production 550B example.** Nemotron 3 Ultra quantizes only MoE routed experts to NVFP4; shared experts
and Mamba mixer linears are FP8 per-tensor; attention linears, latent MoE and Mamba conv1d stay BF16; KV is
FP8. **[read in source]** NVIDIA blog, *Creating the Nemotron 3 Ultra NVFP4 Checkpoint* (2026-06-26).

**Unsloth Dynamic NVFP4**, the source this port imports MLP codes from, keeps "important layers … in FP8
(W8A8) or BF16 and the rest in W4A4". Its measured Qwen3.6-27B table (MMLU-Pro / GPQA / AIME 2025):
Unsloth 86.25 / 86.34 / 93.12; NVIDIA NVFP4 85.96 / 86.87 / 93.12; FP8 86.11 / 86.87 / 93.75; BF16
85.96 / 88.13 / 93.33. **[measured]** Unsloth docs, vendor-run, one serving stack.

**Conventions with a stated reason, not a measurement.** llm-compressor's rule: "Always ignore `lm_head`,
any vision tower layers, and any gating/routing layers"; its FAQ calls `lm_head` "sensitive … will impact
the model's accuracy" **[read in source]**. The only head-isolated *measurement* found is a third-party
shootout of `RadixArk/Qwen3.8-27B-NVFP4` vs its BF16-head sibling: PPL 8.385 vs 8.286, KLD-text 0.0952 vs
0.0592, GSM8K 97.4 vs 97.0, ~12 tok/s and 1.8 GB for the NVFP4 head **[third party]** discussion #192.

### The hybrid half — full attention vs Gated DeltaNet

This is the decisive external result for this model, and it inverts the community's precision map.
**Minima** (arXiv 2609.04098, Sept 2026) quantized **all 496 linear layers** of Qwen3.8-27B to NVFP4 W4A4 —
GDN `qkv`, `z`, `out` and the `a`/`b` gates every other public recipe protects — and measured against BF16
on one RTX PRO 6000 with FP8 KV:

- 5-task average (MMLU-Pro, GSM8K, AIME'25, GPQA-D, LiveCodeBench v6): 85.10 vs BF16 85.62 (−0.52, inside
  seed noise; no pair CI-separated on any task); RULER 100% for all four models at 32K and 64K.
- The two protected gate projections are the **least** sensitive: quantizing `a` and `b` moves layer output
  by 2.1% and 2.6% — the two smallest effects — despite 11.0% and 8.5% GEMM errors, because softplus/exp
  and sigmoid compress the error before it reaches the recurrence. The plain `out`, `qkv`, `z` GEMMs
  contribute the actual error.
- The delta-rule recurrence bounds injected noise at a flat ~12.6% state-error plateau over 32K tokens and
  erases a 1% state impulse within 80–1,382 steps, faster than the decay-gate horizons (44K–62K). The
  weight-quantization gap *shrinks* with position: +0.081 nats in the first half of a 32K window, +0.011
  in the second, −0.053 in the final 2K; PPL gap +0.72 at 4K, +0.49 at 32K.
- Cost: 17.53 GiB weights (vs 50.13 BF16, 20.23 Unsloth, 18.83 RadixArk), +14–19% prefill, decode within
  4% of the other quantized recipes. **[measured]** same paper, Tables 1–3.

**What this means for the port.** The `a`/`b` BF16 exception is a layout constraint, not a quality
decision **[in-tree]** `official_recipes.py:445-446, 222-224`, and the external measurement says these are
the *safest* projections to quantize. The nine BF16 exception parents in the unsloth line were measured
in-tree against DFlash2 acceptance, not model quality **[in-tree]** `official_recipes.py:438-444`,
`swift15-lane-measurement.md`. One serving caveat must travel with any GDN NVFP4 change: llm-compressor
calibrates a global scale per module, and vLLM fuses `in_proj_qkv+z` and `in_proj_b+a` into single GEMMs,
taking the max of the constituent scales. The mismatch (1.82× and 2.75× in every one of the 48 layers)
silently corrupts the gates and *fakes better* long-context perplexity; the repair is a checkpoint-side
scale rewrite. **[measured]** arXiv 2609.04098 §6. The port's fused parents share one activation divisor
for the same reason **[in-tree]** `official_recipes.py:269-283`.

---

## 3. Activation quantization

**How NVFP4 activation scales are obtained.** The per-tensor global scale is static from calibration
(`input_scale = amax / (6·448)` in ModelOpt's export); the per-16 block scales are computed dynamically at
runtime. All three published engines consume the checkpoint's static global scale for NVFP4; TensorRT-LLM
additionally has a `force_dynamic_quantization` branch computing `448·6/amax(input)` per forward, and vLLM
defines a dynamic per-token second-level `QuantKey`. **[read in source]** ModelOpt `nvfp4_tensor.py`;
TRT-LLM `linear.py:1657-1676`; vLLM `quant_utils.py` — recorded with file:line in
`docs/research/activation-scale-method-evidence.md`. NVIDIA's published choices differ: the Qwen3.8-27B
recipe is static max; Nemotron 3 Super used "dynamic max-based scaling for activations"; four-over-six
"falls back to the default NVFP4 on activations". **[read in source]** ModelOpt recipe; NVIDIA blog.

**Calibration corpus and size.** ModelOpt's default is `num_samples=512`, `max_sample_length=512`
(512×512 tokens), left-padded, from `cnn_dailymail` + `nemotron-post-training-dataset-v2`; the Ultra job
calibrated on `nemotron-post-training-dataset-v2`. **[read in source]** `dataset_utils.py:749-750, 998`;
Ultra blog. NVIDIA states PTQ accuracy "is typically robust across different choices of calibration data"
**[read in source]** `hf_ptq/README.md` — a claim about accuracy, not about the resulting amax. The Minima
study calibrated on a frozen **128 samples × 32K tokens** set for a W4A4 model **[measured]** arXiv
2609.04098 §3 — the only NVFP4 calibration recipe found with a published model result attached.

**Calibration size, measured — for PTQ generally, not NVFP4.** Williams & Aletras (ACL 2024; 1,800 models,
four methods, nine LLMs) find "substantial variations in downstream task performance" across calibration
sets from the same dataset, with diminishing returns; D²Quant (arXiv 2602.02546) improves steadily from
16 → 128 samples, best at 128. **[measured]** both. **[not found]** any NVFP4-specific calibration-size
ablation.

**Max-based vs MSE-optimal scale selection.** ModelOpt's `max` is default; `mse` is "an MSE search for
static NVFP4 weight scales, with an FP8-scale sweep over the e4m3 scale values"; activations stay
max-calibrated. **[read in source]** `modelopt_recipes/ptq.md`. The measured comparison is Nemotron 3
Super — MMLU-Pro / GPQA / LiveCodeBench / AA-LCR:

| weight scale | MMLU-Pro | GPQA | LCB | AA-LCR |
|---|---:|---:|---:|---:|
| BF16 | 83.49 | 79.92 | 72.907 | 53.00 |
| max (baseline) | 82.99 | 79.29 | 70.18 | 55.50 |
| per-block MSE | **83.31** | **79.92** | **71.37** | 56.75 |
| output-MSE | 83.05 | 78.98 | 71.00 | **57.06** |
| GPTQ | 83.11 | 80.05 | 69.79 | 57.87 |

**[measured]** NVIDIA blog (2026-06-26), Table 2. On Ultra, four-over-six (per-block choice between an M=6
and M=4 dynamic range, folded into the E4M3 scales) cut the median reconstruction MSE of all 49,152
routed-expert projection weights by **16.4%** vs max and gave "98.5% median recovery relative to BF16,
ahead of max (96.8%) and MSE (98.4%)". Counterweight in the same post: MSE cut per-tensor weight error by
27.1% over four-over-six "yet produced no consistent improvement on downstream benchmarks". **[measured]**
same post.

**This port's own measurement.** The one weight-scale experiment run here moved the NVFP4 block maximum
between 6 and 4 (not an MSE sweep) and measured overall perplexity **+0.36% worse** on the fixed corpus
(1,044,876 tokens, fp8 KV, 4096/2048), with per-domain swings of −2.82% to +3.56% hidden in the aggregate.
**[in-tree]** `docs/research/nvfp4-block-scale-4-vs-6.md`. Activation divisors are max-abs over one forward
pass of the joined corpus; NVIDIA's headroom calibrator (anchor p1 of per-block amaxes, upper p99.99,
`rho=16384`, NVFP4 input quantizers only) is a documented alternative with **no published accuracy
number**. **[read in source]** ModelOpt `nvfp4_act_headroom.py`, PR #2028; **[not found]** an end-to-end
max-vs-headroom accuracy comparison.

---

## 4. QAT for NVFP4

**What the training recipe does (pretraining, not post-hoc QAT).** Every linear GEMM's operands are
fake-quantized to NVFP4; weights use 2D 16×16 blocks for forward/backward consistency (chain-rule
preservation), activations and gradients 1D 1×16; block scales are computed from the current amax, not
learned, so gradients pass through the FP4 cast as a straight-through estimator. Stochastic rounding is
applied **only to gradients** (unbiased there; on weights/activations it increases error and diverges); a
size-16 random Hadamard transform is applied only to Wgrad inputs, with a fixed random sign vector; a
fraction of layers stays higher precision. **[measured]** arXiv 2509.25149 §4, App. E; **[read in
source]** TE NVFP4 docs. Ablations: removing SR, RHT, 2D scaling or the high-precision layers each worsens
convergence; "healing" (forward to BF16 for the last 18%) cuts the loss gap 1.5% → 0.5%. Result: 12B
hybrid, 10T tokens, MMLU-Pro 62.58 vs FP8 62.62. **[measured]** same paper, Table 2.

**QAT for inference recovery, measured.** ModelOpt's QAT quantizes with the same PTQ recipe, then
fine-tunes with the quantized forward pass and high-precision gradients (`QATTrainer`); QAD replaces the
task loss with KL against the frozen BF16 teacher. Learnable-scale variants (LSQ/Dual-LSQ, learnable amax)
exist for QAD. **[read in source]** `examples/llm_qat/README.md`; ModelOpt changelog. Measured recovery
(arXiv 2601.20088):

| model | method | MATH500 | AIME25 | GPQA-D | IFEval-Inst |
|---|---|---:|---:|---:|---:|
| Llama Nemotron Super V1 49B | BF16 | 95.8 | 46.0 | 66.5 | 87.5 |
| | NVFP4 PTQ | 91.4 | 32.3 | 62.1 | 86.9 |
| | NVFP4 QAT | 94.3 | 41.5 | 63.3 | 87.2 |
| | NVFP4 QAD | 94.6 | 45.6 | 64.5 | 87.8 |

On RL-heavy models QAT can *degrade* below PTQ (Nemotron 3 Nano 30B-A3B, AA-LCR/AIME25/GPQA-D/LCB-v5/
SciCode: PTQ 31.3/85.0/71.6/68.9/30.5 → QAT 24.8/83.3/66.0/62.0/25.8 → QAD 34.3/87.9/72.7/68.9/32.3 vs BF16
35.9/89.1/73.0/72.1/33.0), which the authors attribute to QAT acting as an extra post-training stage.
**[measured]** same paper, Tables 1–3. Data: ~0.3B tokens recovered a 49B model, ~2.5B a 30B-A3B; LR
1e-6–1e-5; robust to data quality (even random tokens stay above PTQ). **[measured]** same paper §3.4/§4.1.

**A 27B-class public recipe and checkpoint.** `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` — 19.7 GB, 496/496
transformer linears NVFP4 W4A4 including attention and GDN, one epoch of "loss-aware NVFP4
quantization-aware distillation" against the frozen BF16 teacher (global batch 32, LR 1e-6, 2,446 steps).
Its own measurements: GPQA-Diamond 90.91 vs BF16 91.41, Unsloth 89.39, Inferact 87.63; AIME'26 100.0 vs
100.0; RULER 32K/64K/128K 96.3/96.2/95.8 vs BF16 96.7/96.6/95.9. **[measured]** model card; paper
arXiv 2608.13966. The port already carries a QUASAR-sourced recipe **[in-tree]**
`official_recipes.py:217-253`, so this is an evaluation, not a training, decision.

**Cost to run.** **[not found]** any GPU-hour, wall-clock or dollar figure for QUASAR training; the card
publishes the schedule only. ModelOpt's HF QAT example states a minimum of **2×80 GB GPUs for an 8B** model
and offers QLoRA as the reduced-memory path **[read in source]** `examples/llm_qat/README.md`, so
full-parameter QAT of a 27B on one RTX 5090 is not a documented configuration. Separately, four-over-six +
stochastic rounding is reported as **not** an unbiased backward estimator (Quartet II, arXiv 2601.22813,
replaces it with MS-EDEN) **[third party]**.

---

## 5. FP8 and KV precision

**FP8 weights: per-row vs per-tensor.** NVIDIA's PTQ study recommends per-channel weight scaling
("empirical evidence suggests … reduce rounding errors") but found per-tensor adequate for activations
**[read in source]** arXiv 2309.14592. The end-to-end measurement is weaker: MMLU 5-shot CoT, all four
per-row/per-tensor combinations, differences ≤0.5 points and **not monotone** — Llama-3.1-70B: neither
75.39, weight-only 75.32, both 75.25, activation-only 75.23; Llama-3.1-8B: 62.90 / 62.80 / 62.95 / 62.92
**[measured]** arXiv 2502.01070, App. Table 5. Per-row scaling is not a quality lever at 8 bits.

**FP8 KV, measured on this exact model.** With FP8 KV no task score moved outside seed spread (BF16 and
W4A4 models, four seeds, six suites); capacity grew 1.8–1.9×; RULER stayed 100% at 32K/64K. The one
systematic cost is perplexity at 32K: **+0.13** for the BF16 model and **+0.41** for the W4A4 model —
three times larger, "plausibly because Minima's K/V projections are already W4A4". Adding **calibrated
per-tensor FP8 KV scales** (32 tensors on 16 attention layers) recovers **83%** of it (10.84 → 10.50,
residual +0.07, *below* the BF16 model's own uncalibrated +0.13), throughput unchanged within 0.4%.
**[measured]** arXiv 2609.04098 §7. Unsloth ships calibrated FP8 KV scales with every quant **[read in
source]** Unsloth NVFP4 docs. The port ships FP8 KV **[in-tree]** `docs/research/kv-dtype-evidence.md`;
whether its artifacts carry per-layer KV scales was not checked here.

**4-bit KV (NVFP4).** NVIDIA measured NVFP4 KV vs FP16 and FP8 on Qwen3-480B-A35B: LiveCodeBench ~58% for
all three; MMLU-Pro 77.4 / 78.1 / 78.2; MBPP 79.9 / 79.7 / 80.8; RULER 64K 94.6 / 95.5 / 95.6 (FP16
first). NVFP4 KV is ~50% smaller than FP8 and gives ~2× context budget. **[measured]** NVIDIA blog,
*Optimizing Inference … with NVFP4 KV Cache* (2025-12-08), Figs. 5–6. TensorRT-LLM also supports active
NVFP4 KV and NVFP4 cold-page compression (only cold pages quantized; hot pages stay at the runtime type),
with no accuracy number published. **[read in source]** TRT-LLM `kv-cache-compression.md`,
`quantization.md`.

**The "KV tail" technique.** **[not found]** no technique *named* "KV tail" in the primary literature. The
description matches the **full-precision residual/recent-token window**: KIVI keeps a per-channel K and
per-token V cache at 2 bits but holds the most recent R=128 tokens in FP16 (group size 32). Measured:
KIVI-2 on Llama-2-7B scores GSM8K 12.74 vs 13.50 at 16-bit (and 5.76 with the window removed), CoQA 63.05
vs 63.88, LongBench average 44.27 vs 44.52; the paper states the window "is crucial to maintaining
accuracy for hard generation tasks such as mathematical reasoning". **[measured]** arXiv 2402.02750,
Tables 3–5. The measured critique: KVQuant reports the window "is effective at representing the tail part
of the context, but may provide less benefit for tasks requiring the utilization of the full context
window". **[measured]** arXiv 2401.18079. At 8 bits the trick is largely unnecessary — FP8 KV is already
near-free on tasks — and the window is a 2–4 bit mechanism.

---

## 6. The three most promising for this setup

**1. MSE-optimal (and four-over-six) weight block scales.** Same size, measured gains elsewhere (Nemotron
Super: MMLU-Pro 83.31 vs 82.99 max; four-over-six: 98.5% vs 96.8% median recovery, −16.4% median weight
MSE). **Corrected: the port has already measured this and it lost.** `perplexity-baseline.md:185` records a
per-block MSE scale search on the re-encoded sites — "a searched NVFP4 block scale is worse, and the weight
error says nothing about it" — and the reason is structural rather than a tuning failure: the source's 193
MSE-calibrated weight quantizers are exactly the MLP and `lm_head` sites, which the port imports verbatim,
so the producer's searched scales are already shipped; the 128 groups the port re-encodes were FP8 in the
source, so there was never a searched-scale artifact to match for them. A separate 6-vs-4 constant move lost
0.36% overall while per-domain results swung ±3% **[in-tree]** `nvfp4-block-scale-4-vs-6.md`. What remains
open is narrower than a sweep: only the *re-encoded* sites have no ground truth, and the measured answer for
them so far is to stop re-encoding them at all (see the FP8-attention topology result).

**2. Calibrated per-layer FP8 KV scales.** Same size, measured to recover 83% of the only systematic
long-context cost of FP8 KV on this exact model (10.84 → 10.50 PPL@32K), performance-free, and shipped by
Unsloth. **Needs:** first check whether the `.ninfer` artifact and loader carry a per-layer KV scale field
at all (if not, that is the change); produce scales from the port's calibration corpus (llm-compressor
`kv_cache_scheme` semantics); measure PPL at 32K/64K with and without — the effect is position-dependent
and invisible at 4K. NVFP4 KV (measured within <1% of FP8 on RULER 64K) is the larger size play, but it
needs kernel work and the port's k8v4-vs-fp8 comparison is still unmeasured **[in-tree]**
`kv-dtype-evidence.md`.

**3. Evaluate the public QAT checkpoint, and fix the GDN layout constraint.** QAT/QAD is the only measured
way to improve *task* quality at the same size (GPQA-D 90.91 vs 91.41 BF16 for the 27B QUASAR checkpoint,
496/496 linears, 19.7 GB), and the port already has the recipe — so the cost is a conversion plus
measurement, not training (full-parameter QAT of 27B is not a single-5090 job; ModelOpt's 8B example wants
2×80 GB). In the same pass, remove the `a`/`b` BF16 exception by padding the `(96, 5120)` rows to the
128-row block-scale layout: the GDN study measured those gates as the *least* sensitive projections, so
layout is the only obstacle to the fully-W4A4 configuration it measured at BF16 task parity with +14–19%
prefill. **Needs:** convert QUASAR with the existing recipe and compare per-domain PPL and a task suite
against the shipped PTQ artifacts; pad and encode `a`/`b`, re-measure DFlash2 acceptance and the corpus;
cross-paper numbers (QUASAR vs Minima vs Unsloth) are not comparable — different harnesses and metrics.

**Not promising here:** FP8 per-row → per-tensor (≤0.5-point MMLU differences, not monotone); the KV-tail
window at 8 bits (a 2–4 bit mechanism); headroom activation scaling (documented concern, no published
end-to-end accuracy number; NVIDIA's own recipe for this model still uses plain max).

---

## Sources

- **NVIDIA**: TE *NVFP4* docs; ModelOpt `nvfp4_w4a4_mlp_fp8_attn_max.yaml`, `ptq.md`, `nvfp4_tensor.py`,
  `nvfp4_act_headroom.py`, `examples/llm_qat`, `examples/hf_ptq`, `dataset_utils.py`; blogs of 2026-01-08,
  2026-06-26 and 2025-12-08 (NVFP4 intro / Nemotron 3 Ultra / NVFP4 KV cache).
- **Papers**: 2509.25149, 2512.02010, 2606.06527, 2601.20088, 2609.04098, 2402.02750, 2401.18079,
  2502.01070, 2309.14592, 2602.02546, 2608.13966, 2601.22813, ACL 2024 long 544, ACL 2025 long 631.
- **Hugging Face / other first-party**: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`; Qwen/Qwen3.8-27B
  discussion #192; Unsloth *Run Unsloth Dynamic NVFP4*; llm-compressor FAQ/skills; TensorRT-LLM
  `quantization.md`, `kv-cache-compression.md`.
- **In-tree**: `docs/research/{activation-scale-method-evidence,quantization-coverage-evidence,`
  `nvfp4-block-scale-4-vs-6,kv-dtype-evidence,swift15-lane-measurement}.md`; `tools/convert/official_recipes.py`.
