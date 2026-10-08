# Does the whole attention stack need FP8, or a subset?

Research note, 2026-10-07. One question: the producer's split keeps **every** full-attention and GDN
projection at FP8 and only the MLP at NVFP4, and restoring that split from this port's all-NVFP4
builds is worth 1.4–2.4 % perplexity and 29–37 % relative acceptance at a cost of 3.2–3.9 GiB and
13–14 % prefill **[in-tree]** `attention-topology-2026-10-07.md`. If the quality comes from a subset
of those projections, the same quality could be had for less. This note asks the published record
which subset, and finds that the record ranks the GDN half and does **not** rank q/k/v/o at all.

| label | meaning |
|---|---|
| **[measured]** | someone ran it and published a figure; model, metric and configuration are quoted with it |
| **[read in source]** | read in the owning project's code, config or first-party document |
| **[third party]** | an independent measurement or account, quoted with its setup |
| **[in-tree]** | this repo's own recorded measurement, cited to the document that holds it |
| **[arithmetic]** | computed here from published module shapes; not measured |
| **[not found]** | searched, did not find — stated as a negative with its search scope |

One structural fact governs the whole note, and it is stated by the paper that measured it: **Q, K
and V share the same input, so any activation-only sensitivity metric assigns the three identical
scores**; ranking them requires output-side (gradient) information **[measured]** KronQ, arXiv
2607.07964v2. The field's decisions for this family are consequently made at *block*
granularity (`self_attn`, `linear_attn`), and §4 shows every shipped recipe moves q/k/v/o together.

---

## 1. Per-module sensitivity of q, k, v, o — the literature ranks MLP, not attention

**The only per-sublayer ranking machinery found, and what it actually ranks.** KronQ
(arXiv 2607.07964v2) derives a sensitivity score `s = tr(H_G)·tr(H_X)` per sublayer (activation ×
gradient covariance) and allocates mixed precision with it. On LLaMA-2-7B, incrementally upgrading
the most sensitive sublayers from W2 to W3, the ranking it publishes is **1. `down_proj`,
2. `gate_proj`, 3. `up_proj`** — all MLP — giving WikiText-2 8.19 → 7.21 → 6.70 → 6.38 at 2.17/2.29/
2.43 average bits, against the activation-only ranking (`gate`, `up`, `down`) at 7.40 → 6.96 → 6.38
**[measured]** Table 6. The paper's Figure 3 plots all sublayers including q/k/v/o on LLaMA-2-7B and
LLaMA-3-8B, and its text states the gradient covariance of Q/K/V/O "varies by orders of magnitude",
but **no numeric q-vs-k-vs-v-vs-o ordering is recoverable from the HTML text of the paper** (the
figure is an image; the named top-3 are MLP). Scope: LLaMA-2/3, weight-only W2/W3 and W3A4/W4A4
(Table 18), not NVFP4, not this family. **[not found]** a KronQ-style ranking for Qwen3.8-27B.

**The one explicit q/k/v/o ordering found is on 125M–1.1B models.** A Hugging Face community study
scores each `nn.Linear` by Jensen–Shannon divergence between the full model's and the
single-layer-quantized model's output distribution (INT4 weights, FP16 activations, WikiText
calibration) **[third party]** `badaoui`, *Sensitivity Aware Mixed Precision Quantization V1*
(2025-06-13): `v_proj` is high (1.0e-04 GPT-Neo-125M, 1.1e-05 TinyLlama-1.1B), `k_proj`/`q_proj` are
the lowest (2.7e-05 GPT-Neo; 5.1e-06 and 9.5e-06 OPT-125M), and `c_proj`/`down_proj` are the highest
class (1.0e-04–1.4e-04 GPT-Neo/OPT; 3.3e-05 TinyLlama `down_proj`). Two caveats travel with it: in
the GPT-2 family `c_proj` names both the attention output projection and the MLP down projection, so
the blog's "c_proj" class is not an o_proj number; and it is a vendor-community blog on small
models, not a peer-reviewed ablation.

**The mechanism argument for a q/k/v/o split exists; the measurement does not.** On Gemma-3-1B W8A8
(TensorRT-LLM, SmoothQuant, RTX 5090), errors in `q_proj`/`k_proj` distort the softmax weights while
errors in `v_proj`/`o_proj` distort token representations without touching the weights, so "there is
a basis for arguing" the two pairs need different calibration budgets **[measured]** Safronov,
arXiv 2608.28003 §5.1 — but the same paper names a projection-specific SQNR analysis as future work
and does not perform one. It is W8A8, one 1B model.

**Module-level *weight* sensitivity is per-layer, not per-projection.** Zhang et al.
(arXiv 2503.06518) measure HQQ weight-only sensitivity on Llama-2-7B/13B and Llama-3-8B and show
`self_attn.o_proj` carrying extreme weight outliers in some layers and a flat distribution in others
(their Figure 1); sensitivity is independent of dataset and quantization method, consistent across
bit budgets, and spikes at the first and last layers. Their budget-allocation figure gives
`self_attn.o_proj`, `mlp.down_proj` and `mlp.gate_proj` **no** extra budget on Llama-3-8B at 4.25
bits **[measured]** §3.7–3.8 — i.e. the paper does not rank o_proj as the sensitive one, it ranks
*layers*. **[measured]** The ICML-2024 evaluation (arXiv 2402.18158) has no per-module attention
table either: its Table 2 statistics are per model family and tensor type, and its results are
W8A8/W4A4/W2 per model. (Text-searched its HTML for `q_proj`/`k_proj`/`v_proj`/`o_proj`/`down_proj`:
no match.)

**For this exact model, the only per-tensor error ranking is weight-reconstruction, not output.**
`malaiwah/Qwen3.8-27B-K4`'s conversion log reports LDLQ proxy error `down_proj` ≈ **2.5e-3**,
"consistently the worst projection in every layer", versus `gate_proj` 1.1e-3 and `in_proj_qkv`
1.0e-3; the same card reports online K6 encoding for every attention projection at ~3.2e-4 per
projection **[third party]**. It separates no attention projection from another, and the port's own
record already shows weight-reconstruction error does not order output quality **[in-tree]**
`nvfp4-block-scale-4-vs-6.md`.

**Answer to Q1, plainly: no published study ranks q/k/v/o for this family at 4-bit against 8-bit.**
The closest are a small-model JSD ranking (`v` high, `k`/`q` low, output projection high) and a
Hessian-based ranking method whose published order is MLP-first. Any per-projection decision on this
model has to be measured here (§6).

---

## 2. The GDN / linear-attention side: Minima ranks it, and the port's summary of Minima holds

**What Minima measured.** Kozyrev & Maiboroda, *Why Gated DeltaNet Survives 4-Bit Quantization*
(arXiv 2609.04098, 2026-09-03), build NVFP4 W4A4 on **all 496 linear layers** of Qwen3.8-27B —
per GDN layer `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` (5×48 = 240), per
attention layer q/k/v/o (4×16 = 64), per layer gate/up/down (3×64 = 192) — keeping embeddings,
`lm_head`, the GDN conv1d, norms, `A_log` and `dt_bias` in BF16; calibration is 128 samples × 32K
tokens, llm-compressor, served on one RTX PRO 6000 with FP8 KV **[measured]** §3, §B.4.

**Verification of the port's own summary.** `nvfp4-fp8-technique-survey.md:85-101` records: all 496
linears including the gates at W4A4; matching BF16 within seed noise; a/b least sensitive at 2.1 % /
2.6 % layer-output error despite 11.0 % / 8.5 % GEMM errors; out/qkv/z carrying the error; a flat
~12.6 % state-error plateau; a 1 % impulse erased in 80–1,382 steps; the gap shrinking with position;
17.53 GiB and +14–19 % prefill. **Every one of those is present in the paper** — Table 1 (85.10 vs
85.62 five-task average, "no pair of models is CI-separated on any task"), Table 3 (a 2.1 %, b 2.6 %,
qkv 10.4 %, z 9.9 %, out 12.7 %, all 19.2 %), Table 6 (plateau 12.62 %, impulse 1/e in 80–1,382
steps, horizons 43,970–61,659), §5.4 (+0.081 nats first half → +0.011 second → −0.053 final 2K),
§B.2. One qualification the port's summary omits: **perplexity is not matched.** Table 1 puts Minima
at 7.67/10.84 (4K/32K) against BF16 6.95/10.35 and the two community recipes 7.16/9.91 and 7.35/9.95
— the paper itself calls PPL "the honest residual" and orders the recipes Unsloth < RadixArk <
Minima. "Within seed noise" is a statement about the five task scores, not about PPL. A second
qualification: the shipped checkpoint requires the §6 fused-scale repair (the paired `qkv+z` and
`b+a` global scales differ by 1.82× and 2.75× in every one of the 48 layers; serving without the
repair corrupts the gates and *fakes better* long-context PPL). Minima also notes its mechanism
depends on the log-space gate parameterization; linearly-parameterized decay may not be shielded.

**The per-projection ranking it establishes (GDN only).** Table 3, one projection W4A4 at a time,
median relative error over 96 (layer, sequence) replays of 8K tokens: `out` 12.7 % > `qkv` 10.4 % >
`z` 9.9 % ≫ `b` 2.6 % > `a` 2.1 %; errors combine in quadrature (single-projection errors predict
19.4 % versus 19.2 % measured for all five). So: **`out_proj` is the most sensitive GDN projection,
`a`/`b` are the safest, and `z` is the cheapest of the three plain GEMMs.** No single GDN projection
is measured safe at 4-bit *in isolation* — only the full block at W4A4 was run end-to-end — but the
component errors are separable, which is what a partial split needs.

**The rest of the GDN field is a warning, not a ranking.** NVIDIA's shipped
`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (the earlier GDN hybrid) excludes the **entire** GDN
block from quantization — `conv1d`, `in_proj_qkvz`, `in_proj_ba` for all 36 linear layers stay
BF16 — while its full-attention block is *partly* quantized (§4) **[read in source]**
`hf_quant_config.json`. llm-compressor's own examples for Qwen3-Next and Qwen3.5 ignore
`re:.*linear_attn.*` wholesale **[read in source]** `examples/quantization_w4a4_fp4/`. Kimi-K3's
Kimi Delta Attention projections (`in_proj_qkvgfab`, `f_a/f_b/b_proj`) ship at FP8 weight-only while
its MoE experts are NVFP4 **[read in source]** `nvidia/Kimi-K3-NVFP4`. And LeapQuant
(arXiv 2609.38166) quantizes the **recurrent state**, not the weights: naive NVFP4/MXFP4 state
quantization collapses most benchmarks to ~0 while its per-window method targets near-lossless
**8-bit** state **[measured]** abstract and the 4-bit rows of its results table.
**[not found]** any weight-quantization sensitivity study for KDA or Qwen3-Next GDN projections at
module granularity (searched arXiv for KDA quantization; found only state quantization).

---

## 3. Output projections and the residual path

**The recurrent half has an answer; the attention half does not.** Minima's Table 3 makes GDN
`out_proj` the most sensitive of the five GDN projections (12.7 % layer-output error; Table 2 gives
it W4 error 10.8 % — mid-band against 10.6–11.6 % for the other GDN projections — but the most
outlier-heavy input of the five, max/RMS 298.1 against 63.5). For
**full attention** the paper replays no per-projection output error, and the closest signal is
Table 2's activation error: `o_proj` **9.2 %** against 7.5 % for q/k/v, with 23.6 % one-hot blocks
and input max/RMS 81.2 — the most outlier-heavy input of the four — while its *weight* error 10.7 %
is mid-pack **[measured]**. That is an argument for protecting `o_proj` if exactly one attention
projection is protected, and it is the opposite of what NVIDIA shipped for Qwen3-Next (§4).

**The MLP analogue is well supported, and it is not attention.** `down_proj` is the worst projection
in malaiwah's per-tensor error ranking for this model (2.5e-3 vs 1.1e-3 gate, 1.0e-3 `in_proj_qkv`)
**[third party]**; the same class tops the small-model JSD ranking (c_proj/down_proj) **[third
party]**; and Minima's Table 2 gives MLP down the most extreme input statistics (max/RMS 368.3).
**[not found]** an equivalent published finding that names `o_proj` as the attention-side analogue
(searched the papers above; the one shipped within-attention split goes the other way).

---

## 4. Recipe configs as evidence: the shipped per-module assignments

**[read in source]** each config below; the unit of decision is the block, and q/k/v/o move together
everywhere except NVIDIA's Qwen3-Next row.

| producer | MLP | q/k/v/o | GDN qkv / z / out | GDN a / b | conv | lm_head |
|---|---|---|---|---|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` (AutoQuantize 5.5-bit sweep output) | NVFP4 all 64 layers | FP8 | FP8 | disabled (BF16) | disabled | NVFP4 |
| NVIDIA ModelOpt recipe `w4a16_nvfp4-fp8_attn` (qwen3_5 family) | W4A16 NVFP4 | FP8 | FP8 | disabled | disabled | NVFP4 (re-enabled) |
| `unsloth/Qwen3.8-27B-NVFP4` (Dynamic v3) | NVFP4 0–55, FP8 56–63 | FP8 | FP8 | BF16 | BF16 | FP8 |
| `RedHatAI/Qwen3.8-27B-NVFP4` | NVFP4 0–55, FP8 56–63 | FP8 | FP8 | BF16 | BF16 | BF16 |
| `RadixArk/Qwen3.8-27B-NVFP4` (ModelOpt) | NVFP4 all 64 | FP8 | FP8 | BF16 | BF16 | NVFP4 |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (QAT) | NVFP4 | NVFP4 | NVFP4 | **NVFP4** | BF16 | BF16 |
| `minima-ai/mnma_qwen3.8_27b_nvfp4` | NVFP4 | NVFP4 | NVFP4 | **NVFP4** | BF16 | BF16 |
| `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (older GDN hybrid) | NVFP4 experts | **q/k/v BF16; o_proj NVFP4** | **all BF16 (excluded)** | BF16 | BF16 | BF16 |
| `nvidia/Kimi-K3-NVFP4` (KDA hybrid) | NVFP4 experts | FP8 weight-only | FP8 weight-only | FP8 weight-only | — | — |
| llm-compressor examples (Qwen3-Next, Qwen3.5) | NVFP4 | NVFP4 | **all BF16 (ignored)** | BF16 | BF16 | BF16 |

Exceptions, explicitly: **no recipe assigns different precision to q vs k vs v**; the only
within-attention split anywhere in the survey is NVIDIA's Qwen3-Next, which quantizes `o_proj` and
protects q/k/v — unmeasured and opposite in direction to Minima's activation-error hint. The GDN
block is the only site with a real within-block split (a/b vs the rest) in five of the ten rows, and
the two all-NVFP4 rows (QUASAR, Minima) are the only ones that ship a/b at 4 bits. NVIDIA's own
Qwen3.8-27B recipe and checkpoint are the AutoQuantize output ("the precision assignment exported by
a 5.5-bit NVFP4-max AutoQuantize sweep"), so the whole-attention FP8 assignment is the *searched*
answer for this model, not a hand rule.

**One in-tree correction.** The unsloth recipe's docstring (`official_recipes.py:434-436`, quoted by
`nvfp4-fp8-technique-survey.md:17`) says unsloth leaves "linear-attention including
`in_proj_a`/`in_proj_b` … as row-scaled FP8", but the same file's code path (`:469-471`) and comment
(`:445-446`) keep those two direct. The checkpoint index shows 48 `in_proj_a` and
48 `in_proj_b` weight tensors with **no** scale companion, while `in_proj_qkv`/`z`/`out_proj` and
`self_attn.{q,k,v,o}_proj` each carry a `weight_scale` **[read in source]** measured from
`unsloth/…/model.safetensors.index.json`; the Minima paper agrees ("both keep GDN and attention at
FP8 W8A8 with a/b in BF16"). The 233 FP8 matrices are 208 attention/GDN + 24 MLP 56–63 + `lm_head`.
The FP8-side count is unchanged either way; only the a/b attribution is wrong.

---

## 5. What a partial split would look like — arithmetic, and which measurement predicts it

**Shapes.** The port's artifact doc and the checkpoint config give: hidden 5120; full attention 24 q
heads and 4 KV heads at head_dim 256, with an output gate doubling q — `q_proj [12288,5120]`,
`k_proj`/`v_proj [1024,5120]`, `o_proj [5120,6144]`; GDN 16 key heads and 48 value heads at 128 —
`in_proj_qkv [10240,5120]`, `in_proj_z [6144,5120]`, `out_proj [5120,6144]`, `in_proj_a`/`b [48,5120]`
**[read in source]** `docs/maintainer/qwen3.8-27b-artifact.md` §11.2–11.3, `unsloth` `config.json`.
The task's assumed 32 q heads / 8 KV heads at 128 does not match this checkpoint (it would be
`q_proj [8192,5120]` with the gate); the numbers below use the real shapes.

**Bytes [arithmetic].** FP8 `fp8_e4m3fn_row_bf16` is 1 B/weight + one bf16 row scale (2 B per row of
5120) = 1.00039 B/param. NVFP4 `block_scale_k16_m128x4_v1` is 0.5 B/weight + one E4M3 scale per 16
= 0.5625 B/param, plus one fp32 divisor per parent. Saving = **0.4379 B/param**:

| site | params | layers | saving if moved to NVFP4 |
|---|---:|---:|---:|
| attention q (incl. gate) | 62.9 M/layer | 16 | 441 MB |
| attention k | 5.24 M/layer | 16 | 37 MB |
| attention v | 5.24 M/layer | 16 | 37 MB |
| attention o | 31.5 M/layer | 16 | 220 MB |
| GDN qkv | 52.4 M/layer | 48 | 1,102 MB |
| GDN z | 31.5 M/layer | 48 | 661 MB |
| GDN out | 31.5 M/layer | 48 | 661 MB |
| **attention + GDN total** | **7.214 G** | | **3.16 GB = 2.94 GiB** |

The measured pair deltas are +3.16 GiB (nvidia pair, 22.1 → 18.9 GB) and +3.86 GB (unsloth pair,
23.58 → 19.72 GB) **[in-tree]** `attention-topology-2026-10-07.md`. The nvidia pair matches the
arithmetic to within rounding; the unsloth pair exceeds it by ~0.7 GB, and this note does not
resolve why — the two arms' parent counts (159 NVFP4 + 128 FP8 against 143 + 135) say the unsloth
pair did not move exactly the same set of sites, so the nvidia pair is the byte model's calibration
and the unsloth delta is not directly comparable. The
a/b pair (23.6 M params total) is BF16 today; moving it to NVFP4 would save only ~34 MB and is a
layout question, not a size one (`block_scale_k16_m128x4_v1` needs N divisible by 128; [48,5120] is
not, which is the port's recorded reason they stay BF16).

**Which published measurement predicts survival.** The honest answer is: the GDN side is predicted,
the attention side is not.

- **GDN `z` → NVFP4** is the one move with a component-level prediction: Minima's Table 3 gives it
  the smallest error of the three plain GEMMs (9.9 %) and no state contribution, and quadrature
  composition implies the all-at-once layer error falls from 19.2 % to ≈16.4 %. Saving 0.62 GiB
  (21 % of the delta). Not measured end-to-end anywhere.
- **GDN `out`** is the *worst* projection in the only ranking that exists (12.7 %); do not move it.
- **Attention, any projection** has no published end-to-end prediction. The two available signals
  disagree: Minima's activation error says `o_proj` is the riskiest (9.2 % vs 7.5 %), NVIDIA's
  Qwen3-Next ships `o_proj` at 4-bit and q/k/v at BF16. Neither is an ablation on this model.
- **The port's own 1.41–2.42 % / 29–37 % result moved attention and GDN together**, so it does not
  attribute the damage to either half — and no partial-split effect can exceed it.
- A weight-error pre-screen is explicitly **not** a ranking instrument here: the port measured that
  lower reconstruction error does not imply better output **[in-tree]** `nvfp4-block-scale-4-vs-6.md`.

---

## 6. The experiment to run here

One source, one MLP encoding, one draft encoding; only the named sites change. The recipe mechanism
is the user-recipe file pattern already recorded in `attention-topology-2026-10-07.md` (Appendix).
Each build is ~150 s and each full-corpus PPL is ~200 s, so a six-arm split costs about 15 min of
builds and 20 min of scoring, plus acceptance runs. Arms, in order of what they decide:

1. **A0 `fp8attn`** — the full split, already built (the reference).
2. **A1 GDN → NVFP4, attention stays FP8.** Isolates the half that Minima ranks.
3. **A2 attention → NVFP4, GDN stays FP8.** Isolates the half that has no ranking.
4. **A3 (run if A2 is cheap) `o_proj` stays FP8, q/k/v → NVFP4.** Tests Minima's o_proj hint in the
   direction of protecting o_proj. Saves ~0.51 GB if it survives.
5. **A4 (run if A2 is expensive) q/k/v stay FP8, `o_proj` → NVFP4.** Tests NVIDIA's Qwen3-Next
   direction. Saves ~0.22 GB.
6. **A5 (run if A1 is cheap) GDN `z` → NVFP4, qkv/out stay FP8.** The Minima-predicted subset.
   Saves 0.66 GB.

Score each arm on the fixed four-domain corpus (~200 s), and report per domain rather than the
aggregate — the port's own data shows the aggregate reorders the domains and hides ±3 % swings.
Measure DFlash2 acceptance at the published width (7 draft tokens) for every arm, because it is
deterministic and it is the metric that moved 29–37 % in the original topology comparison, and
because it is drafter-provenance-sensitive: a partial split can win PPL and lose acceptance. Claim
an effect only if it exceeds the pair's repeatability (build each surviving arm's comparison
interleaved, as the original experiment did) and only if no domain regresses beyond the recorded
stream spread. The expected best case from published evidence is modest — `z` alone recovers about a
fifth of the bytes — so the experiment's value is as much in falsifying the cheap options as in
finding one.

## Sources

- **Papers**: arXiv 2609.04098 (Minima, Kozyrev & Maiboroda, 2026-09-03, HTML v1 — Tables 1–6, §5–7,
  §B.4); arXiv 2607.07964v2 (KronQ — abstract, Table 6, Table 18); arXiv 2609.38166
  (LeapQuant — abstract); arXiv 2503.06518 (Zhang et al. — §3.1–3.8); arXiv 2608.28003 (Safronov —
  §4–5.1, §7); arXiv 2402.18158 (ICML 2024 evaluation — §2.3, Table 2/4); arXiv 2509.25149 (NVIDIA
  NVFP4 pretraining — §4.1, §5, App. E); arXiv 2405.14917 (SliM-LLM — group-wise, not per-module).
- **First-party configs and docs**: `NVIDIA/Model-Optimizer` @ `main` —
  `modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_max.yaml` and
  `…_local_hessian.yaml`; `modelopt_recipes/model_type/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast.quant_cfg.yaml`.
  Hugging Face `hf_quant_config.json`/`config.json`: `nvidia/Qwen3.8-27B-NVFP4`,
  `unsloth/Qwen3.8-27B-NVFP4` (+ `model.safetensors.index.json`), `RedHatAI/Qwen3.8-27B-NVFP4`,
  `RadixArk/Qwen3.8-27B-NVFP4`, `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`,
  `minima-ai/mnma_qwen3.8_27b_nvfp4`, `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (+ card),
  `nvidia/Kimi-K3-NVFP4`. `vllm-project/llm-compressor` @ `main`:
  `examples/quantization_w4a4_fp4/qwen3_next_example.py`, `…/qwen3_5_example.py`,
  `docs/key-models/qwen3.5/index.md`.
- **Third party**: `malaiwah/Qwen3.8-27B-K4` model card (LDLQ per-tensor errors, K6 attention
  ~3.2e-4); `badaoui`, HF community article *Sensitivity Aware Mixed Precision Quantization V1*
  (2025-06-13).
- **In-tree**: `docs/research/attention-topology-2026-10-07.md`;
  `docs/research/nvfp4-fp8-technique-survey.md`; `docs/research/quantization-measurement-methods.md`;
  `docs/research/nvfp4-block-scale-4-vs-6.md`; `docs/maintainer/qwen3.8-27b-artifact.md` §11;
  `tools/convert/official_recipes.py`.
- **[not found]** (scoped): a per-module q/k/v/o sensitivity ranking for Qwen3.5/3.8 at 4-bit vs
  8-bit (searched the papers above, ModelOpt's recipe tree, and the checkpoint configs); any
  weight-quantization sensitivity study of KDA or Qwen3-Next GDN projections; a published accuracy
  number for NVIDIA's Qwen3-Next `o_proj`-at-NVFP4 choice.
