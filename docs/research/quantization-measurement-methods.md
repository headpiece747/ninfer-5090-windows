# Quantization measurement methods: what external evidence supports for ranking recipes

Research note. Written 2026-10-07. Every claim carries its source and one of three labels:

| label | meaning |
|---|---|
| **[measured]** | a run produced the number; the configuration travels with it |
| **[read in source]** | read from primary code, config or documentation |
| **[third party]** | an assertion or secondary account, quoted as such |

**[no evidence found]** marks a searched absence. Nothing here was measured on this product except
where it says so; this port's own figures are quoted only to size a recommendation.

**What the port already has**, so the recommendations below are additions rather than restatements:
a 1,044,876-token corpus in four domains scored as 4096/2048 causal perplexity; a built per-domain
KL instrument (`Engine::score_topk`, `tools/release/per_domain_kl.py`, k=60, 478.3 MiB per 1M
positions) measured once on the Swift 1.5 pair; a DFlash2/MTP bench that emits
`accepted_per_position`; and an AIME/GPQA/IFBench EvalScope harness.

---

## 1. What measurement ranks quantization recipes

The literature separates **capability metrics** (perplexity, task accuracy) from **distance
metrics** (KL divergence, flips), and its central result is that the two disagree precisely where
recipes are close.

**Same accuracy, different behaviour.** [measured] Dutta et al., *Accuracy is Not All You Need*
(Microsoft Research, NeurIPS 2024, arXiv:2407.09141), six schemes (bitsandbytes W8A8/W4A4, GPTQ
W8A16/W4A16, AWQ W4A16, SmoothQuant W8A8) on Llama-2-7B/13B/70B-chat and Yi-6B/34B-chat over MMLU,
PIQA, HellaSwag, ARC-E/C, LAMBADA, WinoGrande, GSM8K, TriviaQA and MT-Bench: accuracy stays within
0–2 % of 16-bit for every scheme, while flips (answers changing correct↔incorrect) reach **13.6 %**;
GPTQ W8A16 is the only scheme with negligible flips. KL divergence correlates with flips at
**Spearman 0.981** on MMLU. MT-Bench (GPT-4 judge, Llama-2-70B-chat): 7.431 → 7.018 for bitsandbytes
W4A4, with up to 10 % loss on the harder turn-2 questions; GSM8K flips are 12–30 % for W8A8/W4A4.
The paper's own perplexity argument: symmetric noise on log-probabilities leaves PPL unchanged while
generation quality degrades.

**Same recipe, task-dependent damage.** [measured] Liu et al., *Quantization Hurts Reasoning?*
(Huawei Noah's Ark Lab, arXiv:2504.04823), DeepSeek-R1-Distill-Qwen 1.5B–32B, R1-Distill-LLaMA
8B/70B, QwQ-32B, over AIME-120, MATH-500, GSM8K, GPQA-Diamond, LiveCodeBench: W8A8KV8 is lossless
(≤1 pt) everywhere; W4A16 costs ~1 %; KV4 is lossless at 14B+; at W4A4KV4 the 32B model loses
**4.17 pt on AIME-120, 1.40 on MATH-500 and 0.00 on GSM8K**; RL-trained QwQ is more fragile than
the distilled model (AIME-120 at W3G128: −14.98 vs −9.98 pt).

**Task-suite sensitivity.** [measured] Li et al., *Evaluating Quantized Large Language Models*
(Tsinghua/Infinigence, ICML 2024, arXiv:2402.18158), 11 model families 125M–180B: multi-step
reasoning and self-calibration are the least tolerant abilities; loss grows with context length;
models are more sensitive to KV-cache than to weight-only or weight-activation quantization; W4A4
causes near-complete loss on most models.

**The counterpoint, stated.** [third party] Jin et al., *A Comprehensive Evaluation of Quantization
Strategies* (ACL 2024 Findings, arXiv:2402.16775) concludes that perplexity "can serve as a proxy
metric for quantized LLMs on most benchmarks". The two positions are reconcilable as scope: PPL
orders *distant* recipes, task/KL instruments separate *close* ones.

**A single artifact can move task metrics in both directions.** [measured] `nvidia/Qwen3.8-27B-NVFP4`
model card, BF16 → NVFP4 (temperature 1.0, top_p 0.95, 262,144-token context): GPQA-Diamond
88.92 → 88.01, Terminal-Bench 75.56 → 74.02, IFBench 80.07 → 78.93, but AA-LCR **72.63 → 73.38** and
SciCode **47.93 → 48.41**. No aggregate score is published, and none would be meaningful.

**Distance metrics need their own statistics.** [measured] the malaiwah `Qwen3.8-27B` EXL3 suite
(v5: 5,120 contexts × 2,047 positions, full-vocabulary `KL(BF16 ref || candidate)` through a shared
BF16 head, one RTX PRO 6000, `enforce_eager=True`): FP8 0.005294, K5/K6 0.003210, K4 0.010604,
unsloth NVFP4 0.030115; top-1 96.79 / 97.52 / 95.76 / 93.16 %. Its tail (shard 0, 1,048,064
positions): p99.9 FP8 0.2438 vs K4 0.5555, with 1.2604 % of K4 positions above KLD 0.1 against
FP8's 0.3912 %. Two warnings transfer: absolute KLD is corpus-specific (v3-corrected ÷ v5 spans
2.46× to 3.08× — ordering holds, values do not), and a candidate measured on its own calibration
text flatters itself (§3).

**Two useful protocols.** [measured] two-pass scoring with one model resident at a time (vLLM
score-mode KLD) and a log-probability floor for reference tokens outside the candidate's top-k
(arXiv:2606.19558 Eq. 5). Direction matters: the quoted distances are `KL(reference ‖ candidate)`
(forward, mode-covering); reverse KL penalises the quantized model's overconfidence and is less
stable. [measured] A 60-sample binary task cannot resolve close recipes: the port's own eval note
computes a ~4 pt two-proportion standard error at p≈0.95, n=60.

**Reading for this port.** PPL ranked distant artifacts here (1.8 % overall spread) while the
per-domain table spans 6.4 % and reorders them; per-domain KL spans 5.8× on the Swift 1.5 pair
(Chinese 0.420244, code 0.072512) where PPL moved 1.3 %. The measurement set has the right shape;
what it lacks is the per-site and calibration controls in §6.

---

## 2. Mixed-precision site selection

Published methods fall into four groups. "Cost" is what the method itself reports or implies.

| method | what it computes | what it found / did | cost |
|---|---|---|---|
| **AWQ** (arXiv:2306.00978) [measured] | per-input-channel salience from activation magnitude, not weight magnitude | keeping **1 %** of channels in FP16 selected by activation drops OPT-6.7B INT3-g128 WikiText PPL from 43.16 to 13.02; weight-based and random selection at the same 1 % do not (38.59 / 37.83). Per-channel scaling recovers it without mixed precision | one calibration pass; grid search over α, β |
| **GPTQ** (arXiv:2210.17323) [read in source] | layer-wise OBS/Hessian error compensation | the accuracy baseline AWQ compares against; needs reordering on LLaMA-7B/OPT-66B | inverse Hessian per layer over 128×2048 calibration |
| **SmoothQuant** (arXiv:2211.10438) [read in source] | per-channel scale migrating activation outliers into weights | makes W8A8 work; α controls migration strength | one calibration pass |
| **MixLLM** (arXiv:2412.14590, MLSys 2026) [measured] | **global** output-feature salience: `S_c = |g·(w_q−w)|` from first- plus second-order (Fisher) terms, one pass over all layers, then a global sort | 8-bit for the top 10 % of output features (W4.4A8): Llama-3.1-70B PPL increase within 0.2 vs ~0.5 for GPTQ/AWQ-class SOTA; MMLU-Pro +0.93 over three models | one forward+backward over 128×2048 WikiText2, then any per-group quantizer |
| **NVIDIA AutoQuantize** [read in source] | gradient-based per-layer sensitivity score; searches the format (or skips quantization) per layer under an `effective_bits` budget | **the Qwen3.8-27B PTQ recipe is its output**: the recipe file is headed "the precision assignment exported by a 5.5-bit NVFP4-max AutoQuantize sweep" — MLP and `lm_head` NVFP4, `self_attn` and `linear_attn` FP8 | NVIDIA documents that a large per-layer search space "can result in higher computational costs and longer processing times" |
| **KVTuner** (arXiv:2502.04420, ICML 2025) [measured] | layer-wise sensitivity of attention patterns to KV quantization error; multi-objective search over per-layer precision *pairs*, offline | 3.25-bit KV nearly lossless on Llama-3.1-8B-Instruct, 4.0-bit on the more sensitive Qwen2.5-7B-Instruct; up to +21.25 % throughput vs KIVI-KV8 | offline calibration, then intra-layer pair pruning + inter-layer clustering |
| **OliVe / SqueezeLLM / OWQ / Atom** [third party] | outlier-victim pair encoding; outlier separation; per-input-feature mixed precision | named alternatives to output-feature mixing; Atom's input-feature scheme constrains outlier counts to kernel tiling | not compared here |

The KLD-profiling recipe (HF community article by Rishiraj Acharya, 2025) is the cheapest per-site
instrument: mock-quantize **one site at a time**, score mean KL on a few hundred examples, rank
sites; one forward pass per site per format.

**Gated MLP blocks (gate / up / down).** Two independent measurements point at `down_proj`:
[measured] the malaiwah Qwen3.8-27B conversion log records LDLQ proxy error `down_proj` **2.5e-3**
vs `gate_proj` 1.1e-3 and `in_proj_qkv` 1.0e-3, "consistently the worst projection in every layer",
and its successor responds asymmetrically — "gate/up K5, down K6"; [measured] the ICML 2024
evaluation's Table 4 puts `down_proj` activation kurtosis at 1.5e5–3.8e5 against 15–142 for
gate/up, the worst outlier shape in the block. [read in source] NVIDIA's recipe nevertheless
assigns one NVFP4 format to gate, up and down. **Prior to import: protect `down_proj` first, treat
gate/up together, and do not expect a published gate-vs-up answer.**

**Hybrid attention (full attention + gated linear attention / delta net).** [read in source] NVIDIA's
fixed Qwen3.5-family recipe quantizes `self_attn` and the large `linear_attn` projections
(`in_proj_qkv` / `in_proj_z` / `out_proj`) all at **FP8**, leaves `in_proj_a` / `in_proj_b` and
`conv1d` disabled, and excludes vision/MTP; AutoQuantize reaches the same split. [measured] malaiwah
protects the same projections at K6 while the MLP runs K4/K5, pricing it at 1.80 GB saved versus
FP8. **[no evidence found]** for a published sensitivity study of GDN blocks, or a layer-wise
comparison between this architecture's 48 linear and 16 full-attention layers.

---

## 3. Calibration corpora

| producer | corpus | samples × length | domain-matched? |
|---|---|---|---|
| **GPTQ** [read in source] | C4 | 128 × 2048 | no (web text) |
| **AWQ** [read in source] | The Pile | "a small calibration set"; the paper's own size comparison runs 16 vs 192 sequences; the QHR reproduction uses 128 × 512 | no; explicitly "in order not to overfit to a specific downstream domain" |
| **SmoothQuant** [third party] | The Pile | 512 × 512 as reproduced in later work; the QHR implementation uses 128 × 512 [read in source] | no |
| **NVIDIA ModelOpt** [read in source] | blog: "128 to 512 samples … accuracy is generally stable across different datasets"; `nvidia/Qwen3.8-27B-NVFP4` card: **2,048 samples** of Nemotron-Post-Training-Dataset-v3 with Local-Hessian calibration | 2,048 samples; the port's in-tree record says 512 × 2048 and the checkpoint's `.quant_summary.txt` records MSE/max calibrators — the three statements disagree and are worth reconciling | task/post-training text |
| **Unsloth** [read in source] | "a mix of our dataset optimized for coding, tool-calling and chat alongside UltraChat"; Dynamic 2.0's dataset is 300K–1.5M tokens | count not published | deliberately mixed; code-heavy by vendor statement |
| **llm-compressor** [read in source] | default `open_platypus` (10,000 samples); examples use `ultrachat_200k`; docs say 128–512 samples typical | 512 × 2048 in the FP8-KV example | chat/instruction text |

**Evidence that the corpus changes the result.**

- [measured] Williams & Aletras, *On the Impact of Calibration Data in Post-training Quantization and
  Pruning* (ACL 2024, arXiv:2311.09755): 4 methods × 9 LLMs (LLaMA/Vicuna/OPT 6.7B–33B) × 5 sources ×
  10 sets = 1,800 models, 11 tasks. GPTQ's zero-shot range across sets is 0.9–1.6 %, SpQR's
  0.6–1.0 %, SparseGPT's 2.4–4.8 %; LLaMA-7B SparseGPT on RTE spans 52.7–61.7 % and BoolQ
  66.4–73.0 % across C4 subsets alone. **PPL stays quiet while task accuracy moves**: Vicuna-7B
  SparseGPT on CNN-DM reads 12.72 ± 0.18 WikiText PPL, yet BoolQ spans 57.0–71.6 %.
- [measured] the QHR paper's own domain switch (WikiText2 → model-generated Numina-Math) on
  DeepSeek-R1-Distill-Qwen-1.5B: GPTQ W3 gains **+9.81 pt average** (28.25 → 38.07), AWQ +1.80,
  KVQuant +0.36, SmoothQuant +0.86, FlatQuant +0.15 — calibration dependence is a property of the
  method, not of quantization in general.
- [measured] AWQ's domain-shift experiment (OPT-6.7B INT3-g128, PubMed vs Enron): cross-domain
  calibration costs AWQ 0.5–0.6 PPL but GPTQ 2.3–4.9 PPL.
- [measured] malaiwah's contamination correction: measuring on held-out text instead of the
  quantizer's own calibration text moved K4's mean KLD **+17 %** (0.026231 → 0.030736) and FP8's
  −32 % (0.019309 → 0.013126); 44 of 941 documents were excluded by an exact 12-token overlap scan.
- [third party] Ye et al., *Damage Predicts Recovery* (arXiv:2609.26241, financial tasks): small
  quantization damage ⇒ calibration choice barely matters; large damage (>40 pt from pruning) ⇒
  task-formatted calibration recovers much of it. Measure task-specific damage first.
- **Code vs prose, for a general model:** **[no evidence found]** for a controlled ablation; the
  closest work varies *language* (EACL 2026, eight calibration settings over GPTQ/AWQ), and
  Unsloth's code-heavy mix is a vendor choice with no published ablation. A gap the port can fill
  cheaply (§6).

---

## 4. KV cache quantization

**FP8.** [read in source] vLLM supports per-tensor (one per Q/K/V) and per-attention-head scales;
the per-head scheme requires the Flash Attention backend and llm-compressor calibration, and vLLM
ships `--kv-cache-dtype-skip-layers` because sliding-window layers are "more sensitive". [measured]
a five-arm sweep on one RTX 5090 serving Qwen3.8-27B (context edition, 262,144-token profile, same
flags except KV dtype; fidelity is truncated top-20 KL against bfloat16 KV at 98,304 tokens):

| KV dtype | backend | top-1 | truncated KL | KV tokens | prefill (261,795-token prompt) |
|---|---|---:|---:|---:|---:|
| fp8 (family default) | FLASHINFER | 95.60 % | 0.001655 | 265,122 | 180.4 s |
| int8 per-token-head | TRITON_ATTN | 97.25 % | 0.000914 | 272,453 | 544.3 s |
| fp8 per-token-head | TRITON_ATTN | 98.84 % | 0.001284 | 272,453 | 545.9 s |
| int4 per-token-head | TRITON_ATTN | 94.29 % | 0.005948 | 502,667 | 501.0 s |
| bfloat16 (reference) | FLASH_ATTN | — | — | 138,519 | — |

Two readings: per-token-head scaling *beats* per-tensor FP8 on closeness (its capacity edge is the
backend's CUDA-graph pool, not cheaper bytes); and 4-bit KV buys 1.92× capacity at **3.6× FP8's
error** and 2.78× its prefill, with retrieval still 44/44 — retrieval is not fidelity, which is why
the KL column exists.

**Sub-8-bit methods and what recovers the loss.**

- [measured] KIVI (ICML 2024, arXiv:2402.02750): **keys per-channel, values per-token**, group size
  32, **128-token full-precision residual window**; 2-bit KV with "almost the same quality" and
  2.6× lower peak memory (Llama/Falcon/Mistral), 4× batch size.
- [measured] KVQuant (NeurIPS 2024, arXiv:2401.18079): per-channel keys **before RoPE**, per-token
  values, per-layer non-uniform datatypes, per-vector dense-and-sparse outliers; **<0.1 PPL
  degradation at 3-bit** on WikiText-2 and C4 (LLaMA, Llama-2, Llama-3, Mistral); per-channel key
  quantization alone is worth 3.82 PPL on WikiText-2 over per-token; 1M context on one A100-80GB.
- [measured] QHR: 4-bit KV lossless at ≥14B, 3-bit drops >5 pt on 7B; and for Qwen models whose key
  projections carry large biases, quantizing K **before the bias** recovers +7.81 pt average at
  3-bit (the port's target is a Qwen with 48 linear + 16 full-attention layers).
- [measured] ICML 2024: loss grows with context length, and KV is the more sensitive tensor type —
  which is why long-context probes, not 4K PPL, are the right instrument for a KV decision.

**Applicability to a paged cache.** Per-channel K scales live on the head-dimension axis, which is
contiguous inside a page, so they fit the port's per-row scale plane (`fp8` K is E4M3FN-row256 + 2 B
scale; `k8v4` is that K plus NVFP4-G16 V). Recovery mechanisms that need **two precision levels over
time** (KIVI's 128-token residual, KVQuant's recent window) map onto a page-granular format split —
a second pool, or a per-page format tag — not a single fixed page format. [read in source]
TensorRT-LLM's cold-page compression is the vendor precedent: NVFP4 in the host/disk tier only,
runtime type restored before attention, GDN/SSM/conv state preserved losslessly — but it requires
SM100/103/107 and a nonzero host/disk cache, so not on sm_120a today. [no evidence found] any
published `k8v4`-vs-FP8 quality number.

---

## 5. Draft-model acceptance as a measurement

**How it is measured.** [read in source] DFlash Eq. 1 defines acceptance length τ ∈ [1, γ+1] as
accepted tokens per cycle *including the bonus token*; the DFlash2 card defines it as completion
tokens ÷ verification steps — the port's bench convention. vLLM publishes per-position rates and
`mean_acceptance_length = 1 + accepted_tokens / num_drafts`. **Per-position profiles are the
diagnostic, not the mean**: DFlash2 on Qwen3.5-4B (MATH-500, T=0, block 16) is flat at
88.3 % → 86.5 % over 15 positions, whereas DFlash 1 decays 85.4 % → 72.9 % over 7 — the "suffix
decay" the two-tap convolution addresses (position-6 Recall@1 72.86 → 77.61 %) [measured, author
blog].

**Published acceptance, with configuration.** [measured] DFlash (ICML 2026, arXiv:2602.06036):
Qwen3-8B, block 16, T=0, Transformers backend, ≤2048 new tokens — τ 6.54 (GSM8K), 6.49 average, vs
EAGLE-3 τ ≈ 2.7–3.7 (tree 16/60); Qwen3.5-27B @ 8K τ 4.46/4.17/3.32 on hotpotqa/qasper/gov_report
(base drafter). [measured] DFlash2 on Qwen3.8-27B (SGLang, one H200, block 8 = 7 draft tokens,
T=1.0/top-p 0.95/top-k 20, xhigh, ≤4096 new tokens): τ 5.46 GSM8K, 5.28 MATH-500, 4.39 HumanEval,
4.79 MBPP, 4.10 MT-Bench — against MTP's 4.28 mean and DSpark's 3.62. [measured] DeepSeek-V3 MTP:
second-token acceptance 85–90 %, 1.8× speedup.

**What raises acceptance, and by how much.**

- **Conditioning on target features.** [measured] DFlash ablation (Qwen3-4B, 5-layer, block 8):
  injecting fused target features into **every draft layer's KV** beats EAGLE-3-style input fusion
  (τ 4.2 vs 3.5 on GSM8K; 3.3× vs 2.9× speedup), and in autoregressive drafting too (4.6–4.8 vs
  4.2–4.3). More target layers help (5 hidden features: τ 5.64 vs 5.38 with 3).
- **Draft depth.** [measured] DFlash: 8-layer τ 6.33 but speedup 4.64×, 5-layer τ 5.99 and speedup
  4.71× — acceptance scales with depth, *speedup* peaks at 5 layers. EAGLE-3 tree 60 vs 16 buys
  ~+0.4 τ (3.05 → 3.48 on Q3-4B T=0) at higher verification cost.
- **Draft width / block size.** [measured] DFlash block 16 vs 8 on the same data: τ 6.33 vs 5.21;
  the block-8 model fully accepts entire blocks 35.7 % of the time. Generalisation is asymmetric —
  train-16/test-8 lands close to train-8/test-8, the reverse does not.
- **Training objective.** [measured] DFlash: random anchor sampling per block; exponential loss
  decay `w_k = exp(−(k−1)/γ)` emphasising early positions; shared frozen embedding and LM head;
  training on target-generated responses (≈800K samples from Nemotron Post-Training v2 +
  CodeAlpaca). DFlash2 adds the top-k selector and two-tap dynamic convolutions.
- **Verification.** [measured] Medusa (arXiv:2401.10774): extra heads + tree attention, 2.2× speedup
  losslessly (Medusa-1) and 2.3–3.6× with joint tuning (Medusa-2); self-distillation when no
  training data exists; "typical acceptance" for non-greedy decoding.
- **Quantization interaction.** [third party] QSpec (EMNLP 2025, arXiv:2410.11305): a 4-bit draft
  with a high-precision verifier reaches ~74 % acceptance (γ=6); [measured] a quantized MTP head on
  Qwen3.8-27B reaches 58.2 % acceptance (+101 % throughput). The port's own note records 7–22 % on
  its lane vs upstream's 4.10–5.46 tokens/round, and 49.3 % for the stock drafter at width 7 on a
  finetune.
- **Long context.** [measured] DFlash: base drafter degrades past 4K (τ 4.46 at 8K); 1.6K
  LongAlign-10K samples restore it (5.76 at 8K, 6.05 at 16K on hotpotqa).

**Reading for this port.** Report τ *and* the per-position profile at the published width (7 draft
tokens / block 8) for every target artifact: flat-vs-cliff separates a drafter-wide problem from a
width-boundary defect, and the bench already emits `accepted_per_position`.

---

## 6. Measurements this port could add, and what each costs

Costs are estimates from the recorded instruments, not measurements; the first is already built.

| # | measurement | why it ranks recipes better | cost |
|---|---|---|---|
| 1 | **Run the existing per-domain KL across all five shipping artifacts** against one common near-lossless reference image (BF16 is impossible on this card: 51.75 GiB of BF16 weights vs 31.85 GiB; use the groupwise-int image of the same source, as the Swift 1.5 run did) | gives every artifact a distance metric on the same 1,044,876 positions; today only one pair has it | one reference record (~478 MiB, one full-corpus pass, minutes at ~6k tok/s) + one pass per artifact; disk 0.5 GiB per record |
| 2 | **Publish KLD tail statistics** (p99, p99.9, positions above 0.1/1.0) beside each mean | means and top-1 can hide a 2–6× tail difference between recipes, as measured externally | a `per_domain_kl.py` extension; no new model runs |
| 3 | **Per-site sensitivity profile** — re-encode one site class at a time (`down_proj`, gate/up, `self_attn`, `linear_attn`/GDN) and score PPL and KLD on a subset | the port's allocation is imported from NVIDIA; down_proj is the externally indicated site and is untested here | 4–6 local re-encodes (minutes each) + 4–6 scoring runs on one stream per domain (~261k tokens) |
| 4 | **Calibration-corpus control** — one artifact pair differing only in calibration text (prose vs code), scored on held-out text | no published code-vs-prose ablation exists for a general model; the port's own `calibration.json` corpus is unvalidated | 2 conversions + 2 full-corpus PPL/KL runs |
| 5 | **Acceptance at the published width, per-position, for every target artifact** | acceptance is target-dependent (49.3 % vs 57.1 % on two finetunes here); PPL cannot see it | one `ninfer_bench --spec dflash2 --draft-tokens 7` run per artifact, minutes, interleaved on one card |
| 6 | **Long-context KV probe** — NIAH at 64K/128K plus a 32K-context KLD for `fp8` vs `k8v4` vs bf16-KV | every KV conclusion in §4 was measured at ≥98K context; the port's KV figures are 4K-window PPL and an oracle | one serve run per dtype + one 32K KLD record per arm |

Not recommended as a ranking instrument: more 4K perplexity. The external record shows PPL cannot
see reasoning damage (§1), per-domain spread already reorders artifacts, and a 60-sample binary
score cannot resolve close recipes — a task suite is a tie-break, not the ranking.

---

## Sources

| source | what was taken from it |
|---|---|
| arXiv:2407.09141 (Dutta et al., NeurIPS 2024) | flips up to 13.6 % at 0–2 % accuracy change; KL↔flips Spearman 0.981; MT-Bench and GSM8K flips; PPL-noise argument |
| arXiv:2504.04823 (Liu et al., 2025) | W8A8KV8/W4A16/W4A4KV4 results; task-difficulty and model-origin effects; per-method calibration table |
| arXiv:2402.18158 (Li et al., ICML 2024) | tensor/task sensitivity summary; long-context and KV-vs-weight findings; per-linear kurtosis table |
| arXiv:2402.16775 (Jin et al., ACL 2024 Findings) | PPL-as-proxy counterpoint |
| arXiv:2306.00978 (AWQ) | 1 %-salient result; per-channel scaling; Pile calibration; domain-shift robustness; calibration-set size comparison |
| arXiv:2412.14590 (MixLLM, MLSys 2026) | global output-feature salience; W4.4A8 results; search cost |
| arXiv:2502.04420 (KVTuner, ICML 2025) | layer-wise KV precision search; 3.25/4.0-bit results; throughput |
| NVIDIA Model-Optimizer (recipes, blog) and the `nvidia/Qwen3.8-27B-NVFP4` card | AutoQuantize search; the `nvfp4_w4a4_mlp_fp8_attn_max` assignment and the `w4a16_nvfp4-fp8_attn-kv_fp8_cast` recipe; 2,048-sample Nemotron calibration; per-task BF16→NVFP4 deltas |
| malaiwah Qwen3.8-27B EXL3 collection (v5 receipts) | KLD ladder, tails, head attribution, contamination correction, per-tensor LDLQ errors, 5090 KV-dtype sweep, quantized MTP acceptance |
| arXiv:2311.09755 (Williams & Aletras, ACL 2024) | calibration-set dispersion; PPL-vs-task masking; recommendations |
| arXiv:2609.26241 (Ye et al., 2026) | damage-dependent value of domain-matched calibration |
| llm-compressor docs; unsloth NVFP4 docs | default calibration datasets and counts |
| arXiv:2402.02750 (KIVI); arXiv:2401.18079 (KVQuant) | per-channel/per-token asymmetry; residual window; 3-bit PPL results |
| vLLM `quantized_kvcache.md`; TensorRT-LLM `kv-cache-compression.md` | FP8 schemes and skip-layers; cold-page NVFP4, SM requirements, state preservation |
| arXiv:2602.06036 (DFlash, ICML 2026); DFlash2 model card/blog; arXiv:2401.10774 (Medusa); arXiv:2412.19437 (DeepSeek-V3); arXiv:2410.11305 (QSpec) | τ definitions and values; conditioning/depth/width/training ablations; MTP acceptance; Medusa speedups; 4-bit-target acceptance |
| this tree | `docs/perplexity-baseline.md`; `docs/research/per-domain-kl-instrument.md`; `docs/research/swift15-lane-measurement.md`; `docs/research/dflash2-acceptance-baseline.md`; `docs/research/kv-dtype-evidence.md`; `eval/README.md`; `apps/perplexity/` |
