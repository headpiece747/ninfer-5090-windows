# Target precision and draft acceptance: what is published, and what is not

Research note, 2026-10-07. One question: **when a hidden-state-consuming drafter is held fixed and the
target's precision changes, what does the published record say happens to acceptance, why, and what
remedy is measured?** It was triggered by this port's two topology measurements
(`attention-topology-2026-10-07.md`): keeping the producer's FP8 attention is worth **+77 % relative
acceptance**, while keeping the FP8 GDN as well is worth only **+29 %**, and the two halves therefore
trade perplexity against acceptance. No GPU job was run for this note. Every claim carries its source
and one of these labels:

| label | meaning |
|---|---|
| **[measured]** | someone ran it and published a figure; model, drafter, precision and benchmark are quoted with it |
| **[read in source]** | read in the owning project's code, config or first-party document |
| **[third party]** | an independent account, quoted with its setup and without a first-party figure |
| **[in-tree]** | this repo's own recorded measurement, cited to the document that holds it |
| **[not found]** | searched, did not find — stated as a negative with its search scope |

**Headline.** The direction is real and is stated in production guidance, but **nobody has published
the measurement this port needs**: one fixed drafter's acceptance on one target at two precisions.
Every published figure quantizes the *drafter* (Meta, Nota), quantizes the target *and adapts the
drafter to it* (Nota), or uses a weight-shared quantized copy of the target as the drafter (QSpec,
ML-SpecQD). The one first-party project with this port's exact architecture — NVIDIA's DFlash support —
records the question as an open item: "AR impact of quantization not yet measured". The *mechanism* is
established and quantified in a different experiment (§2): a 10 % RMS perturbation of the drafter's
hidden-state pathway costs a pre-norm EAGLE-3 18 % of its acceptance and a 25 % perturbation costs
72 % **[measured]** *Attention Drift* — a scale large enough to explain +77 %, but no published study
converts a precision change into that perturbation scale. No published study perturbs one target
component (attention vs MLP vs GDN) and reads a drafter's acceptance (§2.3). Measured remedies are all
drafter-side (§3), and the largest and best-measured lever on acceptance is the *sampling*, not the
weights (§4).

## 1. Does target quantization degrade speculation, and by how much?

The table below is everything found that bears on the question, in descending order of how directly it
constrains it. The fixed-drafter-on-a-quantized-target cell is **empty everywhere**.

| source | setup (target / drafter / precision) | acceptance result | label |
|---|---|---|---|
| QSpec, EMNLP 2025 (arXiv 2410.11305v3) | Llama 3.2-3B / Llama 3-8B / Llama 2-7B/13B; the draft is the **same weights** at W4A4, verified by the same weights at W4A16; greedy; γ=2..6 | "at γ=6, the token acceptance rate remains relatively high, approximately **74 %**", against "28∼58 % in [a] 160m-7b draft-target model pair under γ=5 in conventional speculative decoding" | **[measured]** |
| ML-SpecQD, Intel (arXiv 2503.13565) | Llama 2-7B, question answering; the draft is an **MXFP4 direct-cast copy of the target** against a small 68M custom draft | 71.2 % vs 41.7 % acceptance (1.7×) | **[measured]** |
| Nota / AdaptFM competition report (arXiv 2607.04244) | Qwen3.5-4B, AWQ INT4 → QAD INT4 target; 5-layer 537M DFlash drafter; acceptance length (AL) on GSM8K / HumanEval / LongBench-v2, 64 prompts, 256-token cap | BF16-target-trained drafter **4.92** (measured on BF16); adapted to INT4 **5.03** (on INT4); direct INT4 training **4.97**; the public DFlash checkpoint on BF16 **5.69**. **The row "BF16-trained drafter evaluated on the INT4 target" is not published.** | **[measured]** |
| Meta, "Efficient Speculative Decoding for Llama at Scale" (arXiv 2508.08192), Table 2 | Llama 3.1-8B / 3.3-70B / Llama 4 Scout / Maverick; their own EAGLE drafter; **draft-side FFN** BF16 → FP8 → INT4 | TPC 2.79/2.79/**2.76**, 2.95/2.95/**2.96**, 2.86/2.86/**2.87**, 2.81/2.79/**2.78** — draft quantization is nearly free; **the target side is not studied for acceptance** | **[measured]** |
| vLLM / Red Hat Speculators v0.5.0 (2026-05-28) | Gemma 4-31B DFlash speculator trained against the BF16 verifier, then **served with an FP8-quantized verifier** | "Combining DFlash with an FP8 quantized verifier yields even greater gains" — that figure is **latency (ITL)**; the acceptance delta of the FP8 swap is not published | **[read in source]** |
| NVIDIA Model-Optimizer, `examples/speculative_decoding/doc/dflash.md` | DFlash trained against a frozen BF16 target; FP8/NVFP4 export path for the drafter | "PTQ succeeded in testing. **AR impact of quantization not yet measured.**" | **[read in source]** |
| vLLM issue #36629 (closed, stale) | Qwen2.5-14B, 1× 4090D, vLLM 0.16.0; **W4A16 target + EAGLE3** vs **FP8 target + EAGLE3** | W4A16 arm: acceptance 62.22 %, AL 2.87, per-position 76.97 / 60.75 / 48.95 %; FP8 arm's acceptance is not reported (only throughput). At concurrency 16 the W4A16 arm is slower (TPOT 13.73 ms vs 13.16 ms) | **[third party]** |
| GMI Cloud blog (vendor) | "INT4 weights drop acceptance by roughly **3 to 5 percentage points** relative to FP8"; the recommended 2026 pattern is "FP8 target, FP8 draft, FP8 KV" | No model, drafter or measurement quoted | **[third party]** |
| DigitalOcean vLLM configuration guide | "using an **INT4 target model, even with a BF16 draft model**, can noticeably reduce acceptance rates, especially when generation uses sampling instead of deterministic decoding"; priority order "INT8 target over INT4 target" | No measurement shown | **[third party]** |
| this port | `nvidia` source, same MLP, same draft encoding, fixture corpus, K=5 | acceptance 0.1106 → 0.1957 (**+77 %**, attention FP8) and 0.1106 → 0.1429 (**+29 %**, attention+GDN FP8); unsloth source 0.1524 → 0.2083 (**+36.7 %**) | **[in-tree]** |

**Acceptance *improving* with target quantization: [not found].** Searched the papers and engine
trackers above for any report of a fixed drafter accepting more on a quantized target than on the same
target at higher precision; the only "quantization helps acceptance" results found move the *draft*
toward weight-sharing with the target (QSpec 74 %, ML-SpecQD 71.2 %) and are high for that structural
reason, not because quantization helped. This port's +77 % is the only measurement found in which a
target-precision change raises a fixed drafter's acceptance — and its reading, that the target moved
*back toward* the drafter's training distribution, is this port's interpretation rather than a
published result.

**Two port records that are often cited with this effect have narrower scopes.** `_nvdiv` records two
published artifacts at 61.5 % against 53.8 % acceptance (`tools/convert/official_recipes.py:555-556`),
and the Swift note records a **draft-encoding** change losing 3.2 points (57.7 % vs 60.9 %,
`:716-719`). Neither is a target-precision comparison: the first differs in divisor source and MLP
provenance (the divisor-source hypothesis was later rejected by the `_nvdiv` build itself), the second
changes the drafter's own encoding. They corroborate that acceptance is provenance-sensitive; they do
not measure target precision. **[in-tree]**

## 2. The mechanism: what the drafter reads, and how much perturbation it tolerates

### 2.1 Which target components feed each drafter

| drafter | target components consumed | source |
|---|---|---|
| EAGLE-1/2 | top-layer features **immediately before the LM head**, plus the previous token's embedding; the draft token is obtained through the target's LM head | **[read in source]** EAGLE-3 paper §2.2 |
| EAGLE-3 | a fusion of **low-, middle- and high-level hidden states** (reference implementation taps layers `{2, N//2, N−3}`, i.e. 2/16/29 for a 32-layer LLaMA-3.1-8B), plus embeddings; **its own reduced-vocabulary LM head** | **[measured]** EAGLE-3 paper §3.1, Table 2; **[read in source]** `modeling_llama_kv.py` via the MLX write-up; Attention Drift §3 |
| DFlash | **five layers uniformly sampled between layer 2 and the third-to-last layer**, concatenated → FC + RMSNorm, injected as K/V into **every** draft layer; **shares the target's frozen token embedding and LM head** | **[measured]** DFlash paper §4.1-4.2, Tables 7 and 9; **[read in source]** NVIDIA Model-Optimizer example taps `[1, 9, 17, 25, 33]`; Speculators example `--target-layer-ids "2 18 33"` |
| Medusa / Hydra / MTP | the **last hidden state** (pre-LM-head); the MTP head **shares the target's LM head** and is trained jointly | **[read in source]** TRT-LLM Medusa docs; DFlash §2.1; Attention Drift §3 |

The port's DFlash2 companion is the DFlash family: `dflash2/feature_projection [5120, 25600]` is five
taps of hidden width 5120 (`docs/maintainer/qwen3.8-27b-artifact.md:410`), and the artifact config
records them explicitly as **`target_layer_ids: [5, 19, 33, 47, 61]`** of a 64-layer Qwen3.8-27B
**[in-tree]** (read from `out/qwen3_8_27b_nvfp4nvidia.v3.ninfer.conversion.json`,
`components.dflash2.config.dflash_config`). Of those five taps, `components.text.config.layer_types`
gives **`linear_attention` at 5, 33, 61** and **`full_attention` at 19, 47** (48 GDN + 16 full-attention
layers in total). So a GDN precision change lands on three of the five streams the drafter fuses, and a
full-attention change on two — a structural fact the port's topology experiment moves together and
never separates **[in-tree]**.

### 2.2 The drafter reads hidden states that no final norm has sanitised

EAGLE-3 and DFlash both tap *internal* hidden states. Attention Drift states the consequence plainly:
"the target's final pre-LM-head RMSNorm normally absorbs this growth when producing logits, but EAGLE-3
uses verifier's internal states captured before this norm"; it also measures that layer magnitudes grow
monotonically with depth, so the high-layer stream dominates the fused feature **[measured]** Attention
Drift §4.1, Observation 2 (measured for EAGLE-3's fusion; DFlash's is also a concat→FC but its
imbalance is not published). A precision error introduced upstream of a tap therefore reaches the
drafter un-renormalised, and for this artifact an error in a late GDN layer (tap 61) sits in the stream
a magnitude-dominated fusion would weight most.

### 2.3 The one quantified tolerance, and the gap

Attention Drift's α-noise sweep injects Gaussian noise scaled per-tensor RMS,
`x ← x + α·rms(x)·ε`, and reads acceptance length as a percentage of the unperturbed baseline
(Llama-3.1-8B target, EAGLE-3 drafter, MT-Bench) **[measured]** Table 3:

| drafter | pathway | α=0.1 | α=0.25 | α=0.5 | α=1.0 | α=2.0 |
|---|---|---|---|---|---|---|
| pre-norm (τ = 3.06) | **hidden states** | 82 % | 28 % | 5 % | 0 % | 0 % |
| pre-norm | embeddings | 99 % | 93 % | 86 % | 64 % | 27 % |
| post-norm (τ = 3.16) | **hidden states** | 99 % | 86 % | 58 % | 22 % | 5 % |
| post-norm | embeddings | 98 % | 100 % | 93 % | 75 % | 38 % |

Two things follow. First, **the hidden pathway is the fragile one** — pre-norm EAGLE-3 loses 18 % of
its acceptance at α = 0.1 and 72 % at α = 0.25, while the embedding pathway tolerates ~5× more noise.
Second, the paper names this port's exact question as an untested extension: "we further hypothesize
that this tolerance may translate to robustness under other small hidden-state perturbations, **such as
those induced by verifier quantization** or mild distribution shift" **[measured]** §4.4. The noise
sweep is the instrument that would express the port's +77 % in the literature's units, and nobody has
run it against a real precision change.

**Quantizing one target part versus another and reading acceptance: [not found].** Searched EAGLE-3,
DFlash, DFlash2, QSpec, SpecForge/Speculators, TensorRT-LLM and Model-Optimizer docs, and the vLLM /
SGLang / EAGLE trackers for an ablation that quantizes attention, MLP or a linear-attention block
separately and reports a drafter's acceptance. The nearest neighbouring results are: (a) Attention
Drift's pathway split above (hidden ≫ embeddings); (b) KronQ's sublayer ranking, MLP-first, for *weight*
quantization on LLaMA-2, not acceptance **[measured]** and already in `attention-site-sensitivity.md`;
(c) Minima's GDN projection ranking by layer-output error (`out` 12.7 % > `qkv` 10.4 % > `z` 9.9 % ≫
`b` 2.6 % > `a` 2.1 %), again not acceptance **[measured]**. No published work converts any of these
into an acceptance prediction.

## 3. Published remedies, and what each recovered

1. **Adapt the drafter to the quantized target (the only remedy measured against this exact problem).**
   Nota pretrains the DFlash drafter against the BF16 target, then finetunes it on data regenerated by
   the QAD-INT4 target; AL 4.92 (BF16 stage, on BF16) → 5.03 (INT4), against 4.97 for direct training on
   INT4 from random init. The report's own words: Stage 2 "aligns with the quantized target
   distribution better than direct QAD-target training does", and "learning a robust drafting
   initialization from the BF16 target before adapting to the quantized target distribution is
   beneficial" **[measured]**. The size of the recovery in their 4B setting is +0.11 AL. The same
   machinery is available off the shelf: Speculators' online/offline training extracts hidden states
   from a **running vLLM server** (vLLM ≥ 0.18 native hidden-states extraction), so "train the drafter
   against the FP8 verifier" is a configuration choice rather than new code **[read in source]**
   Speculators v0.5.0 blog and docs.
2. **Train the target to follow its BF16 self (QAD).** Nota's QAD recovers the AWQ-INT4 checkpoint's
   quality (MMLU-Pro 0.6557 → 0.6610, IFEval 0.8046 → 0.8285, GPQA 0.6536 → 0.6626 against BF16
   0.6804 / 0.8249 / 0.6798) while keeping the quantization grid **[measured]**. Because the drafter
   was trained on the high-precision target's distribution, moving the quantized target's *output
   distribution* back toward BF16 is a remedy for both quality and drafter alignment at once — though
   the acceptance recovery is not measured in the report.
3. **Architecture, if the drafter can be retrained.** Attention Drift's post-norm drafter plus a
   per-tap RMSNorm before fusion: +10-12 % acceptance length across four models (Llama-3.1-8B, Qwen3-8B,
   Qwen3.5-9B, GPT-OSS-20B), up to 2× under template perturbation, 1.18× on long context, and an
   order-of-magnitude larger tolerance to hidden-path noise (58 % vs 5 % at α = 0.5) **[measured]**. Its
   TTT-2 pre-norm baseline collapses beyond its training depth (per-step acceptance 0.93 → 0.00 at
   k=16) where post-norm holds 0.53 **[measured]** §4.2. This is a drafter redesign, not a post-hoc
   patch, so it cannot rescue a shipped companion.
4. **Accept the loss and ship the pair.** vLLM/Red Hat serve a BF16-trained DFlash speculator against an
   FP8 verifier and report it as a latency win; the Gemma-4 DFlash model card publishes acceptance only
   against the BF16 verifier **[read in source]**. This is the production default, and it means the
   published record does not establish that the acceptance loss is large.
5. **Fine-tune the drafter for the new distribution — it is cheap.** DFlash's long-context adaptation
   takes 1.6K samples for 3 epochs and moves Qwen3.5-27B τ at 8K from 4.46 to 5.76 (hotpotqa), 4.17 to
   5.62 (qasper), 3.32 to 4.04 (gov_report) **[measured]**. If a precision-induced distribution shift
   behaves like a domain shift, the precedent says a small adaptation set can close it.
6. **The proposal-head class (`--lm-head-draft`).** DFlash shares the target's frozen embedding and LM
   head; EAGLE-3 uses its own **reduced-vocabulary** head (Speculators default `draft_vocab_size`
   10,000; DFlash training `--draft-vocab-size 8192`; the LLaMA-3.1 EAGLE-3 checkpoint 32,000); MTP
   shares the target's head. The port's indexed proposal head with token-ID remapping is this class
   **[in-tree]** `docs/maintainer/dflash.md`. Published work on *swapping* the proposal head to
   compensate for target quantization: **[not found]**. The adjacent measured facts are contradictory:
   QSpec kept the EAGLE draft at FP16 because "applying GPTQ quantization to the EAGLE draft model
   resulted in substantial degradation of the acceptance rate" **[measured]**, while Meta's INT4-FFN
   draft is free (Table 2 above) and Nota's GPTQ drafter keeps 4.98 of 5.03 AL **[measured]**. The
   difference is scope: FFN-only INT4 with the rest kept, versus whole-drafter GPTQ.

**A trap worth recording.** vLLM issue #54252 reported that a compressed-tensors config "collapses GDN
linear-attention token distinctness" (per-layer hidden-state self-cosine 0.05 → 0.40/0.55, norm
5.5e8), which reads as a GDN-quantization mechanism; it was closed as invalid when the author found
their own round-to-nearest INT4 pass had ~190 % per-expert relative error once GPTQ error compensation
was added **[third party, invalidated]**. The lesson for the port's measurements is that a hidden-state
symptom on a hybrid model can be the *quantizer*, not the precision; and its per-layer hidden-state
self-cosine sweep is a usable diagnostic shape.

## 4. The training-distribution argument, and how much it can explain

**Supported in direction, unquantified in magnitude.** The standard recipes are explicit that the
drafter is trained against a specific verifier: vLLM states "it is important for the draft model to
align closely with the verifier model, **which is why we train draft models specific to each verifier**"
**[read in source]**; EAGLE-3 and DFlash both generate training responses with the target and extract
its hidden states **[measured]**; Speculators derives target-to-draft vocabulary mappings from the
verifier **[read in source]**. So a target whose hidden states move away from the drafter's training
distribution is expected to lose acceptance. What no source provides is the conversion from a precision
delta to a hidden-state delta to an acceptance delta; the α-noise table is the only tolerance curve
found, and it is noise, not quantization.

**One premise of this port's own telling is unverified.** "The drafter was trained on the stock model's
hidden states" (`official_recipes.py:716-719`) is the port's inference from the companion's design; the
public `incoai/Qwen3.8-27B-DFlash2` card read for `dflash2-acceptance-baseline.md` states the runtime
and sampling but **not the precision of the target used to generate its training hidden states**
**[not found]**. If the companion was trained against a quantized deployment, the "closer to training
distribution" reading of the +77 % changes sign of interpretation. This is cheap to check against the
companion's training config or README.

**Sampling and context dominate everything above.** EAGLE-3 Table 1, τ at temperature 0 → 1:
LLaMA-3.1-8B 6.23 → 4.92 (**−21 %**), Vicuna-13B 6.62 → 5.67 (−14 %), LLaMA-3.3-70B 5.88 → 5.66 (−4 %),
DeepSeek-R1-Distill-8B 5.84 → 4.89 (−16 %) **[measured]**. DFlash Table 1: Qwen3-8B 6.49 → 5.48
(−16 %), Qwen3-4B 6.54 → 5.69 (−13 %) **[measured]**. Template perturbation costs a pre-norm EAGLE-3 up
to 52 % in the worst case, post-norm ≤ 5 % **[measured]**; long context without SWA collapses a
pre-norm drafter to 0.05 acceptance **[measured]**. So the field's own data says the largest, most
reproducible levers on acceptance are the sampling distribution and the context regime — not the
target's weight precision. Any claim about the target's precision has to hold all of those fixed, as
this port's interleaved within-pair comparisons do; it also means the port's absolute bench rates
should never be compared with lane rates measured at different sampling.

## 5. What this port should measure next, ranked by what it would change

Each item names the instrument. Items 1–3 are deterministic, cheap, and each can falsify a distinct
explanation of the +77 % versus +29 % split; item 2 is the only one that makes "adapt the drafter" a
funded decision.

1. **Per-position acceptance for the four arms** (`nvidia`, `attention_fp8`, `gdn_fp8`, full FP8), at
   K=5 and K=7, on the fixture corpus. `ninfer_bench --spec dflash2 --lm-head-draft` already prints
   `accepted_per_position` (`dflash2-acceptance-baseline.md` §9 forecast). A **position-0 drop** means
   the target's first-token distribution moved; an **approximately uniform decay across positions**
   means the drafter's conditioning inputs moved (the drift hypothesis); a **cliff at one position**
   means the drafter's own dispatch or the verify path, not the target. This one run separates the
   three explanations the port currently cannot distinguish and costs two extra bench configurations.
2. **Hidden-state drift at the five taps `[5, 19, 33, 47, 61]` for each arm pair**, reported per tap as
   RMS-relative error, then mapped onto Attention Drift's α scale (is 0.1 % or 10 % of RMS the relevant
   magnitude?). The published instrument is Speculators' vLLM hidden-states extraction
   (`extract_hidden_states`, vLLM ≥ 0.18) run with each artifact as the verifier, or an engine-side
   capture at the five tap layers; the port has no such route today. This converts the port's precision
   deltas into the only noise-tolerance curve that exists, says whether a drafter adaptation could
   recover the GDN half, and localises which of the three GDN taps (5/33/61) carries the effect. Highest
   cost, highest decision value after (1).
3. **Proposal head versus full head under each arm** (`--lm-head-draft` on/off). The indexed head is the
   one component that both consumes the drafter's hidden state and is itself quantized (Q4 in the
   shipped recipe); if the +77 % is concentrated there, the remedy is head-side and no drafter retrain
   is needed. Two extra runs with an existing flag.
4. **Reproduce the split at the lane's sampling settings and at the published width K=7** (`ninfer_bench
   --spec dflash2 --draft-tokens 7`; serving-lane `request-log-jsonl` fields for the lane). The +77 %
   was measured at K=5 on a fixture corpus; the published DFlash2 evaluation is block 8 (7 draft
   tokens), and this port's own baseline note warns the absolute rates are corpus-specific. This is a
   comparability guard, not a mechanism test.
5. **Confirm the training target's precision for the companion** from the DFlash2 checkpoint's own
   config/README and training artifacts. Nearly free; it decides which direction "toward the training
   distribution" points, on which the whole mechanistic story rests.
6. **Only if (1) and (2) say the gap is conditioning drift and is closable: finetune the drafter on the
   quantized target's hidden states.** Precedent: Nota's two-stage training (+0.11 AL in a 4B setup)
   and DFlash's 1.6K-sample long-context adaptation (+1.3 τ at 8K). Instrument: SpecForge/ModelOpt
   DFlash training against a vLLM-served quantized Qwen3.8-27B — a multi-GPU job, out of scope for this
   card. This is the only path that could keep FP8 GDN's perplexity *and* recover acceptance, and it
   should not be funded before (1) and (2) attribute the split.

**A prediction that the measurements can falsify cheaply.** The port's own structural data supplies a
specific hypothesis: GDN precision lands on **3 of the 5 fused taps** while full-attention precision
lands on 2, and the last tap (61, `linear_attention`) is where a magnitude-dominated fusion would put
the most weight. That predicts the GDN half has more room to move acceptance than its perplexity share
suggests — precisely what the +29 % versus +77 % split shows. Items 1 and 2 test it directly; until
then it is a hypothesis from structure, not a result.

## Sources

- **Papers**: arXiv 2410.11305v3 (QSpec, EMNLP 2025 — §2.2 motivation, §3.1 acceptance policy, §4.3-4.4,
  baselines note on GPTQ-EAGLE); arXiv 2503.13565 (ML-SpecQD — §IV-B); arXiv 2607.04244 (Nota/AdaptFM —
  §2.1-2.3, Tables 2-5); arXiv 2508.08192 (Meta — §2.2-2.4, §3.4 "Quantized draft model", Table 2);
  arXiv 2605.09992 (Attention Drift — §3, §4.1-4.4, §5.1-5.2, Tables 1-3, Figures 10, 14); arXiv
  2503.01840v2 (EAGLE-3 — §2.1-2.2, §3, §4.2 Table 2, Table 1 T=0/1); arXiv 2602.06036v2 (DFlash — §3.2,
  §4.1-4.2, §5.4, §5.5.2-5.5.5, Tables 1, 4, 7, 9).
- **First-party docs and repos**: vLLM Speculators v0.5.0 blog (2026-05-28) and v0.3.0 blog (2025-12-13);
  `docs.vllm.ai` Speculative Decoding; vLLM issue #36629 and #54252; `NVIDIA/Model-Optimizer`
  `examples/speculative_decoding/doc/dflash.md`; `nvidia.github.io/TensorRT-LLM` Speculative Sampling
  (Medusa hidden states); `SafeAILab/EAGLE` README and issue #127; Hugging Face
  `RedHatAI/gemma-4-31B-it-speculator.dflash` card (BF16 verifier, acceptance table);
  `ml-explore/mlx-lm` discussion #890 (EAGLE-3 layer taps `{2, N//2, N−3}`).
- **Third-party guidance** (no measurements): GMI Cloud blog, *Speculative Decoding for LLM Inference*
  (INT4-vs-FP8 3-5-point claim); DigitalOcean, *Speculative Decoding on vLLM: A Configuration and
  Decision Framework* (INT4 target with BF16 draft; INT8 > INT4 order).
- **In-tree**: `docs/research/attention-topology-2026-10-07.md` (the +77 %/+29 %/+36.7 % figures, the
  prefill/decode/context costs, the recipe appendix); `docs/research/attention-site-sensitivity.md`
  (KronQ, Minima rankings); `docs/research/dflash2-acceptance-baseline.md` (published DFlash2/DFlash τ,
  per-position instruments); `docs/maintainer/dflash.md` (proposal head and selector path);
  `docs/maintainer/qwen3.8-27b-artifact.md:410` (`feature_projection [5120,25600]`);
  `tools/convert/official_recipes.py:303-313,545-565,700-720` (`_nvdiv` and Swift notes, with their
  scopes); `.audit/models-rebuild.tsv` (the `_nvdiv` rejection); the shipped artifacts'
  `components.dflash2.config.dflash_config.target_layer_ids` = `[5, 19, 33, 47, 61]` and
  `components.text.config.layer_types` (3 GDN + 2 full-attention taps).
- **[not found]** (scoped): any measurement of a *fixed* drafter's acceptance on one target at two
  precisions; any report of acceptance *improving* when a fixed drafter's target is quantized; any
  attention-vs-MLP-vs-linear-attention quantization ablation read through a drafter's acceptance;
  any first-party statement of the precision of the target that generated the DFlash2 companion's
  training hidden states; any acceptance figure for a quantized verifier paired with a BF16-trained
  DFlash speculator (vLLM/Red Hat report ITL only). Searched: the papers above, vLLM/SGLang/EAGLE/
  Model-Optimizer issue trackers and docs, Speculators and SpecForge documentation, and the model
  cards cited above.
