# Do external sources warrant this port's three quantization decisions?

Research note. Written 2026-10-01. Answers a single question: three decisions this port records
as open items because no external evidence could be found for them — are they warranted, or
should they be closed?

| | decision | what external evidence says | verdict-shaped summary |
|---|---|---|---|
| **(A)** | quantize the vision tower (all four artifacts) | Near-universal convention **against**: every engine vendor, every recipe library, and every one of the 130 VLM NVFP4 checkpoints surveyed except two leaves it BF16, with **two documented reasons** (export breakage/garbage embeddings; low accuracy-per-byte). No source *measures* a quantized-vs-BF16 vision tower. | **Contradicted as a convention; unmeasured as a trade.** The burden of proof named in the port's own note ("no precedent, and perplexity cannot see it, so it needs its own check") is now *discharged against* the choice: there is abundant precedent for the opposite and none for this. The specific failure mode is documented upstream and this model is not exempt. |
| **(B)** | keep both W8 endpoints at Q8 | Convention is **split three ways** (BF16 / FP8 / NVFP4), with a stated reason only for the BF16 exclusion. **Q8 appears nowhere** — the port's choice is unique among ~90 surveyed checkpoints. But one clean external ablation exists. | **Does not justify, and mildly undercuts.** "The only choice with a measurement behind it" is true of this port and false as a claim about the field: an external head-only ablation exists and is the *only* head-isolated number I found. No source measures Q8 vs NVFP4 or Q8 vs FP8 for the head. |
| **(C)** | recover `activation_input_divisor` from an FP8-calibrated `input_scale` for A4 | NVIDIA ships a dedicated `NVFP4ActHeadroomCalibrator` whose docstring names **plain max** as the rule it replaces and states its failure mode. **The premise as stated does not hold**: the recovered divisor is arithmetically the port's own max-abs rule, not an 8-bit quantity reused for 4-bit. But it *is* plain max, so the documented concern applies to it by a different route than the note assumes. | **The premise is wrong; the underlying concern survives.** `6/input_scale` on an FP8 site equals `2688/amax` exactly — the port's own `FULL_RANGE/peak` formula, verified against NVIDIA's real tensor bytes at 24 sites. What is left is that it is *max*-derived, which is precisely what NVIDIA documents as leaving no headroom. NVIDIA's replacement is opt-in, recent (2026-08-03), unmeasured on any checkpoint, and **not used by NVIDIA's own Qwen3.8-27B recipe**. |

Evidence classes used below:

| label | meaning |
|---|---|
| **[MEASURED-HERE]** | measured in this session, 2026-10-01, from the named primary artifact or command |
| **[SRC-CODE]** | read in a primary source's source code or config in this session |
| **[VENDOR]** | a model card or doc written by the party that produced the checkpoint; first-party, not independently reviewed |
| **[THIRD]** | an independent measurement by a named third party, quoted with its setup |
| **[IN-TREE]** | already recorded in this repo, with the measurement it rests on |

---

## 0. Two corrections to the framing of the question

Both are **[MEASURED-HERE]** from primary sources and both change what the external evidence
bears on.

### 0.1 The port does not quantize the vision tower to NVFP4

The question states the port "quantizes the vision tower to NVFP4 in all four shipping
artifacts". It does not. `tools/convert/official_recipes.py:24-39` assigns the vision tower
**Q4/Q5/Q6/Q8 grouped-integer** formats (`q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`,
`q8_g32_fp16`), encoded by `grouped_absmax` — not `nvfp4_maxabs`, and not a float format at all:

```python
if name == "vision/patch_embedding":          format = Q6
elif name.startswith("vision/merger/"):        format = Q8
elif name.endswith(("/attention/query", "/attention/key",
                    "/attention/value", "/mlp/fc1")):  format = Q4
else:                                          format = Q5
```

This repo's own review note records the same thing **[IN-TREE]**:
`docs/research/nvfp4-conversion-recipe-review.md` §0 ("vision is Q4/Q5/Q6/Q8") and §3.1b
("Q4 on `attention/{query,key,value}` and `mlp/fc1`, Q8 on the `merger`, Q6 on
`patch_embedding`, Q5 on everything else"). §3.2 of that note measured the endpoints at Q8 and
calls them "the two W8 endpoints", so "W8 endpoints" in item (B) likewise means **Q8, an integer
group format**, not an 8-bit float. Conclusions below use the actual formats.

The correction matters because it moves the external question. No published engine ships a
**4-bit grouped-integer FP16-scale** vision tower either, so the convention evidence is
unchanged — but the reason "NVFP4's block-size divisibility" does **not** apply, and one of
NVIDIA's two documented reasons is about a divisibility failure that cannot occur here (see
§1.3).

### 0.2 The port's A4 divisor is not "sized for 8-bit activations"

Item (C)'s premise is that the recovered divisor is "a divisor sized for 8-bit activations being
used for 4-bit ones". Arithmetically this is not what happens; see §3.2, where I verify against
NVIDIA's real checkpoint bytes that the recovered value is **bit-for-bit the port's own max-abs
formula**. The premise is restated correctly in `docs/perplexity-baseline.md:238-241`
("a divisor sized for an 8-bit activation is not obviously roomy enough for a 4-bit one") — that
is a hypothesis, and §3.2 tests it. It does not survive. What survives is §3.1's separate point
about *max* calibration, which is a real and separately documented concern.

---

## 1. (A) Vision tower quantization

### 1.1 The convention is near-universal, and it is against

**[SRC-CODE]** NVIDIA ModelOpt's shared exclusion list, read at `main`
(`modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml:59-68`), disables the whole
vision branch in *every* recipe that imports it:

```yaml
- quantizer_name: '*embed_vision*'
  enable: false
- quantizer_name: '*vision_tower*'
  enable: false
- quantizer_name: '*visual*'
  enable: false
- quantizer_name: '*vision_model*'
  enable: false
- quantizer_name: '*multi_modal_projector*'
  enable: false
```

**[SRC-CODE]** The same unit's own comment states the reason, with bug IDs:

> "Recipes that enable bare `*weight_quantizer` / `*input_quantizer` or `*mlp*` wildcards otherwise
> also match the vision tower … quantizing the vision branch **crashes export / produces garbage
> image embeddings** on VL models (gemma-4, Qwen3.5-VL — NVBugs 6293731, 6293762, 6294017;
> Llama-4-Scout — NVBug 6359097, where `vision_model.patch_embedding.linear` has in_features=588,
> not divisible by the NVFP4 block size)."

`modelopt_recipes/model_type/gemma4/ptq/README.md:16-19` repeats it and adds a quality claim:
quantizing the vision branch to INT4 "crashes export (`pack_int4_in_uint8` index-out-of-bounds,
NVBug 6294017) **and is accuracy-harmful**".

**[SRC-CODE]** llm-compressor's FAQ states a *different* reason, and it is the one that applies to
a group-quantized tower, since it is about cost rather than format mechanics
(`docs/faq/faq.md`, question 5):

> "the non-textual component is excluded from quantization, as it generally **has fewer
> parameters and is more sensitive**."

and `examples/multimodal_vision/README.md:22`:

> "Most examples do not demonstrate quantizing separate vision encoder parameters if they exist,
> as **compressing these parameters offers little benefit with respect to performance-accuracy
> tradeoff**."

Both are stated reasons, not measurements. Neither cites a number.

**[SRC-CODE]** llm-compresser's own authoring skill states the rule as mandatory
(`examples/.agents/skills/shared_quantization.md:32`): *"**Rule:** Always ignore `lm_head`, any
vision tower layers, and any gating/routing layers."* All 15 of its multimodal examples obey it;
I checked each `ignore=` list under `examples/multimodal_vision/` **[MEASURED-HERE]**: 15/15 name
the vision branch.

**[SRC-CODE]** TensorRT-LLM documents the same position in its support matrix
(`docs/source/features/quantization.md:119-122`):

> "The vision component of multi-modal models … **uses FP16 by default**. The language component
> decides which quantization methods are supported by a given multi-modal model."

and its Model Support Matrix lists **no multimodal model at all** under the NVFP4 column — LLaVA,
VILA and both BLIP2 rows are `.` for NVFP4 **[MEASURED-HERE]**.

**[SRC-CODE]** vLLM hard-codes the exclusion regardless of checkpoint contents
(`vllm/model_executor/layers/quantization/modelopt.py:233-238`), returning
`UnquantizedLinearMethod()` for any prefix containing `vision_tower`, `vision_model` or
`vit_large_projector`.

**[MEASURED-HERE]** The base model vendor agrees. `Qwen/Qwen3.8-27B-FP8`'s own
`quantization_config.modules_to_not_convert` has **882 entries**, of which **330 are the 110
vision Linears** (each named twice, with and without the `model.` prefix), plus `lm_head` and
`embed_tokens`. Its index has **0** packed or scale tensors among 333 visual tensors. This is the
model's own official FP8 release, not a third party's choice.

### 1.2 The checkpoint survey: 128 of 130 leave it BF16

**[MEASURED-HERE]** I enumerated HF models matching `NVFP4-Instruct`, `NVFP4-VL`, `-NVFP4`,
`NVFP4`, filtered to vision-language architectures, and read each `config.json`:

| | count |
|---|---:|
| VLM checkpoints with a `quantization_config` | 130 |
| …that name a vision module in `ignore` (104–207 entries, the usual shape) | **94** |
| …with no vision exclusion, of which most are MLX-format or text-only misdetections | 36 |
| …that actually place a quantizer **on** a vision module | **2** |

The two exceptions:

- **`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`** — `group_mxfp8_vision` targets
  `model.visual.blocks.*.{attn.proj,attn.qkv,mlp.linear_fc1}` (MXFP8) plus
  `group_w4a16_nvfp4_vision_fc2` on the `linear_fc2`s (NVFP4 weight-only). Its card
  **[VENDOR]** states the vision encoder was *"MXFP8 … NVFP4 FC2; remaining parameters BF16"*
  and that *"Vision and MTP retain their weights. They are included in the release but were not
  part of text distillation."* Its only quality table is AA-LCR 79.4 / GPQA 78.8 — **no vision
  benchmark**.
- **`rdtand/Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB`** — all 110 visual Linears at **NVFP4
  W4A4**, i.e. 4-bit weights *and* activations in the vision tower. Its card **[VENDOR]** is the
  most on-point external artifact I found for this exact model: *"All 110 visual Linears are
  fixed to calibrated native NVFP4 W4A4 … rendered from image-conditioned calibration
  activations"*. It also reports a vision-adjacent measurement (§1.4).

NVIDIA's own VLM NVFP4 checkpoints are all on the other side: `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4`
ignores `model.visual*` **[MEASURED-HERE]**; `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` ignores
`visual*`; `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD` ignores `model.layers.vision_model*`.

### 1.3 One of NVIDIA's two documented reasons does not apply here; the other does

NVIDIA's comment gives two failure modes. The divisibility one is model-specific: Llama-4-Scout's
`vision_model.patch_embedding.linear` has `in_features=588`, "not divisible by the NVFP4 block
size". **[MEASURED-HERE]** Qwen3.8-27B's vision patch embedding is `3 × 2 × 16 × 16 = 1536`
in-features against `hidden_size=1152`, and `1536 % 128 == 0`, `1152 % 128 == 0`, merger
`out_hidden_size=5120`, `5120 % 128 == 0`. Every vision matrix in this model is
NVFP4-divisible, so the export-crash mechanism NVIDIA documents cannot fire here.

The other reason is not model-specific and does apply: **"produces garbage image embeddings"**,
plus gemma-4's "accuracy-harmful". Both are asserted, not measured, and neither names a
benchmark. **[VENDOR]** NVIDIA's `fp8_vision` recipes for `qwen3_vl`/`qwen3_5` exist and are
explicitly opt-in, and they *still* hold back the patch embedding and the vision-attention BMMs
in high precision (`model_type/qwen3_vl/ptq/vision_fp8.quant_cfg.yaml:31-34`):

```yaml
- quantizer_name: '*visual.*patch_embed*'
  enable: false
- quantizer_name: '*visual.*_bmm_quantizer'
  enable: false
```

So the most aggressive vision recipe NVIDIA ships quantizes 8-bit, not 4-bit, and still excludes
the patch embedding outright.

### 1.4 No source measures a quantized-vs-BF16 vision tower

This is the negative finding the port's own note predicted, and it holds. I searched:

- ModelOpt's recipe tree for any accuracy report on a vision-quantized checkpoint — **[MEASURED-HERE]**,
  none of the `fp8_vision*` recipes or the model-type READMEs cite a benchmark number.
- llm-compressor's multimodal examples and `tests/lmeval/vl_configs/` — the one VL NVFP4 eval
  config **[SRC-CODE]** (`vl_nvfp4_fp8.yaml`) asserts chartqa `exact_match 0.53 / relaxed 0.75 /
  anywhere 0.79` against an FP8 config's `0.53 / 0.75 / 0.80`. That is a **regression gate on a
  shared recipe**, not a vision-tower ablation: the recipe behind it
  (`tests/e2e/recipes/non_uniform/recipe_nvfp4_fp8_mixed.yaml:4`) ignores only `["lm_head"]` and
  targets `self_attn`/`down_proj`/`gate|up_proj` — the language model. No vision module is
  quantized, so no number in that file bears on (A).
- The two quantizing checkpoints' own cards — `local-inference-lab` publishes no vision metric;
  `rdtand` publishes an eight-image suite (below) and says so.
- `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD` **[VENDOR]**: ChartQA 90.0% FP4 vs 89.7%
  BF16 — but that checkpoint *ignores* `model.layers.vision_model*`, so its vision tower is BF16
  and the number is about the language model.

The closest thing to a measurement anywhere is `rdtand`'s, and it is a fixture, not a benchmark
**[VENDOR]**, reported verbatim:

> "The eight-image suite is intentionally reported **without inflating it into a general vision
> benchmark**. The quantized artifact and BF16 source selected the same multiple-choice letter on
> every image, including the same one miss, in both deterministic replicates. … Free-form captions
> were close paraphrases but **0/8 exact string matches**, and **one image received a materially
> different action interpretation**. The fixture result is therefore **choice parity on this
> suite, not broad caption equivalence**."

Same card, on the cost side: retaining the vision tower under a fixed 20 GB cap raises the
*text* screening KL by 9.16% (0.0402201669 → 0.0439023734) versus its visionless sibling —
i.e. the same trade the port is making, priced by the third party who made it. And the card is
careful that this isolates neither factor: *"These comparisons isolate neither vision nor
allocation method by themselves."*

### 1.5 (A) verdict

- The convention is **against** the port's choice, without exception among the engines and
  libraries that publish recipes, and with two **stated** reasons (crash/garbage embeddings;
  few parameters, more sensitive, little benefit).
- The port's format (Q4/Q5/Q6/Q8 grouped integer) dodges NVIDIA's *divisibility* reason but not
  the *garbage-embeddings* or *accuracy-harmful* claims, and not llm-compressor's
  accuracy-per-byte reasoning.
- **No source measures it**, so nothing here quantifies the downside. One third-party artifact
  quantifies the upside (the ~600 MiB the port estimates) as a 9.16% text-KL cost under its own
  budget — a different allocation, quoted for what it is.
- The port's own note said this "needs its own check". The external record does not perform that
  check; it establishes that the check would be arguing against a unanimous field position.

---

## 2. (B) The two W8 endpoints at Q8

### 2.1 The convention is split three ways, and Q8 is in none of them

**[MEASURED-HERE]** Across ~90 surveyed Qwen3.8-27B / Qwen3.6 checkpoints with a
`quantization_config`:

| head format | examples |
|---|---|
| **left BF16** (in `ignore`) | `QUASAR-QAT/…-QUASAR-NVFP4`, `RedHatAI/Qwen3.8-27B-NVFP4`, `Inferact/…`, `RedHatAI/Qwen3.8-27B-INT4`, `bottlecapai/ThinkingCap-…`, `Qwen/Qwen3.8-27B-FP8` (official) |
| **FP8** (8-bit float) | `unsloth/Qwen3.8-27B-NVFP4`, `Dragoy/Swift-Qwen3.8-27B-abliterated-NVFP4` |
| **NVFP4** (4-bit float) | `nvidia/Qwen3.8-27B-NVFP4`, `RadixArk/Qwen3.8-27B-NVFP4`, `SmokeyPete/…`, `gittensor-model-hub/…` |
| **Q8** (this port) | **none found** |

NVIDIA's own recipe is explicit **[SRC-CODE]**
(`modelopt_recipes/model_type/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast.quant_cfg.yaml:90-93`):

```yaml
  # Re-enable NVFP4 on lm_head weights. Must come after
  # default_disabled_quantizers, which disables `*lm_head*`.
  - quantizer_name: '*lm_head*weight_quantizer'
    cfg: {$import: nvfp4}
```

and its model-type recipe does the same for the dense family at
`w4a16_nvfp4-fp8_attn-kv_fp8_cast.quant_cfg.yaml:76-84`, with the comment
*"Qwen-specific exclusions: linear-attention sub-modules that are not in the reference recipe, and
any visual / MTP siblings on multimodal releases."* So NVIDIA's position is that vision/MTP are
*siblings to be excluded* while the head is *re-enabled*.

The `.quant_summary.txt` in NVIDIA's own checkpoint confirms it fires **[MEASURED-HERE]**:
`lm_head.weight_quantizer  StaticBlockScaleQuantizer((2, 1) bit … calibrator=NVFP4MSECalibrator)`
against 208 `TensorQuantizer(disabled)` weight quantizers, and its `config.json` carries
`lm_head` in the NVFP4 group **[MEASURED-HERE]**.

The only stated reason anywhere for the BF16 exclusion is llm-compressor's **[SRC-CODE]**:

> "Typically, all linear layers are quantized except the `lm_head` layer. This is because the
> `lm_head` layer is the last layer of the model and **sensitive to quantization, which will
> impact the model's accuracy**."

**[SRC-CODE]** TensorRT-LLM documents the opposite conclusion for the same checkpoint and says why
(`tensorrt_llm/_torch/models/modeling_qwen3_5.py:431-438`): *"ModelOpt MIXED_PRECISION exports for
Qwen3.5/3.6 quantize lm_head to W4A16_NVFP4 … cutting the lm_head GEMM's weight traffic 4x vs the
bf16 dequant fallback — the decode lm_head is purely weight-bandwidth-bound."* Its guards include
`vocab_size % tp_size == 0`, with the aside that *"vocab 248320 divides all common tp"*.

### 2.2 A head-only ablation exists externally — and it is the only one

**[THIRD]** `Qwen/Qwen3.8-27B` discussion #192, "NVFP4 Shootout (Quality and Speed)", by `Rieker`
(2026-09-09, updated 2026-09-12). This is the single most useful external finding for (B). It
compares **`RadixArk/Qwen3.8-27B-NVFP4`** against **`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`**,
which RadixArk's own cards confirm differ in exactly one tensor **[VENDOR]**: *"The only
difference from the source checkpoint is that `lm_head` is not quantized … All other tensors are
identical."* **[MEASURED-HERE]** confirmed from their indexes: `lm_head.weight` + 3 scale tensors
vs `lm_head.weight` alone; total_size 21,921,428,072 vs 23,749,063,264 (1.83 GB).

| variant | PPL wiki | KLD text | KLD code | GSM8K | tok/s |
|---|---:|---:|---:|---:|---:|
| RadixArk, **BF16** head | 8.286 (+0.293) | 0.0592 | 0.0263 | 97.0 | 22.74 |
| RadixArk, **NVFP4** head | 8.385 (+0.392) | 0.0952 | 0.0335 | 97.4 | 34.62 |

The author's reading **[THIRD]**: *"Quantizing the LM head to NVFP4 (vs keeping it BF16) costs ~0.1
PPL and ~0.04 KLD-text but buys ~12 tok/s (34.6 vs 22.7) and ~1.8 GB. A clear accuracy↔speed
tradeoff localized to the output head."*

Scope, stated plainly: this is one pair, on one model, on DGX Spark / vLLM nightly, and the
head-formats compared are **NVFP4 vs BF16** — not NVFP4 vs Q8 and not FP8 vs Q8. It ranks RadixArk's
NVFP4-head build *worst* of the six variants on fidelity (KLD 0.0952, worst in the table).

**[THIRD]** The same discussion's headline result for NVIDIA's own checkpoint — PPL +0.146, KLD
text 0.0758, 38.6 tok/s — is the external corroboration that NVIDIA's NVFP4 head is not a
catastrophe on this model.

### 2.3 (B) verdict

- The port's parenthetical, *"the right default and the only choice with a measurement behind
  it"*, is **true within this port** and **not true of the field**: at least one external
  head-isolated measurement exists, and the port's own does not appear in it.
- The reasoning "higher precision at the head" is supported by convention and by one stated
  reason (llm-compressor's "sensitive … will impact the model's accuracy"), plus this port's own
  −3.45 % / −0.79 % **[IN-TREE]**.
- But the field is **not** unanimous that Q8 or even FP8 is right: NVIDIA ships NVFP4 g16 with
  Local-Hessian/MSE scales and publishes end-to-end numbers (GPQA 88.92 → 88.01, MMMU-Pro 75.14 →
  74.86) **[VENDOR]**; TensorRT-LLM deliberately consumes that head for a 4x traffic cut.
- **Nobody measures Q8 vs NVFP4, or Q8 vs FP8, for the head.** The 1.26 GiB the trade would return
  is the port's arithmetic; the matching quality cost has no external measurement at any head
  format. So the specific comparison the port would need to make is unmeasured everywhere,
  including here.

---

## 3. (C) The A4 activation divisor

### 3.1 NVIDIA documents the failure mode of the rule this divisor follows

**[SRC-CODE]** `NVFP4ActHeadroomCalibrator`, `modelopt/torch/quantization/calib/nvfp4_act_headroom.py:36-59`,
written to replace plain max on the NVFP4 activation global scale:

> "NVFP4 scales a tensor in two levels: an FP8-E4M3 scale per 16-element block, and one per-tensor
> global scale (`amax`). **Plain max calibration sets the global scale from the largest block seen
> during calibration, which leaves no room above it: any activation larger than the calibration max
> saturates.**
>
> … `upper_percentile` defaults to 99.99 rather than the literal maximum on purpose. A single freak
> block far above the rest would otherwise drag the global scale up so far that every other block's
> FP8 block scale falls below subnormal and flushes to zero — losing the whole tensor to protect
> one value."

**[SRC-CODE]** Defaults: `anchor_percentile=1.0`, `upper_percentile=99.99`, `rho=16384.0`, with
`amax = max(rho * anchor, upper)`. It applies only to NVFP4 **dynamic-block input** quantizers;
`SequentialQuantizer` activation quantizers raise rather than fall back silently
(`model_calib.py:526-554`). **[SRC-CODE]** It is **opt-in**: the general W4A4 recipe
`nvfp4_default-kv_fp8_cast.yaml:30` is `algorithm: max`, and the headroom variant is a separate
file.

**[SRC-CODE]** The PR that added it (**NVIDIA/Model-Optimizer#2028**, merged 2026-08-03,
`cjluo-nv`) states the class of result and, importantly, its scope: *"on a tensor with a single
block seven orders of magnitude above the rest, that flushes 99.998% of elements to zero, versus
6.7% when the rare blocks are clipped instead. On a benign wide-range tensor the two choices differ
by **0.4% relative MSE**."* That is a **synthetic-tensor** measurement. Its end-to-end section
verifies only that a 30B MoE *exports* correctly — 6,268 activation quantizers calibrated, 19 GB
vs 62 GB BF16 — with **no accuracy number**. 32 of 6,268 quantizers (0.5 %) warned that their
per-block range was too wide for `rho` to clear.

### 3.2 …and the port's divisor is arithmetically its own max-abs rule, not an 8-bit quantity

This is the substantive test of (C)'s premise, and it is **[MEASURED-HERE]** against NVIDIA's real
checkpoint bytes.

NVIDIA's export convention, **[SRC-CODE]** and confirmed numerically: an FP8 per-tensor activation
records `input_scale = amax / 448`; an NVFP4 one records `input_scale = amax / (6 × 448)`. The port
therefore computes `divisor = 6 / input_scale` for an FP8 site, which is

```
6 / (amax_fp8 / 448)  =  6 × 448 / amax_fp8  =  2688 / amax
```

and `tools/convert/calibration.py:15` defines the port's own rule as `d_x = 2688 / amax_site`. **The
recovered divisor and the port's own max-abs divisor are the same expression.** Not "similar" — the
same. The factor of 6 is exactly what converts an FP8 scale into the A4 block-scale orientation, so
nothing "sized for 8-bit" survives into the 4-bit path.

Verified numerically. I read the raw F32 tensors out of
`nvidia/Qwen3.8-27B-NVFP4`'s safetensors shards over HTTP Range requests (3 shards, 9.97 GB each,
offsets resolved through the header; a `Content-Range` check rejects any response that ignores the
range) and compared against `tools/convert/qwen3_8_27b_nvfp4_calibration.json` at 24 sites across
6 site classes and layers {0, 1, 16, 32, 48, 63}:

| site | n | port's local divisor | NVIDIA-derived divisor | ratio (min–max) |
|---|---:|---:|---:|---|
| `gdn/input_projection` | 5 | 34.91 – 114.99 | 34.03 – 114.38 | **0.972 – 1.000** |
| `attention/input_projection` | 1 | 52.19 | 42.00 | 0.805 |
| `attention/output_projection` | 1 | 33.60 | 24.66 | 0.734 |
| `gdn/output_projection` | 5 | 129.54 – 1075.20 | 47.79 – 186.18 | 0.085 – 0.494 |
| `mlp/gate_up_projection` | 6 | 49.32 – 1228.80 | 261.18 – 792.77 | 0.645 – 0.984 |
| `mlp/down_projection` | 6 | 6.65 – 886.76 | 1.28 – 96.86 | 0.025 – 0.665 |

Two things follow. First, the identity `6/input_scale == 2688/amax` holds at every site (it is
algebraic, and the values confirm it). Second, and separately: **NVIDIA's calibrated amax and the
port's own locally measured amax disagree substantially at 20 of 24 sites** — most extreme at
`mlp/down_projection` layer 0, where NVIDIA's implies amax ≈ 2092 and the port measured ≈ 5.12, a
ratio of 0.025 in divisor. The two calibrations used different corpora (NVIDIA: 512 samples ×
2,048 tokens of Nemotron-Post-Training-Dataset-v3; the port: its own corpus). Only the `gdn/input`
sites — the ones whose activation is the residual hidden state, shared across many projections —
agree within 3 %.

So the honest statement of (C)'s exposure is **not** "an 8-bit divisor used for 4-bit". It is:
*the divisor is max-derived, with no headroom above the calibrated maximum* — which is precisely
what §3.1 documents — *and its max came from a different corpus than the port's own measurement
would have used, and the two disagree by up to 40x at some sites.*

### 3.3 The concern applies to the port's *own* calibration too, and to NVIDIA's own recipe

**[SRC-CODE]** The port's `calibration.py:173` computes `divisors = {name: FULL_RANGE / peak}`, with
`peak` the per-site `abs().amax()` over one forward pass. That is the same max-based rule. §3.2's
measurement shows the port's divisors and NVIDIA's are *different numbers for the same tensors*,
not that either is 8-bit-shaped.

**[SRC-CODE]** NVIDIA's own recipe for this exact model uses max:
`modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_max.yaml:25` is
`algorithm: max`, and the Local-Hessian variant only overrides the *weight* algorithm
(`nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml:27-32`, `method: local_hessian`, no activation
override). **[MEASURED-HERE]** confirmed in the shipped checkpoint: all 401 enabled activation
quantizers record `calibrator=MaxCalibrator`, while the 193 NVFP4 weight quantizers record
`NVFP4MSECalibrator`. So **the producer of the very checkpoint this port recovers divisors from
shipped plain max on A4 activations**, and has not adopted its own headroom calibrator on it.

A **[MEASURED-HERE]** data point from this repo corroborates the direction of the concern without
settling it: `mlp.gate_proj` records `amax=[0.0047, 0.4219]`, a 90× spread between the smallest
block and the tensor max **[IN-TREE]** (`docs/perplexity-baseline.md:233`). The headroom
calibrator's `upper_percentile=99.99` exists for exactly that shape.

### 3.4 (C) verdict

- The stated premise ("divisor sized for 8-bit activations, reused for 4-bit") is **false**, and
  demonstrably so from the producer's own bytes. Nothing in the arithmetic is mis-sized.
- What remains is a **real and separately documented** concern: the value is max-derived, and
  NVIDIA ships a calibrator whose docstring says max-derived leaves no headroom and names the
  subnormal-flush failure. That concern is about the *rule*, not the format.
- But NVIDIA's own Qwen3.8-27B recipe and its own shipped checkpoint both use plain max on these
  sites, so the port is following the producer's published choice, not deviating from it.
- NVIDIA's replacement is **opt-in, four months old, and carries no published accuracy number** —
  the PR's evidence is a synthetic-tensor MSE figure plus an export smoke test. The
  `nvfp4_act_headroom-kv_fp8_cast.yaml` recipe's only field experiment (32 of 6,268 quantizers
  warning) was never converted into a score.
- **Nothing external measures an FP8-derived vs FP4-calibrated A4 divisor on any model.** The
  question as posed has no external answer; the question it was *derived from* — max vs headroom —
  has a documented concern and no measurement either.

---

## 4. What I searched and did not find

**Searched, found nothing:**

1. **Any accuracy or perplexity measurement isolating a quantized-vs-BF16 vision tower**, on any
   model, at any format. Searched ModelOpt's recipe tree and model-type READMEs, llm-compressor's
   multimodal examples and `tests/lmeval/vl_configs/`, and the model cards of every checkpoint found
   that quantizes a vision module. The nearest is `rdtand`'s 8-image fixture, which its own author
   declines to call a benchmark.
2. **Any documented reason to quantize a vision tower.** Two reasons are documented for *not*
   doing so; none for doing so. `local-inference-lab` and `rdtand` publish vision-quantized
   checkpoints without a quality argument.
3. **Any Q8 or grouped-integer `lm_head`** in ~90 surveyed Qwen3.8-27B / Qwen3.6 checkpoints. Q8
   endpoints appear to be this port's alone among the formats surveyed.
4. **Any measurement of Q8 vs NVFP4, or Q8 vs FP8, for the output head**, anywhere. The only
   head-isolated number I found is NVFP4 vs BF16 (Rieker, §2.2).
5. **Any NVIDIA documentation that an FP8-calibrated scale is unsuitable for an NVFP4 block** —
   i.e. no source supports (C)'s premise as stated. Searching ModelOpt for `nvfp4_act_headroom`,
   `input_scale`, `448`, and the export paths found no such guidance.
6. **Any published accuracy number for `nvfp4_act_headroom`** on any checkpoint. The PR reports a
   synthetic-tensor MSE comparison and an export smoke test; no benchmark appears in the PR, the
   changelog entry, or the recipes guide.
7. **Any published perplexity for this model with a quantized vision tower.** The only Qwen3.8-27B
   PPL figures I found are text-only (NVIDIA +0.146, unsloth +0.138, QUASAR +0.254, RadixArk
   +0.293/+0.392, `rdtand` +3.0119 % on its own corpus) **[THIRD]**, [VENDOR].

**Searched, and worth recording as a positive finding:**

8. NVIDIA's *own* VLM NVFP4 checkpoint excludes the vision tower with the pattern `model.visual*`
   **[MEASURED-HERE]**, and ModelOpt's shared exclusion unit disables the whole vision branch with
   two stated reasons and five NVIDIA bug IDs **[SRC-CODE]**.
9. NVIDIA's *own* Qwen3.5/3.8-family recipes re-enable NVFP4 on `lm_head` weights immediately after
   that unit disables it **[SRC-CODE]** — so the head and the vision tower are treated differently
   by the same authors, in the same file, on purpose.
10. The divisor arithmetic in §3.2 is verified against NVIDIA's actual tensor bytes, not inferred.

**Known limits of this note:**

- The HF survey is a **census of published configs**, not a review of each checkpoint's index. For
  the four checkpoints that matter most to this port (NVIDIA, unsloth, QUASAR, and the two RadixArk
  variants) I read the index as well as the config **[MEASURED-HERE]**; for the rest the config is
  the evidence, and a checkpoint can in principle disagree with its own config.
- §3.2 samples 24 of the port's 256 calibration sites across 6 site classes. It establishes the
  identity and the *existence* of corpus disagreement; it does not bound the disagreement over all
  sites, and the 40× figure is one site's, not a distribution.
- Every quality number quoted is either **[VENDOR]** (a producer's own card) or **[THIRD]** on one
  engine and one machine. None was reproduced here.

---

## 5. Sources

**NVIDIA Model Optimizer** (`github.com/NVIDIA/Model-Optimizer` @ `main`, read 2026-10-01):
`modelopt/torch/quantization/calib/nvfp4_act_headroom.py`; `modelopt/torch/quantization/model_calib.py`
(`_is_nvfp4_dynamic_activation_quantizer`, `nvfp4_act_headroom_calibrate`);
`modelopt/torch/export/quant_utils.py`; `modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml`;
`configs/ptq/units/base_disable_all.yaml`; `configs/auto_quantize/units/base_disabled_layers.yaml`;
`configs/numerics/nvfp4.yaml`, `nvfp4_static.yaml`;
`modelopt_recipes/general/ptq/nvfp4_default-kv_fp8_cast.yaml`, `nvfp4_act_headroom-kv_fp8_cast.yaml`,
`nvfp4_experts_only_input_scale1-kv_fp8_cast.yaml`;
`modelopt_recipes/model_type/qwen3_5/ptq/w4a16_nvfp4-{fp8_attn,mse-fp8_attn}-kv_fp8_cast.quant_cfg.yaml`,
`fp8_vision-kv_none.yaml`, `fp8_vision_lm-kv_fp8_cast.yaml`; `modelopt_recipes/model_type/qwen3_vl/ptq/vision_fp8.quant_cfg.yaml`;
`modelopt_recipes/model_type/gemma4/ptq/README.md`;
`modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_{max,local_hessian}.yaml`;
`CHANGELOG.rst`; `docs/source/guides/{0_support_matrix,10_recipes,_choosing_quant_methods}.rst`;
`tests/examples/hf_ptq/test_hf_ptq_vision_quantization.py`. PR **#2028** "Add nvfp4_act_headroom
activation calibration for NVFP4" (merged 2026-08-03). Blog "Improving NVFP4 Accuracy with
Local-Hessian Weight Scales", 2026-09-09.

**NVIDIA TensorRT-LLM** (`github.com/NVIDIA/TensorRT-LLM` @ `main`):
`docs/source/features/quantization.md`, `docs/source/features/multi-modality.md`;
`tensorrt_llm/_torch/models/modeling_qwen3_5.py` (`_lm_head_nvfp4_enabled`, `_normalize_exclude_modules`);
`tensorrt_llm/_torch/models/modeling_multimodal_encoder.py`; `examples/quantization/README.md`;
`examples/quantization/quantize_mixed_precision_moe.py`.

**llm-compressor** (`github.com/vllm-project/llm-compressor` @ `main` — note the repo moved out of
`neuralmagic`): `docs/faq/faq.md`; `examples/.agents/skills/{shared_quantization,nvfp4/SKILL}.md`;
`examples/multimodal_vision/README.md` and all 15 `*_example.py` in that directory;
`examples/autoround/quantization_w4a4_fp4/{qwen3_vl_example.py,README.md}`;
`docs/key-models/qwen3.5/nvfp4-vl-example.md`;
`tests/lmeval/vl_configs/*.yaml`, `tests/lmeval/configs/*.yaml`;
`tests/e2e/recipes/non_uniform/recipe_nvfp4_fp8_mixed.yaml`.

**compressed-tensors / vLLM** (`github.com/vllm-project/{compressed-tensors,vllm}` @ `main`):
`quant_scheme.py` (`NVFP4`, `NVFP4A16`), `quant_args.py` (`DynamicType.LOCAL`),
`lifecycle/{initialize,forward}.py`, `compressors/nvfp4/helpers.py`;
`vllm/model_executor/layers/quantization/modelopt.py`,
`.../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`,
`.../utils/nvfp4_emulation_utils.py`.

**HF checkpoints** (read 2026-10-01): `nvidia/Qwen3.8-27B-NVFP4` (`README.md`, `config.json`,
`hf_quant_config.json`, `.quant_summary.txt` (2,257 lines), `model.safetensors.index.json`, and raw
F32 `input_scale` / `weight_scale` / `weight_scale_2` values read by HTTP Range);
`nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4`, `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`,
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`, `nvidia/Qwen3.8-Flash-Next-NVFP4`,
`nvidia/DeepSeek-V4.1-Flash-NVFP4`, `nvidia/Kimi-K3-NVFP4`, `nvidia/Gemma-4-{26B-A4B,31B-IT}-NVFP4`;
`Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.6-27B-FP8`;
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, `unsloth/Qwen3.8-27B-NVFP4`, `Swift-Qwen3.8-27B-NVFP4`;
`RadixArk/Qwen3.8-27B-NVFP4{,-BF16-LMHead}` (config, index, README);
`RedHatAI/{Qwen3.8-27B-NVFP4,Llama-4-Maverick-17B-128E-Instruct-NVFP4}`;
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`;
`rdtand/Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB` and `-Qwen3.6-27B-PrismaSCOUT-…`;
plus ~110 further configs enumerated by script.

**Third-party**: `Qwen/Qwen3.8-27B` discussion #192 "NVFP4 Shootout (Quality and Speed)"; Red Hat
Developers, "Accelerating large language models with NVFP4 quantization", 2026-02-04.

**Read in this tree** (for the corrections in §0 and the divisor rule in §3.2):
`tools/convert/official_recipes.py:12-51, 286-300, 758-816`;
`tools/convert/calibration.py:1-26, 150-208`; `tools/convert/quantization/nvfp4.py:1-31`;
`tools/convert/methods.py:196-220`; `tools/convert/qwen3_8_27b_nvfp4_calibration.json`;
`docs/perplexity-baseline.md:205-241`; `docs/research/nvfp4-conversion-recipe-review.md` §0, §3.1b,
§3.2, §3.4.
