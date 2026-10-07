# Are the four Qwen3.8-27B NVFP4 artifacts built the way they should be?

Research note. Written 2026-09-28. Companion to
[`dflash2-nvfp4-cluster-merge-findings.md`](dflash2-nvfp4-cluster-merge-findings.md) (which owns the
Op side of the NVFP4 drafter route) and to
[`../maintainer/artifact-conventions.md`](../maintainer/artifact-conventions.md) (which owns what a
shipped artifact must *match*). This note asks a different question: whether the *conversion* is
right, judged against NVIDIA ModelOpt's published method, upstream NInfer's own recipe and
published inventory, the source checkpoints' own quantization configs, and the two forks that have
modified `tools/convert/official_recipes.py`.

Every claim carries its evidence class:

| label | meaning |
|---|---|
| **[MEASURED-HERE]** | read out of this tree or these artifacts in this session, 2026-09-28; reproducible from the commands named |
| **[IN-TREE]** | already recorded in this repo's `docs/`, with the measurement it rests on |
| **[NVIDIA-DOC]** | NVIDIA ModelOpt documentation, API reference or engineering blog |
| **[VENDOR]** | a model card written by the party that produced the checkpoint; first-party, not independently reviewed |
| **[UPSTREAM]** | `Neroued/ninfer` — its code, docs, published artifact inventory, or its issue tracker |
| **[THIRD]** | an independent measurement by a named third party, quoted with its setup |

## 0. What the four shipped artifacts actually contain

Not the recipe — the artifact. Read with `tools.artifact.reader.Artifact`, resolving both `object`
and `parts[*].object` binding records (673 of 1,513 bindings are row-range views into fused parents
and are invisible to an `object`-only pass; **[MEASURED-HERE]**).

| artifact | recipe | `text/layers` | W8 endpoints | dflash2 drafter | mtp |
|---|---|---|---|---|---|
| `qwen3_8_27b_nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4_qat` | 512 NVFP4 / 353 BF16 / 96 FP32 | Q8 | **NVFP4** 46, BF16 45 | Q8 9 |
| `qwen3_8_27b_nvfp4full` | `qwen3_8_27b_nvfp4_unsloth` | **485** NVFP4 / **380** BF16 | Q8 | **NVFP4** 46, BF16 45 | Q8 9 |
| `qwen3_8_27b_nvfp4swift` | `qwen3_8_27b_nvfp4_swift` | 512 NVFP4 / 353 BF16 | Q8 | **Q8** 46, BF16 45 | Q8 9 |
| `qwen3_8_27b_nvfp4nvidia` | `qwen3_8_27b_nvfp4_nvidia` | 512 NVFP4 / 353 BF16 | Q8 | **NVFP4** 46, BF16 45 | Q8 9 |

The 27-projection BF16 difference on `nvfp4full` is the transplanted Qwen3.6-27B exception pattern
(`_BF16_EXCEPTION_*`, `official_recipes.py:368-381`), visible as
`text/layers/3/attention/query -> bf16`. All four: vision is Q4/Q5/Q6/Q8, `proposal` is Q4+int32,
`gdn/a_projection` and `gdn/b_projection` are BF16, and no artifact contains a single FP8 tensor.
**The recipes and the shipped artifacts agree.** That is worth stating because the conversion
reports do not all agree with the artifacts — see §9.1.

One row in that table is a convention of ours rather than an inherited one: **vision is quantized in
all four**, and every source checkpoint leaves it BF16 (§3.4). Everything else in the table has a
source's allocation behind it.

## 1. Verdict per artifact

| artifact | verdict | the change, if any |
|---|---|---|
| **`nvfp4nvidia`** (ModelOpt PTQ) | **Closest to NVIDIA's own prescription of the four, and the one artifact where the gap is worth closing.** | Its 192 MLP projections are NVIDIA's own `NVFP4MSECalibrator` codes, imported bit-exactly, which is right. Its **320 locally-encoded text projections — all 80 attention and all 240 GDN — go through `nvfp4_maxabs`, which is the max-scaling rule NVIDIA measured as the worst of the four it compared** (§2.2). Replace it with a per-block FP8-scale sweep (§2.3 option B). Highest-value change in this note. |
| **`nvfp4full`** (unsloth) | **Right, with one inherited exception to re-measure.** | The 27-projection BF16 exception pattern is a Qwen3.6-27B allocation transplanted here; it is unmeasured on *these* weights (it was measured on Swift's, where it lost on both protocols). Measure it here or drop it — see the note under this table. Its 293 locally-encoded projections carry the same max-scaling gap as above. |
| **`nvfp4swift`** | **Right as built; one stale report to fix.** | The shipped drafter is Q8, which matches the recipe and the port's own measurement that NVFP4 lost 3.2 acceptance points on *this* target. The `out/` conversion report for it describes a different build (§9.1). Its 320 locally-encoded projections carry the max-scaling gap. |
| **`nvfp4qat`** (QUASAR) | **Importing is right and must not change; there is nothing in the weights to fix.** | §5. All 512 text NVFP4 projections are QUASAR's own QAT codes, imported; **zero** text projections are locally encoded, so the max-scaling gap does not apply to this artifact at all. The open question is its *drafter* pairing, and the evidence for that is third-party and does not reproduce here. |

The BF16 exception pattern on `nvfp4full`, stated once so the verdict row is not the only place it
appears: `_BF16_EXCEPTION_*` (`official_recipes.py:368-381`) keeps 27 projections BF16 — attention
query/gate/key/value on layers 3, 7, 11, 15, 19, 23, attention output on 3 and 7, GDN output on 4.
`artifact-conventions.md` §1 records that it came from a Qwen3.6-27B nvfp4 artifact with *"no reason
recorded"*, that applying it to Swift measured *worse* than encoding everything (4.7701/4.93254
against 4.68429/4.92432) for 0.77 GB more file, and that **"the pattern remains unmeasured on
nvfp4full's own weights, where it still applies."** The two overrides that would settle it already
exist — reassign those 27 projections to NVFP4 from the BF16 base, keeping the recorded activation
divisors, which is the same `--override` shape `perplexity-baseline.md` used for the Swift variants.
Until then the honest position is the one that file already takes: it is an unmeasured inheritance,
and it is costing 27 projections of NVFP4 for nothing anyone has shown.

**Update, 2026-09-30: the override was run, and the pattern is measured on these weights now.** Encoding
those 27 projections to NVFP4 on `nvfp4full`'s own weights is worth **+11.6 acceptance points to the
DFlash2 route** and costs the **MTP head −21.4** (DFlash2 d7 / MTP d5) for +0.087 % overall perplexity,
so the line ships as two images: `nvfp4full_noex` on DFlash2, the BF16-exception image on MTP. The
pattern is kept where it pays, and the verdict row above is superseded on this point.

Two things that are **not** wrong and should not be "fixed" on the strength of this note:

- **The two W8 endpoints at Q8.** Measured, isolated, and large: **−3.45 % / −0.79 %** against FP8
  endpoints, with each adjacent pair of builds differing in exactly one change, verified by hashing
  every binding **[IN-TREE]**, `artifact-conventions.md` §1. They also cost **2.52 GiB**, 14.3 % of
  everything bound, so the trade is live in both directions — §3.2 states the bounded experiment
  that would decide it. NVIDIA's own ModelOpt recipe puts `lm_head` at NVFP4 (§3.1, confirmed by the
  checkpoint's own quantizer dump), so this is a real alternative; it stays Q8 because it is the one
  choice with a measurement behind it.
- **`--kv-dtype fp8` at 262,144.** §6.

One assignment in all four is a convention of ours with neither a source's precedent nor a
measurement behind it: **the vision tower**, which all three source checkpoints leave BF16 and which
we quantize to Q4/Q5/Q6/Q8, worth an estimated ~600 MiB of device weights (§3.4). It is the cheapest
thing in the four recipes to change and the hardest to evaluate, because perplexity cannot see it.

## 2. The NVFP4 quantization method

### 2.1 What our encoder actually does

`tools/convert/quantization/nvfp4.py`, `encode_block()` (lines 48-58) **[MEASURED-HERE]**:

```python
blocks = (values.float() * divisor).reshape(rows, columns // 16, 16)
scales = (blocks.abs().amax(dim=2) / 6.).clamp(max=448.).to(torch.float8_e4m3fn)
ratios  = blocks / scales
codes   = e2m1_rne_codes(ratios)          # round-to-nearest-even, ties to even
```

The block scale is the per-block **maximum**, divided by 6. The global divisor is
`FULL_RANGE / global_amax` with `FULL_RANGE = 2688 = 6 * 448` (line 93).

That is, in NVIDIA's own vocabulary, the **max-scaling** rule: *"The default way is to set the block
scale based on the per-block maximum value (max scaling)"* **[NVIDIA-DOC]**, Model Optimizer,
"Improving NVFP4 Accuracy with Local-Hessian Weight Scales", 2026-09-09,
`nvidia.github.io/Model-Optimizer/announcements/local-hessian.html`. The group size is right: 16, which is the block size NVFP4 itself defines The rounding is right: RTN, which is what
Local-Hessian uses by default ("Local-Hessian rounds to nearest (RTN) by default" **[NVIDIA-DOC]**).

So the *format* is right and the *scale rule* is the one NVIDIA measured as worst.

### 2.2 What NVIDIA prescribes for this model, and what the calibration is

NVIDIA's reproduce command for the published Qwen3.8-27B checkpoint, verbatim
**[NVIDIA-DOC]**, same page:

```bash
python examples/hf_ptq/hf_ptq.py \
    --pyt_ckpt_path Qwen/Qwen3.8-27B \
    --recipe modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml \
    --dataset nemotron-post-training-v3 \
    --calib_size 512 \
    --calib_seq 2048 \
    --batch_size 1 \
    --export_path <export_dir>
```

with `"algorithm": {"method": "local_hessian", "layerwise": {"enable": True}}`, and the
recommendation to use calibration **batch size 1** so padding tokens cannot contaminate activation
statistics. The recipe name states the layer allocation too: **W4A4 MLP + FP8 attention**.

NVIDIA's own scale-rule comparison, on Qwen3.5-9B, W4A4 everywhere except `lm_head`
**[NVIDIA-DOC]**, Table 1:

| weight scale rule | MMLU | HellaSwag | WinoGrande | GSM8K | avg drop ↓ | WikiText PPL ↓ |
|---|---:|---:|---:|---:|---:|---:|
| BF16 reference | 78.69 | 78.04 | 73.40 | 87.64 | 0.00 | 9.20 |
| **max scale (ours)** | 75.81 | 76.33 | 70.64 | 74.60 | **5.10** | **10.08** |
| MSE scale | 76.49 | 76.61 | 72.45 | 76.72 | 3.87 | 9.98 |
| four-over-six | 75.32 | 76.62 | 70.40 | 76.42 | 4.75 | 10.02 |
| **Local-Hessian** | 76.81 | 76.50 | 71.19 | 80.89 | **3.10** | **9.90** |

**The checkpoint records which rule its producer used**, which is stronger than the blog.
`nvidia/Qwen3.8-27B-NVFP4/.quant_summary.txt` is a complete dump of all 2,256 TensorQuantizers in the
model, and every one of the **193** NVFP4 weight quantizers is a
`StaticBlockScaleQuantizer((2,1) bit fake block_sizes={-1: 16, 'type': 'static', 'scale_bits': (4,3)},
amax=[min, max] … calibrator=NVFP4MSECalibrator)` **[MEASURED-HERE]**. 193 = 192 MLP + `lm_head`.
So the shipped codes came out of an **MSE search over the FP8 scale grid, not max scaling** — which
is the search `local_hessian_calibrate` delegates to (*"fed to `mse_calibrate()`'s weight search via
`error_func`"* **[NVIDIA-DOC]**). The dump records the calibrator object, not the `error_func`
supplied to it, so **the file alone cannot separate plain MSE from Local-Hessian**; the model card
and the reproduce command both say Local-Hessian, and that is the attribution to use.

The same dump settles the *activation* side, and against a reading I had formed: every activation
quantizer on an NVFP4 site is
`TensorQuantizer((2,1) bit fake block_sizes={-1: 16, 'type': 'dynamic', 'scale_bits': (4,3)},
amax=… calibrator=MaxCalibrator)` — a plain observed maximum, with **no headroom**
(`amax` values run 4.25e+01 … 2.10e+03 across sites) **[MEASURED-HERE]**. `nvfp4_act_headroom_calibrate`
is a ModelOpt facility with defaults `anchor_percentile=1.0, upper_percentile=99.99, rho=16384.0`
**[NVIDIA-DOC]**, and **NVIDIA's own published Qwen3.8-27B recipe did not use it.** Our
no-margin activation divisor is therefore the same *policy* NVIDIA used for this checkpoint, and
§2.3 option A has to be argued from the saturation measurement, not from a deviation.

Two sentences from the same page that bear directly on how to read other people's numbers:

- *"Both of these approaches for scale selection consider only weight tensor-level error, which we
  find does not correlate well with downstream accuracy evaluation results."* This is NVIDIA saying
  that a weight-space rel-L2 ranking — the metric used in **[THIRD]** §4.2 — is not a quality
  ranking. It also bounds Local-Hessian + GPTQ composition: *"on Qwen3.8-27B, Local-Hessian + GPTQ
  scored below Local-Hessian alone, so the published checkpoint uses Local-Hessian only."*
- *"Local-Hessian and the other ModelOpt scale-selection algorithms for NVFP4 weight scales are
  free. Weight scales are computed only once, at checkpoint creation."* **There is no runtime cost
  to any of this.** A better scale rule is a one-off conversion-time change, not a serving change.

On the **calibration corpus and sequence length**: NVIDIA's documented configuration is
`nemotron-post-training-v3`, 512 samples of 2,048 tokens, batch size 1, layerwise. Ours, measured
**[MEASURED-HERE]** from `tools/convert/qwen3_8_27b_nvfp4_calibration.json` and
`tools/convert/calibration_corpus.json`:

| | NVIDIA | ours |
|---|---|---|
| corpus | Nemotron-Post-Training-Dataset-v3 | 10 hand-authored snippets, ~9,900 characters: English prose, ML text, Python, financial figures, French, Greek, JSON, Spanish, compiler text, migration notes |
| tokens | 512 × 2,048 = **1,048,576** | **3,625** (the file's own `corpus_tokens`) — **1/289** |
| sites | 193 NVFP4 (192 MLP + `lm_head`) | 247 NVFP4 text sites |
| statistic | per-block 16×16 input second moment, layerwise, batch 1 (**weights**); plain observed max (**activations**) | one global `amax` per site, one forward pass, `device_map` offload |
| headroom | none on activations (`MaxCalibrator`); NVIDIA's `nvfp4_act_headroom_calibrate` exists but this recipe does not use it | **none** — the divisor is `2688 / observed max` |
| which artifacts use it | — | `nvfp4full` only; the other three take the divisor from their source checkpoint |

Our corpus is not wrong to be small — it is *reused verbatim from a fork's `calibrate_nvfp4full.py`*
so that our divisors stay comparable with the published `nvfp4full` profile, which
`calibration.py:8-13` states as the acceptance test and **[IN-TREE]** confirms reproduces those 247
divisors (median ratio 1.0000). That is a real constraint, and it is the reason the corpus is not
being replaced.

The headroom question is separate from the corpus, and it is not a "we deviate from NVIDIA" item —
see above, NVIDIA shipped the same no-margin activation policy. It rests on one measurement.
In **[UPSTREAM]** #70, a third party building the same all-NVFP4 profile from a pinned BF16 base
reported, over its 247 NVFP4 parents with an 8,490-token **held-out** set:

> "Calibration used a deterministic 14,986-token corpus plus a distinct 8,490-token held-out set.
> With 2x activation headroom, the held-out pass saturated 0/247 sites (maximum ratio 0.63757).
> Without that margin, 60/247 sites saturated, with a maximum of 1.8147x."

60 of 247 sites — a quarter of them — overshoot the calibrated range on data the calibration never
saw, the worst by 1.81×, when the divisor carries no margin. That is a measured saturation count on
the same 247-site shape our `nvfp4full` build has, and it is the whole argument for option A. It is
one measurement by one party, and NVIDIA's own recipe shipped without the margin, so the honest
strength of the case is "measured once, against the producer's own choice" — which is why A is
ranked below B and C, both of which NVIDIA measured four ways in one table.

It also bounds the corpus question honestly: a 3,625-token corpus makes 247 independent maxima
thinner than NVIDIA's 1,048,576 tokens makes 193. Enlarging the corpus would move the maxima, which
breaks the comparability constraint the corpus exists to serve. **The headroom factor does not** —
it is a multiplier on an amax, so a margin can be added without touching the amax the acceptance
test compares.

### 2.3 The actionable change, in cost order

| option | what it changes | cost | evidence it is better |
|---|---|---|---|
| **A. Headroom on the activation divisor** | `tools/convert/calibration.py:173`, `d_x = 2688 / amax_site` → `d_x = k * 2688 / amax_site` | one constant; leaves every amax untouched, so the median-ratio-1.0000 acceptance test against the published profile survives | **[THIRD]** saturation count above. **Not** a deviation from NVIDIA — NVIDIA shipped the same no-margin policy for this checkpoint |
| **B. Per-block FP8 scale sweep (ModelOpt "MSE" rule)** | replace `scales = block_amax / 6` in `nvfp4.py:52` with a search over the E4M3 scale grid minimising encoded squared error | contained to one function; no calibration data needed (`mse_calibrate(..., forward_loop=None, fp8_scale_sweep=True)`) | **[NVIDIA-DOC]** Table 1: 3.87 vs 5.10 average drop, PPL 9.98 vs 10.08. The checkpoint itself already records `NVFP4MSECalibrator`, so this is the rule its producer used |
| **C. Local-Hessian scale selection** | same call site, but minimise the Hessian-weighted per-block error; needs 16×16 per-block input second moments from a layerwise, batch-1 calibration forward | a real calibration pipeline this tree does not have; `calibration.py` captures one scalar per site | **[NVIDIA-DOC]** Table 1: 3.10 average drop, PPL 9.90; and it is what NVIDIA's card and reproduce command both name for this checkpoint |

Note for B: NVIDIA's own documents disagree on the candidate count — the blog says 126, the
`local_hessian_calibrate` API doc says *"sweep over all 128 possible FP8 E4M3 scale values"*
**[NVIDIA-DOC]**. Enumerate the format's representable positive finite scales rather than
hardcoding a count.

NVIDIA's own two documents also disagree on the calibration *sample count*, and both are cited in
this note, so it is worth flagging rather than picking one: the model card says *"calibrated on
2,048 samples"* **[VENDOR]**, while the blog's reproduce command passes `--calib_size 512
--calib_seq 2048` **[NVIDIA-DOC]** — 512 samples of 2,048 tokens, which is 1,048,576 tokens either
way. Nothing in either document reconciles "2,048 samples" with `calib_size 512`, and the
checkpoint itself does not record the count. The `512 × 2,048` reading is the one the table in
§2.2 uses, because it is the one the reproducible command states.

The scope of B and C is exactly the locally-encoded sites and nothing else. Counted from the shipped
artifacts **[MEASURED-HERE]**, `text/layers` breaks down as:

| block | NVFP4 | how encoded |
|---|---:|---|
| `mlp` (64 layers × gate/up/down) | 192 | **imported** — the source's own codes |
| `attention` (16 full-attention layers × query/gate/key/value/output) | 80 | **local, `nvfp4_maxabs`** |
| `gdn` (48 linear-attention layers × query/key/value/z/output) | 240 | **local, `nvfp4_maxabs`** |
| total | 512 | **320 local, 192 imported** |

So per artifact:

- `nvfp4nvidia`: 320 locally encoded (attention + GDN) + the 46-projection drafter.
- `nvfp4full`: 293 locally encoded, the other 27 being the BF16 exceptions.
- `nvfp4swift`: 320 locally encoded (its MLP is ModelOpt NVFP4, imported).
- `nvfp4qat`: **zero locally encoded text projections.** Every text projection is QUASAR's own
  code, imported. Its endpoint and drafter encodings are separate matters (§3.2, §5).

The locally-encoded projections are also the ones `ninfer-perplexity` *can* see, and it sees them
as a net small cost: `artifact-conventions.md` §1 measures the Swift attention+GDN re-encode at
**+0.05 % / +0.50 %** against importing the source's FP8. So the exposure B and C address is real
but bounded — which is the honest reason to expect a fraction of a percent, not a point, from a
better scale rule on this axis.

## 3. The per-parameter format assignment

### 3.1 What each source checkpoint says about itself

All three quant configs read directly from the checkpoints **[MEASURED-HERE]**:

| source | config key | assignment |
|---|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` (`hf_quant_config.json`, producer `modelopt 0.47.0.dev80+g913f5e224`) | `quantized_layers.*.quant_algo` | `mlp.{gate,up,down}_proj` **NVFP4, group_size 16, on all 64 layers**; `lm_head` **NVFP4 g16**; `self_attn.{q,k,v,o}_proj` (16 layers) **FP8**; `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (48 layers) **FP8**. No entry for the token embedding. |
| `Swift-Qwen3.8-27B-NVFP4` (`hf_quant_config.json`, `modelopt 0.47.0rc0`) | same | identical to the above |
| `unsloth/Qwen3.8-27B-NVFP4` (`config.json` → `quantization_config`) | `config_groups` | `group_0` `float-quantized` FP8 channel-weights on `self_attn.(q\|k\|v\|o)`, `linear_attn.(in_proj_qkv\|in_proj_z\|out_proj)`, `lm_head`, and MLP 56-63; `group_1` `nvfp4-pack-quantized` g16 `actorder: static` on all `mlp.(gate\|up\|down)_proj` |
| `Qwen3.8-27B-NVFP4-QUASAR` (`config.json` → `quantization_config`) | `config_groups.group_0`, producer `qatfactory 0.1.0` | `nvfp4-pack-quantized` on `targets: ["Linear"]`; `ignore: ["lm_head", "re:.*visual.*", "re:.*mtp.*", ...]`; weights `observer: memoryless_minmax`, `group_size: 16`, `dynamic: false`, `scale_dtype: float8_e4m3fn`, `symmetric: true`; activations `observer: static_minmax`, `dynamic: local`, `group_size: 16` |

The QUASAR index confirms it structurally **[MEASURED-HERE]**,
`Qwen3.8-27B-NVFP4-QUASAR/model.safetensors.index.json`: 496 `weight_packed` + 496 `weight_scale`
+ 496 `input_global_scale`; `lm_head.weight`, `model.language_model.embed_tokens.weight`, all
`model.visual.*` and all `mtp.*` plain BF16. 496 = 240 (`linear_attn` × 5 × 48) + 192 (MLP × 3 × 64)
+ 64 (`self_attn` × 4 × 16). `in_proj_a` and `in_proj_b` **are** quantized there.

QUASAR's own model card **[VENDOR]**: *"QUASAR trains the NVFP4 weights directly against the frozen
BF16 model, then exports standard NVFP4 weights with no custom inference path"*, 496/496 linears,
19.7 GB, 0.909 GPQA-D vs 0.914 BF16, one epoch of loss-aware NVFP4 QAT distillation, global batch 32,
lr 1e-6, 2,446 steps. Its weight rule — `memoryless_minmax`, group 16, symmetric — is **absmax**,
i.e. the same max-scaling rule ours is, applied to weights that were *trained* under it. That is
the strongest argument in this note for not re-encoding QUASAR: the scales are part of what was
trained.

### 3.1b What the NVIDIA quantizer dump says that the config does not

`nvidia/Qwen3.8-27B-NVFP4/.quant_summary.txt`, 2,256 TensorQuantizers by the file's own closing
count, with the calibrator census below taken from it by hand **[MEASURED-HERE]**. This is the only
file in any of the three sources that records the *actual quantizer objects*, and four things in it
are not in the config:

1. **The 193 NVFP4 weight quantizers all used `NVFP4MSECalibrator`** (§2.2). The config records
   `quant_algo: NVFP4` and `group_size: 16` and nothing about the scale rule.
2. **The FP8 is per-tensor, not per-row.** Every non-NVFP4 weight quantizer is
   `TensorQuantizer((4,3) bit fake **per-tensor** amax=… calibrator=MaxCalibrator)`. So
   `fp8_e4m3fn_row_bf16` in our format registry and "row-scaled FP8" in upstream's model card
   describe **upstream's and our** encoding choice, not NVIDIA's. Anything reasoning about "the
   source's FP8" as a row format is reasoning about the wrong granularity.
3. **`linear_attn.in_proj_a` and `linear_attn.in_proj_b` weight quantizers are
   `TensorQuantizer(disabled)`** — 48 each. NVIDIA does not quantize them, exactly as we do not,
   and for the same structural reason (§3.4). This is a fourth independent confirmation, and it
   sharpens the contrast with QUASAR, which *does* quantize them.
4. **The whole vision tower is unquantized**: every `model.visual.*` weight quantizer is
   `TensorQuantizer(disabled)` — `patch_embed.proj`, `pos_embed`, all 27 blocks' `attn.qkv`,
   `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`, and both `merger` linears. And
   `model.language_model.embed_tokens` is disabled too.

Point 4 has a consequence this tree has not recorded. **All three source checkpoints leave the
vision tower in BF16 — NVIDIA with every `model.visual.*` weight quantizer disabled, QUASAR with an
explicit `ignore: ["re:.*visual.*", ...]`, and unsloth with 110 of its 303 `ignore` entries naming the
vision tower (27 blocks × 4 projections, plus both `merger` linears) — and all four of our artifacts
quantize it**, via `_optional` in `official_recipes.py:24-39`, to Q4 on
`attention/{query,key,value}` and `mlp/fc1`, Q8 on the `merger`, Q6 on `patch_embedding`, Q5 on
everything else **[MEASURED-HERE]**. That is 441 vision bindings and 165 quantized objects, and no
source's data covers it. See §3.4.

### 3.2 (a) The two W8 endpoints

Ours: Q8 `grouped_absmax` on `text/token_embedding` and `text/output_head`, from the BF16 base, in
all four **[MEASURED-HERE]**.

**What the endpoints cost, measured on the artifact.** Object bytes by format, from
`tools.artifact.reader.Artifact` over the shipped `qwen3_8_27b_nvfp4nvidia.v3.ninfer`
**[MEASURED-HERE]**:

| component | bytes | breakdown |
|---|---:|---|
| text | 15.309 GiB | NVFP4 13,050.0 MiB, **Q8 2,576.6 MiB**, BF16 50.0 MiB |
| dflash2 | 1.296 GiB | NVFP4 956.3 MiB, BF16 370.5 MiB |
| mtp | 0.420 GiB | Q8 430.3 MiB |
| proposal | 0.333 GiB | Q4 340.0 MiB, int32 0.5 MiB |
| vision | 0.275 GiB | Q4 122.3, Q5 107.1, Q8 45.4, Q6 1.3, BF16 5.8 MiB |
| **bound total** | **17.633 GiB** | |

The 2,576.6 MiB of Q8 is the two W8 endpoints and essentially nothing else: 2 × 248,320 × 5,120 at
one byte plus group scales is 2.52 GiB, which is the whole number. **The endpoints are 16.5 % of the
text stack and 14.3 % of everything bound.** That is not a rounding error, and it is the reason the
endpoint decision is worth more attention than the rest of this note put together — see the size
lever at the end of this section.

| position | who | what | evidence |
|---|---|---|---|
| embedding | every source | **not quantized** (absent from NVIDIA's `quantized_layers`, absent from QUASAR's export, absent from unsloth's `config_groups`) | **[MEASURED-HERE]** |
| `lm_head` | NVIDIA ModelOpt | **NVFP4 g16**; the dump records `calibrator=NVFP4MSECalibrator` and the card and reproduce command name Local-Hessian (§3.1b) | **[VENDOR]** + **[MEASURED-HERE]** |
| `lm_head` | unsloth | **FP8** channel | **[MEASURED-HERE]** |
| `lm_head` | QUASAR | **explicitly ignored**, shipped BF16 | **[MEASURED-HERE]** (`ignore` list) |
| `lm_head` | upstream NInfer's official artifact | **row-scaled FP8**, imported | **[UPSTREAM]** model card |
| both | our four | **Q8** | **[MEASURED-HERE]** + **[IN-TREE]** −3.45 % / −0.79 % |

So the sources disagree three ways, and the honest reading is that nobody has established a rule
here. Two facts constrain the choice:

1. **It is the measured quality lever.** The endpoint pair alone is worth −3.45 % / −0.79 % on
   Swift, isolated from the text-stack re-encode by hashing every binding **[IN-TREE]**.
2. **We cannot import NVIDIA's head even if we wanted to.** The head is 248,320 × 5,120; the
   runtime registers the vocabulary projection for FP8, not NVFP4. A fork that hit this documents
   the mechanism in its recipe docstring: *"The source output head is NVFP4, but the runtime
   registers the vocabulary projection only for FP8, so it is dequantised and re-quantised there"*
   **[THIRD]**, `Wallawalla47/ninfer-custom` `qwen3_8_27b_nvfp4_nvidia`. Our route is strictly
   better than that one — Q8 from the **BF16 base**, so it never inherits the head's NVFP4 error —
   and the per-endpoint cost is the 1.26 GiB measured above.

The counter-argument, and why it does not yet carry: **[THIRD]** §4.2 measures NVIDIA's FP4 head at
rel-L2 0.0848 against a W8 head's 0.0055, "a 15× weight-space error increase traded for −0.59 GiB" —
and the same author explicitly declines to adopt it on that basis, gating it on end-to-end GPQA/LBv2
and deterministic acceptance instead. That is the right posture, and **no end-to-end lane on an
NVFP4 head exists in any source I can reach.** Do not change the endpoints on a weight-space number
NVIDIA itself says does not predict accuracy.

**But the size lever is now measured, and it is not small.** 2.52 GiB of Q8 is 2.52 GiB; the same
two matrices at NVFP4 with a 4-bit global divisor would be roughly 0.63 GiB each, so **about
1.26 GiB back**. `artifact-conventions.md` §3's measured envelope puts the four shipped DFlash2
lanes at 17.2–18.0 GiB of device weights, with 18.0 GiB (Swift) the highest value still reaching
262,144 with vision on and 20.50 GiB the lowest that does not. 1.26 GiB is therefore the same order
as the whole spread between the lanes that reach full context and the build that does not. So the
endpoint question is not "should we trade accuracy for bytes in general" but a bounded, decidable
experiment:

> Build one target with both W8 endpoints at NVFP4 (imported from NVIDIA's Local-Hessian/MSE codes
> where the source has them, so the codes are not ours), and read three things: full-corpus
> perplexity, DFlash2 acceptance, and the `ceiling` ladder for the DFlash2 lane. If acceptance holds
> and the ladder moves, that is a real capability gain; if acceptance drops, the 2.52 GiB is
> correctly spent.

Until that run exists, the current Q8 assignment is the right default: it is the one choice with a
measurement behind it, and it is what the four shipped artifacts bind.

### 3.3 (b) The DFlash2 draft model's projections

**What upstream does.** Upstream's `_optional` in `tools/convert/official_recipes.py` assigns **Q8**
to every `dflash`/`dflash2` projection except a five-suffix skip list (`/moe/router`,
`/moe/shared_score`, `/attention_conv/kernel_projection`, `/mlp_conv/kernel_projection`,
`/candidate_selector/hidden_projection`), and `recipe.share`s each layer's `context_key`/
`context_value` onto `key`/`value` **[UPSTREAM]**, `official_recipes.py:19-54` — the port's file is
identical in this region. Upstream's *published* artifact's drafter is in fact BF16, not Q8: the
published inventory counts 112 NVFP4 and 146 FP8 tensors, which is the text stack, and
`src/models/qwen3_5/load/dflash2.cpp:25` records *"Bound without an exact format: the official
artifact stores BF16 codebooks while the QUASAR checkpoint stores NVFP4 ones"* **[IN-TREE]**. So
upstream has never published a drafter it had to choose a format for.

**What we do**, and the evidence behind each **[MEASURED-HERE]** + **[IN-TREE]**:

| artifact | drafter | why |
|---|---|---|
| QUASAR, nvfp4full, NVIDIA | NVFP4, 46 projections, `A16Only` | interleaved bench on the QUASAR target: Q8 accepts 54.8 %, NVFP4 58.0 % (`official_recipes.py:183-191`) |
| Swift | **Q8**, 46 projections | the NVFP4 rule lost **3.2 acceptance points** on this target (57.7 % vs 60.9 %), `official_recipes.py:303-313` |
| all four | the five-suffix skip list left BF16 | upstream's list; including the conv kernel projections measured **42.2 %** acceptance (`official_recipes.py:188-190`) |

So the drafter format is already a **per-target measured choice**, and the one target that measured
against NVFP4 is the one that ships Q8. That is the right shape for the decision and I would not
change it.

**The fork claim, checked.** `satellitedown/cinference` changes exactly one thing in
`official_recipes.py` **[THIRD]**:

```python
# Target verification makes DFlash2 drafter precision an acceptance choice, not an
# output-quality one. Q4 halves its weight stream with unchanged acceptance; the fused
# QKV input projection (shared by the context key/value projections) keeps Q8, its only
# native form.
if name.startswith("dflash2/") and not name.endswith(
    ("/attention/query", "/attention/key", "/attention/value")
):
    _assign(recipe, name, Q4)
else:
    _assign(recipe, name, Q8)
```

Read to its mechanism, three things are true:

1. **It does apply to a target that is itself NVFP4**, because the change lives inside `_optional`,
   which every recipe calls. So the question's premise is right.
2. **The rationale is a bandwidth argument, not a quality one.** "Halves its weight stream with
   unchanged acceptance" — no measurement, no corpus, no acceptance figure. It sits in `_optional`,
   so it also changes the `qwen3_8_27b` groupwise-int recipe and the Qwen3.6 recipes, not just the
   NVFP4 one.
3. **It is not free here.** The Q8 fused QKV stays Q8, which we already have. But the drafter's
   attention-output projection and its MLP would move to Q4, and this tree's Q4 registry is
   `src/ops/linear/q4/q4_shapes.h:7-16` — it has `n5120_k6144` (the drafter's attention output) and
   `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp:32` covers `{34816, 17408, 5120, 5120}` (the
   drafter's MLP, whose `intermediate_size` is 17,408 like the text stack's). What it does **not**
   have is the drafter's `feature_projection` at `[25600, 5120]`, and the three-output drafter
   `attn_input_proj` is NVFP4-or-Q8 only (`src/ops/wrapper/attn_input_proj.cpp:243-275`) — though
   satellitedown's rule leaves the fused QKV at Q8, so that Op is not the blocker. Adopting it costs
   one new Q4 shape plus its oracle, and buys ~0.3 GiB of drafter weights.

**Verdict: not justified on the evidence offered, and not the change with the best evidence behind
it.** Our own tree has already measured drafter format *per target* and found the answer flips
(Q8 wins on Swift, NVFP4 wins on QUASAR). A single global "Q4 halves the stream" rule is a
size tweak with no acceptance evidence attached, in a region where our measurements show the sign
is not even stable. If the drafter's 0.3 GiB is worth having, the honest form of the experiment is a
Q4 build of *one* target measured on acceptance — not a change to `_optional` that silently moves
five recipes.

One contrary datum, recorded because it does not reproduce here: **[THIRD]**, in **[UPSTREAM]** #214,
Avalonec reports the DFlash2 acceptance rate falling to *"21.4 % / 2.5 tokens per round"* when the
draft weights are NVFP4, against *"64.3 % / 5.5 tokens"* in the unquantized/requant profile. That
is 21.4 % where we measure 58.0 % for the same encoding. "Unquantized/requant" is not defined, and
the two are 2.7× apart, so the setups are not comparable; our figure is the one taken on this port
against the published QUASAR artifact in one interleaved window. Treat Avalonec's as a report of a
different configuration, not as a contradiction.

### 3.4 (c) What should be left unquantized

Our five-suffix skip list plus `gdn/a_projection` and `gdn/b_projection` **[MEASURED-HERE]**:
all BF16 in all four artifacts.

- **The skip list is upstream's**, and upstream's is measured on this engine (42.2 % acceptance with
  the conv kernel projections encoded, `official_recipes.py:188-190`). It is not a guess.
- **`gdn/a_projection` and `gdn/b_projection` cannot be NVFP4 at all**, and this is a hard
  constraint, not a preference: at (96, 5120) the N dimension is not divisible by 128, which
  `block_scale_k16_m128x4_v1` requires, and `nvfp4_maxabs` rejects it outright
  (`nvfp4.py:72`, `if n % 128 or k % 16 ... raise`). **NVIDIA's own quantizer dump has both as
  `TensorQuantizer(disabled)`** (§3.1b point 3), so this is a property of the architecture rather
  than a choice either producer made. The consequence worth stating is the one place where the
  sources disagree with us and we are right: **QUASAR *does* quantize them** (its index shows
  `in_proj_a`/`in_proj_b` with `weight_packed`), so a QUASAR-sourced build has to take these two
  from a BF16 source rather than import them. `official_recipes.py:217-253` does exactly that, and
  this is the only structural reason the QUASAR artifact is not a pure import.
- **The vision tower, the one assignment here that no source covers.** All three source
  checkpoints leave it BF16 (§3.1b point 4) and all four of our artifacts quantize it
  (`official_recipes.py:24-39`). What that costs, measured **[MEASURED-HERE]** on
  `qwen3_8_27b_nvfp4nvidia.v3.ninfer`: the whole vision component is **0.275 GiB (282 MiB)**, of
  which the quantized encodings occupy **276.1 MiB** — Q4 122.3, Q5 107.1, Q8 45.4, Q6 1.3 — and only
  5.8 MiB is still BF16 (the norms, biases and `pos_embed`). Those 276.1 MiB hold roughly 4.7 × 10⁸
  weights at an average of ~4.9 bits each, so the same matrices at BF16 would be **~890 MiB**: the
  assignment is worth **about 600 MiB of device weights**, not the 276 MiB the object census shows.
  Against a 17.1–18.0 GiB envelope that is 3.3–3.5 % — half of what the W8 endpoints cost
  (§3.2), and enough to move a lane that is currently at the top of the envelope.
  So: the convention is ours alone, no source measures it, and it is not a trivial size choice.
  **`_optional`'s vision block is the one assignment in the four recipes with neither a source's
  precedent nor a measurement behind it.** It is also cheap to *change* — one recipe block — and
  expensive to *evaluate*, because no perplexity row in `docs/perplexity-baseline.md` can see it
  (perplexity loads Text weights only, `docs/perplexity.md`). Testing it needs a vision benchmark,
  which §7.3 says we do not have.
- Everything else left BF16 — `gdn/convolution`, `a_log`, `dt_bias`, every norm, all of
  `mtp`'s non-projection tensors, the `candidate_selector` codebooks, the DFlash2 norms and conv
  base kernels — is not a projection or is a control tensor. Nothing here suggests a change.

## 4. Other NInfer forks: what they changed in conversion

### 4.1 `satellitedown/cinference`

Read in full, both files **[THIRD]**. `tools/convert/` has upstream's file set and no calibration
module, no custom NVFP4 quantizer, and no added method. The **only** conversion change is the
DFlash2 Q4 rule in §3.3. `tools/convert/methods.py` is upstream's. **Nothing about format
assignment, calibration or component selection otherwise.**

### 4.2 `Wallawalla47/ninfer-custom`

Default branch is `master`, not `main`. Three changes **[THIRD]**:

**(a) `grouped_mse` — a real method, used by no shipped recipe.**
`tools/convert/quantization/groupwise.py` gains `quantize_matrix_mse`: for each group it evaluates
14 candidate binary16 scales (the max-abs scale, its ±2/±1 power-of-two multiples, a least-squares
estimate, and ±2-ulp word-space neighbours), computes the **exact encoded squared error** under the
engine's own code formula `round(w * binary32(1/scale))`, and keeps the first minimum with the
max-abs scale breaking ties. Deterministic, bit-reproducible across devices, run on the host in
ordered arithmetic. `methods.py` registers it and `METHODS` exposes it. `official_recipes.py` threads
a `method=` parameter through `_assign`/`_optional`/`_dense_groupwise` with `grouped_absmax` as the
default — and **every call site leaves it at the default**, so no shipped recipe uses it. The one
new recipe it does add, `qwen3_8_27b_q6`, is the groupwise-int Dense recipe with MLP gate/up at Q6,
also with `method=grouped_absmax`.

Is it justified? The *mechanism* is the same class of intervention as NVIDIA's MSE rule and it is
implemented properly, with the tie-break preserving the absmax result. But:

- It cannot touch NVFP4 at all. `quantize_matrix_mse` requires `isinstance(spec, QuantFormat)`, and
  `Nvfp4Format` is a separate class from `QuantFormat` — the same fact
  `artifact-conventions.md` §1 records as "the row-split quantizer cannot produce
  `block_scale_k16_m128x4_v1`". So it is irrelevant to the 512 text NVFP4 projections in all four
  artifacts.
- Its scope is exactly the **W8 endpoints** (§3.2) and the grouped-int MTP/vision/proposal weights.
- **No quality evidence accompanies it.** No perplexity, no acceptance, no rel-L2. It is a
  capability, not a result.

That last point is the interesting one, because it lands on the axis that matters. Our endpoints are
the measured quality lever, they are encoded from BF16 locally, and they are encoded with plain
absmax. A per-group scale search is the cheapest available improvement to exactly that axis, it
needs no calibration data, and — per NVIDIA — scale selection is free at runtime. **Porting
`quantize_matrix_mse` and applying it to the two endpoints is a smaller, better-targeted change
than anything else in this note.** It is second in priority only because it has never been measured
on this model, and a numerics change with no oracle is a risk, not a win.

**(b) `qwen3_8_27b_nvfp4_nvidia`** — the ModelOpt layout: MLP all layers NVFP4, attention/GDN FP8
imported, and the head dequantized from the source's NVFP4 and requantized to FP8 *"because the
runtime registers the vocabulary projection only for FP8"*. Two observations: the allocation matches
`hf_quant_config.json` exactly **[MEASURED-HERE]**, and the head handling is the mechanism discussed
in §3.2, where our route is better.

**(c) `qwen3_8_27b_nvfp4_orcarouter`** — a fourth community source, same shape as unsloth's.

Neither fork changes the drafter format; only satellitedown does.

## 5. The QAT route (QUASAR)

**Is re-quantizing a QAT checkpoint to NVFP4 the right move? No — and we already do not do it.**

`qwen3_8_27b_nvfp4_qat` uses `method=import_encoded` for every text projection
(`official_recipes.py:246-252`), which by `docs/weight-conversion.md` *"preserves compatible code
and scale words, including NVFP4's matrix weight divisor… It does not dequantize and requantize
them"*. Since QUASAR exports standard NVFP4 with *"no custom inference path"* **[VENDOR]**, a
bit-exact import is available, and it is what the artifact does. The docstring records the
verification: 496 fused NVFP4 sites, every text projection resolving to a real NVFP4 site, and the
activation divisor imported from the checkpoint's own `input_global_scale` via
`methods.py`'s `import_encoded` path. `artifact-conventions.md` §1 puts it plainly: *"No local
encoder run and no calibration corpus are involved."*

Why re-encoding would be wrong, from the source's own words **[VENDOR]**:

- the scales were **trained**, not fitted: one epoch of loss-aware QAT distillation against the
  frozen BF16 teacher, and *"trains the NVFP4 weights directly against the frozen BF16 model"*;
- QUASAR's weight rule is `memoryless_minmax` group 16 — the *same* absmax rule as ours. Re-encoding
  with absmax would land on a very similar but not identical point and throw away the gradient
  information that distinguishes a QAT checkpoint from PTQ;
- **[NVIDIA-DOC]** says weight-space error "does not correlate well with downstream accuracy", and
  **[THIRD]** §4.2 says the same about rel-L2 on a co-adapted QAT checkpoint: *"for a co-adapted
  QAT checkpoint it measures training drift from the teacher, not quantization error"*.

**So: is there a published recommendation to ship a QAT checkpoint in its native format?** **No —
that question is not answered by any source I could reach**, and I would rather say so than dress an
inference as a citation. What *is* published is QUASAR's own serving recommendation for the
checkpoint as released, plus the training description above. Our answer to the question we actually
face — import or re-encode — is an inference from those two: the format is standard NVFP4, so a
bit-exact import exists; the scales were trained; and re-encoding would discard the training. That
inference is strong, and it is also the reason the recipe was written this way in the first place.
Labelling it as an inference rather than attributing it to a source that does not say it.

**The real QUASAR finding is not the weights — it is the drafter.** Two third-party observations,
both in **[UPSTREAM]**:

- cometkim, #70: *"One important observation is that when using QAT weights, the DFlash2 token
  acceptance rate drops significantly compared to the existing nvfp4full for the same prompt. Not
  always, but it is frequently observed. This is a major issue for inference speed."* And on the
  quality side, the same campaign found the two **equivalent**: GPQA-Diamond 89.22 ± 2.49 (nvfp4qat)
  against 87.88 ± 2.62 (nvfp4full), AIME26 91.11 ± 3.85 against 93.33 ± 3.34, LBv2 short 66.30 ± 0.86
  against 67.41 ± 1.94 — overlapping, with cometkim's own reading *"Both are getting equivalent
  scores in GPQA-Diamond and AIME26"*, and *"the difference is not as significant as expected:
  16.02 GiB vs. 15.31 GiB"*.
- cometkim, #214: *"Measurements using the ninfer fixed corpus show that while PPL does indeed
  improve, the MTP/DFlash2 acceptance rate drops significantly."* And end to end, on his own
  hardware: MTP-3 **−8.9 % / −9.7 %** tok/s and DFlash2-K7 **−4.1 % / −4.2 %** for the NVIDIA-sourced
  build against his nvfp4full — with the MTP-0 control at −0.00 % / +0.15 %, which is what makes it a
  speculative-decode result rather than a target-model one.

**Our own measurements do not reproduce the acceptance collapse.** Our QUASAR drafter is NVFP4 and
measured **58.0 %** acceptance against a Q8 drafter's 54.8 % in the interleaved bench that motivated
`_nvfp4_draft` (`official_recipes.py:183-191`) — the opposite sign to cometkim's report, and on the
same target family. So the port and the tracker disagree on the direction of this effect. What both
agree on is that **the drafter is the sensitive axis and the target weights are not** — which is the
actionable reading, and which is why §3.3's verdict is "keep the per-target measurement" rather than
"pick a format". A same-hardware A/B of the two published QAT-vs-PTQ target pairings, DFlash2 lane,
would settle it; nothing in this note does.

One caveat on the QUASAR card that a comparison should carry: it states the BF16 model at
**GPQA-D 0.914**, while `Qwen/Qwen3.8-27B`'s own card states **89.2** for the same model, and
upstream's artifact card quotes 89.2 **[VENDOR]**/**[UPSTREAM]**. So the card's "0.909 vs 0.914"
comparison uses a BF16 reference that does not match the base model's published figure, and its
NVFP4 90.91 is *above* the base card's 89.2. The QAT-beats-PTQ reading rests on that comparison, so
it should be quoted with the discrepancy attached.

And our own perplexity points the other way from the benchmark tables: on the fixed corpus the
QUASAR artifact is the **worst** of the group — 5.00234 (fetched) / 4.99097 (rebuilt) / 4.99744,
against the official stock's 4.90169, the NVIDIA rebuild's 4.90168 and Swift's 4.92432
**[IN-TREE]**. Perplexity on a corpus that is 25 % Chinese reference text is not GPQA, and a QAT
checkpoint can be better at reasoning while being worse at next-token likelihood; but "QAT is the
best artifact" is not a claim the evidence supports.

## 6. KV cache representation

**Ship `fp8`. Unchanged. Three independent reasons and no contrary evidence.**

| KV | quality evidence | provenance |
|---|---|---|
| `bf16` | 4.89838 vs 4.90169 fp8 on the official stock, full corpus — **0.07 %** | **[IN-TREE]** |
| `fp8` | the shipped choice; all three `--kv-dtype fp8` occurrences in upstream's README are in launcher examples; NVIDIA's own ModelOpt card serves this checkpoint with `--kv-cache-dtype fp8_e4m3` | **[UPSTREAM]** README L73/106/212; **[VENDOR]** |
| `int8` (g64) | cometkim, after re-measuring both profiles: *"re-confirmed there is no difference in quality compared to int8 KV. It is just a difference of speed"* — LBv2 short 67.41 (INT8 G64) vs 66.67 (hq-e8-2b) at 262,144; medium 59.07; long 38.89 | **[THIRD]**, #70 |
| `nvfp4` | **on this model**: −1.01 pp GPQA-D, −1.60 pp SWE-bench Verified (381/500 vs 389/500), −0.31 pp GSM8K, equal on AIME 2025 (98.33 %) — SGLang/Qwen/NVIDIA, FP8 weights, FP8 vs NVFP4 KV. On a different model (Qwen3-480B-A35B): MMLU-PRO 77.4 vs 78.1, Ruler 64K 94.6 vs 95.5. Ours: +0.74 % PPL on Swift, concentrated in zhwiki (+2.10 %) | **[VENDOR]**-grade first-party blog, 2026-09-16; NVIDIA developer blog 2025-12-08; **[IN-TREE]** |
| `k8v4` | **no published quality evidence found** — not in the upstream tracker, not in either NVIDIA document, not in the SGLang post | — |

Two corrections to the premise of the question, because both would otherwise be cited:

- **The "0.08 %" figure is not k8v4, and not against BF16.** It is NVFP4-KV vs **FP8**-KV on
  **Qwen3.5-397B-A17B** GSM8K: *"NVFP4 produced one fewer correct GSM8K answer, a difference of
  approximately 0.08 percentage points"*, with aggregate correct counts *identical* on GPQA-Diamond
  and AIME 2025. It is a different model, a different comparison baseline, and a different KV mode.
- **NVIDIA's "<1 % accuracy loss" is measured on a different model.** The Qwen3-480B-A35B figures
  above are the source of that claim; on Qwen3.8-27B the same team's own measurement is 1.0–1.6
  points on two of four benchmarks.

The one *upside* in that post is a lead, not a reason to switch: *"The experiments we performed did
not make use of the per-tensor FP32 global scale of NVFP4 (we use 1.0 for simplicity). Proper
calibration may reduce numeric overflow/underflow and may further close the accuracy gap."* The same
headroom question as §2.2, on the KV axis. Long-context failure modes: neither NVIDIA nor SGLang
reports a *failure* on NVFP4 KV at 64K or beyond; the effect is a graded accuracy cost
(Ruler 64K −1.0 pp), not a breakdown, and SGLang's 1M-token run is explicitly a performance-only
experiment past the model's 262,144 native limit.

## 7. Perplexity and benchmark targets

### 7.1 There is no published NVFP4 perplexity for this model family

Stated plainly because it is the useful answer. Searched and not found in: `Qwen/Qwen3.8-27B`
(no perplexity anywhere on the card), `nvidia/Qwen3.8-27B-NVFP4` (GPQA-D 88.01, Terminal-Bench 74.02,
AA-LCR 73.38, MMMU-Pro 74.86, SciCode 48.41, IFBench 78.93 against BF16 88.92/75.56/72.63/75.14/
47.93/80.07 — no perplexity), `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (GPQA-D 90.91, AIME'26 100.0 —
no perplexity), upstream's own `docs/perplexity.md` (protocol only, no numbers), and upstream's
artifact card (IFBench/AIME/GPQA/ERQA/RealWorldQA, no perplexity).

**The corpus is `ninfer-ppl-1m-v1`: 16 UTF-8 streams over four domains (English reference,
English long-form, Chinese reference, NInfer C++/CUDA code), 1,044,876 scored tokens on `full`,
261,223 on `--quick` (stream 00 of each domain), 4,096-token context with a 2,048-token stride
**[IN-TREE]**, `eval/corpora/perplexity-1m/manifest.json`.** For KV comparisons upstream recommends
the full corpus at `--context 65536 --stride 32768` without `--quick` — a protocol none of our
recorded rows uses.

### 7.2 Exactly one third-party perplexity exists, and our baseline is not the same file

`docs/perplexity-baseline.md` records an unexplained 6.1 % gap: *"A third-party conversion recipe
publishes PPL 4.617 for the official stock artifact on this same corpus. Measured here it is 4.898
with bf16 KV and 4.903 with fp8."* I traced the 4.617 to its source **[THIRD]**, **[UPSTREAM]** #298:

> "its full-corpus perplexity under `ninfer-perplexity` (int8 KV, 4096/2048) is 4.448 vs **4.617 for
> the official stock artifact on the same box**."

`koldfrontier`, NInfer `master` `9e163eee` (2026-09-18), official artifact
`qwen3_8_27b_nvfp4.ninfer` v3, sha256 `74d2c571…`. Two of the three candidates the note lists are
now excluded **[MEASURED-HERE]**:

- **Corpus revision: excluded.** `eval/corpora/perplexity-1m` has exactly two commits —
  `11e76d8d` (2026-08-29, added the corpus) and `a2761ec1` (2026-09-02, KV refactor) — and
  `git diff upstream/master upstream/dev -- eval/corpora` is empty. This tree's copy is
  byte-identical to `upstream/dev`. Both runs postdate the last change by three and two weeks.
- **Scoring-path revision: excluded.** `apps/perplexity` last changed at `a2b7ed11` (2026-09-15,
  logging only), before koldfrontier's `9e163eee` on 09-18. The metric definition, the window rule
  and the aggregation are the ones in `docs/perplexity.md` today.
- **KV dtype cannot be it, in the direction required.** 4.617 is *below* our bf16-KV 4.89838. No KV
  representation coarser than bf16 can produce a better score than bf16 on the same arithmetic.

That leaves a third candidate, and it is checkable and **it is ours**: our local copy of the official
stock is not the published file **[MEASURED-HERE]**.

| | published `neroued/Qwen3.8-27B-nvfp4-NInfer` | `C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer` |
|---|---|---|
| sha256 | `74d2c57145e6ff11d4…` (SHA256SUMS, and `artifact-manifest.json`) | **`9f35ba7407b45e42…`** |
| file bytes | 23,719,715,844 | 23,719,760,043 |
| payload span | 23,719,322,628 | 23,719,317,675 |
| objects / tensors | 1246 / 1240 | **1190 / 1184** |
| bindings / uses | 1513 / 844 | **1518 / 851** |
| bf16 / fp32 / fp8 / int32 / nvfp4 / q4 / q5 / q6 / q8 | 579 / **264** / 146 / 1 / 112 / 55 / 54 / 1 / 28 | 579 / **208** / 146 / 1 / 112 / 55 / 54 / 1 / 28 |
| contiguous_le_v1 | 844 | 788 |

**Every weight-format count matches exactly; the whole difference is that 56 `fp32` objects are
absent, and those are exactly the 56 `contiguous_le_v1` layouts counted the same way.** Net: 5 more
bindings and 7 more uses against 56 fewer tensors. Per `artifact-conventions.md` §2 a differing
file hash is *expected* for a rebuild (`artifact_id` is seeded from `uuid4()`), so the hash alone
proves nothing — but the inventory does not match, and every "official stock" perplexity row in
this repo was measured on the local copy. I cannot tell from here whether the difference is a
rebuild (different packing, same weights) or a different conversion, because the published file is
not present on this machine under any name. **The check that would settle it** is
`python3 tools/release/compare_artifacts.py <published> C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer`,
which diffs the index and digests the payload region separately; it needs the published file
downloaded first. Until then the 6.1 % gap is unexplained, and `docs/perplexity-baseline.md`'s
"two candidates remain" should be corrected to name this one.

### 7.3 Are our numbers in the expected band?

Against the only like-for-like reference available — the shipped NVIDIA ModelOpt PTQ codes, which
NVIDIA calibrated with 2,048 Nemotron samples under Local-Hessian — the rebuilt NVIDIA artifact
measures **4.90168** against the official stock's **4.90169**: a 0.00 % difference on the full
corpus **[IN-TREE]**. That is the strongest single statement available about our conversion
quality, and it says the all-NVFP4 text stack is not costing anything measurable on this corpus. It
is also the one comparison with a caveat attached: `--quick` reads −1.78 % for the rebuild, and the
domain breakdown shows why (wins `chinese_reference` −1.80 %, loses `english_long_form` +1.53 % and
`ninfer_code` +1.21 %, aggregate carried by one `zhwiki-00` stream at −8.85 %). Four streams cannot
carry a conclusion, and `docs/perplexity-baseline.md` already says so.

Published benchmark numbers we can be compared against, with the protocol attached:

| benchmark | BF16 (Qwen card) | NVIDIA ModelOpt PTQ | QUASAR QAT | upstream NInfer NVFP4 (unsloth codes) | our artifacts |
|---|---:|---:|---:|---:|---|
| GPQA-D | 89.2 | 88.01 | 90.91 | 90.40 (179/198) | not measured in this tree |
| AIME 2026 | — | — | 100.0 | 96.67 (29/30) | not measured |
| IFBench | 79.5 | 78.93 | — | 77.00 (231/300) | not measured |
| ERQA | 65.5 | — | — | 66.25 (265/400) | not measured |
| RealWorldQA | 85.9 | — | — | 83.53 (639/765) | not measured |
| LongBenchV2 short | — | — | 66.30 ± 0.86 (INT8, 262,144) | — | not measured |
| perplexity, `ninfer-ppl-1m-v1` full | — | — | — | 4.90169 (local copy; published file differs) | see §0 and `perplexity-baseline.md` |

**The gap in our own evidence is that we have never run a published benchmark on any of the four
artifacts.** Every number above for a `.ninfer` artifact is somebody else's. That is a coverage
hole, not a quality finding, and it is the one I would close first after §2.3.

## 8. Known conversion defects in the upstream tracker

Searched the full issue list (331 issues, `gh issue list --repo Neroued/ninfer --state all`). What
is there, and what is not:

| issue | what it says | bearing |
|---|---|---|
| **#214** (closed) | *"Switch Qwen 3.8 NVFP4 base to `nvidia/Qwen3.8-27B-NVFP4`"*; *"The current NVFP4 quant for Qwen 3.8 27B is based on `unsloth/Qwen3.8-27B-NVFP4`"* | confirms our reading of upstream's official artifact **[MEASURED-HERE]** against the model card's `base_model` field |
| **#70** (closed) | fuller-NVFP4 profile; the 247-site headroom/saturation measurement (§2.2); QAT acceptance drop (§5) | the single most useful issue for this question |
| **#298** (open) | the 4.617 third-party perplexity (§7.2); and a conversion failure worth knowing: a finetune's `tokenizer_config.json` written by a newer `transformers` omits `add_bos_token`, and the reporter traces the refusal to `validate_tokenizer_config` in `src/models/qwen3_5/frontend/frontend.cpp` defaulting that key to `true`. The reporter's suggested doc line is unimplemented. | not a weight problem, but it is a conversion failure the guide does not mention. I did not re-verify the `frontend.cpp` line; the attribution is the reporter's |
| **#245** (open) | upstream ships the custom-recipe mechanism; lists #231, #192, #81, #214, #70 as the related requests | the feature that makes every recipe in this note expressible |
| **#119** (open) | NVFP4's MTP acceptance on *code* is lower than the groupwise-int artifact's at short context (84.2 % → 71.6 %), inverting above 8K prompt tokens on production traffic | a target-model/drafter interaction, not a format-assignment defect |
| **#123** (closed) | `k8v4` and `nvfp4` KV landed in `4ac73c47` (2026-09-01) | the mode's origin; no quality data attached |
| **#285** (closed) | the approximate SwiGLU epilogue cost 0.61 % relative corpus perplexity | already fixed here; recorded because it sets the scale for "one arithmetic choice" |
| **#8**, **#187** (closed) | tensor-descriptor mismatch on `text/token_embedding`; startup failure on `dflash2/feature_projection` | both endpoint/drafter binding sites; both closed, both are the shapes §3.2 and §3.3 touch |
| **#209** (closed) | `download_model.bat: option 4 downloads nvfp4full instead of nvfp4` | the file-naming hazard, live on this machine (§9.2) |

**What is not there: no open or closed issue reports a wrong scale, a bad format assignment, a
mis-bound component, or an incorrect artifact produced by `official_recipes.py` on upstream.** The
two conversion reports in this tree are `official_recipes.py`-adjacent and the only defects I found
are local (§9).

## 9. Two local findings that need action, independent of the research

### 9.1 `out/qwen3_8_27b_nvfp4swift.v3.ninfer.conversion.json` does not describe the shipped Swift artifact

**[MEASURED-HERE]**, 2026-09-28:

| | report | shipped artifact |
|---|---|---|
| payload_bytes | 18,946,410,244 | **19,781,986,308** |
| dflash2 drafter | NVFP4 46 | **Q8 46** |
| timestamp | 2026-09-24 22:17 | 2026-09-24 00:32 |

The report is a **stale** conversion of a Swift build made with the NVFP4 drafter — the encoding
the recipe's own docstring says lost 3.2 acceptance points and was removed. The shipped artifact
matches the current recipe. Three reports in `out/` (`…qat…`, `…nvidia…`, `…swift…`) all carry
`payload_bytes: 18946410244`; that is *expected* rather than alarming, because QUASAR and NVIDIA
recipes assign identical formats over identical shapes and so produce identical sizes from different
sources — but for Swift it is a symptom of the stale report, not a coincidence to explain away.
Per `artifact-conventions.md` §2 the conversion report is the authority on what a build contains,
and this one is not the build that ships. **Renamed to `…conversion.json.stale` on 2026-09-29**, with
the other five reports that fail the same check, so no lane can be resolved from it. The valid report
per lane is now the only unrenamed one: `nvfp4full…conversion.json` and `…v4`, `nvfp4nvidia…`,
`nvfp4qat…`.

### 9.1a No conversion report describes the shipped Swift artifact, or the official stock **[MEASURED-HERE]**, 2026-09-29

**Size is not identity, and for this model family it is actively misleading.** A conversion report's
`payload_bytes` is the tensor payload; the file adds framing, measured at 458,752-466,944 bytes across
the pairings that are known to be correct. Subtracting gives an offset that is either framing-sized
(under 1 MB, so the report matches its lane) or three orders of magnitude out (769,186,816 for the
superseded `nvfp4full` v1/v2/v3, 836,038,912 for the Swift report, 279,319,296 for the
`nvfp4qat.nvfp4draft` report), which is a different build. Exact equality is the wrong test and always
fails, because payload and file size can never be equal.

Applying that test properly, the only report whose payload matches the shipped Swift file's is
`qwen3_8_27b_nvfp4qat.v3.ninfer.old.conversion.json` at 19,781,986,308, an offset of 462,848 — squarely
framing-sized. **It is not a Swift report.** Its sources are `Qwen3.8-27B` and
`Qwen3.8-27B-NVFP4-QUASAR`, and it has 1,072 method records against the shipped Swift file's shape. It
matches on size for the reason this section already gives for QUASAR and NVIDIA: *"QUASAR and NVIDIA
recipes assign identical formats over identical shapes and so produce identical sizes from different
sources."* A size match here is a coincidence of recipe, not evidence of provenance.

So: **no conversion report on disk describes the shipped Swift artifact, and none describes the
official stock `qwen3_8_27b_nvfp4.v3.ninfer`.** Every other report resolves to a lane it genuinely
describes. Per `artifact-conventions.md` §2 that leaves two shipped artifacts with no build record, so
for those the file is the only authority there is.

### 9.1b Retraction: the 9-tensor `nvfp4 -> bf16` set is QUASAR vs NVFP4-full, not Swift **[MEASURED-HERE]**, 2026-09-29

A comparison run on 2026-09-29 against `out/qwen3_8_27b_nvfp4swift.v3.ninfer.conversion.json` — the
stale report this section names — reported nine tensors stored `bf16` in NVFP4-full and `nvfp4` in
"Swift": `self_attn.q_proj` at layers 3, 7, 11, 15, 19, 23 (stride 4), plus `o_proj` at 3 and 7 and
`linear_attn.out_proj` at 4. **That attribution is withdrawn.** The Swift side of that comparison was
the stale report, so the finding is a QUASAR-vs-NVFP4-full difference and nothing more. Re-run against
the QUASAR report, the nine tensors and the one-directional `nvfp4 -> bf16` transition reproduce
exactly, and the module roles and layer stride are unchanged; what changes is only which lane the
"Swift" side actually was.

**The nine are not a discovery; they are NVFP4-full's own documented recipe.**
`docs/maintainer/qwen3.8-27b-artifact.md` already records that artifact's Text allocation as
*applying the Qwen3.6-27B nvfp4 exception pattern*: `attention/query_key_gate_value` bf16 on layers
3, 7, 11, 15, 19, 23 and nvfp4 on the other ten full-attention layers; `attention/output` bf16 on
layers 3, 7; `gdn/output` bf16 only on layer 4. Six plus two plus one is the nine measured here, on the
same layers and the same roles, which is a correctness check on the measurement as well as a
coincidence ruled out. So the real recipe difference is that **QUASAR does not carry the exception
pattern and NVFP4-full deliberately does**, inherited from Qwen3.6-27B — not an unexplained divergence
between two producers doing the same thing.

**The Swift-vs-NVFP4-full recipe comparison still cannot be made from the reports on disk** — §9.1a is
why. The two artifacts come from different source checkpoints (`Swift-Qwen3.8-27b` against
`Qwen3.8-27B`), which is established by the reports only for NVFP4-full's side. Whether Swift's recipe
differs in anything but source is **not established**, and the 1.13 % perplexity gap between Swift and
NVFP4-full is not attributed.

Also in `out/` under a shipped name: `qwen3_8_27b_nvfp4qat.v3.ninfer.old` (19,782,449,156 bytes) and
`qwen3_8_27b_nvfp4qat.v3.ninfer` (18,946,877,188) are different builds of the same artifact, and
`C:\AI\models\qwen3_8_27b_nvfp4full.v3.ninfer` is the `.v4`/`.rebuilt` build, not the one the
recorded 4.98768 / 4.97532 rows were taken on. The `20260928-173142` report shows that file scoring
**5.00285**, which is not any recorded row.

**Cleared 2026-09-29.** `C:\AI\models` held eleven `qwen3_8_27b*` files for five shipped artifacts.
It now holds exactly the five, named by `tools/release/profiles.py` plus the official stock. One file
was deleted — `qwen3_8_27b_nvfp4swift.v3.ninfer.before-draft`, byte-identical to the shipped Swift
artifact by size, by sampled content and by mtime, so its loss is provably nil. The other ten, 177.7
GiB including both original downloads and every staging copy that sat in `out/` under a shipped name,
were moved to `C:\AI\models\_superseded\` rather than deleted, which clears every glob and name lookup
at the top level while losing nothing; the move is reversible. The five shipped artifacts were
re-hashed afterwards and all five are unchanged. The six hazardous conversion reports were renamed
with a `.stale` suffix rather than deleted, per the recommendation in this section, so the provenance
survives and the report can no longer be resolved.

### 9.2 Every "official stock" baseline in this repo is the wrong file

§7.2. The local copy is not the published artifact, and the published artifact is not on this
machine. This is the highest-value cheap check outstanding, because it decides whether a 6.1 %
discrepancy in our own documentation is a conversion problem or a bookkeeping one.

## 10. Two conclusions that changed while this was written

Recorded because the method error, not the individual claim, is what recurs — and because both of
these were written down before the file that corrected them was read.

| # | claim as first written | corrected by | error |
|---|---|---|---|
| 1 | "NVIDIA prescribes a headroom margin on the NVFP4 activation global scale (`rho=16384`, percentile-anchored), and we have none — option A is a deviation from NVIDIA's recipe" | `Qwen3.8-27B-NVFP4/.quant_summary.txt`: every NVFP4 site's `input_quantizer` is `calibrator=MaxCalibrator`, a plain observed maximum | I read a ModelOpt *facility* as a *prescription for this checkpoint*. The facility exists and has defaults; the recipe that produced this checkpoint did not use it. Option A survives on **[THIRD]**'s saturation count alone, and is ranked lower because of it |
| 2 | "the vision tower's Q4/Q5/Q6/Q8 assignment is covered by the source checkpoints' convention, like every other row in §0" | the same dump, plus QUASAR's `ignore: ["re:.*visual.*"]` and unsloth's 303-entry `ignore` list, of which **110 name the vision tower** (27 blocks × 4 projections, both `merger` linears) and 48 each name `in_proj_a`, `in_proj_b` and `linear_attn.norm` | I generalised "every source leaves the embedding and the a/b projections unquantized" into "every source leaves the optional components unquantized", and did not check the vision row against a single source. It does not hold, and it is the only assignment in the four recipes that is purely ours |

Also worth recording because it nearly produced a wrong table: **673 of the 1,513 bindings in a v3
artifact are `parts` records** — row-range views into a fused parent — and a format census that
resolves only the `object` key silently reports a third of the artifact. The first pass through the
four shipped artifacts undercounted the drafter as 21 NVFP4 projections when it is 46, and the text
stack as 128 when it is 512. Any future format census must resolve `parts[*].object`; the counts in
§0 and §2.3 do.

## 11. What is too thin to act on

Named explicitly, because each of these would be easy to mistake for a finding:

1. **Avalonec's 21.4 % DFlash2 acceptance under an NVFP4 draft** (#214). Contradicts our 58.0 % on
   the same encoding by 2.7×; "unquantized/requant" is undefined; no protocol. Not actionable, and
   it does not support changing any drafter format.
2. **The 15× weight-space error on NVIDIA's NVFP4 `lm_head`** (#214). NVIDIA's own documentation
   says weight-space error does not predict downstream accuracy, and the author who measured it
   declined to act on it. It is a reason to keep measuring the head, not a reason to change it.
3. **The 0.08 % figure.** Different model, different baseline, different KV mode (§6). It is not
   evidence for k8v4 and must not be cited as such.
4. **NVIDIA's "<1 % accuracy loss" for NVFP4 KV.** Measured on Qwen3-480B-A35B. On Qwen3.8-27B the
   same team's measurement is 1.0–1.6 points on two of four benchmarks.
5. **`k8v4` entirely.** No quality evidence exists in any source I could reach. The mode is real
   (`4ac73c47`) and its density argument is arithmetic, but "no evidence" is not "evidence of no
   harm", and it is not a reason to consider it.
6. **Local-Hessian's expected gain on *our* artifacts.** The 5.10 → 3.10 average drop is measured on
   **Qwen3.5-9B**. NVIDIA also reports that composing it with GPTQ *hurt* on Qwen3.8-27B. The
   direction is supported by the producer and by the checkpoint's own recorded calibrator; the
   magnitude for a 27B dense model with vision is not established, and `artifact-conventions.md` §1
   puts the whole locally-encoded axis at +0.05 % / +0.50 % of perplexity, so the upside is bounded
   by something well under a point.
7. **"QAT is best."** It rests on a comparison whose BF16 reference (91.41) does not match the base
   model card's (89.2), and our own perplexity puts QUASAR last of the group.
8. **Whether a per-block scale sweep helps *this* model.** It is free at runtime and NVIDIA-measured
   as better on another model, but nothing measures it on Qwen3.8-27B, and a numerics change with no
   oracle is a risk. Measure the weight-space error first (free, offline) and the perplexity second.
9. **The vision tower's quantization.** Not thin as a *decision* — §3.4 argues it is ours alone and
   worth an estimated ~600 MiB — but thin as an *evaluation*: `ninfer-perplexity` loads Text weights
   only (`docs/perplexity.md`), so no number we can currently produce bears on it either way. A
   vision benchmark is a prerequisite, not a detail. The 600 MiB is derived from the object's byte
   count and its bits-per-weight, not measured as a BF16 rebuild.
10. **The 1.26 GiB an NVFP4 endpoint pair would return.** The arithmetic is firm (2.52 GiB of Q8 is
    measured; NVFP4 on the same two matrices is 0.63 GiB each per **[THIRD]**'s own figure for the
    head). What is missing is the acceptance and perplexity cost on our engine, and the runtime does
    not register an NVFP4 vocabulary projection, so the build is not a recipe edit alone. §3.2.

## 12. Sources

**Primary, read in full:** NVIDIA Model Optimizer, "Improving NVFP4 Accuracy with Local-Hessian
Weight Scales", 2026-09-09 (`nvidia.github.io/Model-Optimizer/announcements/local-hessian.html`);
`modelopt.torch.quantization.model_calib` API reference (same site); NVIDIA developer blog,
"Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache", 2025-12-08;
SGLang/Qwen/NVIDIA, "Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache", 2026-09-16;
`nvidia/Qwen3.8-27B-NVFP4` README, `hf_quant_config.json`, `.quant_summary.txt`;
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` README, `config.json`, `model.safetensors.index.json`;
`unsloth/Qwen3.8-27B-NVFP4` `config.json`; `Swift-Qwen3.8-27B-NVFP4` `hf_quant_config.json`;
`Qwen/Qwen3.8-27B` README; `neroued/Qwen3.8-27B-nvfp4-NInfer` README, `SHA256SUMS`,
`artifact-manifest.json`; `Neroued/ninfer` issues #70, #119, #123, #164, #214, #245, #298, commit
`4ac73c47`, `docs/perplexity.md`, `docs/weight-conversion.md`, `tools/convert/official_recipes.py`
at `upstream/dev` (`7f6aafed`); `satellitedown/cinference` `tools/convert/{official_recipes,
methods}.py` and `tools/convert/` listing at `main`; `Wallawalla47/ninfer-custom`
`tools/convert/{official_recipes,methods}.py`, `tools/convert/quantization/groupwise.py` at
`master`.

**Primary, read in this tree:** `tools/convert/official_recipes.py`, `calibration.py`,
`calibration_corpus.json`, `qwen3_8_27b_nvfp4_calibration.json`, `quantization/nvfp4.py`,
`docs/weight-conversion.md`, `docs/perplexity-baseline.md`,
`docs/maintainer/artifact-conventions.md`, `src/ops/wrapper/attn_input_proj.cpp`,
`src/ops/linear/q4/q4_shapes.h`, `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp`;
`C:\AI\models\hf-src\{Qwen3.8-27B, Qwen3.8-27B-NVFP4-nvidia, Qwen3.8-27B-NVFP4-QUASAR,
Qwen3.8-27B-NVFP4-unsloth, Swift-Qwen3.8-27B-NVFP4, Qwen3.8-27B-DFlash2}`;
the four shipped `.ninfer` artifacts and every `profiles/perplexity/**/report.json`;
`out/*.conversion.json`.

**Commands that reproduce the [MEASURED-HERE] claims:** `python3 -m tools.artifact.inspect
<artifact> --json` for the inventory counts; a `tools.artifact.reader.Artifact` walk that resolves
both `object` and `parts[*].object` for the per-binding formats and the per-component byte split
(§10, third row, is the failure mode if the second is skipped); `certutil -hashfile` for the stock
artifact's SHA-256; `git log --format="%h %ad %s" --date=short upstream/{dev,master} -- eval/corpora`
and `-- apps/perplexity` plus `git merge-base --is-ancestor a2b7ed11 9e163eee` for the two excluded
revisions; `gh issue view <n> --repo Neroued/ninfer --comments` for the tracker quotes;
`Select-String` over `.quant_summary.txt` for the calibrator census.
