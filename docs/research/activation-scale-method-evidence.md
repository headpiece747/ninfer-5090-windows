# Activation scale derivation: primary-source evidence

Research note, 2026-10-01. Scope: how an FP8 or NVFP4 **activation** scale is derived canonically,
whether a large per-site disagreement between two max calibrations is explainable, what published
FP8/NVFP4 inference engines actually do at runtime, whether any accuracy comparison exists, and
whether the NVFP4 subnormal-flush mechanism is detectable after the fact.

Every claim below is tagged:

| Tag | Meaning |
|---|---|
| `[SRC-CODE]` | Read in the owning project's own source file (line numbers from `main`/default branch on 2026-10-01) |
| `[SRC-DOC]` | Read in a first-party document (NVIDIA doc, repo README, doc page) |
| `[SRC-ISSUE]` | Read in a first-party issue/PR on the owning project's tracker, including maintainer replies |
| `[VENDOR-CLAIM]` | A first-party statement of a result, with no independent measurement I could find |
| `[INFERRED]` | Arithmetic or reasoning I performed from cited constants. Not measured. |
| `[NOT-FOUND]` | Searched, did not find. Stated as a negative result, not padded. |
| `[IN-TREE]` | This repository's own files, read for context. |

Nothing in this note is a measurement performed on this machine. No GPU job, no calibration run, no
artifact read. Where a number is arithmetic I say so.

---

## Verdict table

| # | Question | Answer | Confidence | Primary source |
|---|---|---|---|---|
| 1 | Canonical way to derive an FP8/NVFP4 activation scale? | `amax` **is** the per-tensor absolute max. Exported divisor = `amax/448` (FP8) and `amax/2688` (NVFP4). Per-16 block scales are FP8-E4M3 and are **dynamic at inference**. A documented rule for the global scale does exist, and it is stated in terms of where block scales land in the E4M3 exponent range — not as a margin on the amax. Four documented policies exist: `max`, `nvfp4_act_headroom`, `constant_amax`/`input_scale1`, and runtime recomputation. | High | `config.py:674-688`; `tensor_quantizer.py:315-321,833`; `nvfp4_tensor.py:32-46,141-162,206-228`; `modelopt_recipes/ptq.md:186-257`; TensorRT-LLM#3037 |
| 2 | Is a 40x per-site disagreement plausible from corpus choice alone? | **Structurally yes, and corpus is not the only candidate cause.** Both numbers are the same statistic (per-tensor abs max of the same module input), so "per-tensor vs per-block" does not explain it. But four other measured-from-source differences also separate the two calibrations, any one of which is sufficient. The in-tree note's own arithmetic does not reconcile. | Medium | `calibration.py:151,159-165`; `max.py:54-90`; `model_calib.py` `max_calibrate`; `ptq.md`; see §2.3 |
| 3 | Do published engines use static checkpoint activation scales or dynamic runtime ones? | **It splits by format, and the split is consistent across all three engines.** FP8: **dynamic per-tensor at runtime by default** in vLLM, SGLang and TensorRT-LLM. NVFP4: **per-tensor global scale is baked into the checkpoint and consumed statically** in all three; only the 16-wide block scale is dynamic. TensorRT-LLM carries both paths and picks static when the checkpoint supplies a scale. NVIDIA's own best published NVFP4 recipe used *dynamic max-based* activation scaling. | High | `fp8.py:98,511-515`; `_custom_ops.py:1823-1891,1513-1588`; `compressed_tensors_w4a4_nvfp4.py:86-138`; sglang `fp8.py:181,493`; sglang `compressed_tensors_w4a4_nvfp4.py:88-96,145`; TRT-LLM `linear.py:1657-1676,1853-1864,2502-2510`; Nemotron blog Table 2 |
| 4 | Published accuracy comparison, max vs headroom/percentile activation scaling for NVFP4? | **No perplexity or benchmark-accuracy comparison exists.** What exists: (a) one generation-*length* figure on one model/scope, which did not reach its own gate; (b) two tensor-level figures from the PR that added the calibrator. No activation-side headroom row appears in NVIDIA's published NVFP4 accuracy table, whose alternative rows are all weight-side. | High (for the negative) | `modelopt_recipes/ptq.md:212-217`; Model-Optimizer#2028; Nemotron blog Table 2; `CHANGELOG.rst:126` |
| 5 | Is max calibration documented as dangerous for NVFP4, with a threshold, and is it detectable after the fact? | **Documented, with a named threshold.** The E4M3 normal-range ratio `28672` *is* the threshold, and it is in NVIDIA's source as `_FP8_NORMAL_DYNAMIC_RANGE`. Detectable after the fact by reading the stored E4M3 block scales: a block is subnormal below `2^-6` and flushes below `2^-9`. For **activations** this needs a runtime pass, because the block scales are dynamic and not in the artifact. | High | `nvfp4_act_headroom.py:29-36,` docstring; `nvfp4_tensor.py:37,44-46`; `CHANGELOG.rst:178`; `ptq.md`; TRT-LLM#3037 |

---

## 1. What is the canonical way to derive an activation scale

### 1.1 `amax` is the per-tensor absolute max, and the export is a divisor

`MaxCalibrator` reduces each collected tensor with `convert_quantization_axis_to_reduce_axis(x, axis)`
and takes a running elementwise max; with `axis=None` the reduction is over the whole tensor, so the
result is a scalar per-tensor absolute max. No margin, no clipping, no percentile. `[SRC-CODE]`
`modelopt/torch/quantization/calib/max.py:54-90`.

The mapping from `amax` to a scale is stated in three places, and they agree:

- **FP8:** the real-quantize path computes `scales=self.amax / 448.0`. `[SRC-CODE]`
  `modelopt/torch/quantization/nn/modules/tensor_quantizer.py:824-834`.
- **NVFP4:** `maxbound` returns `6.0` for `num_bits == (2,1)` with `scale_bits == (4,3)`. `[SRC-CODE]`
  `tensor_quantizer.py:314-321`. The export is
  `activation_scaling_factor = amax / (quantizer.maxbound * E4M3_MAX) = amax / (6 * 448)`. `[SRC-CODE]`
  `modelopt/torch/quantization/qtensor/nvfp4_tensor.py:210-228`.
- **Stated in prose, in NVIDIA's own config schema:** "For NVFP4 activations the exported
  `input_scale` equals `amax / (E2M1_MAX * E4M3_MAX) = amax / (6 * 448)`; setting `constant_amax` to
  `2688.0` therefore yields an exported `input_scale` of `1.0`." `[SRC-CODE]`
  `modelopt/torch/quantization/config.py:674-688`.

So `FULL_RANGE = 2688.0` in `tools/convert/calibration.py:26` is not a convention invented by this
port; it is the denominator NVIDIA names in its own schema. This confirms the in-tree background
statement and I am not re-opening it.

`export_amax` makes the two-level structure explicit: for a dynamic-block quantizer it returns the
per-tensor `amax` unchanged, commented as "the NVFP4 second-level scale". `[SRC-CODE]`
`tensor_quantizer.py:1113-1118`.

**Divisor, not multiplier — confirmed by a fourth independent codebase.**
compressed-tensors' ModelOpt→compressed-tensors NVFP4 converter lists the incoming parameter names
`["input_scale", "weight", "weight_scale", "weight_scale_2"]` and converts
`input_scale x -> 1/x` into `input_global_scale`, and `weight_scale_2 x -> 1/x` into
`weight_global_scale`. `[SRC-CODE]` `vllm-project/compressed-tensors`,
`src/compressed_tensors/entrypoints/convert/converters/modelopt_nvfp4.py:35,77-107`.
TensorRT-LLM carries the same comment: "modelopt ckpt stores `amax/(448*6)`, convert to
`(448*6)/amax`". `[SRC-CODE]` `tensorrt_llm/_torch/modules/linear.py:1857-1858`.

### 1.2 The stored per-block scale is `448 * block_amax / global_amax`

This is the identity that makes the whole question decidable from bytes, and it is readable
directly from ModelOpt's quantizer tensor code. On the static-export path: `[SRC-CODE]`
`nvfp4_tensor.py:141-162`

```
per_block_scale      = per_block_amax / E2M1_MAX
per_block_scale_max  = global_amax / E2M1_MAX
# then, in _cast_per_block_scale_to_fp8 (lines 32-46):
per_block_scale      = per_block_scale * E4M3_MAX / per_block_scale_max
                     = 448 * per_block_amax / global_amax          # (algebraically)
return per_block_scale.clamp(min=2**-9, max=E4M3_MAX).to(float8_e4m3fn)
```

On the fully dynamic path (no calibrated amax at all) both levels come from the tensor: `[SRC-CODE]`
`nvfp4_tensor.py:185-208`

```
per_block_amax       = reduce_block_amax(input, block_sizes={-1: block_size})
weights_scaling_factor_2 = reduce_amax(input) / (E2M1_MAX * E4M3_MAX)
per_block_scale      = per_block_amax / (E2M1_MAX * weights_scaling_factor_2)
```

In both cases the per-tensor global scale is the literal per-tensor absolute max when it is not
supplied, and the block scale is the block's amax normalized against it. The E4M3 block scale is a
*normalized* quantity in `[0, 448]` by construction, and 448 is the largest value it will take.

### 1.3 Is there a documented rule for what the global activation scale SHOULD be?

**Yes — four of them, all first-party, and they disagree with each other.** That disagreement is
itself the finding.

**(a) `max` — the default.** "amax/max calibration. Fast, one calibration pass; the baseline
choice." `[SRC-DOC]` `modelopt_recipes/ptq.md:186-187`. This is what
`modelopt_recipes/configs/ptq/presets/model/nvfp4.yaml` ships (`algorithm: max`) `[SRC-CODE]`, and
what the shipped Qwen3.8-27B recipe ships. `[SRC-CODE]`
`modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_max.yaml` → `algorithm: max`,
with `*language_model.layers.*.mlp.*.input_quantizer: nvfp4` (dynamic block scales),
`*.self_attn.*_proj.input_quantizer: fp8`, `*.linear_attn.*_proj*.input_quantizer: fp8`, and
`linear_attn.in_proj_a*` / `in_proj_b*` disabled. Its `local_hessian` variant overrides only the
weight algorithm and adds `get_qdq_activations_from_prev_layer: true`. `[SRC-CODE]`
`nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml`.

**(b) `nvfp4_act_headroom`.** The documented rule is explicitly about exponent placement:

> Plain `max` anchors that scale to the largest per-block amax seen during calibration, so any
> larger activation at inference saturates. This variant instead sets
> `amax = max(rho * anchor, upper)` from the per-block amaxes at `anchor_percentile` (default 1) and
> `upper_percentile` (default 99.99), with headroom factor `rho` (default 16384): **calibrated
> blocks sit low in the FP8 block-scale range, leaving the rest for unseen outliers.**

`[SRC-DOC]` `modelopt_recipes/ptq.md:194-217`. Implementation `[SRC-CODE]`
`calib/nvfp4_act_headroom.py` (`anchor_percentile=1.0, upper_percentile=99.99, rho=16384.0`,
validated `0 < rho < 28672`).

**(c) `input_scale1` / `constant_amax`.** "Pins the expert **activation** per-tensor amax to a
constant `2688.0` (= E2M1_MAX × E4M3_MAX = 6 × 448) via `constant_amax`, so the exported NVFP4
`input_scale` is exactly **1.0** and those quantizers skip activation calibration entirely (no
forward statistics collected) … the per-block E4M3 activation scales remain dynamic." `[SRC-DOC]`
`modelopt_recipes/ptq.md:224-238`; `[SRC-CODE]` `config.py:674-695`.
This is a third distinct answer to "what should the global scale be": a constant, independent of
the data.

**(d) Recompute it at runtime from the current input.** TensorRT-LLM's dynamic branch is exactly
`input_scale = 448 * 6 / amax(input)` on every forward. `[SRC-CODE]` TRT-LLM `linear.py:1657-1668`.

The engineer-level statement of the principle is on TensorRT-LLM's own tracker: `[SRC-ISSUE]`
NVIDIA/TensorRT-LLM#3037, `juney-nvidia`, 2025-03-24 and 2025-03-25:

> "For NVFP4, it introduces two-levels quantization method, the first top-level is the per-Tensor
> quantization scaling factor, the second level is the fine-grained blockwise quantization scaling
> factor and yes each group has its own range and scaling factor."
> "Yes, the needs for two levels of scaling factors are for accuracy purpose."
> "For activation, **the per-Tensor scaling factor is computed offline, the blockwise scaling factor
> is computed online.**"

(The README sentence quoted in that issue — "Activation global scale are calibrated" — is no longer
in `examples/quantization/README.md`, which now defers to ModelOpt and to pre-quantized Hub
checkpoints. `[SRC-CODE]` TRT-LLM `examples/quantization/README.md`, read 2026-10-01.)

### 1.4 `PercentileCalibrator` no longer exists as a separate class

`[SRC-CODE/SEARCH]` A GitHub code search for `PercentileCalibrator` in `NVIDIA/Model-Optimizer`
returns 3 hits: two under `examples/diffusers/quantization/` and one test. None is in the core
quantization path. The current structure is a single `HistogramCalibrator` whose
`compute_amax(method=...)` accepts `"entropy"`, `"mse"` or `"percentile"`, with
`percentile: float = 99.99` as the default keyword. `[SRC-CODE]`
`modelopt/torch/quantization/calib/histogram.py:32-36,135-190`. A GitHub search for
`class MaxCalibrator`-adjacent layout confirms the `calib/` package is
`__init__.py, bias.py, calibrator.py, histogram.py, max.py, mse.py, nvfp4_act_headroom.py`. `[SRC-CODE]`

Note the direction of each: the element-level histogram percentile **clips a high-tail fraction**;
`NVFP4ActHeadroomCalibrator` histograms **per-block amaxes** (a derived quantity) on a **log2** grid
and reads the **low** tail as an anchor. PR #2028 states the reason explicitly: block amaxes "span
many decades and the anchor is read from the **low** tail, where linear bins have almost no
resolution". `[SRC-ISSUE]` NVIDIA/Model-Optimizer#2028.

### 1.5 llm-compressor

`[NOT-FOUND]` I did not read `neuralmagic/llm-compressor` for this note. ModelOpt is the tool that
produced the checkpoint in question and the three engines' loaders all trace to ModelOpt's export
format, so llm-compressor was not on the critical path. If a second producer's calibration policy
matters later, that is an open read.

---

## 2. The 40x disagreement: plausible, and probably not only the corpus

### 2.1 What the two numbers actually are

The port: a forward pre-hook on the target module, `hidden.detach().abs().amax()`, accumulated as a
running max, over **one** `model(**encoded)` call on the **joined corpus as a single sequence**,
with **BF16 weights** (no fake-quant). `[IN-TREE]` `tools/convert/calibration.py:146-175`
(`hidden = args[0]`, `value = hidden.detach().abs().amax().item()`,
`joined = "\n\n".join(documents)`, `with torch.no_grad(): model(**encoded)`).

NVIDIA: `MaxCalibrator` on the same module input quantizer, i.e. the same statistic, accumulated
over many rows and many forward batches. `[SRC-CODE]` `calib/max.py:54-90`.

**So both numbers are the per-tensor absolute max of the same module input.** The
"per-tensor vs per-block" hypothesis is *not* supported for the port's number.

But the headroom calibrator's `upper` **is** a different statistic — the 99.99th percentile of
per-block amaxes, not the per-tensor max. If a headroom-calibrated `amax` were ever compared
against a max-calibrated `amax`, a ratio of tens would be expected by construction. `[SRC-CODE]`
`nvfp4_act_headroom.py:181-200`. I flag this because the in-tree prior finding compared two max
calibrations, and the two cases must not be conflated.

### 2.2 Differences between the two calibrations that are readable in source

Each of the following is sufficient on its own to move a per-tensor max by a large factor, and none
of them is the corpus.

1. **Weights are fake-quantized during NVIDIA's activation forward; the port's are BF16.**
   `max_calibrate` calls `enable_stats_collection(model)`, then `weight_only_quantize(model)`, and
   only then `forward_loop(model)`. `[SRC-CODE]` `modelopt/torch/quantization/model_calib.py`,
   `max_calibrate`. The activation statistics are therefore collected through already-quantized
   weights. The port's `calibration.py:112-116,164-165` loads `dtype=torch.bfloat16` and never
   installs a quantizer.
2. **Layerwise calibration propagates QDQ activations.** The shipped Qwen3.8-27B local-Hessian
   variant sets `get_qdq_activations_from_prev_layer: true`, i.e. a layer's input quantizer sees the
   *previous quantized layer's output*, not the BF16 hidden state. `[SRC-CODE]`
   `nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml`. If the shipped checkpoint used this variant, every
   layer past the first is measuring a different tensor from the port's. The prior note read
   `calibrator=MaxCalibrator` on the activations and `NVFP4MSECalibrator` on the weights `[IN-TREE]`
   `docs/research/quantization-coverage-evidence.md:434-436`, which is consistent with either the
   `max` recipe or the local-Hessian one — the metadata does not distinguish them.
3. **Sequence structure and padding.** ModelOpt warns that "padding tokens participate in attention,
   skewing calibration statistics" and forces `padding_side=left`; the default path tokenizes with
   `truncation=True, max_length=max_sample_length`. `[SRC-CODE]`
   `modelopt/torch/utils/dataset_utils.py:798-800,887-916`. The port's single joined sequence has no
   padding and no row structure — a different attention context for the same text.
4. **Token count.** ModelOpt's published default is `num_samples=512` rows of
   `max_sample_length=512` tokens. `[SRC-CODE]` `dataset_utils.py:749-750,998,1188-1189`. A max over
   512×512 tokens and a max over the port's corpus are maxima over different-sized sets; the larger
   set's max is the larger one, and the ratio is bounded only by the activation range.

### 2.3 On the prior note's own arithmetic

`[IN-TREE]` `docs/research/quantization-coverage-evidence.md:410-421` states, of the same site
(`mlp/down_projection` layer 0): "NVIDIA's implies amax ≈ 2092 and the port measured ≈ 5.12, a ratio
of 0.025 in divisor." Those three numbers are not mutually consistent:

- `2688 / 5.12 = 525.0` and `2688 / 2092 = 1.285`; their ratio is **410**, not 0.025.
- Equivalently `2092 / 5.12 = 408.6`, whose reciprocal is 0.0024, not 0.025.

And the table's `mlp/down_projection` row gives port divisors `6.65 – 886.76` and NVIDIA-derived
divisors `1.28 – 96.86`, with a ratio range of `0.025 – 0.665`. A divisor ratio of 0.025 means the
**NVIDIA** divisor is 40× the port's, i.e. NVIDIA's `amax` is 40× **smaller** — the opposite
direction from the two amax values quoted in the sentence. So "up to 40x" and the amax pair in that
sentence describe different quantities, and the sentence as written cannot be read off the table.
I did not re-derive the 24-site table; I am reporting that its prose and its own numbers disagree.
`[IN-TREE]`

**On plausibility itself:** `[INFERRED]` Nothing in the format bounds how far apart two per-tensor
maxima over two different token sets can be. 40× is 5.3 binades, which is unremarkable for a
max-of-magnitudes statistic and well inside the dynamic range of a BF16 activation. So 40× from
corpus alone is plausible. But "plausible" is not "attributable": given the four differences in
§2.2, corpus choice is one of five candidate causes and there is no evidence here that ranks them.

### 2.4 NVIDIA's own claim that calibration data barely matters

`[SRC-DOC/VENDOR-CLAIM]` `examples/hf_ptq/README.md:52`: "The accuracy of PTQ is typically robust
across different choices of calibration data, by default Model Optimizer uses a mix of
`cnn_dailymail` and `nemotron-post-training-dataset-v2`."

This is a claim about **accuracy**, not about the `amax` value. Two calibrations can produce wildly
different amaxes and identical accuracy, so it does not contradict a 40× amax difference — but it
does mean a 40× amax difference is not, on its own, evidence of an accuracy problem. The two
questions have to be kept apart.

### 2.5 Is there a reproducible recipe? Yes, and it is not the one the prior note describes

`[SRC-DOC]` `plugins/modelopt/skills/ptq/SKILL.md:85-104,126` (NVIDIA's own version-controlled
skill) documents: for listed models run `--calib_size 512` directly; for text-only LLM PTQ
"Prefer the representative `nemotron-post-training-v3` blend.
`modelopt/torch/utils/dataset_utils.py` expands it to seven registered Nemotron SFT domains";
`cnn_dailymail` is a constrained-environment fallback, not the preferred set.

Invocation: `python examples/hf_ptq/hf_ptq.py --pyt_ckpt_path <model> --recipe <yaml> --dataset <name>
--calib_size 512`. `[SRC-ISSUE]` #2028 gives the same invocation; `[SRC-DOC]` `hf_ptq/README.md:292,
301` shows `--calib_size 512` in the shipped examples.

`[SRC-CODE]` Defaults read from `modelopt/torch/utils/dataset_utils.py`:
`num_samples: int = 512` (lines 749, 1188), `max_sample_length: int = 512` (lines 750, 998),
`batch_size: int = 1` (line 730), `padding_side` forced left (798-800), `attention_mask` always
included (908-916). An alternative `_pack_documents_into_rows` path (684-726) concatenates
documents into one EOS-separated stream and slices uniform rows, "Megatron-LM pretraining style".

`[NOT-FOUND]` / **not independently verified:** the in-tree note's claim that the shipped
`nvidia/Qwen3.8-27B-NVFP4` checkpoint used "512 samples × **2,048** tokens". I found no public
artifact stating the sequence length used for that specific checkpoint, and the current published
default is 512 tokens, not 2048. I have not read the checkpoint to check. The dataset name
(`nemotron-post-training-v3`) is consistent with the current SKILL.md recommendation, but the
2048 figure is unverified and should be treated as inherited.

`[SRC-DOC]` Note on reproducibility drift: TRT-LLM's `examples/quantization/README.md` points at
"TensorRT-Model-Optimizer Hugging Face export flow (`examples/llm_ptq` in that repository)", which
in Model-Optimizer is now `examples/hf_ptq`. Third-party reproduction from a blog link will 404 on
the example path.

---

## 3. State of the art in published FP8/NVFP4 engines

**The answer splits by format, and all three engines agree on where the line is.** This is the most
decision-relevant section of this note.

### 3.1 TensorRT-LLM — both paths, explicitly selected

The static/dynamic branch is written out, with no indirection. `[SRC-CODE]`
`tensorrt_llm/_torch/modules/linear.py:1657-1676` (NVFP4):

```python
# Dynamic vs static quantization
if module.input_scale is None or module.force_dynamic_quantization:
    FP8_MAX, E2M1_MAX = 448.0, 6.0
    amax_input = torch.amax(torch.abs(input)).float()
    input_scale = FP8_MAX * E2M1_MAX / amax_input
    alpha = (amax_input / (FP8_MAX * E2M1_MAX)) * module.weight_scale_2
else:
    input_scale = module.input_scale          # static, from checkpoint
    alpha = module.alpha
```

and the loader `[SRC-CODE]` `linear.py:1839-1864`:

```python
if input_scale is not None:
    # modelopt ckpt stores amax/(448*6), convert to (448*6)/amax
    input_scale = 1.0 / input_scale
else:
    # Dynamic mode: input_scale and alpha computed at runtime
    alpha = None
```

So a ModelOpt NVFP4 checkpoint **takes the static path**. FP8 per-tensor has the same structure
(`static_quantize_e4m3_per_tensor` vs `quantize_e4m3_per_tensor`) `[SRC-CODE]` `linear.py:2498-2510`.
There is also a `force_dynamic_quantization` per-module flag `[SRC-CODE]` `linear.py:3732,3770`.

### 3.2 vLLM — FP8 dynamic by default, NVFP4 static global + dynamic block

- FP8: `Fp8Config(..., activation_scheme: str = "dynamic")` is the default, validated against
  `ACTIVATION_SCHEMES` `[SRC-CODE]` `vllm/model_executor/layers/quantization/fp8.py:98,114-116`. For
  MoE the activation key is `kFp8StaticTensorSym` only when `activation_scheme == "static"`, else
  `kFp8DynamicTensorSym` `[SRC-CODE]` `fp8.py:510-515`. `scaled_fp8_quant`'s docstring: "This
  function supports both static and dynamic quantization: If you provide the scale, it will use
  static scaling and if you omit it, the scale will be determined dynamically", dispatching to
  `dynamic_scaled_fp8_quant` (per-tensor) or `dynamic_per_token_scaled_fp8_quant`. `[SRC-CODE]`
  `vllm/_custom_ops.py:1823-1891`.
- NVFP4: `CompressedTensorsW4A4Fp4.create_weights` registers
  `input_global_scale = PerTensorScaleParameter(...)` loaded from the checkpoint `[SRC-CODE]`
  `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:86-92`; `process_weights_after_loading`
  comments "Process input global scale and pre-compute alpha for W4A4 mode" and stores
  `input_global_scale_inv` "for runtime quantization" `[SRC-CODE]` same file, 116-138. The kernel
  calls `scaled_fp4_quant(x, layer.input_global_scale_inv, ...)` `[SRC-CODE]`
  `vllm/model_executor/kernels/linear/nvfp4/cutlass.py:54-60`. The op's docstring is the canonical
  one-liner: **"For every 16 consecutive elements, a single dynamically computed scaling factor is
  shared. This scaling factor is quantized using the `input_global_scale`"** `[SRC-CODE]`
  `vllm/_custom_ops.py:1520-1531`.
- vLLM's scale taxonomy makes the static/dynamic split explicit as first-class objects:
  `kNvfp4Dynamic = QuantKey(FP4, scale=(dynamic, (1,16) E4M3), scale2=kStaticTensorScale)` vs
  `kNvfp4DynamicToken = QuantKey(FP4, scale=(dynamic, (1,16)), scale2=kDynamicTokenScale)`. `[SRC-CODE]`
  `vllm/model_executor/layers/quantization/utils/quant_utils.py:192-218`.
- A load-time warning exists and names the failure mode: "In NVFP4 linear, the input global scale
  is different for parallel layers (e.g. q_proj, k_proj, v_proj). **This will likely result in
  reduced accuracy.** Please verify the model accuracy. Consider using a checkpoint with a shared
  global NVFP4 scale for fused layers." `[SRC-CODE]` `compressed_tensors_w4a4_nvfp4.py:117-124`.
  The same warning exists for `weight_global_scale` at 100-108.

### 3.3 SGLang — same split

- FP8: `ACTIVATION_SCHEMES = ["static", "dynamic"]`, default `"dynamic"` `[SRC-CODE]`
  `python/sglang/srt/layers/quantization/fp8.py:181,259,277-279`. Docstring: "The activation
  quantization scheme can be static or dynamic. **The dynamic activation quantization is more
  commonly used.**" `[SRC-CODE]` same file, 493.
- NVFP4: `input_global_scale = PerTensorScaleParameter(...)` from the checkpoint `[SRC-CODE]`
  `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:88-92`, and
  `fp4_quantize(x, layer.input_global_scale)` at the forward `[SRC-CODE]` same file, 145.

One discrepancy worth recording, not resolving: vLLM inverts the loaded value
(`layer.input_global_scale = 1/divisor`) and passes `input_global_scale_inv` (the divisor) to the
quantizer `[SRC-CODE]` vLLM `compressed_tensors_w4a4_nvfp4.py:110-134`; SGLang keeps the loaded value
as-is and passes it to `fp4_quantize` `[SRC-CODE]` SGLang same file, 95-96 and 145. One of the two
must be compensating in the weight loader. I did not trace `PerTensorScaleParameter` far enough to
say which, and this is not load-bearing for anything else in this note.

### 3.4 What that means for this project

`[INFERRED]` Three facts, in order of how much they should change a plan:

1. **The corpus question is real for the NVFP4 path as published.** All three engines read a
   per-tensor activation global scale out of the checkpoint and use it. A checkpoint whose
   activation divisor came from a different corpus is a checkpoint every one of them will consume
   as-is. The static/dynamic split is not a vLLM quirk.
2. **The same three engines all contain a working dynamic per-tensor implementation** for exactly
   this quantity, and TensorRT-LLM's dynamic rule is `448*6/amax(current input)` — the same
   statistic a max calibration measures, recomputed every forward. So the "which corpus" question
   and the "static vs dynamic" question are alternatives, not additions.
3. **For FP8 the corpus question is already moot by default.** None of the three bakes a per-tensor
   FP8 activation scale in unless the checkpoint's `activation_scheme` says `static`. This port's
   `AllowA4` sites are the ones where it bites.

### 3.5 NVIDIA has shipped both, and published the dynamic one

`[SRC-DOC/VENDOR-CLAIM]` NVIDIA technical blog, *Creating the NVIDIA Nemotron 3 Ultra NVFP4
Checkpoint with NVIDIA Model Optimizer*, 2026-06-26, on Nemotron 3 **Super**:

> "For our previous model, NVIDIA Nemotron 3 Super, the final quantization recipe combined MSE-based
> block scaling for weights with a per-tensor FP8 sweep and **dynamic max-based scaling for
> activations**."

and the same post's Table 2 describes its own baseline row as: "**Static** per-tensor scales are
computed using max-value calibration; per-block scales are computed dynamically from block maximum
values." (MMLU-Pro 82.99 / GPQA 79.29 / LiveCodeBench 70.18 / AA-LCR 55.50, vs BF16 83.49 / 79.92 /
72.907 / 53.00.)

So NVIDIA's published record contains **both** a static max-calibrated per-tensor activation scale
(the Qwen3.8-27B recipe, §1.3a) and a dynamic max-based per-tensor activation scale (Nemotron 3
Super's chosen recipe). Note what the dynamic rule still is: **a max**, recomputed. Nothing in any
published engine uses a percentile or headroom-anchored global activation scale at runtime.

---

## 4. Published accuracy comparison: max vs headroom/percentile for NVFP4 activations

### 4.1 What exists

**One end-to-end figure, and it is not an accuracy metric.** `[SRC-DOC]`
`modelopt_recipes/ptq.md:206-217`:

> "Reach for it when a W4A4 recipe regresses and the **activations**, not the weights, are to
> blame … On a GLM-5.3-Flash experts-only W4A4 study (SciCode, temperature 1.0) it cut the median
> **generation-length regression** from +38% to +19% and the mean from +19% to +4% with no capped
> generations: the best strict-W4A4 result there, **but still short of the p50/p75 gate**. A strong
> first lever, not a guaranteed fix — and sweep `rho`, since headroom above the calibrated range
> costs resolution inside it."

Scope of that claim, stated plainly: one model (GLM-5.3-Flash), one scope (experts-only W4A4), one
temperature, one benchmark, metric is generation *length* not task accuracy, and it did **not**
reach the gate it was measured against. The accompanying diagnostic guidance is qualitative
("the symptom is behavioral (verbose or runaway generations, hitting the generation cap) rather
than a flat score drop").

**Two tensor-level figures, from the PR that added the calibrator.** `[SRC-ISSUE]`
NVIDIA/Model-Optimizer#2028 (PR body):

> "Flooring at the literal max means one freak block drags the global scale up until every other
> block's FP8 block scale falls below subnormal: on a tensor with a single block seven orders of
> magnitude above the rest, that flushes **99.998% of elements to zero**, versus **6.7%** when the
> rare blocks are clipped instead. On a benign wide-range tensor the two choices differ by **0.4%
> relative MSE**."

The first is a constructed adversarial tensor (one block 10⁷ above the rest); the second is described
only as "a benign wide-range tensor", and the post does not say whether that was a real model
activation. The `CHANGELOG.rst:126` entry restates the algorithm and records **no** accuracy number.

**Adjacent, weight-side, for contrast.** `[SRC-DOC/VENDOR-CLAIM]` The Nemotron blog's Table 2 *is*
a published end-to-end accuracy comparison of NVFP4 calibration strategies (max / per-block MSE /
output-MSE / GPTQ, on MMLU-Pro, GPQA, LiveCodeBench, AA-LCR). Every alternative row is a **weight**
change; the activation side is held fixed. Four-over-six is likewise weight-only — "Four-over-six
works on weights and falls back to the default NVFP4 on activations" — and its reported result is
"98.5% median recovery relative to BF16, ahead of max (96.8%) and MSE (98.4%)", with the caveat
"MSE calibration reduced per-tensor weight error by 27.1% over four-over-six scaling … yet produced
no consistent improvement on downstream benchmarks."

### 4.2 What does not exist

`[NOT-FOUND]` **No published perplexity or benchmark-accuracy measurement comparing max-calibrated
against headroom- or percentile-calibrated NVFP4 activation global scales.** Specifically, I did not
find one in:

- the Model-Optimizer `CHANGELOG.rst` entry for `nvfp4_act_headroom` (algorithm description only);
- `modelopt_recipes/ptq.md` (one generation-length figure, §4.1);
- the PR that introduced it, `NVIDIA/Model-Optimizer#2028` (tensor-level figures only);
- the docs PR `NVIDIA/Model-Optimizer#2439`;
- the unit tests `tests/unit/torch/quantization/test_nvfp4_act_headroom.py` — 30+ tests, all on
  scale arithmetic and quantizer plumbing (`test_amax_is_rho_times_anchor`,
  `test_default_clips_rare_outliers_to_protect_the_bulk`, `test_headroom_exceeds_plain_max`,
  `test_rho_out_of_range_rejected`, `test_sequential_quantizer_activation_is_rejected`, …); **no
  test evaluates a loss or a metric**;
- the Nemotron 3 Ultra NVFP4 blog post's accuracy tables (Table 1, Table 2, Table 3 — no
  activation-global-scale row);
- a general web search for an NVFP4 activation-scale comparison returned only weight-side work
  (ReSET, ARCQuant, four-over-six, local-Hessian) and a third-party NVFP4-vs-MXFP4 format comparison.

`[NOT-FOUND]` I did **not** read the Nemotron 3 Ultra or Nemotron 3 Super technical reports
(arXiv 2606.15007 and 2604.12374) in full. If an activation-scale comparison exists there, this note
has not found it; I am not claiming it does not exist.

---

## 5. The subnormal-flush mechanism, its threshold, and detection

### 5.1 The mechanism, in NVIDIA's own words

`[SRC-CODE]` `calib/nvfp4_act_headroom.py`, class docstring:

> "Plain max calibration sets the global scale from the largest block seen during calibration, which
> leaves no room above it: any activation larger than the calibration max saturates. …
> `upper_percentile` defaults to 99.99 rather than the literal maximum on purpose. **A single freak
> block far above the rest would otherwise drag the global scale up so far that every other block's
> FP8 block scale falls below subnormal and flushes to zero — losing the whole tensor to protect one
> value.**"

and the two constants that make it quantitative `[SRC-CODE]` lines 29-36:

```python
# FP8-E4M3 normal dynamic range (max_normal / min_normal = 448 / 2**-6). Per-block scales
# outside this ratio cannot all be represented as normal FP8 values by one global scale.
_FP8_NORMAL_DYNAMIC_RANGE = 28672.0
```

### 5.2 The thresholds, from the identity in §1.2

`[INFERRED]` — this is arithmetic on NVIDIA's own constants (`E4M3 min normal = 2**-6`,
`E4M3 min subnormal = 2**-9`, `_FP8_NORMAL_DYNAMIC_RANGE = 28672.0`, and
`block_scale = 448 * block_amax / global_amax`), not a measurement I performed.

With `s_b = 448 * block_amax / global_amax` the stored E4M3 block scale:

| Condition on `block_amax / global_amax` | Condition on `s_b` | Consequence |
|---|---|---|
| `> 1/28672` | `> 2**-6` | block scale is an E4M3 **normal**; full 3 mantissa bits |
| `< 1/28672` | `< 2**-6` | block scale is E4M3 **subnormal**; fewer mantissa bits |
| `< 1/229376` | `< 2**-9` | block scale hits the clamp floor / underflows |

`229376 = 448 * 512 = 2**9 * 448`. So: **one global scale can carry at most a 28672× spread in
per-block amax before some block's E4M3 scale leaves the normal range**, and a 229376× spread before
ModelOpt's own clamp turns a block into the `2**-9` floor. NVIDIA's own `rho` is validated against
the first number: `rho` must satisfy `0 < rho < 28672`, and the calibrator warns when the per-block
range `upper/anchor` exceeds it, adding "The range also exceeds the FP8 scale range (28672), so no
permitted rho can clear it and **no single global scale holds both ends. Reduce the range with
outlier mitigation (SmoothQuant / per-channel / higher precision).**" `[SRC-CODE]`
`nvfp4_act_headroom.py:91-97,206-224`.

The clamp itself `[SRC-CODE]` `nvfp4_tensor.py:32-46`, docstring: "Clamp to FP8 E4M3FN range
`[2**-9, 448]` and cast — **avoids underflow→0** / overflow→NaN." And the changelog records it as a
fix: "Block scales below `2**-9` are now clamped to that minimum, and non-finite or negative scales
raise an error." `[SRC-CODE]` `CHANGELOG.rst:178`.

So there are **two distinct failure modes**, and they should not be conflated: an *exponent* problem
(a scale leaves the E4M3 normal range and loses mantissa bits) and an *underflow* problem (a scale
reaches the `2**-9` floor, or flushes to 0 in a kernel without the clamp). Only the second is what
"flushes to zero" means.

### 5.3 Can it be detected after the fact? Yes for weights; for activations, not from bytes alone

`[INFERRED]` The test is: read the stored per-block E4M3 scales, count how many are below `2**-6`
(subnormal) and how many are at the `2**-9` floor or zero. **The per-tensor global scale is not
needed for the test** — the block scale is the quantity whose range matters, and the threshold is a
property of E4M3 alone.

That is directly implementable against an artifact that stores E4M3 block scales. This port's
artifact stores FP8 block scales for weights. It does **not** store activation block scales, because
all three engines compute those at runtime from the tensor in flight (§3). So:

- **Weight path:** the diagnostic is a byte-level read of the artifact. Count subnormal/floored
  E4M3 block scales per packed parent.
- **Activation path:** the stored bytes contain only the per-tensor divisor. Whether a flush occurs
  is a property of the *runtime activation distribution*, so detection requires running a tensor
  through the quantizer and inspecting the block scales it produces — not reading the artifact.
  `[INFERRED]`

The headroom calibrator's own in-calibration warning (§5.2) is a fourth diagnostic, available only
during calibration, not after the fact.

### 5.4 A correction to the in-tree note's corroborating datum

`[IN-TREE]` `docs/research/quantization-coverage-evidence.md:439-442` offers as a
"data point … that corroborates the direction of the concern":
`mlp.gate_proj` records `amax=[0.0047, 0.4219]`, "a 90× spread between the smallest block and the
tensor max".

`[INFERRED]` On that reading (min per-block amax `0.0047`, per-tensor max `0.4219`), the smallest
E4M3 block scale under the identity of §1.2 is `448 * 0.0047 / 0.4219 = 4.99`, and the largest is
`448`. Both are E4M3 **normal** values. Reaching subnormal needs a spread of 28672×, not 90×. So
that datum does not show a flush, and as stated it does not corroborate the subnormal concern.

Two further limits on that datum, independent of the arithmetic: (a) a weight quantizer's per-block
`amax` array is exactly what `nvfp4_act_headroom` does **not** touch (it is restricted to NVFP4
dynamic-block **input** quantizers — `_is_nvfp4_dynamic_input_quantizer` requires the module name to
end in `input_quantizer`) `[SRC-CODE]` `model_calib.py`,
`nvfp4_act_headroom.py`; and (b) the port's own weight path uses `type: static` per-block scales,
so those scales are fixed at build time, whereas the activation block scales are recomputed per
forward. The flush mechanism is an activation-side, dynamic phenomenon; a static weight scale dump
cannot exhibit it at inference.

---

## What I searched and did not find

**Searched, found nothing (stated as negatives, not padded):**

1. **No published perplexity / benchmark-accuracy comparison of max vs headroom/percentile NVFP4
   activation global scales.** Searched: Model-Optimizer `CHANGELOG.rst`; `modelopt_recipes/ptq.md`;
   PR #2028; docs PR #2439; `tests/unit/torch/quantization/test_nvfp4_act_headroom.py`; the
   Nemotron 3 Ultra NVFP4 blog post; GitHub code and issue search across
   `NVIDIA/Model-Optimizer`, `vllm-project/vllm`, `sgl-project/sglang`, `NVIDIA/TensorRT-LLM`,
   `vllm-project/compressed-tensors`; and a general web search. The one end-to-end number is a
   generation-length regression on SciCode, one model, one scope, which did not reach its own gate.
2. **No published engine that uses a headroom- or percentile-anchored global activation scale at
   runtime.** TRT-LLM's dynamic rule is `448*6/amax(input)` — still a max. vLLM and SGLang use
   static checkpoint scales for NVFP4. `nvfp4_act_headroom` exists only as an offline ModelOpt
   calibration algorithm.
3. **No shipped ModelOpt preset that uses `nvfp4_act_headroom` for any LLM.** I enumerated
   `modelopt_recipes/configs/ptq/presets/model/` on `main` (39 `*.yaml` files, listed in full via the
   git tree; no name contains `headroom`) and read
   `modelopt_recipes/ptq.md:39`: the headroom recipe ships only as
   `nvfp4_act_headroom-kv_fp8_cast` in `general/ptq`, a KV-cache-cast recipe, not a model recipe. The
   two shipped Qwen3.8-27B recipes are `nvfp4_w4a4_mlp_fp8_attn_max` and its local-Hessian
   variant, both `algorithm: max` on activations.
4. **No FP8 analogue of the headroom calibrator.** `NVFP4ActHeadroomCalibrator` is selected only for
   quantizers with `num_bits == (2,1)` and `scale_bits == (4,3)`; the selection predicate
   `_is_nvfp4_dynamic_input_quantizer` rejects everything else, and `ptq.md:203-204` states
   "Affects NVFP4 **input** quantizers only — a no-op for FP8 and weight-only recipes."
5. **No FP8 subnormal-flush analogue, because FP8 per-tensor has no per-block scale level.** With a
   single per-tensor E4M3 scale there is nothing to fall out of range relative to. The mechanism is
   specific to the two-level NVFP4 structure.
6. **No public statement of the sequence length used to calibrate the shipped
   `nvidia/Qwen3.8-27B-NVFP4` checkpoint.** The current published default is 512 samples × 512
   tokens; the in-tree note's "2,048 tokens" is inherited and unverified by me.
7. **`llm-compressor` not read.** Not on the critical path for a ModelOpt-produced checkpoint; left
   as an open read rather than a claimed negative.
8. **Nemotron 3 Ultra / Super technical reports not read in full** (arXiv 2606.15007, 2604.12374).
   My negative in item 1 covers the sources I actually read.

**Searched, found, and worth carrying forward:** the two-level NVFP4 activation scheme is
consistent across four independent codebases (ModelOpt, vLLM, SGLang, TensorRT-LLM) plus
compressed-tensors' converter, and NVIDIA's own engineer stated it on a first-party tracker. The
`28672` normal-range ratio is in NVIDIA's source as a named constant and is reused as the `rho`
validation bound. The per-block scale identity `448 * block_amax / global_amax` makes every question
in this note decidable from stored bytes — for weights.
