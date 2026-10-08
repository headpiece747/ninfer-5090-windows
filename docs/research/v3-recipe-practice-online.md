# Recipe practice as published online

**Research note. Written 2026-10-07. Sources read that day: upstream `Neroued/ninfer` at
`81c8ce09` (fetched into this repo as `upstream/dev`; live `master` was at the same commit), the
upstream issue tracker via `gh`, public fork READMEs and model cards, and the Hugging Face Hub.
Nothing here was measured by this session.**

Labels used per claim: **[read in source]** — read in code, docs, config or model card;
**[measured]** — a figure the named party reports having measured; **[third party]** — a
community source, not upstream and not this port.

## 1. Upstream's own practice

### 1.1 The official artifacts

Five, per upstream `README.md` [read in source]:

| Model | Weights profile | Artifact | Download |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | `neroued/Qwen3.6-27B-NInfer` |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | `neroued/Qwen3.6-27B-nvfp4-NInfer` |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | `neroued/Qwen3.8-27B-NInfer` |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | `neroued/Qwen3.8-27B-nvfp4-NInfer` |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | `neroued/Qwen3.6-35B-A3B-NInfer` |

### 1.2 The five official recipes

`tools/convert/official_recipes.py` [read in source]. Format names: `q4_g64_fp16`,
`q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16`, `fp8_e4m3fn_row_bf16`, plus imported `nvfp4`.

| Recipe | Vocabulary | Text-layer allocation |
|---|---|---|
| `qwen3_6_27b` | Q6 | attention Q/K, GDN Q/K, MLP gate/up → Q4; everything else Q5; GDN a/b kept separate |
| `qwen3_8_27b` | Q8 | same allocation as above |
| `qwen3_6_35b_a3b` | Q8 embed / Q6 head | expert `down` → Q6 on layers 34/38/39 else Q5; other expert matrices Q4; shared/projection weights Q8; router/shared_score/a/b excluded |
| `qwen3_6_27b_nvfp4` | Q8 | imported NVFP4, except attention (not output) in layers 0–23, attention output on layers 3 and 7, and GDN output on layer 4, which keep the source's format |
| `qwen3_8_27b_nvfp4` | FP8 | imported NVFP4 for MLP on layers 0–55; row-scaled FP8 for embedding, attention, GDN and layers 56–63 MLP; embedding encoded locally from BF16 |

For 3.8 NVFP4 the model card adds [read in source]: 112 NVFP4 tensors, 146 row-scaled FP8,
1,246 stored objects (1,240 tensors + 6 resources), 22.09 GiB; "source-derived NVFP4 and FP8
words are preserved without decode and requantization; only the official BF16 token embedding is
encoded locally". Recipe `qwen3_8_27b_nvfp4`; sources `Qwen/Qwen3.8-27B` rev `1d4bf0f2` and
`unsloth/Qwen3.8-27B-NVFP4` rev `60e813d4`; embedding encoder `fp8_row_maxabs`. The 3.6 NVFP4
artifact imports from `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`.

### 1.3 Formats, block sizes, calibration

From `docs/maintainer/tensor-formats.md` and the v3 container spec [read in source]:

| Format | Codes / scale |
|---|---|
| `q4_g64_fp16` / `q5_g64_fp16` / `q6_g64_fp16` | signed codes, group 64, one FP16 scale per group |
| `q8_g32_fp16` | codes `[-127,127]`, group 32, one FP16 scale per group |
| `nvfp4` | E2M1 codes, block 16, one E4M3FN scale per block, one FP32 weight divisor per tensor |
| `fp8_e4m3fn_row_bf16` | E4M3FN codes, one BF16 multiplier per row |

Activation permissions are independent of the stored weight format: `A16Only`, `AllowA8`,
`AllowA4`. An NVFP4 A4 input needs a positive finite activation divisor; `import_encoded` takes it
from the source, or the recipe supplies one [read in source: `docs/weight-conversion.md`].

**Calibration: upstream has none.** The converter's four built-in methods are `cast_direct`,
`grouped_absmax`, `fp8_row_maxabs` and `import_encoded` [read in source: `methods.py`]; there is no
floating-point-to-NVFP4 quantizer ("NInfer currently provides no built-in floating-point-to-NVFP4
quantizer"), and `tensor-formats.md` states "calibration is not part of this weight format". The
NVFP4 artifacts therefore import the source checkpoint's codes, scales and divisors; the parts
encoded locally are groupwise-integer or row-FP8, which need no calibration. The port's
`calibration.py`, `calibration_corpus.json` and `quantization/nvfp4.py` have no upstream
counterpart [read in source].

### 1.4 Published results

Throughput, RTX 5090, upstream README excerpts [measured, upstream]:

- Qwen3.8 MTP3 decode saturation C=1→8: `groupwise-int` 136.5 / 253.3 / 398.1 / 582.4 tok/s
  (acceptance 44.4–46.4 %); `nvfp4` 147.7 / 291.0 / 522.2 / 922.4 (45.8–48.7 %).
- Qwen3.8 no-spec prefill, 7,680 tokens: 3,331.9 vs 12,819.1 tok/s; at 260,096: 2,139.4 vs
  4,016.4; structured MTP3 decode 214.7 vs 231.7 tok/s.
- Resident weight arenas, no speculation: 15.920 vs 18.976 GiB.

Quality, upstream evaluation table (EvalScope 1.9.0, 0-shot, one sample) [measured, upstream]:
3.8 `groupwise-int` AIME25 96.67 / AIME26 96.67 / GPQA-D 87.37 / ERQA 66.25 / RealWorldQA 82.22;
3.8 `nvfp4` 96.67 / 96.67 / 90.40 / 66.25 / 83.53. The nvfp4 model card compares against the
official Qwen BF16 card and labels the comparison not same-protocol: IFBench 77.00 vs 79.5,
GPQA-D 90.40 vs 89.2, ERQA 66.25 vs 65.5, RealWorldQA 83.53 vs 85.9, "deltas stay within ±2.5
points on the four overlapping benchmarks" [measured, upstream].

Perplexity: upstream defines the fixed corpus `ninfer-ppl-1m-v1` (16 streams, 4 domains;
4096/2048 default; 65536/32768 recommended for KV comparisons) and the metric, but publishes no
per-recipe perplexity table in `docs/` or the model cards [read in source; absence].

## 2. The upstream tracker

| # | State | Claim | Measurement attached |
|---|---|---|---|
| 23 | closed | Plan: NVFP4 support from unsloth's "nvfp4 + fp8 mixed quant", ~20 GB ≈ 6 bpw, 262,144 context | None (plan) [read in source] |
| 25 | closed | Register `qwen3.8-27b/nvfp4`; a comment reports the community Ostfralla artifact "noticeably faster and supports the full 262,144-token context, unlike the 21 GB upstream FP8 artifact"; PR #107 detects the wire format from `text/token_embedding` | A/B numbers in #38 [third party] |
| 38 | closed | Official NVFP4 artifact slower than an older community one, with more invalid structured responses | TTFT 50.5 vs 276.4 ms; C2 prefill 2,482 vs 2,062 tok/s, decode 290.7 vs 234.0; C8 5,243 vs 3,930 and 602.1 vs 441.4; valid structured 226/250 vs 203/250 and 222/250 vs 211/250. The two artifacts ran on different runtimes; "valid" is schema only [third party] |
| 70 | closed | Fuller-NVFP4 profile; QUASAR QAT | Apoze's independent rebuild: 247 NVFP4 parents, 9 BF16 exceptions, calibration 14,986-token + 8,490 held-out; with 2× activation headroom 0/247 sites saturated (max 0.63757) vs 60/247 without (max 1.8147); GPQA-D 175/198 = 88.38 %. cometkim: QUASAR GPQA-D 89.22 ± 2.49, AIME26 91.11 ± 3.85; QAT drops DFlash2 acceptance; `nvfp4qat` 15.31 vs 16.02 GiB device weights [measured, third party] |
| 214 | closed (duplicate) | Switch the 3.8 NVFP4 source to `nvidia/Qwen3.8-27B-NVFP4` ("likely better calibrated") | ndizazzo interleaved: pp2048 +0.04 %, pp512 +0.77 %, decode −0.07 %; tg128 −29.9 % attributed to the DFlash2 draft head. cometkim weight-space rel-L2: NVIDIA beats unsloth on 168/168 comparable tensors, the local encoder on 190/192, the `nvfp4full` payload on 191/192; QUASAR QAT ahead on 117/192 but "cannot be ranked by this metric and must never be mixed with PTQ codes"; FP4 `lm_head` 15× weight-space error for −0.59 GiB; end-to-end MTP3 −8.9/−9.7 %, DFlash2 −4.1/−4.2 %; "PPL does indeed improve, the MTP/DFlash2 acceptance rate drops significantly" [measured, third party] |
| 285 | closed | `silu_approx` in the NVFP4 SwiGLU epilogue costs quality | +0.61 % relative corpus perplexity (1.540180 vs 1.530857), prefill +1.4 % wall clock, decode unchanged; reproduced twice [third party] |
| 249 | closed | Approximate silu op-level analysis | median −10.8 % on the fused TMA route; 17.9 % of kernel time and +2.9 % prefill; in-band rounding disagreement 1 in 31,128 for `silu_approx` vs 1 in 41,237 for `silu` [third party] |
| 174 | open | Full-vocabulary Q4G64 MTP proposal head | FP8 head 752 µs / 1.272 GB vs full-vocab Q4 402 µs / 0.675 GB; structured JSON 277.6 vs 302.5 tok/s, both at 99.0 % acceptance; costs 0.63 GiB [third party] |
| 164 / 375 | open | KV precision tail (`--kv-tail-tokens`), fork implementation | #375: text and vision 64/64 byte-identical, memory bit-identical 64/64, decode median −0.18 %, KLD improves 24/24 cells, cost 64 MiB/sequence at N=1024, ~6 % decode; inert on `fp8`/`nvfp4`/`k8v4` [measured, third party] |
| 370 | open | Parallel CPU workers for conversion | workers=1 1,012.175 s vs workers=2 708.447 s (1.429×, −30.0 %); one full-model run per count, peak memory not measured; serial/parallel object bytes compared [measured, third party] |
| 298 | open | WSL2 report + finetune conversion notes | `tokenizer_config.json` must carry `add_bos_token`; the DFlash2 draft does not transfer to finetunes (3.3–5.1 % acceptance vs MTP 67–85 %); the author's finetune artifact perplexity 4.448 vs 4.617 for official stock [third party] |

By topic, from the searches run for this note (nvfp4, quantization, fp8, recipe, precision, block
scale, QAT, calibration, unsloth, perplexity) plus the issues read in full [all third party unless
noted]:

- **NVFP4/FP8 quality**: quality is discussed for NVFP4 only, and mostly as acceptance or
  capability scores (#38, #70, #214). No issue measures FP8 weight quality; upstream's FP8 comes
  from the source import [read in source].
- **High-precision layers**: #70/#214 — 9 BF16 exception parents transplanted from the Qwen3.6-27B
  NVFP4 pattern; QUASAR removes every exception; MirkoCovizzi and Ostfralla keep GDN
  `in_proj_a`/`in_proj_b` BF16 because N=48 (CUTLASS FP4 needs N % 64 == 0) [third party].
- **Activation scaling**: no quality ticket. The only measured activation-headroom claim is
  Apoze's 2× margin in #70; the activation-scale issues found (#247, #252, #263) are kernel-route
  tickets [third party].
- **Block scale**: no issue proposes a different block-scale constant. Upstream imports the
  source's scales; the 4-vs-6 experiment exists only in this port
  (`docs/research/nvfp4-block-scale-4-vs-6.md`) [read in source; absence].
- **KV precision**: #164 asks for the tail feature, #375 supplies the fork's measurements above
  [third party].
- **Conversion speed**: #370 above [third party].
- **QAT**: no upstream QAT. QAT enters through the community QUASAR checkpoint and is imported
  (#70, #214); the port's `qwen3_8_27b_nvfp4_qat` has the same shape [third party].
- **A better recipe**: #214 was closed as a duplicate of #245, the custom-recipes announcement,
  which moves recipe choice to the user rather than changing the official one [read in source].

## 3. Forks and community artifacts

Hugging Face's model search for "ninfer" returns 100+ entries (the API page limit), most of them
re-quantizations for other GPUs or finetunes. The sources below publish a recipe or a measured
claim.

| Fork / publisher | What it publishes | Recipe and measured claim |
|---|---|---|
| `cometkim/ninfer` (13★) | `nvfp4full` and `nvfp4qat` model cards, branches `feat/qwen3.8-nvfp4full`, `feat/qwen3.8-nvfp4qat` | `nvfp4full`: 247 NVFP4 parents, encoder `NVFP4_MAXABS_DIVISOR_RNE_V1` (per-tensor FP32 divisor, per-16 E4M3FN block scales, RNE codes), 135-site calibration input, 9 BF16 exceptions transplanted from the Qwen3.6 pattern, W8 endpoints; GPQA-D 87.88 ± 2.62, AIME26 93.33 ± 3.34, LBv2 67.41 ± 1.95. `nvfp4qat`: QUASAR import, 256 NVFP4 parents, 0 exceptions; GPQA-D 89.22 ± 2.49, AIME26 91.11 ± 3.85, LBv2 66.30 ± 0.85. DFlash2 drafter as NVFP4: 5.50 tok/round at 64.3 % vs 5.75 at 67.9 % W8; the QAT profile's drafter 2.50 at 21.4 %. Adds `hq-e8-2b` KV and YaRN. [read in source, measured, third party] |
| `MirkoCovizzi/ninfer-rtx5090-mobile` (19★) | HF QUASAR NVFP4 v3 artifact (19.8 GB) | 256 NVFP4 parents (32 attention / 96 GDN / 128 MLP) from 400 of 496 QUASAR matrices; GDN `in_proj_a`/`in_proj_b` decoded to BF16; stored formats bf16 579 / fp32 352 / int32 1 / nvfp4 256 / q4 55 / q5 54 / q6 1 / q8 30; the release makes "no new throughput or quality claim"; validation is greedy parity + 121 CTest checks [read in source, third party] |
| `ValerioDolci/ninfer-tp2` (21★) | Two-GPU TP fork; QUASAR conversion | Every large projection NVFP4, GDN a/b BF16, head/embedding FP8; `_dflash2_nvfp4_gate_up` (`nvfp4_mse`, A16) decodes +1.5–2.4 % at C=1; GSM8K 0.975–0.985; TP numbers against the published 5090 runs [third party] |
| `Don-Chad/ninfer-3090` (436★) | sm_86 port | Loads the official `groupwise-int` artifact; no NVFP4/FP8 execution; C1–C8 end-to-end 70.2→161.3 tok/s (decode 71.0→165.3), prefill ~844–862 tok/s; builds no artifacts [third party] |
| `Wallawalla47/ninfer-custom` (15★), `5258MF/ninfer-rtx3060-27b`, `sybrix/qwen3.8-flash-next-ninfer` | Modified `official_recipes.py` in public forks | Between them: `qwen3_8_27b_nvfp4_nvidia`, `qwen3_8_27b_nvfp4_orcarouter` (GPTQ compressed-tensors), `qwen3_8_27b_q6`, `qwen3_6_35b_a3b_nvfp4`, `qwen3_8_flash_next_nvfp4`; `nvfp4_mse` method and `_dflash2_nvfp4_gate_up` [read in source] |
| `koldfrontier/ninfer-finetune-nvfp4` | Finetune recipe variant | FP8 parts generated with `fp8_row_maxabs` from BF16, NVFP4 MLP imported from a community compressed-tensors quant; artifact perplexity 4.448 vs 4.617 official stock [third party] |
| Ostfralla (HF) | 18.3 GB NVFP4 artifact, pre-v3 engine | llm-compressor NVFP4; GDN projections quantized, `in_proj_a`/`in_proj_b` BF16 (N=48, CUTLASS needs N % 64 == 0); fused-group global scale taken as the smallest member scale and re-quantized from BF16 (mean reconstruction error 0.09471 → 0.09470); HumanEval+ 152/164 both profiles, AIME25/26 55/60 both; 1.56–1.98× wall clock [third party] |
| `qwased/…-precision-tail` | GSQ-RCO IQ3_XXS artifact + KV precision tail | The #375 measurements above [third party] |

Upstream's own file defines only the five recipe names in §1.2; a GitHub code search for
`qwen3_8_27b_nvfp4` returns upstream's own files, this port, and the forks above, with no upstream
recipe name beyond those five [read in source].

## 4. The artifact format, publicly

- **Current authority**: `docs/maintainer/artifact-container.md`, "NInfer v3 容器规范" (Chinese),
  12 sections: file set and address space, binary framing, JSON directory, physical objects, v3
  format and layout names, logical bindings, Uses/permissions, resources, provenance,
  reader/writer, worked examples [read in source].
- **Framing** (from §3): entry header is 32 bytes — magic `NINFER\0\x03`, `json_bytes` (LE),
  16-byte `artifact_id`; JSON starts at 32; payload at `align_up(32 + json_bytes, 4096)`.
  Continuation files use magic `NINPRT\0\x03` with `part_index` and the same `artifact_id`,
  payload at 4096. The root JSON carries `components`, `objects`, `bindings`, `uses`; the whole
  directory can be moved. Reader and writer are meant to be implementable from the spec alone
  [read in source].
- **Formats and layouts** (§6): the format names in §1.3; layouts `contiguous_le_v1`,
  `row_split_k128_v1` (groupwise), `block_scale_k16_m128x4_v1` (NVFP4, requires N % 128 == 0 and
  K % 64 == 0), `row_scale_v1` (FP8), `raw_bytes_v1` (resources). NVFP4 reconstructs as
  `code_value * block_scale / weight_divisor` [read in source].
- **Numeric contract**: `docs/maintainer/tensor-formats.md` (English). **Byte packing**:
  `docs/maintainer/storage-layouts.md` (English) [read in source].
- **Builder entry point**: `docs/weight-conversion.md` — official recipes, format table,
  activation policies, methods, custom recipe and method examples, alternate sources, resources,
  `--max-file-bytes` sharding, the `.conversion.json` report, and `tools.artifact.inspect` [read in
  source]. Examples live in `docs/maintainer/examples/artifact-v3-{text,mixed-sharded}.json`.
- **Model cards** carry inventory and provenance per artifact: object counts, tensor format
  counts, source revisions, minimum runtime revision [read in source].
- **History**: per-model storage contracts existed in the v2 era
  (`docs/maintainer/qwen3.6-27b-artifact.md`) and were removed by upstream commit `9b884034`
  "docs: consolidate references for the v3 architecture" (2026-09-14); no per-model v3 reference
  is published upstream today, and this port's `docs/maintainer/qwen3.8-27b-artifact.md` is port
  work [read in source: git history].
- **No upstream blog post or separate specification** was found. The one third-party write-up
  located is an auto-generated DeepWiki page for `sergiuszm/ninfer-4090`, which describes the
  **v2** framing (`NINFER\0\2`, JSON directory, 4096-byte alignment) and is stale for v3
  [third party].

## 5. What is missing publicly

- No upstream-published perplexity figure per recipe, and no upstream calibration implementation,
  corpus or NVFP4 encoder.
- No upstream study of block-scale choice or activation-scale selection as quality levers; the
  strongest evidence there is community rel-L2 screens (#214) and Apoze's saturation numbers
  (#70), neither of which is an end-to-end comparison against the official artifact.
- QAT is entirely community-sourced (QUASAR), imported rather than trained by NInfer, and the one
  repeated finding across #70, #214 and the fork cards is that QAT or re-sourced NVFP4 improves
  perplexity/capability while reducing MTP/DFlash2 acceptance.
- Every community recipe number above carries its own protocol; none of them is comparable to
  upstream's evaluation table without re-running both arms.
