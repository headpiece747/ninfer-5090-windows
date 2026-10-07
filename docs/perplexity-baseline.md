# Perplexity baseline (Windows port)

The protocol, so a number is comparable: `ninfer-perplexity.exe <artifact> --corpus
eval/corpora/perplexity-1m/manifest.json --kv-dtype <dtype>`. That is upstream's fixed corpus
(`ninfer-ppl-1m-v1`: 16 streams, 1,044,573 input tokens of which 1,044,557 are scored, 496 windows,
4096/2048 context and stride). Every row carries the date it was taken; all of them are this machine,
which is a different box and clock from any published figure. Removing `--quick` gives the full corpus.

**Rows taken before 2026-10-04 are not comparable with later ones, for two separate reasons.** Upstream
rewrote the tokenizer in `b9114396`/`a8e212ac` (merged 2026-10-04), and the rewrite *fixed* an
over-segmentation: the old tokenizer produced 319 more tokens on this corpus (12 / 186 / 121 / 0 across
the four domains), while the current one matches the reference tokenizer exactly -- `tokenizers` on the
artifact's own `tokenizer.json` yields 262,022 / 261,408 / 261,223 / 259,904 scored tokens per domain,
which is what the engine scores today (verified 2026-10-07). Separately, the *logits* moved:
`ninfer_code` scores the **same 259,904 tokens in both engines** and still reads lower by 3,390 nats on
the retired image (133,296 -> 129,906) and 3,675 nats on `nvfp4full` (137,248 -> 133,574). That domain is
the control: a token stream that did not change cannot move under the tokenizer, so an engine-side
numerics change is present too, and it is **unattributed**. Both effects point the same way, and neither
is the artifacts: their weights are unchanged, and the five images rebuilt on 2026-10-07 are
byte-identical to the ones before them apart from the embedded chat template. The serving lanes moved in
the same window and are recorded separately in
[the lane regression record](research/lane-regression-2026-10-07.md); whether the two share a cause is open.

**What this measurement cannot see.** `ninfer-perplexity` has no `--spec` or draft option at all --
`apps/perplexity/main.cpp` contains no reference to either -- so it scores the target model only.
These figures are therefore evidence about the *text* model's arithmetic, and say nothing about a
drafter: not the DFlash2 projection kernels, and not the drafter's own MLP path. Two of this
project's routes are drafter-only and invisible here, which is why a green perplexity result is
not by itself evidence that drafter work is correct. For that, the drafter's routes are covered by
`ninfer_dflash2_nvfp4_routes_test` (oracle over the NVFP4 routes) and
`ninfer_qwen3_5_dflash2_real_test` (real artifact, end to end, at `K=15 B=8` so the drafter's SwiGLU
is driven at `T=128`).

**Reading the 2026-09-26 pair against the band.** The acceptance criterion recorded before the
upstream `e31bc99b` merge was ±1% around the 2026-09-24 figures. The full corpus lands at -0.098%,
comfortably inside. The `--quick` corpus lands at -0.888%, inside but 0.11% above the lower edge,
so it should be treated as marginal rather than as a clean pass. Note the quick corpus is a quarter
the size of the full one and its own recorded history already spans 1.05% (4.89741 on 2026-09-22
against 4.94879 on 2026-09-24), which is the same order as the band being tested; a band that tight
cannot separate a real regression from the measurement's own spread.

| artifact | KV | PPL | notes |
|---|---|---|---|
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | fp8 | **4.90295** | full corpus, 4.21k tok/s, recorded 2026-09-22 |
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | fp8 | **4.90169** | full corpus, 2026-09-24; re-measured |
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | bf16 | **4.89838** | full corpus, 4.20k tok/s, 2026-09-22 |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT, as published) | fp8 | **4.89741** | `--quick`, recorded 2026-09-22 |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT, as published) | fp8 | **4.94879** | `--quick`, 2026-09-24; re-measured, and it does not reproduce the figure above |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT, as published) | fp8 | **5.00234** | full corpus, 2026-09-24 |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT, as published) | fp8 | **4.90484** | `--quick`, 2026-09-26, build with upstream `e31bc99b` merged; 6.48k tok/s, 261,223 tokens. -0.888% on the 2026-09-24 figure: inside the ±1% band, but only 0.11% above its lower edge |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT, as published) | fp8 | **4.99744** | full corpus, 2026-09-26, same build; 6.53k tok/s, 1,044,876 tokens. **-0.098%** on the 2026-09-24 figure -- the text model's arithmetic is effectively unmoved by the merge |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR, rebuilt from source) | fp8 | **4.88817** | `--quick`, 2026-09-24 |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR, rebuilt from source) | fp8 | **4.99097** | full corpus, 2026-09-24 |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT) | fp8 | **5.88829** | custom corpus, 177,400 tokens |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (NVFP4-full) | fp8 | **5.92007** | custom corpus, same text |
| `qwen3_8_27b_nvfp4swift.v3.ninfer` (Swift, re-encoded) | fp8 | **4.68429** | `--quick`, 2026-09-24; the same recipe importing the checkpoint's FP8 scored **4.84938** |
| `qwen3_8_27b_nvfp4swift.v3.ninfer` (Swift, re-encoded) | fp8 | **4.92432** | full corpus, 6.03k tok/s |
| Swift, FP8 imported (superseded) | fp8 | **4.93874** | full corpus, 4.72k tok/s; the build the re-encode replaced |
| Swift, re-encoded with NVFP4-full's bf16 exceptions | fp8 | **4.7701** | `--quick`; 27 projections kept bf16 |
| Swift, re-encoded with NVFP4-full's bf16 exceptions | fp8 | **4.93254** | full corpus |
| Swift, re-encoded, endpoints as FP8 | fp8 | **4.85155** | `--quick`; isolates the endpoints, which are Q8 in the rows above |
| Swift, re-encoded, endpoints as FP8 | fp8 | **4.96342** | full corpus |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (NVFP4-full, as published) | fp8 | **4.82452** | `--quick`, 2026-09-24 |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (NVFP4-full, as published) | fp8 | **4.98768** | full corpus, 2026-09-24 |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (unsloth, rebuilt from source) | fp8 | **4.75750** | `--quick`, 2026-09-24 |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (unsloth, rebuilt from source) | fp8 | **4.97532** | full corpus, 2026-09-24 |
| `qwen3_8_27b_nvfp4nvidia.v3.ninfer` (NVIDIA, built from source) | fp8 | **4.71979** | `--quick`, 2026-09-24 |
| `qwen3_8_27b_nvfp4nvidia.v3.ninfer` (NVIDIA, built from source) | fp8 | **4.90168** | full corpus, 2026-09-24; the official stock measures 4.90169 on the same protocol |

## All four shipping artifacts, re-measured 2026-09-28 after `a012e2bc`

One row per shipped lane, full corpus, fp8 KV, 1,044,876 tokens, on the build carrying the masked-draft
attention reorganisation. `qwen3_8_27b_nvfp4nvidia.v3.ninfer` is the official stock: its 4.90168 sits
against the official stock's 4.90169 on the same protocol.

| artifact | source of its weights | 2026-09-24/26 | 2026-09-28 | change |
|---|---|---:|---:|---:|
| QUASAR QAT (`nvfp4qat`) | `Qwen3.8-27B-NVFP4-QUASAR` | 4.99744 | **4.997441** | identical |
| Swift (`nvfp4swift`) | `Swift-Qwen3.8-27B-NVFP4` | 4.92432 | **4.931761** | +0.151 % |
| NVFP4-full (`nvfp4full`) | `Qwen3.8-27B-NVFP4-unsloth` | 4.98768 | **5.002854** | +0.305 % |
| NVIDIA ModelOpt (`nvfp4nvidia`) | `Qwen3.8-27B-NVFP4-nvidia` | 4.90168 | **4.915181** | +0.275 % |

Two things this establishes, and one it does not.

**The attention reorganisation is output-neutral.** QUASAR's baseline was re-taken on 2026-09-26, one
merge later than the other three, and it is unchanged to every digit the table records. The other
three moved +0.15 % to +0.31 %, but their baselines predate the `e31bc99b` merge that QUASAR's does
not, and that merge is independently recorded at -0.098 % on QUASAR. So the drift on those three is
attributable to the older baseline, not to this merge, and the +0.3 % must not be cited as an
attention regression without a same-day control on the same artifact.

**The `nvfp4full` lane is built from a community quantization and is measurably worse than the
official one.** Its recipe is `qwen3_8_27b_nvfp4_unsloth` over `Qwen3.8-27B-NVFP4-unsloth`, while
`nvfp4nvidia` is `qwen3_8_27b_nvfp4_nvidia` over NVIDIA's. On the same protocol the official source
scores **4.915181 against 5.002854** on 2026-09-28, a 1.75 % gap, and **4.911188 against 4.998419**
on 2026-09-29, a 1.78 % gap. The name appears to describe a format property
rather than a source -- an earlier row reads "Swift, re-encoded with NVFP4-full's bf16 exceptions" --
which would make the community checkpoint a deliberate choice as the one permitting complete NVFP4
coverage. That reasoning is not written down anywhere and has not been tested. Whether a
full-coverage artifact can be built from NVIDIA's source is open, and if it can, it would be strictly
better than the lane we ship under that name.

**What this does not establish:** which source is *better for decode acceptance*. Perplexity scores
the target model only -- `ninfer-perplexity` has no `--spec` option -- so these rows say nothing about
whether the drafter accepts well against either target.

The custom corpus is this repo's `docs/` and `tests/` markdown, concatenated in sorted order
(177,400 tokens). On it the two shipped artifacts sit 0.5% apart with QUASAR marginally better, which
is the like-for-like check: same text, same protocol, two artifacts. The full-corpus rows are not
comparable with the `--quick` row, since the corpus subset differs. Give every row the date it was
taken: the QUASAR figure and the official stock's full-corpus figure both failed to reproduce when
re-measured, which is a claim about the revision, not about the arithmetic.

## All four shipping artifacts, re-measured 2026-09-29 after the `d44ab584` merge

Same protocol as the table above -- full corpus, fp8 KV, 1,044,876 tokens, 4096/2048 -- on build
`4b3acfc2`, which is the merge of `upstream/dev` `d44ab584` (14 commits). Re-measurement was not
optional: that merge changed the FP8 and NVFP4 linear routes and attention, so the 2026-09-28 column
was pinned to a revision that no longer existed. The binary on disk was itself dated 2026-09-27, two
days older than the merge, so measuring before rebuilding would have reported the pre-merge code under
a post-merge heading. One recipe built and measured, in that order.

| artifact | 2026-09-28 | 2026-09-29 | change |
|---|---:|---:|---:|
| NVIDIA ModelOpt (`nvfp4nvidia`) | 4.915181 | **4.911188** | -0.081 % |
| Swift (`nvfp4swift`) | 4.931761 | **4.936397** | +0.094 % |
| QUASAR QAT (`nvfp4qat`) | 4.997441 | **4.994346** | -0.062 % |
| NVFP4-full (`nvfp4full`) | 5.002854 | **4.998419** | -0.089 % |

**The merge is perplexity-neutral.** All four move within +/-0.1 %, which is inside this measurement's
own spread: the `--quick` corpus is recorded elsewhere in this file as spanning 1.05 % between two
runs of the same build. The sign differs per artifact, so this is noise around an unmoved centre, not
a shared drift. The ranking is unchanged. The one derived figure that moves is the official-source
gap quoted above, from 1.75 % to **1.78 %** (4.911188 against 4.998419).

**The per-domain spread is far wider than the overall gap, and that is the argument for a per-domain
instrument.** The evaluator already prints a per-domain breakdown, and it disagrees with the
aggregate's ordering:

| domain | nvidia | swift | qat | full | spread |
|---|---:|---:|---:|---:|---:|
| `chinese_reference` | 6.371428 | **6.229361** | 6.625568 | 6.509076 | 6.4 % |
| `english_long_form` | 8.300561 | 8.350429 | **8.243521** | 8.304946 | 1.3 % |
| `english_reference` | **6.459213** | 6.707160 | 6.718965 | 6.779242 | 5.0 % |
| `ninfer_code` | 1.691042 | 1.690048 | **1.683223** | 1.691167 | 0.5 % |
| `overall` | **4.911188** | 4.936397 | 4.994346 | 4.998419 | 1.8 % |

Three things follow, and the first is the uncomfortable one. **The 1.8 % overall gap understates the
real disagreement by 3.5x**: on `chinese_reference` the same four artifacts span 6.4 %, and Swift
beats QUASAR there by 5.9 % while sitting 1.2 % *behind* it overall. **No artifact is best
everywhere**: nvidia takes the overall and both English domains, swift takes Chinese by a clear
margin, qat takes code, and the code column is three orders of magnitude tighter in spread than
Chinese. **The aggregate is not a summary of the four domains** -- it is a token-weighted mean that
lets one domain's ordering decide the headline. Any claim of the form "lane X is better quality"
should name the domain, because on this evidence the answer changes with it.

That is also why the missing instrument is per-domain **KL against the BF16 reference** rather than
more overall perplexity: `qwen3_8_27b_nvfp4.v3.ninfer` at bf16 KV reads 4.89838, and a per-domain KL
to it would say which domain each artifact's quantization actually damaged, which perplexity's
ordering cannot.

**Not re-measured here, and therefore suspect.** The official stock `qwen3_8_27b_nvfp4.v3.ninfer`
(4.90169 full corpus) was not in this run, so that row and the comparison at the head of this section
still rest on 2026-09-24. `tools/release/profiles.py` carries decode `tok` and acceptance figures per
lane; those are decode throughput from `v3_profile_matrix.py`, a different instrument from the
*scoring* rate reported here (5,904-6,644 tok/s across these four runs), and they are **not** covered
by this re-measurement. The merge changed linear routes those figures depend on, so they need a bench
run before any of them is cited again. All four reports are persisted under
`profiles/perplexity/qwen3.8-27b/`, one directory per run.

## A searched NVFP4 block scale is worse, and the weight error says nothing about it

The local NVFP4 encoder derives each block's E4M3FN scale from the block's absolute maximum, so the
block's largest value maps onto 6.0, the format maximum. An alternative is to search the scale: pick,
per block, the shrink factor of the max-abs scale that minimises that block's own squared
reconstruction error.

The premise was that the NVIDIA source records `calibrator=NVFP4MSECalibrator` on all 193 of its NVFP4
weight quantizers, so a max-abs scale reproduces a max-calibrated producer rather than matching a
searched one. **That premise does not apply to the weights this experiment changed.** The 193 are
disjoint from the re-encoded set, as `.quant_summary.txt` in the source checkpoint records:

| calibrator | count | which weights |
|---|---:|---|
| `NVFP4MSECalibrator` | 193 | 192 `mlp.*` + `lm_head` |
| `MaxCalibrator` | 609 | 128 `self_attn.*` + 288 `linear_attn.*` + 193 input quantizers |

`hessian` and `local_hessian` appear **zero** times, so the producer used the plain squared-error path
rather than the Hessian-weighted one that NVIDIA's source says "wins over the plain path" — the
objective here was the right one to test. But the 193 searched sites are exactly the MLP and
`lm_head`, which `qwen3_8_27b_nvfp4_nvidia` already imports verbatim with the producer's own codes and
scales. The 128 object groups re-encoded from BF16 here are the attention and linear-attention
projections, which the producer quantized as **FP8** with plain max calibration. There was never a
searched-scale artifact to match for them, and the searched scales the source does contain are already
in the shipped lane.

So this measures the right question -- what to do about weights the producer never encoded as NVFP4 --
and the answer is max-abs. It does not measure, and cannot, whether faithfully reproducing
`NVFP4MSECalibrator` helps: that part of the recipe was never a re-encoding.

One difference from the producer's search remains and is not closed by this result. ModelOpt sweeps
**126** valid FP8-E4M3 candidates anchored to the tensor's global amax (`block_amax = global_amax *
candidate`, `candidate = fp8_e4m3/448`), a log-spaced grid over the whole representable range. This
experiment swept 11 ratios linear in [0.5, 1.0] of each block's own amax. Same objective, different
and much narrower grid.

Built as `nvfp4_mse` (commit `900a0f78`, reverted in `b891e1e9`) with 11 shrink factors from 1.0
down to 0.5, candidates cast to E4M3FN before being scored so only representable scales were
compared, and the max-abs scale in the candidate set so the search could only match or beat it. Wired
into the `qwen3_8_27b_nvfp4_nvidia` recipe for the 128 object groups it re-encodes locally; the 31
drafter groups were left on max-abs so one variable moved. The method tallies in the two conversion
reports differ in exactly those 128 entries and nowhere else.

Paired in one window, full corpus, fp8 KV, both scored on the same revision:

| build | weight reconstruction error | perplexity |
|---|---:|---:|
| max-abs, as shipped | baseline | **4.915181334** |
| searched scale | −40 % to −66 % | **4.925917194** |

**+0.218 %**, and the two builds differ in nothing but weight values. The reports agree on
`prefill_signature` `ed709b7d…`, on `formats` (`bf16 fp32 nvfp4 q8_g32_fp16`), on every field of
`execution` (4096 context, `fp8-e4m3-r256`, stride 2048, tile 1024), on the corpus, and on all
1,044,876 scored tokens. So the searched build changed no format and no plan -- it re-derived the
E4M3FN scale of 128 object groups and nothing else. The max-abs run reproduces the 4.915181 recorded
above to every digit, so the harness is sound and the difference is attributable to the scale.

**The search loses, and it loses while doing exactly what it was asked to do.** Reconstruction error
on 27B-shaped projections fell by 40 % to 66 %, with 37-38 % of blocks shrunk, and perplexity rose
0.218 %. Per-weight MSE and output perplexity are decoupled here: the blocks that gained the most
were the ones whose scale shrank, and shrinking means clipping the block's largest values, so what
the search buys is resolution on the fifteen ordinary values at the cost of the one large one. In a
projection that large value is carrying signal.

Scope of the claim: one grid, one lane, the 128 groups this recipe re-encodes locally, and this
corpus scored against the target only. It does not show that a searched scale is worse in general, and
1.0 is the ceiling of this search space rather than a sampled point -- a factor above 1.0 maps every
block's maximum past 6.0, which is unconditional clipping. What it does settle is that for weights with
no producer evidence behind them, max-abs is the right default, and that a converter change justified
by lower weight error has to be measured on perplexity before it is believed.

**The untested axis is the activation scale, and it is the better-motivated one.** ModelOpt ships a
separate `NVFP4ActHeadroomCalibrator` for the activation *global* scale, and documents the failure mode
of max calibration in terms that match the shape measured here: a single freak block far above the rest
drags the global scale up until every other block's scale falls below subnormal and flushes to zero,
"losing the whole tensor to protect one value", so the default anchors to a 99.99th percentile and
clips the rare blocks deliberately. The source checkpoint shows exactly that spread --
`mlp.gate_proj` records `amax=[0.0047, 0.4219]`, a 90x gap between the smallest block and the tensor
maximum.

Those 128 re-encoded sites are assigned `activation_policy="AllowA4"`, and their
`activation_input_divisor` is recovered from a source `input_scale` that the producer calibrated for
**FP8** -- `6 / input_scale` for an FP8 site, against `1 / input_scale` for an already-NVFP4 one. A
divisor sized for an 8-bit activation is not obviously roomy enough for a 4-bit one, and this
experiment did not touch it. That is the axis with a documented failure mode behind it, and it is
open.

## A `--quick` comparison is decided by four streams

`--quick` scores one stream per domain, 261,223 tokens; `full` scores four per domain, 1,044,876.
Comparing the rebuilt NVIDIA artifact against the official stock, `--quick` reads **−1.78%** and the
full corpus reads **−0.00%**, and the domain breakdown says why:

| domain | official stock | NVIDIA build | change |
|---|---:|---:|---:|
| `chinese_reference` | 6.40455 | 6.28941 | **−1.80%** |
| `english_reference` | 6.55706 | 6.49852 | −0.89% |
| `english_long_form` | 8.17251 | 8.29735 | +1.53% |
| `ninfer_code` | 1.67007 | 1.69030 | +1.21% |

It wins two domains and loses two and they cancel, so "the same on average" is the honest description
and "1.78% better" is not. The `--quick` figure is carried by its `zhwiki-00` stream, which moves
**−8.85%** where the full corpus's four Chinese streams move −1.80% together. That singleton is the
same `zhwiki` stream the `nvfp4` KV finding above singled out, which suggests the Chinese domain is
where a quantization difference shows largest — but four streams cannot establish it, and a comparison
that rests on one of them cannot be quoted as a result.

### The same breakdown for the other two pairs

The two lines whose published files exist were measured the same way, so the concentration can be seen
across all three rather than in one comparison.

| domain | QAT rebuilt | published | change | unsloth rebuilt | published | change |
|---|---:|---:|---:|---:|---:|---:|
| `chinese_reference` | 6.59476 | 6.66432 | **−1.04%** | 6.42705 | 6.46918 | **−0.65%** |
| `english_reference` | 6.72791 | 6.72041 | +0.11% | 6.74142 | 6.76224 | −0.31% |
| `english_long_form` | 8.24524 | 8.24943 | −0.05% | 8.30183 | 8.30162 | 0.00% |
| `ninfer_code` | 1.68395 | 1.68257 | +0.08% | 1.69145 | 1.69198 | −0.03% |
| **overall** | 4.99097 | 5.00234 | **−0.23%** | 4.97532 | 4.98768 | **−0.25%** |

Every rebuilt line is lower on `chinese_reference`, and that is the domain carrying the aggregate in
each: −1.04%, −0.65% and −1.80% against changes of at most a tenth of that elsewhere. The QAT line is
lower on two domains of four and the other two move against it slightly; the unsloth line is lower or
exactly tied on all four. So "same or better" is accurate in aggregate for both, and the honest
description of *where* is this domain, three times over.

## Measuring a template

`--chat-template` was added to this tool and then removed: perplexity scores **raw text**, so it never
renders a chat and the flag could not change a score. Measured before removing it, three templates --
the artifact's embedded one, `tools/chat_templates/qwen3_8.jinja`, and
froggeric/Qwen-Fixed-Chat-Templates v22.5 -- produced byte-identical perplexities on every artifact.

What the run did establish is the artifact ranking this file was missing:

| artifact | PPL (--quick, fp8) |
|---|---|
| `qwen3_8_27b_nvfp4swift` (Swift, re-encoded) | **4.68429** |
| `qwen3_8_27b_nvfp4full` | **4.77136** |
| `qwen3_8_27b_nvfp4` (official) | 4.82676 |
| `qwen3_8_27b_nvfp4qat` (QUASAR) | 4.89741 |

On `--quick`, Swift's re-encoded artifact is lowest, NVFP4-full 1.9% above it, the official artifact
3.0% above that and QUASAR 4.6% above that. Swift is also the one artifact here whose text weights
are not imported: its attention and GDN are encoded to NVFP4 from the finetune's BF16 source,
because the same recipe importing ModelOpt's per-tensor FP8 scored 4.84938 -- a per-tensor FP8 scale
is coarser than NVFP4's one scale per 16-element block, so the block scales more than pay for the
narrower codes. That comparison is like-for-like: one recipe, one checkpoint, only the attention and
GDN encoding differs.

**The `--quick` ranking does not generalize, and this is the row that shows it.** On the full corpus
the official stock is 4.90295, the re-encoded Swift 4.92432 (0.44% above it) and the superseded Swift
4.93874 (0.73% above). So Swift is lowest on the four-stream subset and *not* lowest on the 496-window
corpus: a finetune can win one subset and lose the corpus, and `--quick` selects one stream per
domain while `full` scores every window. Quote a `--quick` ranking as a `--quick` ranking. The
re-encode's own gain is the part that holds on both protocols: 3.4% on the subset, 0.29% on the
corpus, same direction, one recipe and one checkpoint.

**The bf16 exception pattern is a source allocation, not a rule, and it measures worse here.** The
third build is the same re-encode with NVFP4-full's pattern applied -- 27 projections kept bf16, 247
NVFP4 parents exactly as that artifact has. It scores 4.7701 and 4.93254 against the all-NVFP4
build's 4.68429 and 4.92432, on both protocols, while costing 0.77 GB more file and about 0.5 GiB
more resident. So the ordering on both is all-NVFP4, then exceptions, then the FP8 import. That
pattern came from a third party's mixed-precision checkpoint, was transplanted by the fork to a
different one, and nothing recorded why; on Swift's weights it does not pay for itself.

**The endpoints, not the block scales, are where the re-encode's win comes from.** The re-encoded
build differs from the FP8-importing one in two ways, and three builds isolate them, because each
adjacent pair differs in exactly one change — verified by hashing every binding, which leaves the two
endpoints as two objects and the text stack as 320. The endpoints alone are worth **−3.45% / −0.79%**
(Q8 against FP8, two rows above); re-encoding attention and GDN alone is worth **+0.05% / +0.50%**, a
small cost. The net **−3.40% / −0.29%** is the figure an earlier reading credited to NVFP4's block
scales being finer than a per-tensor FP8 scale, and that reading was wrong. Re-encoding the text stack
is bought for resident bytes and context, not for accuracy — the artifact conventions now state it as
that trade.

**Every variant row is reproducible without editing the recipe.** Each is the shipped recipe plus a
`--override` file that reassigns one thing, from the same sources, components and resources. The
bf16-exception rows reassign the 27 projections NVFP4-full keeps bf16 to `bf16`/`cast_direct` from
`swift_bf16` and drop the activation divisors the recipe had recorded for them; the FP8-endpoint rows
reassign `text/token_embedding` and `text/output_head` to `fp8`/`fp8_row_maxabs` from `swift_bf16`.
The published artifact's own invocation is in Section 16 of the artifact reference.

**A chat template needs a different instrument.** Perplexity cannot see one. What can are the rendered
prompt -- the token counts the CLI reports, which is how the reasoning-effort alias gap was caught -- and
any chat-shaped scoring route. Recorded here so the flag is not added again.

## The accurate-activation change (2026-09-23)

`nvfp4_linear_swiglu_w4a4_tma.cuh` was the only one of twelve SwiGLU activation sites using the
approximate `silu_approx`; every other site (fp8 ×2, nvfp4 decode, nvfp4 small-t, nvfp4 non-TMA W4A4,
q4 ×4, q8 ×3) already called the accurate `silu`. Upstream took the approximation in PR #250 and
Neroued/ninfer#285 measured a model-level cost for it, so the TMA epilogue was restored to `silu` and
`silu_approx` deleted. All four `ninfer_linear_swiglu_*_test` cases pass against their FP64 oracle,
including the TMA route, which `kA4Cases` reaches at 256, 512 and 1024 tokens (`kNvfp4TmaBlockM` is
256).

The change is engine-level, so it moves every artifact. Measured fp8 on this machine:

| artifact | protocol | before | after | Δ |
|---|---|---|---|---|
| official stock | full corpus | 4.90295 | **4.901690** | −0.03% |
| official stock | `--quick` | 4.82676 | **4.805574** | −0.44% |
| NVFP4-full | full corpus | not measured | **4.987682** | — |
| NVFP4-full | `--quick` | 4.77136 | **4.824524** | +1.11% |
| QUASAR QAT | full corpus | not measured | **5.002337** | — |
| QUASAR QAT | `--quick` | 4.89741 | **4.948789** | +1.05% |

Two things follow, and neither is the naive reading of #285.

**The accurate form is not a strict perplexity improvement.** It lowers the official stock artifact
and raises the other two. Perplexity is not monotone in numerical accuracy: a perturbation moves a
checkpoint toward or away from its training distribution depending on the checkpoint. The accurate
form is still the contract-correct one -- it is the FP64 oracle's own definition, eleven of twelve
sites already used it, and upstream's published artifacts were produced with it -- but the quality
argument is "matches the oracle and upstream", not "always scores better". The full-corpus effect on
the two artifacts whose baseline was not measured is therefore unknown, not zero: only the official
artifact has a full-corpus before-and-after.

**`--quick` is not a subset of the full corpus.** The manifest's two modes name different text:
`quick` is stream 00 of each domain (four documents, 261,223 tokens), `full` is all sixteen
(1,044,876). Their magnitudes differ by more than an order of magnitude for the same change (0.44%
against 0.03% on the official artifact), so a `--quick` delta is not an effect size for the full
corpus.

The ranking recorded in the section above (`--quick`: NVFP4-full 4.77136 < official 4.82676 < QUASAR
4.89741) was measured before this change. After it, on both protocols, the official stock artifact is
lowest: full corpus 4.9017 < 4.9877 < 5.0023, `--quick` 4.8056 < 4.8245 < 4.9488.

## The one open discrepancy

A third-party conversion recipe publishes PPL **4.617** for the official stock artifact on this same
corpus. Measured here it is **4.898** with bf16 KV and **4.903** with fp8, so the 6.1% gap is not the
KV dtype -- and the bf16-versus-fp8 agreement of 0.1% says the KV path is sound in general.

Two candidates remain and neither is checked: the **corpus revision**, since the recipe may have been
run against an earlier `perplexity-1m`; and the **artifact revision**, since theirs may be a newer
export of the stock model than the one this port carries. The scoring path itself is upstream's, not
this port's, so a port-specific defect is not the leading explanation -- but it is not excluded either,
and the protocol above is what would settle it.

Checked and excluded since: the accurate-activation change recorded above moves the full corpus by
−0.03%, so the engine's SwiGLU activation is not the cause of the gap. That leaves the two candidates
named here.
