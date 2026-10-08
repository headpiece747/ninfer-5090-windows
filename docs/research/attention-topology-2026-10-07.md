# The attention/GDN precision the producer chose, and what re-encoding it to NVFP4 costs

**MEASURED-HERE 2026-10-07.** One artifact pair, one difference, full corpus. The port's `nvidia` line
re-encodes the source checkpoint's FP8 attention and linear-attention projections to NVFP4; importing the
producer's own FP8 words instead is worth **1.41 % perplexity and 29 % relative acceptance**, and costs
**14.2 % prefill throughput** and **3.16 GiB**. Decode moved −5.3 %, which is inside this card's arm-to-arm
spread and is therefore *not* claimed.

## What the two sources actually contain, and what the port does with it

Both checkpoints the port builds from put only the MLP in NVFP4 and leave the rest of the text stack in
row-scaled FP8:

| source | what its own metadata records | what the shipped recipe does |
|---|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` | "MLP on all 64 layers and `lm_head` are NVFP4 (193 sites with a `weight_scale_2`), while attention and linear-attention are FP8 (208 sites)" — `official_recipes.py:772-774` | MLP imported; **attention/GDN re-encoded to NVFP4** with `nvfp4_maxabs` and the producer's Local-Hessian divisors |
| `unsloth/Qwen3.8-27B-NVFP4` | "unsloth quantizes the MLP of layers 0-55 as NVFP4 (168 matrices) and leaves attention, linear-attention including `in_proj_a`/`in_proj_b` and MLP 56-63 as row-scaled FP8 (233 matrices)" — `official_recipes.py:434-436` | same: MLP imported, the FP8 side re-encoded |

This is not a decision either source made. It is also not upstream's: upstream has no NVFP4 encoder and
imports its sources' words. And it is not what the published third-party artifacts do — NVIDIA's own
AutoQuantize assignment for Qwen3.8-27B is MLP + `lm_head` NVFP4 with both attention families at FP8, and
`d0xin/Swift-Qwen3.8-27B-Uncensored-NVFP4-LocalHessian-ActivationHeadroom-NInfer` ships the same split
(MLP 0-55 NVFP4, MLP 56-63 FP8, attention/GDN FP8, `lm_head` and embedding FP8). The port's *own* stock
recipe already implements it — `official_recipes.py:172`, `"/mlp/" in name and layer < 56` — which is why
the stock artifact holds the best perplexity this port has recorded (4.615687).

Why re-encoding those groups is hard is recorded in this port already: the source's 193 MSE-calibrated
weight quantizers are exactly the MLP and `lm_head`, which the port imports verbatim, and "the 128 object
groups re-encoded from BF16 here are the attention and linear-attention projections, which the producer
quantized as FP8 with plain max calibration. There was never a searched-scale artifact to match for them"
(`perplexity-baseline.md:206-208`). The port had to invent scales for them, and this measures the price.

## The experiment

Same source checkpoint, same MLP words, same imported divisors, same draft encoding, same template. The
only change is the attention/GDN branch: `import_encoded` of the producer's FP8 words with
`activation_policy="AllowA8"` instead of `nvfp4_maxabs` with `AllowA4` and a derived divisor.

Built as a *user recipe file* — `tools.convert` accepts `file.py:function`, so no tracked file was edited:

```
python -m tools.convert \
  --model C:\AI\models\hf-src\Qwen3.8-27B \
  --recipe <user recipe>.py:configure \
  --source quantized=C:\AI\models\hf-src\Qwen3.8-27B-NVFP4-nvidia \
  --source dflash2=C:\AI\models\hf-src\Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 --name qwen3.8-27b-fp8attn \
  --out C:\AI\models\_rebuild\qwen3_8_27b_nvfp4nvidia_fp8attn.v3.ninfer
```

160 s, 22.1 GB, 1,280 objects: **159 NVFP4 + 128 FP8** against the shipped line's 287 NVFP4. Everything
else — 579 BF16, 96 FP32, the Q4/Q5/Q6/Q8 vision and proposal encodings — is identical.

## Perplexity: −1.41 %, and every stream

Full corpus, fp8 KV, 4096/2048, `ninfer-perplexity`, on the engine at HEAD:

| line | PPL | vs this build |
|---|---|---|
| **fp8attn** | **4.620757** | — |
| `nvidia` — *same source*, topology only | 4.686759 | **−1.41 %** |
| `qat` (the recommended line) | 4.684860 | −1.37 % |
| stock (retired, FP8 attention) | 4.615687 | +0.11 % |
| `full` | 4.724219 | −2.19 % |
| `full_noex` | 4.727636 | −2.26 % |
| `swift15` | 4.755739 | −2.84 % |

All **16 of 16** streams are better than the same-source baseline (−0.93 % to −2.56 %), and all four
domains: `ninfer_code` −1.01 %, `english_long_form` −1.20 %, `english_reference` −1.93 %,
`chinese_reference` −1.48 %. Against the stock line it is +0.11 % — winning `english_reference`, losing
the other three — which is this port's recorded pattern that no lane wins everywhere. Only the `nvidia`
row is topology-isolated; the others are different checkpoints and are context, not attribution.

## The lane: prefill is a real regression, decode is inside the noise, acceptance improves

`ninfer_bench --spec dflash2 --draft-tokens 5 --lm-head-draft --kv-dtype fp8 -p 2048 -n 128 -r 5 --warmup 2`,
arms alternating fp8attn / nvidia / fp8attn / nvidia on the bench fixture corpus:

| metric | fp8attn | nvidia | delta | arm-to-arm spread of the baseline |
|---|---|---|---|---|
| prefill pp2048 | 9,459 tok/s | 11,024 tok/s | **−14.2 %** | 3.5 % (11,216 / 10,833) — regression is real |
| decode tg128 | 74.07 tok/s | 78.18 tok/s | −5.3 % | 9.2 % (81.61 / 74.75) — **inside the noise, not claimed** |
| DFlash2 acceptance | 0.1429 | 0.1106 | **+29 % relative** | deterministic, identical across both reps |
| rounds for 128 tokens | 375 | 415 | −9.6 % | follows from acceptance |
| artifact | 22.1 GB | 18.9 GB | +3.16 GiB | — |

The acceptance gain is consistent with the drafter's provenance rather than with the re-encode being
"better" in the abstract: DFlash2 was trained on the stock model's hidden states — the port's own recipe
docstring records the same change *losing* 3.2 acceptance points on Swift 1.0 for that reason — and the
`_nvdiv` docstring records the same direction between two published artifacts. The absolute rates here are
the bench fixture corpus and are **not** comparable with the lanes' recorded 53–61 % on the default code
domain; only the relative difference is like-for-like.

### The unsloth pair, same protocol, and what the extra 3.86 GiB costs the lane

| metric | fp8attn | `full` | delta | arm-to-arm spread of the baseline |
|---|---|---|---|---|
| prefill pp2048 | 10,064 tok/s | 11,617 tok/s | **−13.4 %** | 0.3 % — real, and it reproduces the nvidia pair's −14.2 % |
| decode tg128 | 104.64 tok/s | 102.78 tok/s | **+1.8 %** | 0.4 % — a small real gain, the opposite sign to the nvidia pair's −5.3 % |
| DFlash2 acceptance | 0.2083 | 0.1524 | **+36.7 % relative** | deterministic, identical across both reps |
| available after weights | 9,679,405,056 B | 13,540,261,888 B | **−3.86 GB** | — |
| implied KV context | **293,102 tokens** | 410,012 tokens | **−28.5 %** | arithmetic, not measured |

The context row is derived and labelled so: `available_after_weights_bytes` divided by the 33,024 bytes per
KV token the bench reported for `fp8-e4m3-r256` on this model. It is an upper bound on a text-only lane
rather than a served capacity — the lanes run with Vision and a runtime reservation on top — so the **ratio**
is the defensible part and the absolute is not. At 293,102 implied the variant still covers the model's
262,144 ceiling, but with little room left, and that is the cost most likely to decide the question in
practice.

Two sources now agree on the shape: prefill −13 to −14 %, acceptance +29 to +37 % relative, perplexity −1.4
to −2.4 %, and a large slice of KV capacity. Decode disagrees in sign between the pairs and both magnitudes
sit near the noise, so no decode claim is made either way.

## Per-domain KL: the distributional damage, which perplexity cannot order

`ninfer-perplexity --topk-record` on both artifacts over the same corpus, then
`tools/release/per_domain_kl.py` with the higher-fidelity image as the reference. The module's own
`check_comparable` guard passed — the streams, positions and per-stream token digests agree, which is also
independent evidence that both artifacts tokenize this corpus identically. KL is between two *quantized*
images, not against BF16: 27.781 B parameters is 51.75 GiB of BF16 against 31.85 GiB of card, which the
module states in its output rather than leaving to be discovered.

| domain | mean KL (nats) | positions | floored entries | reference mass in top-60 |
|---|---|---|---|---|
| `chinese_reference` | **0.197228** | 262,022 | 2,513,736 | 0.9424 |
| `english_reference` | 0.153235 | 261,223 | 2,079,344 | 0.9661 |
| `english_long_form` | 0.148772 | 261,408 | 1,760,108 | 0.9586 |
| `ninfer_code` | 0.059384 | 259,904 | 3,126,385 | **0.9917** |
| **overall** | **0.139802** | 1,044,557 | 9,479,573 | — |

Two things this says that the perplexity table does not.

**KL and perplexity order the domains differently.** Perplexity put the largest gain on
`english_reference` (−1.93 %) and the smallest on `ninfer_code` (−1.01 %); KL puts the largest damage on
`chinese_reference` (0.197 nats) and the smallest on `ninfer_code` (0.059). They agree that code moved
least and disagree about which moved most, because perplexity reduces each position to the reference
token's log-probability while KL sees the whole distribution. This is the third independent confirmation
of this port's recorded claim that "the aggregate is not a summary of the four domains" — and the domain
where the lanes already spanned 6.4 % on perplexity is the domain where the re-encode moves the
distribution most.

**The `reference mass in top-60` column explains the ordering.** The code domain is peaked — the top-60
support holds 99.2 % of its own mass, which is why its perplexity is 1.65 — and Chinese is the flattest at
94.2 %. A flatter distribution has more mass outside any fixed top-k, so it has more room to move, and it
moved most. That is a property of the domain rather than of the artifacts, and it is worth carrying: any
top-k divergence comparison will rank a flatter domain as more damaged unless the support is accounted for.

## The same result on a second source, where it overrides the fork's choice instead

unsloth's checkpoint carries the same split, so the same variant is buildable there — but it is a different
kind of experiment. For the nvidia line the port was overriding a producer choice the fork had made
deliberately (the fork's profile quantized the FP8 side to NVFP4), so that measurement restored what the
producer did. For unsloth the same re-encode is present, and importing its FP8 words overrides the fork's
own decision. Both come out the same way, which is what makes the mechanism the *topology* rather than the
provenance.

Built the same way — a user recipe file, the port's measured BF16 exception pattern kept so that one thing
changes — 149.6 s, 23.58 GB, 1,256 objects: **143 NVFP4 + 135 FP8** against `full`'s 278 NVFP4.

| line | source | PPL | size |
|---|---|---|---|
| **unsloth fp8attn** | unsloth | **4.613442** | 23.58 GB |
| stock (retired) | Qwen | 4.615687 | 23.72 GB |
| nvidia fp8attn | nvidia | 4.620757 | 22.11 GB |
| `qat` (recommended) | QUASAR | 4.684860 | 18.95 GB |
| `nvidia` | nvidia | 4.686759 | 18.95 GB |
| `full` | unsloth | 4.724219 | 19.72 GB |
| `full_noex` | unsloth | 4.727636 | 18.95 GB |
| `swift15` | Swift | 4.755739 | 18.95 GB |

Against its own source it is **−2.35 %** (`full`) and **−2.42 %** (`full_noex`), with all four domains
better (−0.024 code to −0.229 English reference). Against the nvidia variant, a different source and a
different finetune, it is −0.16 %. Against the stock line it is −0.05 % — effectively equal, on a
different checkpoint. It is the lowest perplexity this port has recorded on this corpus, and the two
variants together are the evidence: **every shipped line pays 1.4–2.4 % perplexity for re-encoding the
producer's FP8 attention and GDN projections to NVFP4.**

### Why the prefill costs what it costs: the tensor rate, not the bytes

`ncu` cannot collect counters on this host (`ERR_NVGPUCTRPERM`, which needs an elevation this port does not
take), but `nsys` needs none and answers duration questions directly. Traces of both artifacts at pp8192:

| kernel instance | fp8attn | nvidia |
|---|---|---|
| `fp8_a8_tma_mma_kernel<Fp8A8SplitKSchedule<…128,256,128,2…>>` | 369.27 ms | — |
| `fp8_a8_tma_mma_kernel<…>` (second shape) | 187.40 ms | — |
| `fp8_a8_tma_mma_kernel<…>` (third shape) | 109.48 ms | — |
| the `nvfp4_a4_tma_kernel` instances that remain | 550.46 + 281.63 ms | 562.55 + 292.12 ms |
| **total** | **2117.5 ms** | **1864.9 ms** (+13.6 %) |

The entire regression is those three FP8 instances — 666.15 ms of work the shipped arm does not do, because
it runs the same sites through `nvfp4_a4_tma_kernel` instead — and the surviving NVFP4 kernels get *faster*
(−22.58 ms) for having fewer sites. Comparing the arm-exclusive instances like for like:

| the sites that switched route | total GPU time |
|---|---|
| FP8 A8 | **708.88 ms** |
| NVFP4 A4 | **430.22 ms** |
| ratio | **1.65×** |

**The first reading of this section was wrong, and the arithmetic that refutes it was one division away.**
It said the cost was the weight bytes — an FP8 row weight is 1 byte per element against NVFP4's 0.5, so
1.65× looked *better* than the 2.0× byte ratio and the route looked healthy. But these kernels are not
bandwidth-bound at all: one pp2048 pass reads its ~15.9 GB of text weights in 185.8 ms, which is
**85.6 GB/s, 4.8 % of this card's ~1.79 TB/s bus**, and the switched sites alone read 7.2 GB at FP8 in
708.88 ms, **10.2 GB/s, 0.6 % of bus**. A byte ratio cannot explain a cost on a bus that is 95 % idle, and
the 1.65× is equally consistent with the real cause.

**The cause is the tensor rate.** On GB202 the block-scaled NVFP4 MMA is `m16n8k64` and the FP8 one is
`m16n8k32` — the same instruction family at half the K per issue — and NVIDIA publishes FP4 dense at 2× FP8.
Moving 28.2 % of the pass's FLOPs onto a half-rate path costs 17.6 ms on a 70.9 ms GEMM-only pass at peak
(+24.9 %), and the measured delta of +15.4–16.5 % is that same effect on a 185.8 ms pass. A two-term fit
reproduces both measured pairs with the FP8 rate at exactly half the FP4 rate plus the same ~30 µs/token
fixed term — which is also why the ratio is 1.65× and not 2.0×: part of each kernel's time is not tensor work.

**And the port's FP8 route is the better-utilised of the two.** Upstream's measured MXFP8
fused-projection throughput is 660–712 TFLOP/s at T=1024, 79–85 % of the 838 FP8 peak, against this port's
NVFP4 A4 at 882–985 TFLOP/s, 53–59 % of the 1676 peak. So there is no defect to fix in the 8-bit path; if
anything the *NVFP4* kernels have headroom.

**One discrepancy is left visible rather than smoothed over.** The two Op-level rates predict about +6 %
where the model measures +13–15 µs/token, 2.4–2.7× more. Closing that gap is what an interleaved
`ninfer_linear_bench` A/B at the four real geometries would do, and it matters because the decision below
turns on the *exchange rate* between quality and prefill: with the cost being tensor FLOPs rather than bytes,
a partial split's price is proportional to the FLOPs moved, not to the bytes saved.

The lever is therefore **how many sites need FP8**, which is what a per-projection sensitivity ranking
decides, and the reason the KV lever matters is that the context cost cannot be recovered from the kernel side.

## Appendix: the exact recipe change, so both variants are re-derivable

Each variant is its source's official recipe with **one branch replaced** — everything else, including the
BF16 exception pattern on the unsloth side and the draft encoding, is unchanged. `tools.convert` accepts a
user recipe file, so neither was ever a tracked file; this is the change they made, and it is what a future
re-derivation needs, because the artifacts themselves are disposable caches (`_rebuild\`, 22–24 GB each,
~150 s to rebuild).

For `qwen3_8_27b_nvfp4_nvidia` — the branch that replaces the `nvfp4_maxabs` + `AllowA4` + derived-divisor
arm for every non-MLP text projection:

```python
# Attention and linear-attention: keep the source's own FP8 words, no re-encode.
recipe.assign(
    name,
    format=FP8,                     # "fp8_e4m3fn_row_bf16"
    method=import_encoded,
    source=model.source(name, quantized, FP8),
    activation_policy="AllowA8",
)
```

For `qwen3_8_27b_nvfp4_unsloth` — the same replacement, after the BF16 exception check and the MLP branch,
so that the exceptions and the MLP stay exactly as the shipped line has them:

```python
# Attention, linear-attention and MLP 56-63: the source's own FP8 words, no re-encode.
recipe.assign(
    name,
    format=FP8,
    method=import_encoded,
    source=model.source(name, quantized, FP8),
    activation_policy="AllowA8",
)
```

Both need `FP8 = "fp8_e4m3fn_row_bf16"`, `import_encoded`, and the helpers the official recipe already
imports (`_optional`, `_nvfp4_draft`, `_assign`, `add_proposal`, and `_is_bf16_exception` for the unsloth
one). The conversion command for each is the one in the experiment section, with the matching
`--source quantized=` and `--name`.

## What is not established

- **The served context with less KV** — the unsloth pair's *ratio* is derived (−28.5 %, 410,012 → 293,102
  implied tokens at 33,024 bytes per KV token), and neither pair has been *served* with `--kv-capacity auto`
  to read the resolved figure the way the lanes report it. The ratio is the defensible part; the absolute is
  an upper bound that ignores Vision and the runtime reservation.
- **Decode** — the A/B cannot resolve a 5 % effect at this repetition count, and the two pairs disagree in
  sign, so no decode claim is made either way.
- **Two sources remain untested, for different reasons.** `qat` re-encodes nothing — QUASAR's export covers
  every text linear, so its NVFP4 attention is what its own QAT training produced, and an FP8 variant would
  be a train/test mismatch. Swift is a third case where importing its ModelOpt FP8 attention is what the port
  rejected for *size* ("9 GiB of 8-bit weights"), so a Swift variant trades roughly +9 GiB rather than the
  +3.2 to +3.9 GiB measured here — and the port's own notes record its BF16 exception pattern measuring
  *worse* there, so it is not a copy of these two.
- **A task-accuracy measurement** — everything here is perplexity and acceptance. `d0xin` measured
  225/280 against FP8's 224/280 on a fixed MMLU-Pro subset; nothing on this port has been measured on a
  task suite for these two artifacts.
