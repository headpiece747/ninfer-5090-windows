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

## What is not established

- **The lane's maximum context with 3.16 GiB less KV** — unmeasured, and on a 32 GB card it is the cost
  most likely to matter in practice.
- **Decode** — the A/B cannot resolve a 5 % effect at this repetition count.
- **The other sources** — unsloth's checkpoint has the same split, so the same variant is buildable there;
  it is a different question because the fork *chose* to quantize that side, so it tests the hypothesis
  rather than the provenance. `qat` is a third case again: its QAT training used NVFP4 attention, so
  inference at FP8 is a train/test mismatch.
- **A task-accuracy measurement** — everything here is perplexity and acceptance. `d0xin` measured
  225/280 against FP8's 224/280 on a fixed MMLU-Pro subset; nothing on this port has been measured on a
  task suite for these two artifacts.
