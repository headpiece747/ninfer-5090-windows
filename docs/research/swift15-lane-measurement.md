# Swift 1.5 replaces the Swift lanes: what was measured, and what it settles

**MEASURED HERE 2026-09-30 on an RTX 5090 (32 GB), one binary, one card, every comparison
interleaved and rotated. The Swift 1.0 control is same-day and same-binary throughout, because the
2026-09-28 table it would otherwise be compared against does not reproduce — see the last section,
which is a finding in its own right.**

## The question

Replace the two lanes that ship `ukisai/Swift-Qwen3.8-27B-NVFP4` — `start_swift_v3_dflash2_vision`
and `start_swift_v3_mtp4_vision` — with `ukisai/Swift-1.5-Qwen3.8-27b-NVFP4` converted to v3, at the
best encoding and the best draft window, with dflash2, mtp, perplexity and per-domain KL checked.

## The conversion needs no new recipe, and that is a measurement rather than an assumption

Swift 1.5's ModelOpt export is structurally identical to Swift 1.0's: **401 quantized sites, the same
193 NVFP4 (MLP) and 208 FP8 (attention and GDN) module names, no site added, removed or
re-algoed.** Only the producer string moved, `0.47.0rc0` → `0.47.0rc1.dev90+gb311c054d`. So
`qwen3_8_27b_nvfp4_swift` applies unchanged and the build differs from the Swift 1.0 build only in
which weights it read — which
[swift-lane-build-record.md](swift-lane-build-record.md) has already established reproduces the
shipped Swift 1.0 artifact to every perplexity digit.

| | Swift 1.0 (shipped) | Swift 1.5 |
|---|---|---|
| recipe | `qwen3_8_27b_nvfp4_swift` | the same, unmodified |
| sources | 1.0 NVFP4 + 1.0 BF16 + DFlash2 | 1.5 NVFP4 + 1.5 BF16 + DFlash2 |
| components | text, vision, mtp, dflash2 | the same four |
| objects | 1590 | 1590 |
| bytes | 18.42 GiB | 18.42 GiB |
| conversion time | — | 148.2 s |
| structural verifier | PASS 4383 checks | **PASS 4383 checks** |

The verifier's five counters are identical between the two — `auxiliaries 512`, `bindings 1513`,
`input_divisors 512`, `objects 1590`, `weight_divisors 256` — which is the artifact-conventions
section 2 contract applied to the new image and the strongest structural statement available without
decoding every payload.

Both finetunes are also the same size and the same tensor names: 1199 BF16 tensors, `model.visual.*`,
`mtp.*` with 15 tensors, `lm_head`, 18 shards. Swift 1.5 ships **its own MTP head and its own vision
tower**, so the MTP lane's drafter is matched to its target by construction. It ships **no DFlash2
draft**.

## Perplexity: the control reproduces, and 1.5 is a different model rather than a worse build

Full corpus, 4096 context / 2048 stride, fp8 KV, 1,044,876 scored tokens, `fp8-e4m3-r256`.

| domain | Swift 1.0 (control) | Swift 1.5 | change |
|---|---:|---:|---:|
| `chinese_reference` | 6.229361 | 6.466866 | +3.81 % |
| `english_long_form` | 8.350429 | 8.374790 | +0.29 % |
| `english_reference` | 6.707160 | 6.774238 | +1.00 % |
| `ninfer_code` | 1.690048 | 1.692300 | +0.13 % |
| **overall** | **4.936397** | **5.000654** | **+1.30 %** |

The Swift 1.0 column reproduces `perplexity-baseline.md`'s recorded 4.936397 **to every digit** on the
same binary, which is what makes the 1.30 % a property of the two checkpoints rather than of this
port's building of either. That distinction is the one
[swift-lane-build-record.md](swift-lane-build-record.md) drew for the 1.0-versus-`nvfp4full` gap: a
perplexity difference between two *different finetunes* is not a build-quality measurement. Swift
1.5's own card reports it scoring higher than the base on GPQA-Diamond (88.59 % against 88.28 %) with
58.5 % fewer thinking tokens, and a model trained to compress its reasoning can score worse on a
corpus of English reference prose and C++ while scoring better on the tasks it was trained for. So
**+1.30 % is reported, not acted on.** Acting on it would be reading a corpus likelihood as a model
ranking.

## Context ceiling: two shipping lanes are dead today, and it is not these artifacts

Probed with `ceiling` mode, which renders the launcher's own flag set, on 2026-09-30.

| artifact | DFlash2 text | DFlash2 + Vision | MTP text | MTP + Vision |
|---|---:|---:|---:|---:|
| QUASAR | 262,144 | 262,144 | not re-probed | not re-probed |
| NVFP4-full | 262,144 | **240,000** | not re-probed | not re-probed |
| Swift 1.0 | 262,144 | **240,000** | 262,144 | 262,144 |
| Swift 1.5 | 262,144 | **240,000** | 262,144 | 262,144 |
| NVIDIA | 262,144 | 262,144 | not re-probed | not re-probed |

`CEILINGS` in `tools/release/v3_profile_matrix.py` said every DFlash2 combination reached 262,144.
Two do not. The cause is not the images: device weights are 18.0 GiB on the Swift build and QUASAR
and NVIDIA still serve 262,144 on 1.51 and 1.37 GiB free. **Runtime grew from 10.6/10.7 GiB on
2026-09-24 to 11.5/11.6 GiB on 2026-09-30**, and the DFlash2 + Vision lanes had under a gigabyte of
margin left. `profile` mode then measured `start_swift_v3_dflash2_vision` **REFUSED** through the real
launcher — the fastest lane in the product, 370.9 tok/s in the table, does not start today.

## Draft window: the shipped widths are already right, and the code domain hides it

Every width each backend accepts, from the startup validation rather than from convenience:
**MTP is hard-capped at 5** (`kMaximumMtpDraftTokens`, `startup.cpp:785`, error text `"[1,5]"`) and
**DFlash2 at 15** (`startup.cpp:796`). `docs/active-work.md` item 8 proposes raising the MTP window to
10 on the strength of a fork that did it; this engine refuses to start at 6, so that sweep cannot be
run on this tree at all. That is a correction to the project's own list, not a finding about MTP.

Swift 1.5, Vision, the shipped context, **two interleaved rotated rounds**, the `code` domain:

| width | tok/s | spread | accept | tok/round |
|---|---:|---:|---:|---:|
| none (control) | 85.3 | 4.9 % | — | — |
| d1 | 141.6 | 1.9 % | 90.4 % | 0.90 |
| d2 | 183.1 | 0.6 % | 77.6 % | 1.55 |
| d3 | 226.7 | 0.3 % | 76.5 % | 2.29 |
| d4 | 239.7 | 0.3 % | 62.4 % | 2.49 |
| d5 | 294.9 | 0.4 % | 73.9 % | 3.69 |
| d6 | 305.6 | 1.7 % | 66.1 % | 3.93 |
| **d7 (shipped)** | **297.9** | 0.9 % | 49.3 % | 3.43 |
| d8 | 296.6 | 0.2 % | 44.8 % | 3.52 |
| d9 | 353.1 | 0.2 % | 50.5 % | 4.47 |
| d10 | 355.1 | 2.8 % | 46.4 % | 4.47 |
| d11 | 318.2 | 0.8 % | 35.9 % | 3.93 |
| d12 | 373.4 | 2.7 % | 41.7 % | 4.96 |
| **d13** | **396.0** | 0.6 % | 42.9 % | 5.33 |
| d14 | 330.4 | 0.4 % | 31.8 % | 4.32 |
| d15 | 340.6 | 0.3 % | 31.6 % | 4.54 |

**Token speed ceiling: 396.0 tok/s at DFlash2 d13, 4.64× the non-speculative control's 85.3.** The
non-monotonicity at d7/d8 and d11 is reproducible, not noise — those widths' two rounds span 0.9 %
and 0.2 %, 0.8 %.

**On the code domain alone, d13 is +33 % over the shipped d7.** That is the number a single-domain
table would publish, and it is the wrong basis for the decision. The harness's own header says code
moves tokens per round by more than a factor of two against Chinese and that *"every width and depth
decision taken from it reversed once other domains were included"*. So:

**DFlash2, Swift 1.5, tok/s:**

| domain | d7 | d9 | d13 | best |
|---|---:|---:|---:|---|
| code | 297.9 | 353.1 | **396.0** | d13, +33 % |
| chinese | **170.8** | — | 158.5 | d7, +7.7 % |
| prose | **172.2** | 169.4 | 154.3 | d7, +11.6 % |
| dialogue | **226.9** | 219.3 | 215.0 | d7, +5.5 % |
| repetition | 225.9 | **252.7** | 203.6 | d9, +11.9 % |

**MTP, Swift 1.5, tok/s:**

| domain | d4 | d5 | best |
|---|---:|---:|---|
| code | 224.3 | **237.2** | d5, +5.8 % |
| prose | **122.8** | 119.4 | d4, +2.8 % |
| dialogue | **170.8** | 150.7 | d4, +13.3 % |
| repetition | **186.8** | 164.6 | d4, +13.5 % |

**d7 wins three domains of five and d13 one; d4 wins three of four and d5 one.** Choosing the width
that maximises the worst domain — the only choice that does not assume a traffic mix — gives **d7 and
d4, which is what the product already ships.** Acceptance explains it: it falls monotonically with
width (90.4 % → 31.6 % on code, and to 8.6–21.8 % on the non-code domains), so a wide window buys
tokens per round and loses more than that in accept probability except where the drafter is unusually
strong, which on this drafter means code.

The per-position profile, newly reported by the `widths` mode from
`speculative_accepted_per_position`, shows where that goes. On `repetition` at d13:
`75/69/65/49/90/72/77/70/86/50/100/33/0` — position 13 accepts nothing and position 14 is never
reached. On `chinese` at d7 it is `66/52/62/53/78/64/56`, a healthy flat profile. The late-position
figures at wide settings rest on very few rounds, which is why the mode prints `-` for a position no
round reached rather than a zero.

## The artifact comparison: Swift 1.0 against Swift 1.5, same day, same binary

`code` domain, two interleaved rounds, Vision:

| configuration | Swift 1.0 | Swift 1.5 | change |
|---|---:|---:|---:|
| none (control) | 84.2 | 85.3 | +1.3 % |
| DFlash2 d7 | **325.0** (57.1 %) | 297.9 (49.3 %) | **−8.3 %, −7.8 pp** |
| DFlash2 d13 | 384.6 (41.4 %) | **396.0** (42.9 %) | +3.0 %, +1.5 pp |
| MTP d4 | **231.1** (64.8 %) | 224.3 (58.7 %) | −2.9 %, −6.1 pp |
| MTP d5 | 153.6 (34.9 %) | **237.2** (60.2 %) | **+54 %, +25.3 pp** |

`chinese` domain, two interleaved rounds:

| configuration | Swift 1.0 | Swift 1.5 | change |
|---|---:|---:|---:|
| DFlash2 d7 | 133.4 (14.2 %) | **170.8** (21.8 %) | **+28.0 %, +7.6 pp** |
| DFlash2 d13 | 131.9 (8.6 %) | **158.5** (11.5 %) | **+20.2 %, +2.9 pp** |
| MTP d4 | 121.3 (20.8 %) | see below | |
| MTP d5 | 127.8 (23.6 %) | see below | |

**The DFlash2 draft is the interesting part, and the prediction came from the project's own upstream
tracker.** `Neroued/ninfer#298` §3 reports that on a *finetune* the stock `z-lab` DFlash2 draft
accepted **3.3–5.1 %** and decode fell to 67–75 tok/s, against MTP at 67–85 %, and concludes "the
DFlash2 draft does not transfer to finetunes". Swift 1.0 is the counterexample that made this worth
measuring rather than assuming — it is a finetune and it accepted 57.1 % at d7. Swift 1.5 is a further
RL and OPD round on top of Swift 1.0, and its DFlash2 acceptance at d7 falls to **49.3 %, −7.8 pp**,
while its **own** MTP head — the one that ships inside the checkpoint — improves on Swift 1.0's at
depth 5 by 25.3 pp. The drafter is stock-trained; Swift 1.5's hidden states have moved further from
the stock ones; its own head moved with them.

That is also why the artifact comparison **reverses by domain**: on code, at the widths the product
ships, Swift 1.0 is 8.3 % faster; on chinese, Swift 1.5 is 28 % faster. Neither figure is the answer
on its own, and the honest statement is that the choice depends on the traffic mix, with Swift 1.5
ahead on the domain where the drafter is weakest.

## Per-domain KL: a BF16 reference is arithmetically impossible on this card

`docs/research/per-domain-kl-instrument.md` specifies per-domain KL against a **BF16** reference.
Measured from the safetensors headers rather than estimated: Qwen3.8-27B is **27.781 B parameters**,
so its BF16 weights are **51.75 GiB**, and the RTX 5090 has **31.85 GiB**. A BF16 reference cannot be
resident, and CPU-offloaded weight streaming is not something this engine has. The design's premise
does not hold on this machine and the instrument has to name a different reference or not exist.

The `topk_logprobs` Op and its qualification were already in the tree (`9b39205e`); the three
missing steps — a top-k scoring mode, the record format, the per-domain reduction — are built here:

* `include/ninfer/types.h` gains `CausalTopk` and `Engine::score_topk`; `Program::causal_score_topk`
  shares one staging loop with `causal_score` rather than duplicating it, because the two routes must
  agree about *which positions were scored* for a comparison between them to mean anything.
* `apps/perplexity/topk_record.{h,cpp}` writes the record. Positions are interleaved one at a time
  (k indices then k log-probabilities), every integer is emitted little-endian by explicit shifts
  rather than by struct overlay, and each stream carries a SHA-256 of its own token ids so the reader
  can **refuse** a comparison against an image that tokenized the same text differently instead of
  reporting a mean over two unrelated contexts.
* `tools/release/per_domain_kl.py` reads two records and reduces them per domain, with arXiv
  2606.19558's log-probability floor for a reference token the candidate's top-k does not carry, and
  the floored-entry count printed beside every mean so a mean that is really a floor in disguise is
  visible rather than plausible.

`tests/test_topk_record.cpp` pins the digest against `hashlib` — the two implementations have to
agree or the instrument fails closed and is simply dead — including the multi-block case and a
negative token id, and takes a written record apart to check the layout is the reader's. It caught
two real defects during this work: the writer emitted per-tile blocks where the reader expects
per-position, and `put_padded` rejected a field that fills exactly, which every real prefill
signature does.

### The measurement

Reference: the groupwise-int image of the **same Swift 1.5 BF16 source checkpoint**, so the
comparison is between two images of one set of weights and no finetune difference is folded in.
Candidate: the shipping NVFP4 image. Both records are k = 60, full corpus, 1,044,876 positions,
478.3 MiB each, and both carry the token digest that makes the pairing checkable.

| domain | mean KL (nats) | positions | floored entries | reference mass/position |
|---|---:|---:|---:|---:|
| `chinese_reference` | **0.420244** | 262,034 | 3,232,078 | 0.9448 |
| `english_long_form` | 0.192605 | 261,594 | 2,088,707 | 0.9604 |
| `english_reference` | 0.242014 | 261,344 | 2,508,864 | 0.9703 |
| `ninfer_code` | **0.072512** | 259,904 | 3,381,468 | 0.9923 |
| **overall** | **0.232178** | 1,044,876 | 11,211,117 | |

**The domains disagree by 5.8× where perplexity disagrees by 1.3 %**, and in the same direction as
the perplexity table: the NVFP4 image is furthest from its reference on the domain where it also
scores worst. That is the instrument doing what `per-domain-kl-instrument.md` specified it for —
perplexity's single number let one domain decide the headline, and here the four domains separate
cleanly instead.

**The control that had to be zero, and was.** The Q8-draft and NVFP4-draft images have bit-identical
text stacks — the recipe differs only in how the `dflash2` component is encoded — so their
divergence must be exactly zero. It is: **0.000000 in every domain, with zero floored entries.**
Without that, a non-zero figure would have been uninterpretable; with it, the 11.2 M floored entries
above are a real property of the quantization rather than an artefact of the record path.

## The draft encoding is the one that decides the context ceiling, and it reverses on this target

Swift 1.5's own MTP head is inside the checkpoint, so the MTP lane is unaffected by how the drafter
is encoded. The DFlash2 lane is not: `_nvfp4_draft` encodes the `dflash2` component to NVFP4, which
Swift 1.0's recipe deliberately did **not** do, because on Swift 1.0 that change lost 3.2 acceptance
points (57.7 % against 60.9 %) — the draft was trained on the stock model's hidden states and Swift
1.0's are not the stock ones. That is the rule this file keeps restating: a draft encoding is
measured per target.

Built as `qwen3_8_27b_nvfp4_swift15_nvdraft`, it is the **same recipe with the one component's
encoding changed**, so the MTP head and the whole text stack are bit-identical:

| | Q8 draft | NVFP4 draft |
|---|---:|---:|
| objects | 1590 | 1600 |
| bytes | 18.42 GiB | **17.65 GiB** |
| structural verifier | PASS 4383 | **PASS 4424** |
| DFlash2 + Vision context | 240,000 (refused at 262,144) | **262,144** |
| DFlash2 + Vision free | 1.41 GiB | **1.51 GiB** |

**0.77 GiB of device weight is exactly the margin.** Its 1.51 GiB free at 262,144 + Vision is
identical to QUASAR's on the same configuration, so it lands where the lanes that serve the full
context land rather than merely clearing the bar.

Throughput, interleaved with the Q8 build in the same window, two rounds, `code` and `chinese`:

| domain | Q8 draft | NVFP4 draft | change |
|---|---:|---:|---:|
| code, d7 | 288.4 (49.3 %) | **349.4 (63.0 %)** | **+21.2 %, +13.7 pp** |
| chinese, d7 | **166.3 (21.8 %)** | 146.9 (17.2 %) | −11.7 %, −4.6 pp |
| MTP d4, code | 216.0 | 215.7 | −0.1 % |
| MTP d4, chinese | 125.8 | 125.5 | −0.2 % |

The MTP rows are the control: **0.1 % and 0.2 %**, because the recipe does not touch the MTP head. A
variant that moved the MTP lane by that much would be the harness measuring noise, and it confirms
the DFlash2 differences are the encoding rather than the run.

So the encoding **reverses on this target too**, exactly as the Swift 1.0 measurement predicted it
might: on code the NVFP4 draft is 13.7 points better, on chinese 4.6 points worse. The choice is
therefore not "which is faster" but "which defect is acceptable". The Q8 build's defect is a lane
that **cannot start at the context the product advertises**; the NVFP4 build's is 11.7 % on the one
domain where the drafter is weakest. Recorded, measured, and shipped as the NVFP4 draft, with the
per-domain figures here rather than a single headline.

## The 308 MiB that was missing, and why it is not the same fix on every lane

The runtime growth is not a defect in this port. Commit `e621c7d6` ("derive launch plans from device
sm count") is on `upstream/dev`, and it threads the device's SM count through
`causal_softmax_attention_workspace_capacity_bytes`, which raises the split count and with it the
partials the workspace reserves. `e621c7d6`'s own message says it preserves launch decisions at 170
SM, which is this card, so the larger reservation is upstream's deliberate memory-for-parallelism
trade rather than a regression introduced here. It is recorded here rather than treated as a
candidate for reversal: undoing it would trade throughput back for context on every lane at once,
which is a larger decision than the one this document answers.

What the refusal actually needs is small. Reading the engine's own arithmetic for
`start_ninfer_v3_dflash2_vision` at 262,144 with Vision, with `--log-level debug`:

```text
weights ready | 17.9 GiB
FATAL | minimum Engine runtime reservation requires 12487716865 bytes in addition to
       1073741824 bytes of automatic headroom, but only 13244563456 bytes are available
       after weights
```

11.63 GiB of reservation plus 1 GiB of headroom against 12.33 GiB available after weights —
**short by 308 MiB.** For scale, the KV pool at 262,144 is 8.0 GiB of that 11.63 (4,096 pages at
2 MiB), so the non-KV part is about 3.6 GiB, of which the DFlash2 graph allowance at width 7 is
480 MiB across its six visible-context tiers.

**308 MiB is reachable from the artifact side, and the encoding that reaches it is worth different
things to different routes.** NVFP4-full keeps nine fused parents in BF16; encoding them to NVFP4
frees 0.7 GiB, which clears the refusal with margin — 17.2 GiB of device weights against 17.9, and
the lane then reads the same 1.51 GiB free that QUASAR does. Interleaved against the BF16-exception
build, `code`, two rounds:

| route | BF16 exceptions | NVFP4 no-exception | change |
|---|---:|---:|---:|
| DFlash2 d7, context | 240,000 (refused at 262,144) | **262,144** | +9.4 % |
| DFlash2 d7 tok/s | 230.4 | **287.6** | **+24.8 %** |
| DFlash2 d7 acceptance | 37.1 % | **48.7 %** | **+11.6 pp** |
| MTP d5 tok/s | **206.5** | 151.3 | **−26.8 %** |
| MTP d5 acceptance | **53.7 %** | 32.3 % | **−21.4 pp** |
| full-corpus perplexity | **4.998419** | 5.002751 | +0.087 % |

**The two routes want opposite encodings**, which is the same rule as the DFlash2 draft: a pattern is
measured per target. The BF16 exceptions are worth 21.4 acceptance points to the MTP head and cost
the DFlash2 route 11.6. So the line is split — `start_ninfer_v3_dflash2_vision` runs the
no-exception image at 262,144, and `start_ninfer_v3_mtp5_vision` keeps the BF16-exception image at
262,144 — and both were verified through their own launchers at the native context.

The no-exception image's cost is the honest part of this trade: **+0.087 % overall perplexity**,
5.002751 against 4.998419 on the same binary and day, and one domain moves the other way
(`english_reference` improves 0.32 %). The older note in `official_recipes.py` recorded the same
change as +0.65 % on `--quick` and +0.167 % on the full corpus; the full-corpus figure here is half
that, so the earlier number was not reproduced exactly and the measured one is the one quoted.

## The recorded table does not reproduce, which is a finding and not a footnote

`profiles.py` recorded the Swift DFlash2 lane at **370.9 tok/s and 67.3 % acceptance**. Same lane, same
domain, same binary, today: **325.0 tok/s and 57.1 %** for Swift 1.0 and **349.4 / 63.0 %** for
Swift 1.5 — and the lane was refused at its shipped context until the rebuild above. The Swift MTP
lane's recorded 242.0 reproduces at 231.1, −4.5 %, which is inside the harness's own spread.
NVFP4-full's recorded MTP 231.2 reads 204.4 today, −11.5 %, outside it. The DFlash2 figures do not
reproduce; the MTP ones are mixed.

`profiles.py`'s docstring already records that the 2026-09-28 table came from one interleaved window
and that the merges since invalidated the performance table. This is that invalidation arriving, and
it is why the numbers in the table are being re-measured rather than adjusted.
