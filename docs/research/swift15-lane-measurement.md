# Swift 1.5 replaces the Swift lanes: what was measured, and what it settles

**MEASURED HERE 2026-09-30 on an RTX 5090 (32 GB), one binary, one card, every comparison
interleaved and rotated. The Swift 1.0 control is same-day and same-binary throughout, because the
2026-09-28 table it would otherwise be compared against does not reproduce — see the last section,
which is a finding in its own right.**

**The lane configuration changed after most of this document was written.** Every table below was
taken at `--prefill-chunk 8192`, which is what ships; 4096 was measured and rejected (see
"`--prefill-chunk` is a memory lever"). The figures in `tools/release/profiles.py` and the launcher
headers were re-measured across all eight lanes on 2026-09-30 and supersede several generations of
recorded numbers that no longer reproduced — including ones this document quotes, which are left as
they were recorded rather than rewritten.

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

## Which component grew, measured rather than inferred

The growth was attributed to an upstream commit from its message before it was checked. It is now
measured, component by component, by rebuilding the tree that produced the 2026-09-24 figure
(`80e911bd`) and starting the **same artifact** with the same flags on both binaries. The
`server_start` event in the request log carries the per-arena breakdown, which the single `capacity`
startup line does not.

Swift 1.5 Q8-draft, DFlash2 draft window 7, vision, `--max-context 240000`:

| component | 09-24 binary | current binary | change |
|---|---:|---:|---:|
| weights | 17.991 GiB | 17.991 GiB | — |
| sequence arena (KV) | 8.234 GiB | 8.234 GiB | — |
| CUDA Graph allowance | 0.469 GiB | 0.469 GiB | — |
| **unified workspace (capacity)** | **1.354 GiB** | **2.246 GiB** | **+0.892 GiB** |
| unified workspace (peak used) | 1.198 GiB | 2.090 GiB | +0.892 GiB |
| **runtime reservation** | **10.057 GiB** | **10.949 GiB** | **+0.892 GiB** |
| available after startup | 2.334 GiB | 1.211 GiB | −1.123 GiB |

The whole increase is the unified workspace, and its **peak** grew by the same amount, so this is
genuine demand rather than over-reservation: some width in the band really does allocate that much.
It is not the graph allowance, which is what the earlier reading of the commit history suggested.

**Read in the source.** `causal_softmax_attention_workspace_capacity_bytes` used to take a
`max_width` and derive its splits from `detail::causal_attention_split_capacity(q_heads, width,
cache_storage, envelope, batch_size)` — a function of shapes alone. That function no longer exists.
It now takes a `DeviceExecutionView` and dispatches per storage type with
`execution.multiprocessor_count`, and `src/ops/softmax_attention/dense/causal_cache/fp8/plan.cpp`
budgets splits against the SM count directly: decode budgets `2 * sms`, and the reservation is the
**maximum** partials buffer over every width in the band. A 170-SM 5090 therefore reserves
substantially more partial scratch than a card with fewer SMs would.

**The ngram change is not the cause.** It added a second graph family (`d0d419f6`) and
`8c7242e6` withdrew it — that family was charged twice for executables the tree never built, and the
charge is gone. The graph allowance measures 0.469 GiB on *both* binaries, and the growth is in the
workspace, which that code did not size.

**What the merges cost, measured.** Same artifact, one full-length warmup discarded, no server-side
seed:

| binary | context | decode | acceptance |
|---|---:|---:|---:|
| 09-24 | 262,144 | 329.0 tok/s (n=5) | 60.1% (n=5) |
| current | 240,000 | 328.9 tok/s (n=24) | 58.4% ± 1.1 pp (n=24) |

Per-record acceptance spans 46–67%, so these are indistinguishable: the workspace growth bought no
measurable throughput and cost no acceptance. Two caveats on that row: the old side is n=5 at a
larger context, and a first comparison of 60.1% against the harness's 49.3% was **invalid** — the
harness fixes a seed and repeats one draw, so its figure is reproducible but is a single sample of
the distribution, and the two do not share a sampling method.

**The process gap.** `docs/active-work.md` item 1 recorded that the merge rewrote 88 files of
`src/ops/softmax_attention` and that "what the merge invalidated is the recorded performance table."
That was right about throughput and it left the memory reservation unchecked: no pre-merge control
was run against the same artifact, which is why this took a rebuild to locate.

## `--prefill-chunk` is a memory lever, 4096 was measured, and it is rejected

`startup.cpp:271` sets `max_width = min(prefill_chunk, capacity)`, and that is the width the
reservation is maximised over. So a flag every lane already ships is a direct memory lever, and
changing it needs no CUDA change and no revert.

Swift 1.5 DFlash2 d7 + Vision at `--max-context 262,144`, measured with the engine's own per-request
`timings` (`prompt_ms`, `prompt_per_second`, `predicted_per_second`):

| `--prefill-chunk` | workspace | workspace peak | reservation | free after startup | prefill tok/s @19.2k |
|---:|---:|---:|---:|---:|---:|
| 8192 | 2.246 GiB | 2.090 | 11.630 GiB | 1.505 GiB | 13,077 |
| **4096** | **1.130 GiB** | 0.974 | **10.280 GiB** | **2.856 GiB** | 12,770 |
| 2048 | 0.807 GiB | 0.499 | 9.839 GiB | 3.018 GiB | 11,621 |

KV capacity is 262,144 with pages 4,096/4,096 in all three, so nothing is bought by serving less
context.

**Neither alternative ships.** The rejection is not that 4096 fails — it does exactly what it says,
recovering 1.12 GiB of workspace for -2.3% prefill. It is that **262,144 is already served at 8192 on
all eight lanes**, each started through its own launcher. The experiment bought *margin, not context*,
and margin was not the binding constraint: it spent measured throughput on headroom nothing was short
of. A flag that reduces the worst-case memory without enabling anything is a robustness purchase, and
it has to be argued as one rather than presented as a fix. Recorded in `profiles.py` so the
measurement is not repeated.

### The lever that was actually worth changing — and where my first attribution was wrong

The field standard for choosing split-KV splits is FlashAttention's `num_splits_heuristic`
(`hopper/heuristics.h`, and the same file vendored by vLLM). It selects by wave-quantization
efficiency and then returns **the smallest split count reaching 85% of peak efficiency**, with the
stated reason that "we also don't want too many splits as that would incur more HBM reads/writes".
It also returns **1 split whenever work tiles already fill 80% of the SMs**, and caps splits at
`min(max_splits, num_SMs, num_n_blocks)`. A 2026 study of FlashAttention-3 on Hopper (arXiv
2604.00028) sweeps a low-head decode point and finds a **broad low-latency plateau from s=3**, with
the best tested value (s=64) under ~2% better — over-splitting buys almost nothing and costs HBM
traffic and scratch.

**The first attribution here was wrong and is withdrawn.** It blamed `causal_partition_target`'s flat
`2 * multiprocessor_count` CTA budget divided by independent tiles. That rule governs the small-width
families, whose partials are tens of MiB. It is not where the workspace is.

The workspace is the **tiled prefill family**, and the rule there was already an occupancy rule that
simply had no economic term. `mxfp8_tiled_partition` minimises waves per partition and retains fewer
on a tie — but more splits always balance better, so on this product's shapes it walks to its cap of 8
at every step:

| splits | next waves | `next*selected < waves*splits` |
|---:|---:|---|
| 1 | 25 | — |
| 2 | 49 | 49 < 50 ✓ |
| 3 | 73 | 146 < 147 ✓ |
| 4 | 97 | 291 < 292 ✓ |
| 5 | 121 | 484 < 485 ✓ |
| 6 | 145 | 725 < 726 ✓ |
| 7 | 169 | 1014 < 1015 ✓ |
| 8 | 193 | 1351 < 1352 ✓ |

Each split costs a full FP32 accumulator of `kCausalHeadDim` depth per head per width. `allocate_causal_partials`
reserves `{kCausalHeadDim, heads, width, splits*batch}` FP32 plus two more `{heads, width, splits*batch}`,
so with `head_dim` 256 and `num_attention_heads` 24 (`docs/research/dflash2-acceptance-baseline.md:308`)
that is `8 × 8192 × 24 × (4·256 + 8) B` = **1.62 GiB** of the 2.246 GiB workspace, held for work that
was never short of occupancy: `ctas = 24 × 64 = 1536` query tiles against a threshold of
`0.8 × 170 = 136`. The comparison is not marginal.

**The fix adds the missing term**, and it is the one FA states: 1 split once the query tiles already
fill 80% of the SMs. Below that threshold the wave-balance search still runs, because there the splits
are what fill the machine.

| | before | with the guard |
|---|---:|---:|
| runtime, DFlash2 lane | 11.6 GiB | **10.3 GiB** |
| free VRAM | 1.50 GiB | **2.83 GiB** |
| prefill @ ~45k, interleaved, 3 rounds/arm | 10,482 tok/s | **10,839 tok/s (+3.4%, disjoint arms)** |
| prefill @ ~131k, interleaved, 3 rounds/arm | 6,462 tok/s | 6,444 tok/s (−0.3%, overlapping arms) |
| acceptance / digest | 63.0% / `c04e7c1d` | **63.0% / `c04e7c1d`** |

So it returns 1.3 GiB per lane **at no prefill cost**, and slightly positive at moderate depth.

Two things about how that was measured are worth recording, because both would have produced a wrong
answer:

* **A single sample said the guard cost 3.6–12.4% prefill.** Three interleaved rounds per arm said
  +3.4% at one depth and −0.3% at another. This card's decode figures drift several percent between
  windows; the prefill column was the repeatable one (0.4% within-arm) and the single-sample reading
  was pure window.
* **Both arms had to be in one binary.** Rebuilding per arm cannot be interleaved, so the guard's
  threshold was temporarily readable from the environment, the two arms were measured alternately
  with the same executable, and the probe was then removed before anything was committed. The
  permanent form is the constant, and the probe is gone.

**Qualification.** All five KV storage types pass the FP64 oracle (`ninfer_softmax_attention_test`,
bf16 / int8-g64 / fp8 / nvfp4-g16 / k8v4, plus packed and context attention), and all eight lanes
return byte-identical digests with unchanged acceptance. The change alters the split count, which
alters FP32 reduction order in principle; with 1 split there is no reduction to reorder, and the
digests confirm it empirically.

### What the tracker already decided, and why its justification does not transfer

Upstream issue **#277** ("Raise default `prefill_chunk` 1024 → 2048", open) measured the same lever and
reached the opposite direction from 4096: every chunk boundary costs about 10 ms, so **larger** chunks
are faster, and chunking is **math-neutral** — identical greedy token across 512/1024/2048, which
independently corroborates the byte-identical digests measured here at 4096 and 8192. Its case for not
going further was that "workspace grows linearly with chunk size and stays small vs model weights
(~15 GiB)" — but that was measured on a **2332-token prompt**, where the workspace peak was 76–305 MiB.
At this product's native 262,144 the same reservation is **2.246 GiB**, because the plan sizes splits
against `max_visible_keys` at full capacity. The conclusion does not carry to a lane that actually
serves the native context, and the −2.3% measured here is consistent with #277's direction: more chunk
boundaries cost time. 8192 is the right shipped value on both counts.

So the change worth making is not `--prefill-chunk` and not a revert of the split budget. It is the
saturation guard above, which lands in `mxfp8_tiled_partition` and returns the memory **without**
paying prefill for it — unlike this flag, which pays prefill and leaves the split count untouched.

## #217: not needed, and the arithmetic agrees

Upstream **#217** ("Concurrency ceiling 8 is too low for large-VRAM GPUs") measured aggregate
throughput on a 96 GB RTX 5090 running Qwen3.8-27B at 600 tok/request: **429 tok/s at C8, 556 at C16,
~732 at C32**, flat or collapsing at C64, with zero failures. Its audit found the cap is
**validation-only** — every kernel takes batch as a runtime grid dimension, and the ceiling is
`kMaximumConcurrency` sizing `std::array` members plus scattered `batch > 8` checks. It was closed
COMPLETED on 2026-09-09. **It never landed here**: `include/ninfer/types.h:20` still reads
`kMaximumConcurrency = 8`, and `git log -S` over every ref shows only the original value ever touched
it.

**It should not land here.** Concurrency 1 is this product's operating point by decision. The lanes
are single-request latency configurations, and a second in-flight sequence divides the same device
between two requests rather than making either one faster. Higher concurrency is an
aggregate-throughput lever, and aggregate throughput is not what these lanes are for.

Two independent findings support leaving it, which is why this is settled rather than merely declined:

* **The gain is bounded and then reverses.** vLLM on the same class of card goes 101 tok/s at c=1 to
  2,907 at c=50 and saturates there; *"adding more users yields no throughput gain — marginal TPS
  actually regresses at c=128"*, and raising `max_num_seqs` from 64 to 128 was measured to hurt *both*
  latency and throughput through scheduling contention. A ceiling raised to 64 would not buy 64.
* **At the native context it is arithmetically unavailable anyway.** vLLM V1 documents that it
  *"reserves KV / CUDA-graph capacity per in-flight slot in proportion to `max_model_len`"*. On this
  card one 262,144-token slot costs 8.681 GiB of sequence arena, against 17.213 GiB of weights and
  1.599 GiB of workspace plus graphs. Two slots need 17.36 GiB of KV where roughly 10.5 GiB is
  available, so C=2 at 262,144 is short by about 6.9 GiB whatever the validation constant permits. C=2
  would cap context near 158k — a direct trade against the context this product ships.

A concurrency-2 startup was **not** attempted. The product decision settles the question, and spending
a GPU measurement on a configuration that will not ship would be measuring something already decided.

One related caution from the same research, recorded because it bears on the `--prefill-chunk` work
above: a vLLM write-up reports that raising the prefill chunk to 8,192 *"silently disabled the draft
model's CUDA graph. Decode collapsed from roughly 109 to 19 tok/s while speculative acceptance still
looked healthy."* These lanes ship 8,192 and measure 339–362 tok/s with acceptance unchanged, so this
product does not hit that interaction — but acceptance alone would not have revealed it, which is why
the decode column is measured rather than inferred.

The prefill cost is **-2.3%**, from three interleaved rounds per arm: 13,102/13,075/13,054 against
12,780/12,799/12,732. The two clusters are disjoint and the within-arm spread is 0.37% and 0.52%. An
earlier reading of this as "unchanged" came from two samples per arm and was wrong; it is a real cost,
just a small one. Decode is unaffected — 232 against 232 in the same rounds.

**This corrects the claim above that the growth bought no throughput.** That conclusion came from a
400-token decode comparison, where the split budget barely binds (splits ≈ 2) and the SM count cannot
show up. What the growth buys is *prefill*, and a short-context decode measurement cannot see it. The
general form: a memory change sized by a width that only wide operations reach will look free on any
benchmark that never reaches that width.

At 4096 the attention workspace is 1.130 GiB, below the 1.354 GiB the same card showed on the
pre-merge binary at 8192 — so more headroom was available without reverting `e621c7d6`. That is why
the split rule, not a revert, is the interesting target.

**The change is output-preserving.** Three lanes were measured at both chunk widths:

| lane | 8192 | 4096 |
|---|---|---|
| QUASAR MTP5 | 183.5 tok/s, 43.0%, `4b7c11dd` | 184.9 tok/s, 43.0%, `4b7c11dd` |
| NVIDIA DFlash2 | 323.9 tok/s, 56.2%, `558e4ba6` | 319.9 tok/s, 56.2%, `558e4ba6` |
| NVIDIA MTP4 | 202.5 tok/s, 53.1%, `26331348` | 208.5 tok/s, 53.1%, `26331348` |

Identical acceptance and byte-identical digests on all three, which is the control that matters:
acceptance is draft-versus-target agreement and cannot depend on how a ~50-token test prompt is
chunked, so an acceptance difference would have meant the change altered output rather than cost
time.

**The recorded table did not reproduce, and this is how it surfaced.** Re-measuring all eight lanes
moved QUASAR MTP5 from 250.4/60.5% to 184.9/43.0% and both NVIDIA lanes down 5-9 points. Those
figures were **not** caused by the flag — the controls above hold acceptance and digests fixed — and
at 8192 QUASAR MTP5 reads 183.5/43.0% and NVIDIA DFlash2 323.9/56.2% as well. Three recorded
generations of figures on these lanes had already been withdrawn for not reproducing. Two consequences
are recorded in `profiles.py` rather than left implicit: the QUASAR and NVIDIA **draft-depth choices
were made from the stale population and have not been re-measured since**, and at 8192 *every*
DFlash2 lane read 11.6 GiB with 1.51 GiB free, not the two that were believed to be thin — QUASAR and
NVIDIA looked comfortable only because their rows still carried 2026-09-24 numbers.

## Depth: 262,144 has been shown reachable, and now fast enough to name

The ceiling figures above are startup acceptance. What a request at that depth actually costs was
never measured, so it is measured here, on Swift 1.5 DFlash2 d7 + Vision at the **shipped**
`--prefill-chunk 8192`, fresh text per depth so no request is answered from the context cache, depth
read from the engine's own `prompt_n`:

| prompt depth | prefill | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 3,068 | 0.22 s | 13,705 | 215.1 |
| 45,097 | 4.21 s | 10,706 | 194.7 |
| 131,172 | 19.95 s | 6,574 | 188.6 |
| **248,839** (95% of the ceiling) | **58.49 s** | 4,254 | **119.0** |

So a request near the ceiling spends about a minute in prefill before the first token and then decodes
at roughly 120–190 tok/s. **262,144 is a capacity claim, not a latency claim**, and this table is the
latency one.

The same sweep at 4096, for the decision above: 11,868 / 10,463 / 6,468 / 4,201 prefill tok/s at the
four depths, so 8192 is faster at every one (+15% shallow, +2.3% at 45k, +1.3% at 249k). The decode
column is a single sample per depth and is not used for any decision here — this card's decode figures
drift several percent between windows with host load, which is why the depth work above reports
prefill, which is GPU-bound and repeatable to 0.4%, and why the lane table's decode comes from three
runs through the harness instead.

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
| DFlash2 + Vision context | 240,000 (refused at 262,144) † | **262,144** |
| DFlash2 + Vision free | 1.41 GiB † | **1.51 GiB** |

† **The two Q8-draft figures in this table are superseded and were stale for as long as they stood
here.** Both were measured 2026-09-30 and both were still quoted afterwards, including as this file's
own justification for the encoding choice. The tiled saturation guard changed the runtime memory
picture and neither was re-measured. Re-measured 2026-10-04 with `verify`, both arms minutes apart on
the q8draft image already in `_superseded`: the Q8 build **SERVES at 262,144 with Vision** on 2.05 GiB
free, against this table's 240,000 and 1.41 GiB.

**The encoding choice survives the correction, on better grounds.** On the `code` domain — the default
`verify` domain, and the one the `code` row of the table below was taken on — NVFP4 is not merely the
build that fits; it is better on all three measured axes at once:

| `code`, DFlash2 d7, Vision | Q8 draft | NVFP4 draft | change |
|---|---:|---:|---:|
| context served | 262,144 | 262,144 | — |
| free VRAM | 2.05 GiB | **2.83 GiB** | +0.78 GiB |
| decode tok/s | 290.3 | **348.3** | **+20.5 %** |
| acceptance | 49.3 % | **63.0 %** | **+13.7 pp** |

Those throughput and acceptance figures reproduce this file's own interleaved `code` row (288.4 /
49.3 % and 349.4 / 63.0 %) to within 0.7 %, on a re-run five weeks later and a different day, which
is the reproducibility check the original numbers were missing.

**This does not generalise, and the table below is why.** On `chinese` the ordering reverses: Q8 is
11.7 % faster and 4.6 points better accepted. So the honest statement is that NVFP4 wins the `code`
domain outright and loses `chinese`, exactly as before — what the re-probe removes is the *capacity*
argument, not the per-domain split. A build that is faster, more accurate and smaller on one domain
is still not the better artifact everywhere, and nothing here should be quoted without its domain.

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
therefore not "which is faster" but "which defect is acceptable". This paragraph used to name the Q8
build's defect as "a lane that cannot start at the context the product advertises" — that was the
capacity claim, and the 2026-10-04 re-probe falsified it, since the Q8 build now starts at 262,144 with
Vision. What remains is a pure per-domain quality trade with no capacity wall behind it. Shipped as the
NVFP4 draft: it is the stronger arm on `code`, which is the documented default domain, and the weaker
one only on `chinese`. Per-domain figures here rather than a single headline.

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
no-exception image at 262,144, and `start_ninfer_v3_mtp4_vision` keeps the BF16-exception image at
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
