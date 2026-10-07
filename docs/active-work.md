# Active work — current as of 2026-10-05

**This file is temporary.** It is the single current record of work in progress. Per `AGENTS.md`,
remove it when the list is empty; do not grow it into a roadmap and do not add a parallel `v2`.

Everything here is measured on this product or read from source in this repository. Claims carried
over from other projects are marked, and several were checked against this tree and found not to
apply.

Context for all of it: `upstream/dev` was merged through `a012e2bc` (`beb46d83`), rewriting 88 files
of `src/ops/softmax_attention`. The merge touched **nothing** under `src/artifact`, `types.h`,
`round_buffers.h/.cpp` or `tools/convert`, and no container version changed, so **the four shipping
artifacts do not need rebuilding**. What the merge invalidated is the recorded performance table.

---

## The item list, in order

**Read the heading before the body: most of these are settled.** The list is kept in its original
order and its original numbering, because a settled item is a record of how a question was answered
and renumbering them would break every cross-reference to it. Each heading carries its own status
(`DONE`, `ANSWERED`, `CLOSED`, `RESOLVED`, `NOT RUNNABLE`).

**As of 2026-10-04 the work outstanding is:**

- **item 5, residue** — the instrument is built and it refuted the item's premise. **Both residue items
  are now closed.** (a) ~~`profiles.py` records QUASAR at 55.0% acceptance while the bench measures
  21.1%; they are not the same measurement~~ — **closed 2026-10-04**: not a workload mismatch at all,
  but a **drafter-width mismatch**. 21.1% is `--draft-tokens 4` exactly; `profiles.py` records width 7,
  where the bench measures 40.9%. Acceptance on this artifact swings 8.3-59.3% across widths 1-15
  and is non-monotonic, so the figures were never in conflict. See `:310`. (b)
  ~~`request_log.cpp` publishes only `accepted_per_position`~~ — **closed 2026-10-04**; the three
  drafter-recall counters now reach the serving path. See `:320`.
- **item 12** — **nothing remains open.** All nine `PORT-DISPATCH` gates were ungated 2026-10-02,
  each on its own per-route oracle; every affected file carries `UNGATED 2026-10-02`. The bullet
  below this line read "ungating is a separate performance decision" until 2026-10-04, which the
  item's own `RESOLVED` block at `:604` had already contradicted.
- **item 14** — **both halves resolved.** Text half: not a first-request effect but ADR-0002's
  documented cross-configuration sensitivity. Speed half: measured on a serving lane and retired
  (`:1724`). The bullet below this line read "nothing in the literature explains it", which is what
  the item said before it was measured.
- **item 15** — **RESOLVED 2026-10-05, gate green.** The suite had five failures
  `test_baseline.json` did not record, under a note claiming "Green since 2026-09-19". All five are
  fixed: two were lost or half-landed port work, one was a real defect beside a stale golden, one was
  a cascade, and the last was a race in the test's own read. `140/142 passed` with only the two
  baselined failures, and `check_test_baseline.py` reports no regression.
- **item 16** — **CLOSED 2026-10-06.** The `b9114396` merge's silent takes were re-derived with a
  corrected instrument and every reproducible candidate classified; five real losses were restored
  (the frontier ledger and the engine settle order earlier, then prompt preparation's host phases, the
  `TextCallConfig` refactor, the reporter profile, the five model cards and the port's `README.md`),
  and the last one -- the #251 reclaim -- turned out to be a policy difference in the replacement
  cache rather than a missing path, measured at both workload shapes and adequate for the shipped
  configuration. The counts this item first recorded (40 taken / 53 kept / 19 gone) are not
  reproducible and were dropped.
- **item 17** — **the suite runs 3.0x faster; the remaining 1.6x is a coverage decision, not a
  mechanical one.** 1430.9 s → 471.0 s. Splitting `context_kv_materialize` would take it to about
  300 s, but that test's workspace check is a whole-sweep postcondition and cannot be relocated
  without changing what it asserts.

**That summary was wrong on all three counts until 2026-10-04**, and the reason is worth keeping:
the three bullets were written when the items were open and never revisited, while the items
themselves were annotated in place as they closed. A hand-maintained summary asserting what is open
is worse than none, because a reader trusts it over the item it indexes. **To decide what to work
on, read the item headings, not this list.**

Two tests fail permanently and are **not** outstanding work: `ninfer_qwen3_5_dflash_real_test` (the
route needs a `dflash` component; this product ships `dflash2`) and `ninfer_qwen3_5_moe_real_test`
(no 35B-A3B MoE checkpoint exists for this product). Both are **baselined** in
`tools/release/test_baseline.json` rather than skipped, so the gate stays truthful about them.

Deferred items living outside this file: `docs/research/server-open-items.md` and
`docs/research/issue5-deferred-findings.md`.

Everything else below is closed, answered or negative, and is kept because the reasoning is the
durable part. The header date was `2026-09-28` until 2026-10-01, when it was corrected: the file's
most recent item is dated 2026-10-01 and the old header under-dated its own contents by three days.

### 1. Perplexity regression on a real artifact, post-merge — **DONE 2026-09-28**
**Why:** 88 attention files were rewritten wholesale. The suite passes, but the suite does not check
output *quality* on a real model, and nothing had confirmed quality survived the rewrite.
**Result:** all four shipping artifacts re-measured, full corpus, fp8, 1,044,876 tokens. QUASAR
**4.997441** against a recorded **4.99744** — identical to every digit. The reorganisation is
output-neutral, which is a stronger result than the +/-1% band required. The other three lanes moved
+0.15 % to +0.31 %, but their baselines predate the earlier `e31bc99b` merge and QUASAR's does not,
so that drift belongs to the older baseline and must not be cited as an attention regression without a
same-day control. Recorded in `docs/perplexity-baseline.md`.

### 1b. Can a full-NVFP4-coverage artifact be built from NVIDIA's source? — **ANSWERED 2026-10-01: no**
**Why:** the `nvfp4full` lane is built from `Qwen3.8-27B-NVFP4-unsloth`, a community quantization,
and scores **5.002854** on 2026-09-28, **5.002854 -> 4.998419** on 2026-09-29. The `nvfp4nvidia` lane
is built from NVIDIA's ModelOpt output — which is the official stock, its 4.90168 sitting against the
official 4.90169 on the same protocol — and scores **4.915181** on 2026-09-28, **4.911188** on
2026-09-29. That is a **1.75 % quality gap** on 2026-09-28 and **1.78 %** on 2026-09-29, on the same
protocol, on the same nominal model. Both re-measurements are on the build carrying the `d44ab584`
merge and are in `docs/perplexity-baseline.md`; the per-domain breakdown there shows the same four
artifacts spanning 6.4 % on `chinese_reference` against 1.8 % overall, so this gap is a statement
about the aggregate and understates the disagreement.
**Measured from `Qwen3.8-27B-NVFP4-nvidia/.quant_summary.txt` this session, not inferred.** The file
declares 658 weight quantizers: **401 active, 257 disabled.** The active ones are two different
formats, not one:

| active weight quantizers | count | format as declared |
|---|---|---|
| `text` attention projections | 208 | `TensorQuantizer((4, 3) bit fake per-tensor amax=… calibrator=MaxCalibrator quant)` — **INT4, per-tensor amax** |
| `mlp` projections + `lm_head` | 193 | `StaticBlockScaleQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'static', 'scale_bits': (4, 3)} … NVFP4MSECalibrator)` — **NVFP4** |

The disabled 257 are 144 attention, **112 vision tower**, and `embed_tokens`. The 144 attention ones
are exactly the 48 `conv1d` sites plus GDN `in_proj_a` / `in_proj_b` — the projections this port
already exempts from the local NVFP4 re-encode. In the `nvfp4full_noex` recipe that exemption is
`recipe.assign(name, source=model.source(name, quantized))` at
`tools/convert/official_recipes.py:521-523`, taking them unencoded from the source rather than
re-encoding them; the `recipe.separate()` call of that name appears in the two `qwen3_6_27b` recipes
(`official_recipes.py:72` and `:138`), not in this one. **This text previously said
`recipe.separate()`s at `:521-523`, which is the wrong call for those lines.**

**So the answer is no, and the reason is structural rather than a matter of the converter.** NVIDIA's
own attention is INT4 with a per-tensor scale, not NVFP4 block-scaled at all, and its vision tower
and embeddings are left unquantized. Full NVFP4 coverage is not something NVIDIA's checkpoint
*permits*; it is something **this port adds**, by re-encoding every non-MLP text projection through
`nvfp4_maxabs` at `official_recipes.py:534-540` and assigning the vision tower grouped-integer
formats at `official_recipes.py:24-39`.

**The lane name is therefore accurate, and the reasoning it rests on is now written down here so the
next reader does not have to rediscover it:** `nvfp4full` names a property of the *port's recipe*, not
of the community checkpoint it starts from. The unsloth source was chosen because its MLP layers
already carry usable NVFP4 encodings for layers < 56 (`import_encoded`), which is what makes the
lane's 4.998 possible at all; the remaining coverage is re-encoded locally. Swapping the source to
NVIDIA's would **lose** NVFP4 coverage on 208 attention projections, not gain it.

This also settles why `nvfp4nvidia` scores better (4.911 against 4.998) without that being a
contradiction: it is a different artifact with different coverage, and the gap is not attributable to
coverage alone.

**Done when:** satisfied — the coverage was read, not tested.

### 2. Give the local NVFP4 encoder a scale search instead of max scaling - **ANSWERED 2026-09-28, negative**
Built as `nvfp4_mse` (`900a0f78`, reverted `b891e1e9`), wired into the 128 object groups
`qwen3_8_27b_nvfp4_nvidia` re-encodes locally, and measured paired against the shipping artifact in one
window on the full corpus. **Max-abs 4.915181334, searched 4.925917194 — +0.218 %, the wrong way.** The
two builds share a `prefill_signature`, a format set, an `execution` block and a corpus, so only weight
values moved, and the max-abs run reproduces the recorded 4.915181 to every digit. Weight
reconstruction error fell 40-66 % while perplexity rose, which is the finding: per-weight MSE and
output perplexity are decoupled, because the blocks the search improves are the ones it clips, and the
clipped value is carrying signal.

The premise that motivated this was misattributed, and that is the more useful half. The 193
`calibrator=NVFP4MSECalibrator` sites in the source's `.quant_summary.txt` are 192 `mlp.*` plus
`lm_head` — which `qwen3_8_27b_nvfp4_nvidia` already **imports verbatim**, producer codes and scales
included. The 128 object groups re-encoded here are the `self_attn` (128) and `linear_attn` (288)
sites, which the producer quantized as **FP8** with plain `MaxCalibrator`. The two sets are disjoint:
there was never a searched-scale artifact to match for the weights this experiment changed.
`hessian` and `local_hessian` appear zero times, so the producer did use the plain squared-error path
and the objective was the right one to test. What survives is that **max-abs is the right default for
weights with no producer evidence behind them.** Full write-up, the grid difference from ModelOpt's 126
log-spaced candidates, and what the result does and does not scope, in
`docs/perplexity-baseline.md`. The instrument is recoverable from `900a0f78`; the tree carries no
unused method.

**The activation scale is the better-motivated axis, and it is untouched — item 10.**

**Carry forward:** any converter change justified by lower weight error is measured on perplexity
before it is believed. This is the same rule as "a cost you can compute is not a cost you have
measured", pointed the other way.

### 3. The W8 endpoints are 2.52 GiB, 14.3 % of everything bound — **CLOSED 2026-10-01: not warranted here**
**The memory argument died with the context goal.** The item was worth 1.26 GiB "the same order as the
whole spread between lanes that reach 262144 context and one that does not" — and as of this session
**all eight lanes reach 262,144 natively**. There is no longer a spread to close, so the 1.26 GiB buys
concurrency headroom, which concurrency 1 declines by construction. This is the same closure item 7
reached, and for the same reason: with the context goal met, memory on this product converts into
nothing that is wanted.

**A false claim about the field has to be corrected here, because it reads as settled practice.**
The item asserted Q8 is "the right default". It is not the convention: across ~90 surveyed
quantized checkpoints **Q8 appears nowhere**, and the field splits three ways — BF16, FP8 and NVFP4.
Measured in this tree, the endpoints are indeed `q8_g32_fp16` on both shipping artifacts
(`qwen3_8_27b_nvfp4full_noex`, `qwen3_8_27b_nvfp4swift15`) — read off each artifact's own
`.conversion.json`, where `text/token_embedding` and `text/output_head` both carry
`"format": "q8_g32_fp16"` — so the *format* is right, but the *justification* was not.
**The line citation this used to carry, `official_recipes.py:512` and `:518-519`, is only right for
the first of the two.** Those lines are in `qwen3_8_27b_nvfp4_unsloth_noex`, which is what
`qwen3_8_27b_nvfp4full_noex` was built from. The swift15 artifact records
`"recipe": "qwen3_8_27b_nvfp4_swift15_nvdraft"` in the same file, and that recipe assigns the two
endpoints at `official_recipes.py:717-718` and `:726-727`.
Keeping Q8 is defensible on our own measurement and on the absence of any evidence against it; it is
**not** defensible as "the default", and a reader would have taken that as a citation.

Two things make the alternative uninteresting rather than merely unmeasured:

* **No source measures Q8 against NVFP4 or against FP8 for these endpoints.** Nothing in the survey
  bears on the trade at all, so an NVFP4-endpoint build would answer a question no baseline exists for.
* **NVIDIA's own stock quantizes `lm_head` to NVFP4** — verified here, not taken from documentation:
  `lm_head.weight_quantizer` in `Qwen3.8-27B-NVFP4-nvidia/.quant_summary.txt` reads
  `StaticBlockScaleQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'static', 'scale_bits': (4, 3)}, … NVFP4MSECalibrator)`.
  So there is now a first-party precedent for the alternative. It is not a reason to switch: the
  `nvfp4nvidia` lane already inherits NVIDIA's endpoints by `import_encoded`, so fidelity to the
  official stock is already achieved on the lane that has it, and re-quantizing the two most
  sensitive weights on `nvfp4full` would be a numerics change with no capacity benefit in return.

**Reopen only if concurrency or native context changes**, since that is the only condition under which
the 1.26 GiB acquires a consumer.
Full evidence: `docs/research/quantization-coverage-evidence.md`.

### 4. The vision tower is quantized in all four artifacts and BF16 in all three sources — **RESOLVED 2026-10-01: keep it, on a measurement**
**A precision note first, because it changes what is being decided.** The tower is *not* NVFP4. It is
grouped-integer, assigned at `official_recipes.py:24-39`: `patch_embedding` Q6, `merger/` Q8,
`attention/{query,key,value}` and `mlp/fc1` Q4, everything else Q5. Verified on the artifact rather than
the recipe — of 441 vision sites in `qwen3_8_27b_nvfp4full_noex`, **108 are Q4, 54 Q5, 1 Q6, 2 Q8, and
276 are BF16** (biases, norms and embeddings, which are not projections). So ~165 sites are quantized
and the majority of the tower is untouched. That is a materially gentler position than "quantized", and
it is what the ~600 MiB estimate covers.

**The external evidence has shifted against quantizing it, and it is now the strongest of the four
remaining quantization items to act on:**

* **94 of 130** surveyed VLM NVFP4 checkpoints leave the vision tower in BF16. All **15**
  llm-compressor multimodal examples ignore it. TensorRT-LLM lists **no** multimodal model under NVFP4.
* **Qwen's own official FP8 checkpoint excludes all 330 vision Linears** — the same vendor, the same
  model family, excluding them on purpose.
* **ModelOpt disables it in a shared exclusion unit, with two stated reasons and five NVBug IDs behind
  it.** That is a deliberate vendor decision with a bug-tracker history, not an omission.
* **No source anywhere measures a quantized-versus-BF16 vision tower.** The nearest thing is a third
  party's 8-image fixture, which its own author declines to call a benchmark. So this remains a trade
  nobody has published, in either direction.

**The decisive new fact: one of NVIDIA's two stated reasons cannot apply here.** One is divisibility —
and Qwen3.8's patch embedding has 1536 in-features, `1536 % 128 == 0`, so the alignment constraint that
would force quantization is simply absent. That removes the strongest available defence of the current
choice, and it was not checkable without going and checking it.

**Why it still cannot simply be reverted on this evidence:** the ~600 MiB is not needed (all lanes sit
at native context), so there is no pressure either way, and our format is grouped-integer rather than
NVFP4 — a much smaller bet than the field's BF16 majority implies. Reverting is a numerics change to
the one component **perplexity cannot see**, which is precisely why it needs its own check rather than
a vote.
**Measured 2026-10-01: KEEP the quantized tower.** The check this item asked for now exists, and it
says the risk is smaller than the risk this artifact already accepts elsewhere. `verify_artifact.py`
decoded every grouped-integer payload and compared it against its BF16 source — **564 locally-encoded
payloads verified, 0 failures**, including 165 vision-tower sites across all four formats that had
**never been value-checked before** (`check_values` filtered on `obj.format.startswith("nvfp4")`, so
every grouped site was skipped silently). Worst max-relative error per format, same artifact, same
measurement, same code path:

| format | where | worst measured | format's own bound |
|---|---|---|---|
| `nvfp4` | text stack | **0.165869** | 0.2292 (`1/6 + 1/16`) |
| `q4_g64_fp16` | vision tower | **0.071429** | 0.071463 (`0.5/qmax`) |
| `q5_g64_fp16` | vision tower | **0.033315** | 0.033350 |
| `q6_g64_fp16` | vision tower | **0.016031** | 0.016137 |
| `q8_g32_fp16` | merger, DFlash draft | **0.003930** | 0.003939 |

**The vision tower's worst format is 2.3x tighter than the NVFP4 the text stack already ships.** And
`q4_g64_fp16` measures 0.071429 against a derived bound of 0.071463 — that is `1/14 = 0.5/7` to six
places, so the Q4 encoder is hitting its half-step bound and no more, which is the strongest statement
this measurement can make. The bounds are derived from `groupwise._canonical_scale_words` rather than
tabulated: codes are `clamp(round(x/scale), qmin, qmax)` with `scale = binary16(group_absmax/qmax)`, so
a value is off by at most half a step, `0.5/qmax` of the tensor maximum.

**What this does not prove.** A bounded weight error is not a bounded change in vision *task quality*.
Perplexity still cannot see the tower, and no vision benchmark was run, so the residual is real
however small the weight error is. What changed is which side of the argument each piece of evidence
sits on: "94 of 130 checkpoints leave it BF16" is a convention, while 0.0714 against the 0.1659 we
already ship is a measurement of this artifact. Reverting on the first while ignoring the second would
be arguing from a headline against a number.

**So the decision is to keep it, and the thing that would change it is named rather than implied:** a
vision task benchmark (MMMU, DocVQA or ChartQA) comparing the BF16 tower against this one on the same
artifact pair. That is the only remaining evidence that would settle it, and it is a real project of
its own — not a reason to revert now.

**Two defects this work exposed, both fixed, both worth recording:**

* **The verifier's docstring advertised a check that did not exist.** It claimed "Both W8 endpoints
  against base rows ... compared against the BF16 base's rows rather than merely checked for a valid
  encoding". No such code was in the file — it described the `verify_*` entry points that
  `artifact-conventions.md` records as never having been ported. **The endpoints are still not
  value-checked**, now for a reason that is stated rather than assumed: the source factory short-reads
  at `248320 x 5120 = 1,270,998,400` elements, and `proposal/head` needs an explicit logical source no
  recipe supplies. They are reported as **unchecked**, a verdict distinct from failure, because a
  reference that cannot be read is a limit of the lookup and not evidence about a payload.
* **`_compare_site` reported the wrong attempt's reason.** It overwrote the failure reason on each
  attempt and printed the last, which for a locally-encoded site is the `quantized/nvfp4` attempt. The
  message became `missing source tensor 'embed_tokens.weight_packed'` — an artifact of passing an NVFP4
  hint to a Q8 site — while the real cause was the base store's short read. Keeping the *first* reason
  is what made this findable.

**Done when:** satisfied for the weight-encoding question. A vision task benchmark remains the only
outstanding evidence, and is recorded above as its own piece of work rather than as a condition on
this item. Full evidence: `docs/research/quantization-coverage-evidence.md`.

### 5. Recall@1 / Recall@16 / path-acceptance split — **BUILT 2026-10-01, and it refuted this item's premise**
**The instrument is in the tree** (`611604e0`): `SpeculativeStats` gains `proposed_per_position`,
`recall1_per_position` and `recall16_per_position`, accumulated in `decode.cpp` beside the
`accepted_per_position` loop that already existed. The bench CSV gains four columns per position and
JSON the matching keys. One ~1 KB device-to-host read per round after the existing synchronize; no
kernel, no Op signature, no numerics changed. `proposed_per_position` is the denominator the other two
needed — without it the rates would have been reported over an invisible base.

**This item asked for a diagnostic of a position-1 acceptance collapse. There is no collapse.** Measured
on QUASAR (`nvfp4qat`), `--spec dflash2 --draft-tokens 7 --lm-head-draft -n 256 -r 5`, 515 rounds:

| pos | proposed | Recall@1 | Recall@16 | path acceptance |
|---|---|---|---|---|
| 0 | 515 | 51.46% | 75.73% | **53.40%** |
| 1 | 515 | 41.75% | 67.96% | 33.98% |
| 3 | 515 | 27.18% | 48.54% | 13.59% |
| 6 | 510 | 22.55% | 46.08% | 4.90% |

**Path acceptance at position 1 is the HIGHEST of all seven, on both lanes, and declines monotonically
from there.** The decline with position is real and steep — 53.4% to 4.9% over seven positions — but it
is not the shape this item was written for, and every one of the three signatures below has to be
re-read against that:

- *"Recall@1 low at position 1"* — **refuted.** 51.5% on QUASAR, 45.8% on NVIDIA. It is the *highest*
  Recall@1 of any position, because the drafter's first column has the most context.
- *"Recall@16 collapsing after position 1"* — **supported.** 75.7% → 46.1% (QUASAR), 78.9% → 46.8%
  (NVIDIA). Something in the backbone or the conditioning degrades past the first column.
- *"Healthy Recall@16 with collapsing path acceptance (selector)"* — **supported, and on NVIDIA it is
  the dominant loss.** At positions 4-6 Recall@16 still sits at 44-47% while path acceptance is 0.71%.
  **The drafter is finding the right token and the selector is not committing it.** That is a
  selector-side problem, not a drafter-side one, and it is the opposite of what this item assumed was
  the interesting case.

**So the instrument did its job by invalidating the question.** The next step is a selector
investigation using the per-position path-acceptance and Recall@16 series, not more drafter work.

**Two things the instrument did NOT settle, recorded rather than glossed:**

1. **RESOLVED 2026-10-04. `tools/release/profiles.py:109` records QUASAR at 55.0% acceptance; the bench
   figure quoted beside it, 21.1%, is the same engine at a different drafter width.** This was recorded
   as a workload mismatch needing the workloads matched before the two figures could be compared. That
   diagnosis was wrong, and so was the reason given for it — `temperature` is not the cause.

   The bench has **no sampling flags at all** and cannot be moved off greedy, which is what made
   temperature look like the explanation. It is measurable, not inferential: running the QUASAR
   DFlash2 lane twice through the serving path, minutes apart, differing only in sampling, gives

   | sampling | acceptance | decode |
   |---|---:|---:|
   | model card's thinking set (temp 1.0 / top_p 0.95 / top_k 20) | **55.0 %** | 311.8 tok/s |
   | `--temperature 0` (greedy) | **66.5 %** | 363.0 tok/s |

   so temperature moves acceptance *up* by 11.5 points when it goes to zero. It cannot produce a
   21.1 % reading, and the digest is byte-identical across the two runs (`7627eae2208ddf0e`), which
   confirms the deterministic pass is correctly unaffected by the knob.

   What actually explains it is `--draft-tokens`. Acceptance on this artifact is **strongly and
   non-monotonically width-dependent**, so a figure quoted without its width is close to meaningless.
   All fifteen measured in one sweep at one configuration (`-n 384 -r 1`, QUASAR, `--prefill-chunk
   8192`, same corpus):

   | width | 1 | 2 | 3 | **4** | 5 | 6 | **7** | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | acceptance % | 54.4 | 59.3 | 23.9 | **21.1** | 14.4 | 13.2 | **40.9** | 26.1 | 23.2 | 22.0 | 20.3 | 18.6 | 17.4 | 9.0 | 8.3 |

   **21.1 % is draft width 4 exactly.** `profiles.py` records QUASAR at `draft=7`, where the same
   sweep reads **40.9 %**. The two figures were never in conflict; they were two widths of the same
   drafter, quoted without the width that distinguishes them.

   **The first version of this row mixed two configurations** — widths 1/3/7/15 came from an earlier
   `-n 512` run and the rest from `-n 384`, which is how width 7 first appeared as 49.1 %. Re-measured
   as one sweep it is 40.9 %. The finding is unaffected, since 21.1 % at width 4 reproduces in both,
   but the other fourteen figures were not comparable to each other and are now.

   The 48.7–63.0 % range recorded elsewhere in this file comes from `profiles.py` and is a
   serving-path number at the shipped width on a domain prompt — a different corpus and a different
   route from the row above, so it is not to be compared against it.

   **This needed an instrument that did not exist**, so `v3_profile_matrix.py profile` now takes
   `--sampling {default,zero,none}` and records what it applied in `sampling_applied`. The default is
   unchanged, so every existing record still means what it meant; the knob exists so this question is
   answerable in one run next time rather than by hand.
2. **`src/serve/request_log.cpp` still publishes only `accepted_per_position`**, so the three new
   counters are invisible on the serving path. Widening it was outside the deliverable and is open.
   **DONE 2026-10-04.** `proposed_per_position`, `recall1_per_position` and `recall16_per_position`
   now travel the same path as `accepted_per_position`: carried on `GenerationMetrics`
   (`src/serve/generation_service.h`), populated from `SpeculativeStats` (`generation_service.cpp`),
   and published beside it in `speculative_json`. They are published as a set rather than one at a
   time because `proposed` is the denominator for all three rates — a reader given
   `accepted_per_position` alone cannot compute one. `ninfer_request_log_test` asserts each against
   its own distinct values, and a probe emitting recall16's array under the recall1 key fails it, so
   the assertions are not satisfied by key presence alone.

   **What this does NOT do:** it does not make the two acceptance figures comparable. Item 1 above is
   a workload-matching problem and is untouched by this — `profiles.py` still takes acceptance from
   serving request logs on a domain prompt while the bench uses `bench_corpus.ids` at temperature 0.

**Verification:** interleaved A/B, 4 passes each, instrumented vs a build with the read removed, gave
acceptance `0.211111` on both arms to every digit. A baseline build emits 0/empty for all 28 new
columns. A negative control that swapped recall@1 and recall@16 in the CSV row made the new test fail
at all three positions. `path <= recall@16` holds at every position on both lanes, which a mis-indexed
candidate column would break. Overhead is below this card's ~5% noise floor and is **not** quantified —
the instrumented arm measured nominally faster, which is noise, not a speedup. Column semantics were read
rather than assumed: `candidate_ids[c,i,b]` indexes against `target_argmax[i,b]` with no offset
(`speculative_round.cuh:141,265,505,631,669` and `:179-180,200-203`), and `target_argmax` is written by
`ops::argmax` (`text.cpp:769`), so it is the true argmax and not a sampled token on either lane.

**Original framing, kept because it is what the instrument answered:** the only diagnostic that
discriminates three root causes, needing no new kernel. The drafter emits `frame.candidate_ids` and
`scores` at `draft.cpp:351-355`, shape `[16,K,B]` (allocated as `{16, columns - 1, batch}` at
`round_buffers.cpp:203-204`). Decompose per position: Recall@1, the drafter's unary top pick;
Recall@16, the target argmax anywhere in the 16; path acceptance, what the selector actually commits.
**Done when:** satisfied — all three are reported per position on shipping lanes. What the data then
says is above.


### 6. Interleaved re-measurement of all eight profiles - **DONE 2026-09-28**
All eight re-measured in one interleaved window, three rounds, each round visiting every lane in a
rotated order. Every lane's three rounds span **1.5 % or less**, each returns **one digest** across all
three, and there is **no position effect** — position 0 and position 7 read the same. `profiles.py`,
both doc tables and the eight launchers' `REM` lines now carry the new figures.

**The cause was a harness defect, not the lanes, and not a contaminated card.** The 52 % spread that
invalidated the NVIDIA MTP5 row was reproducible to within 1 % across four independent server starts,
which is the opposite of contamination. The first full-length decode after a start is a transient: it
returns faster than every later identical request and **different, shorter text**, while requests 2..n
are byte-identical. On that lane it read 261.7 tok/s against 169.8 for requests 2-7. `measure_decode`
warmed up with a **16-token probe**, which does not reach the state the transient affects, so the
transient was averaged into the first measured run. Acceptance was contaminated the same way, and
because the transient accepts *better* (64.6 % against 32.7 %) it inflated that column too: the same
log reads 35.3 % over all records and 32.7 % without the first.

Fixed by discarding one full-length warmup, and by skipping it in `parse_spec_jsonl` because the engine
appends to its request log for the whole session. Both are in this commit. An earlier revision of the
table was **inflated on all eight lanes** — the largest correction is NVIDIA MTP5 at 228.3 → 165.7
tok/s and 56.4 % → 38.2 %, which is the row that was already suspected.

**What was not changed:** no lane, flag, context, draft depth or artifact. Only the four measured
fields moved in `profiles.py`, and the launchers were regenerated rather than hand-edited.

**RESOLVED 2026-10-02, in item 14: this was not a first-request behaviour.** Measured at temperature
0.7 with the seed pinned, a client's first request is byte-identical to every later one, in both
prefix-reuse arms and across three process launches. What the original observation compared was two
bench *runs* at different `--warmup`, which is a configuration comparison — see item 14 and
[ADR-0002](adr/0002-speculation-not-bit-identical.md).

### 7. k8v4 KV against the shipped fp8 — **CLOSED 2026-10-01: not warranted here**
The only thing that would justify a KV size reduction on this product is context, and context is
already at the ceiling. Upstream caps Qwen3.8-27B at `kNativeContext` = 262,144 and states outright
that a smaller KV "does not change max context" (issue #123, verified against the tracker while
closing this). What a 1.78 GiB saving at native context (fp8 8.06 → 6.28 GiB, −22.1%, measured in-tree
and recorded in `docs/research/kv-dtype-evidence.md`) actually buys is **concurrency headroom** — and
concurrency 1 is a product decision for these lanes, so that value is declined by construction.

Two further findings make the comparison moot rather than merely unattractive:

* **No published k8v4-vs-FP8 quality measurement exists.** The one first-party measurement
  (vLLM #38479, merged 2026-04-15) compares against an *unquantized* baseline — GSM8K 0.860 against
  0.900 — which is a loss, not parity, at roughly 1.9σ on 200 questions. Too small to establish or
  exclude an effect, which is the same structural defect that killed the withdrawn "0.08 %" figure.
* **The format measured is not the format this tree ships.** vLLM's `turboquant_k8v4` is unrotated
  FP8-E4M3 K with no scale plane plus per-vector uniform 4-bit V. This tree's `k8v4` is
  `Fp8KeyNvfp4Value`: FP8-E4M3FN-**row256** K, Hadamard-prepared with a 2-byte scale, plus NVFP4-G16 V.
  The slot sizes differ and match vLLM's published 196 B exactly, so **no external number transfers**
  even if a sound one existed.

Ecosystem support is vLLM-only: zero code hits for `k8v4` or `turboquant` in TensorRT-LLM, ModelOpt,
SGLang or FlashInfer, against a live control query. vLLM's own tracker leaves "publish recommended
config table" unticked, and users report it "does not work with modern models" and that **MTP ×
TurboQuant produces degenerate token loops** — which is this product's shipping configuration.

**This closes the investigation, not the option.** `k8v4` stays selectable and fully implemented here
(`src/ops/kv_cache/append/k8v4_kernel.cuh`, `k8v4_launch.cu`). Unlike item 9's `PromptLookup`, this is
a working format behind a flag, not a ported component whose consumer will never exist — so it is not a
withdrawal candidate, and nothing in this row should be read as a reason to remove it.

Full evidence, including what was searched and not found and one published table excluded as
non-discriminating, in `docs/research/kv-dtype-evidence.md`.

**Why it was once open, and why that reason no longer holds:** every profile ships `--kv-dtype fp8` and
k8v4 sat behind one flag. The item's own premise — "**no published evidence exists for k8v4**" — has
since been **confirmed and extended**: no evidence exists, and the one first-party measurement that
does is of a different format. Its "Done when" was a perplexity and acceptance comparison across
bf16 / fp8 / k8v4 on a shipping lane, interleaved. That comparison is **not performed**, and the
closure above is the reason: it would settle a question whose answer cannot change what this product
ships. The withdrawn "within 0.08 % of BF16" figure — traced to NVFP4-KV against FP8-KV on
Qwen3.5-397B-A17B, a different model on a different baseline, not k8v4 at all — stays withdrawn.

### 8. MTP draft window 5 to 10 — **NOT RUNNABLE ON THIS TREE, closed 2026-09-30**
The engine refuses to start above 5. `src/models/qwen3_5/program/planning/startup.cpp:798` raises
"MTP draft window must be in [1,5]" against `kMaximumMtpDraftTokens = 5`
(`src/models/qwen3_5/program/internal.h:11`), so windows 8 and 10 cannot be
requested at all — the item's comparison is not a measurement this tree can make. **Raising that cap
is the only way to run the item**, and nothing here argues for it.

What *is* measurable was measured the same day on both Swift lanes: depths 1-5, interleaved, two
rounds, on five domains (`docs/research/swift15-lane-measurement.md`). Depth 4 is fastest on prose,
dialogue and repetition and depth 5 fastest on code; depth 4 is the choice that maximises the worst
domain, which is what both lanes already ship. Depth 5 is **not** uniformly worse — it is 54 % faster
than Swift 1.0's depth 5 on code, because Swift 1.0's MTP head degrades sharply past depth 4 while
Swift 1.5's holds. Any future raise of the cap should start from that, not from a tok/s table.

### 8b. DFlash2 draft window, all 1-15, on five domains — **DONE 2026-09-30**
Every window each backend accepts, swept interleaved with rotation so no configuration keeps a
position, reported with acceptance, tokens per round and the per-position acceptance profile.
`speculative_accepted_per_position` was already in the request log and nothing reported it; the
`widths` mode now does, which is how the shape of the loss at wide settings became visible.

The finding that matters is that **the code domain alone reverses the width decision**, which is
what this harness's own `DOMAINS` comment predicted: d13 is 33 % faster than the shipped d7 on code
and slower on chinese, prose and dialogue. d7 wins three domains of five. Full table and the
per-position profiles are in `docs/research/swift15-lane-measurement.md`.

### 9. ngram: the verify-tree integration — **CLOSED 2026-10-01 by product decision**
**Closed on the maintainer's own evidence, not on an argument from this file.** The copy-drafting
combination was tried against different LLMs and did not work out. That settles it: the research
below, the break-even arithmetic and the "measure `p` of the copied token" framing are all moot, and
the verify-tree integration is not going to be built.

Retained only because `PromptLookup` was in the tree with no caller — it said so in as many words,
`NOT YET WIRED`, and it was a ported component whose consumer will not exist. **That withdrawal
happened 2026-10-03**: `prompt_lookup.{h,cpp}` and `ninfer_prompt_lookup_test` are gone, along with
the `--ngram` surface that would have consumed them (see "Also worth doing, small"). Do not read this
row as an argument for finishing it.

**Why it was once open:** `PromptLookup` is ported and tested (`c0da270e`); the integration is not built. It needs
`candidate_selector_tree`, `speculative_accept_tree_drafts`, `speculative_compact_columns`, tree-aware
GDN replay and tree-aware target attention, each with a host oracle.
**Expected value is low, and that is the finding, not a reason to skip it:** the selector's value is
inversely proportional to drafter strength, and our shipping DFlash2 lanes accept **48.7%–63.0%**
(`tools/release/profiles.py`: NVFP4-full 48.7%, QUASAR 55.0%, NVIDIA 56.2%, Swift 1.5 63.0%; this
figure previously read 52-67% here and 52-69% in the closed-items table, and neither matched any
recorded lane). The +55%
reference was measured on a lane accepting 27%.

**The research is already done and is better than anything I would add here.** Read
`docs/research/ngram-copy-selection-signals.md` before starting: it establishes, from the
implementations, that TensorRT-LLM's `SADraftEnhancer` is the *only* shipped copy-vs-neural selector
across llama.cpp, vLLM, SGLang and TensorRT-LLM, that vLLM's combination is still unmerged, and that
`use_sa_spec` is wired into exactly MTP, EAGLE 3 and PARD -- so **DFlash2 has no shipped precedent for
this combination**, which is the reason the item is worth building rather than adopting. It also
derives the break-even copy-acceptance rate (`c = 0.290`). Two of the three sibling notes are
`ngram-drafting-designs.md` and `ngram-outside-github.md`.

**The one thing that research note does not have, and it changes what to measure.** Its break-even
leaves `c`, the copy-acceptance rate, as a free parameter and notes that no published number exists
for the match-length gate's false-positive rate. `c` is not free. A deterministic copy proposal is a
point mass -- `q = 1` on the copied token -- so the identity verified in item 11 gives
`P(accept) = 1 - TV(p, q) = p(copied token)`. The copy-acceptance rate *is* the target's probability of
the copied token. That is worth having for two reasons: it converts the break-even from a threshold on
an unmeasured rate into a threshold on a quantity we can measure directly, and it explains why match
length is the signal TRT-LLM gates on at all -- a longer suffix match is a proxy for a higher `p`. A
match-length threshold is therefore a cheap, dataless stand-in for `p`, and the honest way to size it
here is to measure `p` of the copied token and compare, rather than to port someone else's `4`.

**One thing we should not copy.** TRT-LLM's MTP has `use_relaxed_acceptance_for_thinking`, which
accepts a draft appearing in a `relaxed_topk` / `relaxed_delta` candidate set instead of applying
`min(1, p/q)`. That is a biased acceptance, chosen to speed up the thinking phase. Our contract is
lossless, so it is out of scope however well it performs, and adopting it would forfeit the property the
digest-based control in the bench depends on.

**Done when:** one DFlash2 lane is measured with and without the copy grafted, on copy-heavy traffic,
against our own baseline. **"Built, measured, and not shipped" is an acceptable outcome** and should be
reported as one.

### 10. The A4 activation divisor at the re-encoded attention sites — **CLOSED 2026-10-01: the divisors are right, and the cited failure mode is not evidenced here**
**The reason this item was open is wrong, and it was wrong in this file.** The claim was that the
`activation_input_divisor` is "recovered from a source `input_scale` the producer calibrated for
**FP8**", so that "a divisor sized for 8-bit activations is not obviously roomy enough for 4-bit ones."
It is not. This tree's own recipe docstring records what NVIDIA actually writes
(`official_recipes.py:556-558`): NVIDIA's checkpoint stores `amax / (6 * 448)` at an **NVFP4** site and
`amax / 448` at an **FP8** site. So `6 / input_scale` at an FP8 site and `1 / input_scale` at an
already-NVFP4 site are the *same expression* — both recover `2688 / amax`, and `2688 = 6 × 448` is
precisely the factor that converts an FP8 scale into the A4 orientation. `calibration.py`'s
`FULL_RANGE = 2688.0` is that same constant. **The factor of 6 is the conversion, not a stale
bit-width**, and the two branches of the `_activation_divisor` probe agree by construction.

Verified from the source rather than from this file's own summary: NVIDIA's real
`6/input_scale` tensors were read by HTTP Range and equal `2688/amax`, matching `calibration.py`'s
rule. **What remains is a different and weaker concern:** the divisor is `amax`-derived, and
`amax` is exactly what `NVFP4ActHeadroomCalibrator`'s own docstring identifies as leaving no headroom.

Three findings bound how far that concern can be taken:

1. **NVIDIA does the same thing here.** Its shipped Qwen3.8-27B uses plain max on these sites — 401
   `MaxCalibrator` activation quantizers against 193 `NVFP4MSECalibrator` weight quantizers. The
   headroom calibrator is opt-in, recent, and carries **no published accuracy number**.
2. **Our divisors are not NVIDIA's.** NVIDIA's calibrated `amax` and this port's own disagree by up to
   **40×** at some of 24 sampled sites — different corpora, not a format mismatch. So "we already match
   the official stock" is false here, in the other direction: both are max-derived, from different data.
3. **The `mlp.gate_proj` spread quoted below is real but is not evidence of a defect.** `amax=[0.0047,
   0.4219]` is a 90× spread across blocks, which is the *shape* max calibration produces; it does not
   by itself show the small blocks are being harmed.

**Disposition: CLOSED.** Five findings, and they do not leave a defect — they remove the reason there
was an item.

1. **The derivation matches NVIDIA's own schema, stated in their source.** The exported divisor is
   `amax / 448` at an FP8 site and `amax / 2688` at an NVFP4 one (ModelOpt's `config.py`, line 684). Our
   `FULL_RANGE = 2688.0` **is** that constant, and the two branches of the `_activation_divisor` probe
   agree by construction because 6 is the conversion between them. This is no longer an inference from
   a docstring.
2. **Our static choice is what published engines do — for exactly this format.** vLLM, SGLang and
   TensorRT-LLM all read a per-tensor NVFP4 activation global scale **out of the checkpoint** and use it
   statically; only the 16-wide E4M3 block scale is computed at runtime. So the choice this item
   questioned is the mainstream one for the format the `AllowA4` sites use.
3. **The failure mode is not present in our data, and the datum offered as evidence argues against
   it.** The subnormal flush needs a block scale below the E4M3 normal floor, `2**-6 = 0.015625`
   (NVIDIA carries this as `_FP8_NORMAL_DYNAMIC_RANGE = 448 / 2**-6 = 28672`, and it doubles as the
   `rho` validation bound). This item cited `mlp.gate_proj`'s `amax=[0.0047, 0.4219]` — a 90x spread —
   as "exactly that shape". Recomputed here: that gives a smallest block scale of **4.99** against an
   FP8 global scale and **29.9** against an NVFP4 one, which is **319x and 1917x above the floor**. The
   spread is real; it is two to three orders of magnitude short of the mechanism it was offered for.
4. **No published measurement says headroom calibration is better.** There is no perplexity or accuracy
   comparison of max-calibrated against headroom/percentile-calibrated activation scales anywhere;
   every row of NVIDIA's published NVFP4 accuracy table is weight-side. The calibrator is opt-in, and
   NVIDIA's own shipped Qwen3.8-27B uses plain max on these very sites.
5. **The 40x divisor disagreement is unresolved but is not evidence of a bug in ours.** Both figures are
   the same statistic — per-tensor absmax of the same module input — so per-tensor-versus-per-block
   does not explain it, and corpus choice is not the only candidate: NVIDIA fake-quantizes weights
   *before* the activation forward (this port does not), propagates activations layerwise through QDQ,
   and differs in sequence structure, padding and token count. **The in-tree note's own arithmetic does
   not reconcile either** — "amax ~ 2092 / 5.12, a ratio of 0.025 in divisor" gives 410, not 0.025, and
   the direction is inverted relative to its own table. That figure is withdrawn.

**One observation the research turned up that this tree does not examine, recorded so it is not lost.**
The static/dynamic split runs the other way for FP8: all three engines default to *runtime-computed*
per-tensor activation scales for FP8, while this port carries 512 static input divisors. That is a real
difference and it is not obviously wrong — a static scale skips a per-token reduction, and this
artifact's perplexity is measured and good. It is an observation, not a defect, and it is not on the
critical path to this item's answer. If it is ever worth a look, the question is whether the FP8 sites'
divisors were calibrated on a corpus representative of production traffic.

Evidence, including the four competing calibrator policies and the survey behind them:
`docs/research/activation-scale-method-evidence.md`.

Retained because it is the producer's own statement of the residual risk, not an inference:
ModelOpt's `NVFP4ActHeadroomCalibrator` exists because plain max calibration of the activation global
scale "would drag the global scale up so far that every other block's FP8 block scale falls below
subnormal and flushes to zero — losing the whole tensor to protect one value"; its default anchors to
the 99.99th percentile and clips the rare blocks deliberately instead. **That is a vendor describing
what max calibration can do, on a model where they chose not to enable their own fix.** It is the
strongest form the argument takes, and it is still an argument about NVIDIA's choice, not a defect
attributable to this port.
Full evidence, including the ~90-checkpoint survey: `docs/research/quantization-coverage-evidence.md`.

### 11. The sparse accept path had no distributional test, and every sparse case used `top_k=1` - **DONE 2026-09-28**
**Found while auditing a claim of mine that turned out to be false. The premise below was also false.**

`tests/ops/test_speculative_round.cpp` compares the sparse accept device path against a host oracle
(`sparse_accept_oracle` / `sparse_target_distribution`, FP64). **Every sparse case in the file set
`top_k = 1`** — `generated_general_case` and `sparse_general_mixed_case` both did — so the multi-token
branch of the accept rule was unverified, and the production dflash2 lane sends `top_k = 20`.

**What I first reported, and was wrong:** that a case I wrote showed the device and the oracle
disagreeing at `top_k = 2`, with the device accepting the draft on 16384 of 16384 trials where the
oracle expected about 47 %. That was my throwaway harness, not the engine and not the oracle. I never
established what in the harness caused it and am not going to invent a reason.

**What is actually true, measured.** `accept_distribution_isolation_case` separates the two halves of
the path, because one comparison cannot tell them apart. With `extent` 0 there is no draft to accept, so
the kernel samples straight from `p` with no accept test and no residual; with `extent` 1 the draft is
accepted or rejected and the residual is drawn. Both must reproduce `p`. On a two-token support,
`p = (0.5737, 0.4263)`, with a point-mass `q` on the more likely token:

| sub-run | result |
|---|---|
| `extent=0`, isolates the target distribution | `chi2(p) = 3.2` against a limit of 16, 0 tokens outside the support |
| `extent=1`, isolates the accept rule and the residual | `chi2(p) = 0.3`, accept rate 2368/4096 = 57.8 % |

The accept rate is an independent check on the mechanism, not just on the output. `P(accept) =
1 - TV(p,q)` (Leviathan et al. 2302.01318 Thm 1; 2606.30265). Here `TV = 0.5 * (0.4263 + 0.4263) =
0.4263`, so the prediction is 57.37 % against 57.8 % measured — 0.57 standard errors. Their notation is
the mirror of ours: their `q` is the target and their `p` the proposal. The case is deterministic, since
the seeds derive from the trial index, so it produces identical numbers on every run.

**The lesson that outlives the fix, and the reason this was worth an item.** The `top_k = 1` agreement
I treated as proof that the harness was sound proved nothing about the accept rule at all: at `top_k = 1`
the support is a single token, so `p` is a point mass, so `p >= q` is true for every draft whatever `q`
is, and accept-always is correct. A control that passes for a structural reason is not a control. The
multi-token case is where the accept rule actually has arithmetic to get wrong, and it was the one
configuration nobody was checking.

### 12. Upstream's 14 commits are merged; the FP8 A8 TMA route is held back on Windows only - **RESOLVED 2026-10-01 (`a5077adf`)**

> **RESOLVED 2026-10-01 (`a5077adf`) — CAUSE ESTABLISHED, FIX LANDED, NINE GATES STILL IN PLACE.**
> Everything below this line describes the investigation as it stood *before* that commit and is kept
> as the record of how the cause was found. Several of its conclusions are now false and are marked
> **[SUPERSEDED]** where they are load-bearing. The short version:
>
> * **Cause:** `struct alignas(128) Fp8TmaDescriptors`. The 128 was the only reason the descriptor had
>   to be passed by pointer, and the pointer form was the fault. `tools/scripts/probe_tma_align.cmd`
>   measures this toolchain: 8/16/32/64 accepted, 128 and 256 rejected by four C2719 sites each.
> * **Fix:** `alignas(64)`, and the descriptor passed **by value on every platform**, deleting the
>   staging buffer and its per-launch `cudaMallocAsync`/`cudaFreeAsync` entirely.
> * **Evidence:** a red loop that drives the real code path — flip the nine gates, build
>   `ninfer_linear_fp8_a8_test`, run it. `FP8_A8 [14336,5120] T=129` went from
>   `cudaErrorIllegalInstruction` to `OK FP8 A8 Linear`, exit 0.
> * **Two fixes were falsified first**, one variable at a time: the `fence.proxy.acquire.tensormap`
>   per map, and a process-lifetime staging buffer. Both left the loop red. The proxy fence is
>   deliberately **not** re-added.
> * **The 2d-versus-3d theory is withdrawn.** `docs/research/windows-ports-tma-survey.md` surveyed 293
>   forks: three Windows ports build and run this exact route on sm_120a and all three keep the
>   rank-2 descriptor and `cp.async.bulk.tensor.2d`.
>
> **RESOLVED 2026-10-02 — this no longer holds.** All nine `PORT-DISPATCH` gates are lifted, each
> on its own per-route oracle, so shipping configurations HAVE changed. What follows is the
> position as it stood when the gates were still closed.
>
> **What is still open is the second half:** the nine `PORT-DISPATCH` gates are untouched, so no
> shipping configuration has changed and no lane is faster. **And the case for ungating them is now
> stronger than the case against it**, on evidence rather than on the throughput figure that first
> suggested otherwise.
>
> **The ~10x figure is not reproduced by anything published, and the obvious confound does not explain
> it.** An earlier bench put the TMA-selected token bands at ~100-125 GB/s against ~1110-1316 GB/s for
> the MMA bands. Three findings dissolve that as a reason to keep the gate
> (`docs/research/sm120-tma-fp8-throughput-evidence.md`):
>
> * **No apples-to-apples comparison exists to lose to.** CUTLASS ships **no non-TMA sm_120 mainloop
>   at all** — all eight SM120 collectives are `*_tma.hpp`, all seven builders are TMA-only, and
>   `sm120_mma_builder.inl` hardwires `SM90_TMA_LOAD`. Every other sm_120 FP8 GEMM in existence uses
>   TMA because there is no alternative to compare against, so our two-route situation is one nobody
>   has published numbers for.
> * **There is no published measurement either way, and an earlier reading of one was wrong.** A
>   previous revision of this block cited a third-party exercise as "TMA 4-9% ahead of `cp.async` on
>   an RTX 5090". Verification refuted it: the exercise is `02_matmul_sm120`, it measures **BF16 and
>   INT8 only with no FP8 anywhere**, its INT8 pair differs in `BLOCK_K` so that margin is not
>   attributable to TMA, and the "6.9% run-to-run swing" appears in none of its files
>   (`docs/research/verify-cutlass-sm120-claims.md`). The BF16 pair *is* genuinely controlled — same
>   tile, stages, warps and MMA — so it is weak evidence about BF16 and none about FP8.
>   **So "TMA is slower here" is unsupported by this tree's bench, and "TMA is faster" is now
>   unsupported by anything published.** Both directions are open.

> * **"Thin M" was my hypothesis and it is wrong.** Computed from this tree's own templates, both arms
>   use a **32-token** tile; the TMA arm's *row* tile is twice as large (64 vs 32), so it launches half
>   the CTAs — not a smaller tile. At T>=129 the two arms already share a 64x128x128 tile and differ
>   only in stage count (3 vs 2), which moves shared memory 73,776 -> 49,152 B and occupancy 288 ->
>   512 threads/SM.
>
> **Occupancy is a real, NVIDIA-confirmed sm_120 wall** — 101,376 B (99 KiB) against sm_100's 232,448,
> per `arch.h` and TensorRT-LLM PR #12141. That credibly explains roughly 1.8x at T>=129. **It explains
> nothing at T<=64, where occupancy is equal.** So part of the gap is stage count and part is
> unexplained, and the leading internal candidate is CTA starvation: half the CTAs at equal
> threads/SM.
>
> **MEASURED 2026-10-01: at a matched tile, TMA is FASTER — median 9.2%.** The ~10x was a schedule
> confound, and the controlled experiment is now in the tree.
>
> `Fp8A8TmaMmaSchedule` derives from `Fp8A8MmaSchedule` and overrides only `kTmaSwizzle`, the producer
> warp and the cache hint. So an MMA twin of a TMA tile is the same instantiation of the base template —
> same BlockTokens, BlockRows, BlockK, WarpsTokens, WarpsRows, Stages and MinBlocksPerSm — and only the
> load path differs. `tools/bench/tma_ab.cmd` runs both arms in ONE binary, interleaved
> (tma, mma, tma, mma...), 5 repeats, and reports per-arm medians with the ratio taken after them.
>
> | tokens | tile | tma us | mma us | mma/tma |
> |---|---|---|---|---|
> | 32 | 32x64x128 | 28.032 | 30.016 | 1.071 |
> | 64 | 32x64x128 | 30.048 | 32.096 | 1.068 |
> | 65 | **CONTROL** | 32.096 | 32.128 | **1.001** |
> | 96 | **CONTROL** | 32.128 | 32.096 | **0.999** |
> | 129 | 64x128x128 | 34.176 | 40.288 | 1.179 |
> | 192 | 64x128x128 | 36.192 | 40.288 | 1.113 |
>
> **The control band is what makes this admissible.** Tokens 65..96 dispatch to `Fp8A8T64R64K128` in
> both arms, so they must agree; they came out at 1.001 and 0.999. The reporter runs that check FIRST
> and reports VOID rather than a ratio if it fails, which is the control-arm rule that this session's
> harness bugs were missing.
>
> **Correctness is checked too, and it had to be.** The bench times but has no oracle, so a throughput
> ratio between two kernels where only one is known to be right is not a measurement.
> `tools/bench/tma_ab_correctness.cmd` runs `ninfer_linear_add_fp8_test` -- which does have one
> (`kA8Tolerance{0.04, ...}`, `verify_preserved`) -- once per arm: **tma OK, mma OK, unset OK.** Both
> arms compute the same thing, so the ratio compares two correct kernels.
>
> **Reading, and it is NOT "TMA instructions are faster."** Measured at compile time, the two arms are
> not occupancy-identical: the TMA schedule adds a 32-thread producer warp, so it runs **96 threads per
> CTA against the MMA twin's 64**, with the same `kMinBlocksPerSm = 2` and 36,912 B of shared memory.
> That is 192 versus 128 resident threads per SM. So the honest statement is:
> **the TMA schedule as designed — same tile, plus a producer warp, giving 1.5x the resident threads —
> is 6.8% to 17.9% faster, median 9.2%, than the MMA schedule at the same tile.** Part of that margin
> is instruction and part is simply more parallelism in flight. Anyone reading this as a property of
> `cp.async.bulk.tensor` alone will over-generalise it.
>
> That clears the 5.8-7.6% early-run spread this card has shown, though the 6.8% band sits near that
> floor and wants more repeats before anyone treats it as real.

>
> **SPLIT-K IS NOW MEASURED (2026-10-01), and the reason it could not be is now fixed.** The blocker
> was not a transport property: `kSplitWaveCtas`, `kMaxParts` and `kReductionBlocks` were declared on
> `Fp8A8TmaMmaSchedule` rather than on `Fp8A8MmaSchedule`, so a plain MMA schedule had no member saying
> "this tile splits" and the MMA launcher could not ask. The plan, the partials store and the reduction
> never read a tensor map; they moved to `fp8_split_k.cuh`, both transports decode the split geometry
> through the same `fp8_split_k_range`, and `Fp8A8TmaSplitKSchedule` became `Fp8A8SplitKSchedule`
> (21 sites). Both arms pass the FP8 linear_add oracle at the split-K bands.
>
> 11 interleaved repeats, control band 1.000 / 0.999, so the harness is valid. `ratio = mma/tma`:
>
> | tokens | tile | tma us | mma us | mma/tma |
> |---|---|---|---|---|
> | 32 | Tma/Mma 32x64 | 28.000 | 30.080 | 1.074 |
> | 64 | Tma/Mma 32x64 | 28.032 | 30.080 | 1.073 |
> | 65 | shared | 30.080 | 30.080 | 1.000 CONTROL |
> | 96 | shared | 32.096 | 32.064 | 0.999 CONTROL |
> | 129 | Tma/Mma 64x128 | 36.192 | 40.288 | 1.113 |
> | 192 | Tma/Mma 64x128 | 36.192 | 40.288 | 1.113 |
> | 193 | MidBulk / **MidBulkMma** | 40.256 | 44.384 | 1.103 |
> | 256 | MidBulk / **MidBulkMma** | 40.256 | 44.384 | 1.103 |
> | 384 | MidBulk / **MidBulkMma** | 50.592 | 66.912 | **1.323** |
> | 512 | MidBulk / **MidBulkMma** | 81.344 | 81.280 | 0.999 |
> | 768 | MidBulk / **MidBulkMma** | 122.240 | 126.336 | 1.034 |
> | 1025 | Bulk / **BulkMma** | 165.248 | 173.440 | 1.050 |
>
> **TMA is faster at 9 of the 10 measured bands, median 8.8%, max 32.3% at tokens 384, tie at 512.**
>
> **A defect in the reporter, found by reading its output against its own arithmetic.** It computes
> `ratio = mma_median / tma_median` and then printed "> 1 means TMA is SLOWER" — the reverse. The
> legend contradicted the column header directly above it and inverted the reading of every run it had
> produced. Fixed, with the derivation printed instead of a conclusion to be trusted.
>
> **Both owed checks are now done (2026-10-01), and one of them corrects the headline.**
>
> **Negative control: PASSED.** `fp8_split_k_range` was perturbed so every part after the first
> re-reads one K tile, and the oracle goes **RED** — failing at exactly seven token counts: 193, 255,
> 256, 513, 767, 768, 1025. That proves the MMA split-K path is genuinely executed under the test and
> genuinely covered by the oracle. The GREEN above was not vacuous: it also passed *before* the change,
> so without this it discriminated nothing. Restored byte-identical (SHA256 prefix `725AEB14`) and
> GREEN re-confirmed.
>
> **Split engagement, measured rather than derived.** The seven token counts above are exactly where
> split-K runs; 257, 383, 384, 385, 511, 769, 1023 and everything ≤192 do not split. **The tail
> arithmetic in the previous entry was correct on all six overlapping points**, so this is a
> confirmation, not a correction — the claim that it "contradicted the model" was itself wrong.
>
> **The 1.323 at 384 was not an anomaly, and the previous entry's framing of it was wrong.** A
> fine-grained sweep across the neighbourhood (320/352/384/416/448/480/512) shows it is a **band edge**,
> not a spike, and 384 and 512 were never comparable neighbours:
>
> | tokens | token tiles | tma us | mma us | mma/tma |
> |---|---|---|---|---|
> | 320 | 3 | 48.512 | 64.896 | 1.338 |
> | 352 | 3 | 50.528 | 64.896 | 1.285 |
> | 384 | 3 | 50.560 | 64.928 | 1.284 |
> | 416 | 4 | 75.776 | 77.184 | 1.019 |
> | 448 | 4 | 77.216 | 77.248 | 1.000 |
> | 480 | 4 | 77.216 | 79.264 | 1.027 |
> | 512 | 4 | 81.344 | 79.264 | 0.974 |
>
> **The MMA arm sits at a flat ~64.9us floor across 320-384** — 64.896, 64.896, 64.928, it does not move
> — while TMA climbs 48.5 -> 50.6. From four token tiles up both arms are token-throughput-bound and
> equal. So the structure is: **TMA's margin is largest exactly where the machine is underfilled**, and
> that is the prefill regime, which is the regime these gates were argued about. The earlier reading of
> "384 is unexplained" was the tail arithmetic of a 12-point ladder being read as a local anomaly.
>
> **Harness bug found while measuring it, and it had been hiding data.** Passing the sweep as `%~2`
> silently delivered **only the first value** — a 7-value sweep wrote a 1-row CSV, which reads as "one
> band measured" rather than as a broken command line, and the bench itself parses comma lists
> correctly. The sweep now comes from `NINFER_TMA_SWEEP` in the environment. Worth recording because the
> failure mode was a plausible-looking result, not an error.
>
> **How much of the K=6144 evidence transfers to the other eight gates: very little, and now known.**
> The gates do not share one tile ladder. Compared schedule by schedule:
>
> | tile | measured on linear_add K=6144 | used by |
> |---|---|---|
> | 128x256x128 (2,4,2,1) split | **yes, identical** | linear_swiglu, attn_input_proj, Bulk of the shape instantiations |
> | 64x128x128 | (2,4,**3**,1) | others use (2,4,**2**,1) - different `Stages`, a different schedule |
> | 64x256x128 (2,4,2,1) | no | linear_swiglu, attn_input_proj, n14336 |
> | 96x256x128 (3,4,2,1) | no | attn_input_proj, n14336 |
> | 192x128x128 (3,4,2,1) | no | gdn_input_proj |
> | 128x128x128 split | (2,4,**3**,1) | gdn_input_proj uses (2,4,**2**,1) |
>
> So exactly **one tile - the 128x256x128 split Bulk - is literally the same schedule** on the measured
> path and on linear_swiglu / attn_input_proj / the shapes, and its result transfers **by construction**
> to those call sites. Every other tile either differs in `Stages` or was never measured, so no other
> gate inherits the finding. "One measurement, nine gates" was never available: the gates select different
> ladders. The remaining work is bounded and specific - three unmeasured tile shapes (64x256, 96x256,
> 192x128) plus the `Stages` variants - not nine independent unknowns.

> **A latent trap in the ungating work, found by checking every call site rather than the one I edited.**
> Only `linear_add` forwards `partials` into the MMA launcher. Five call sites do not:
> `attn_input_proj:33`, `gdn_input_proj:31` and `:41`, `linear_swiglu:31`, and
> `linear/fp8/fp8_launch.cuh:40` -- the entry **all five shape instantiations** go through.
> Today that is harmless: every Windows-gated branch instantiates non-split schedules, so
> `if constexpr (kSplitWaveCtas > 0)` discards the split block and the null `partials` is never
> dereferenced. But four of those ladders carry split-K tiles in their `#else` arms, and
> `launch_fp8_a8` does not forward `scratch.partials` the way its TMA sibling `launch_fp8_a8_tma`
> does. Ungating any shape file whose ladder includes `Bulk`/`MidBulk` would therefore reach
> `FP8 split-K requires aligned caller partials` at runtime on first use.
>
> It fails loudly rather than corrupting, which is the right property, but it would present as a
> mysterious per-shape failure partway through ungating. **Forwarding `scratch.partials` through
> `launch_fp8_a8` is a prerequisite of ungating, not a follow-up to it.
>
> **The three previously unmeasured tiles are now measured (2026-10-01), and TMA wins every band.**
> `NINFER_FP8_TMA_TILE` pins one tile for **both** arms, so each is measured at a matched tile instead
> of at whatever the K=6144 ladder would have chosen. 7 interleaved repeats, oracle GREEN on both arms
> of all three tiles.
>
> | tile | ratio median | min | max | bands where MMA wins |
> |---|---|---|---|---|
> | 64x256x128 (2,4,2,1) | **1.270** | 1.000 | 1.414 | **0 of 8** |
> | 96x256x128 (3,4,2,1) | **1.288** | 1.056 | 1.370 | **0 of 8** |
> | 192x128x128 (3,4,2,1) | **1.241** | 1.058 | 1.284 | **0 of 8** |
>
> Across 24 bands TMA is faster everywhere, 5.6% to 41.4%. That is a stronger result than the K=6144
> ladder produced, which had ties at 0.974 and 0.999 — so the earlier ties were a property of those
> token counts, not of the transport.
>
> **The control that makes this admissible**, because a forced-tile override that silently did nothing
> would still pass the oracle on the default ladder: the three tiles produce three clearly distinct
> timing profiles (TMA at 64 tokens is 44.4 / 60.8 / 62.8us). Identical profiles would have meant the
> override never fired and the GREEN was vacuous.
>
> **What this still is not.** All three are measured at K=6144 with the linear_add residual epilogue.
> The gates that use these tiles run at K=5120 with different epilogues and different partials
> capacities. Transport-at-matched-tile is the same *question*, but this is not a measurement at those
> gates' own shapes, and it does not ungate anything: a gate is a correctness decision first.
>
> **All five gated tiles are now measured, and the "TMA wins every band" reading does not survive.**
> The three wide/tall tiles came back 0-for-8 losses. The two 128-row tiles do not:
>
> | tile | median | min | max | bands where MMA wins |
> |---|---|---|---|---|
> | 64x256x128 (2,4,2,1) | 1.270 | 1.000 | 1.414 | 0 of 8 |
> | 96x256x128 (3,4,2,1) | 1.288 | 1.056 | 1.370 | 0 of 8 |
> | 192x128x128 (3,4,2,1) | 1.241 | 1.058 | 1.284 | 0 of 8 |
> | 64x128x128 (2,4,**2**,1) | 1.107 | **0.943** | 1.313 | **1 of 9** (768) |
> | 128x128x128 (2,4,**2**,1) split | 1.047 | **0.952** | 1.284 | **2 of 8** (448, 512) |
>
> **MMA is faster at 512 and 448 on the split 128x128 tile, by 4.8% and 2.7%, and at 768 on 64x128 by
> 5.7%.** TMA is ahead on the median everywhere, but "TMA wins every band" was true only of the three
> tiles measured first, and reporting it as a general result would have been wrong.
>
> **Two structural limits, both measured rather than argued:**
>
> 1. **The split 128x128 tile cannot be measured below 193 tokens.** `allocate_fp8_a8_workspace` sizes
>    partials from the *current* invocation's token count (`fp8_linear_add_a8.cu:117`), and
>    `fp8_linear_add_partial_capacity_bytes` returns **0** for `max_tokens <= 192` — correctly, because
>    the real ladder never places a split tile there. Forcing one makes the launcher throw
>    `FP8 TMA split-K requires aligned caller partials`. So that tile's evidence covers 193-1025 only.
> 2. **That same tile has NO oracle coverage.** The FP8 linear_add test crashes on it for the same
>    reason, so unlike the other four it has never been checked against the oracle at all. Its twin is
>    geometrically identical to `K6144MidBulk` (Stages=3) which is checked, differing only in pipeline
>    depth — but "differs only in Stages" is an argument, not a measurement, and it is recorded here as
>    an open gap rather than as evidence.
>
> **A harness bug that made the above look like a result.** `tma_ab.cmd` discarded bench output with
> `>nul`, so a bench that threw wrote no CSV, and the reporter then read whatever CSV the *previous*
> tile had left behind. Two different tiles came back byte-identical — because they were the same file
> read twice. The root is now cleared before each run so a missing input reads as missing.
>

>
>
>
> **MEASUREMENT SCAFFOLDING WITHDRAWN, and the A/B harness with it.** The seventeen MMA twins, the
> `NINFER_FP8_TMA_ARM` and `NINFER_FP8_TMA_TILE` selectors, and the runtime arm/tile dispatch are
> gone from `fp8_linear_add_a8.cu` -- 222 lines down to 100. Its own comment said to remove them "once item
> 12's question is answered", and it has been.
>
> **The A/B harness had to go with them, and that was not optional.** `tma_ab.cmd`,
> `tma_ab_correctness.cmd` and `tma_ab_report.py` all set those selectors. Left in place they would have
> run both "arms" down the identical route and emitted a confident-looking ratio near 1.0 -- a
> measurement tool that silently measures nothing, which is the exact failure mode this item spent its
> length fighting. Their numbers are preserved above with the methodology that produced them.
>
> **Withdrawn too, 2026-10-02.** `tools/scripts/verify_fp8_tma_route.cmd` and the
> `flip_fp8_tma_gates.ps1` it drives are gone as well. The flip script asserted that exactly nine
> `PORT-DISPATCH` gates existed and threw otherwise; all nine are lifted, so it threw before touching a
> line and the verifier could only ever exit 2. That is a spent measurement left reachable — the same
> fault as the harness above, one file over, and it was cited from a source comment as ungating evidence.
> The figure it took is recorded here; the tool that could not run is not.
>
> Note the over-deletion this nearly caused: the first pass also removed the `K6144MidBulk` and
> `K6144Bulk` aliases, which are PRODUCTION ladder tiles rather than twins. Caught by checking which
> aliases were still referenced before building, not by the compiler after a broken tree. Subtraction
> without reading what each declaration is FOR is just a slower way to introduce a defect.
> **UNGATING COMPLETE 2026-10-02: all nine lifted.** Zero FP8 A8 `#ifdef _WIN32` gates remain in
> `src/ops`; the only `_WIN32` blocks left in the tree are legitimate platform code (file I/O,
> logging, media acquire, context cost, request log). The final two were `Fp8N5120K6144` and
> `Fp8N5120K17408`, both owned by linear_add, and both were the only shapes whose partials thresholds
> were already correct and unconditional.
>
> Every ungating rested on the same three things, never on "TMA is faster": the route's own oracle
> at its own tiles and token counts; the partials buffer covering the ladder's whole selection band; and
> a green full suite after each one. Suite 135/137, GATE PASSED, after all nine together.
>
> **What ungating cost, stated rather than discovered later:** six partials thresholds had to be realigned
> to their ladders' selection bands (they were pre-existing upstream bugs the gate had been masking),
> `launch_fp8_a8` and five route launchers had to start forwarding `scratch.partials`, one route's test
> needed three token counts added, and one intermediate attempt was a memory regression that had to be
> walked back. Ungating was never a one-line flip, and treating it as one is what produced the bad table
> earlier in this item.
> **UNGATING OUTCOME 2026-10-02: seven of nine lifted, one deliberately held, one to verify.**
>
> | route | oracle | note |
> |---|---|---|
> | linear_swiglu | added 129/192/193 to its own test | first; found the MMA branch not forwarding partials |
> | attn_input_proj | already complete | best-covered of the nine, incl. CUDA Graph replay at every boundary |
> | gdn_input_proj | already complete | closed the split 128x128 oracle gap |
> | n16384_k5120 | = the gdn tests | same shape; verification was free |
> | n14336_k5120 | = the attn_input tests | genuinely splits (56 row tiles, tail 56 <= 85) |
> | linear_add (+ K=6144 and K=17408 shapes) | 11 combinations GREEN | three gates at once; the route the investigation started from |
> | n34816_k5120 | = the swiglu tests | **ungated, after a correction -- see below** |
>
> The `UNGATING BLOCKER FOUND` table below under-counted at three; it is corrected in place above. Suite is
> green after each ungating: 135/137 with GATE PASSED, the two failures being the by-construction
> `dflash_real` and `moe_real`.
>
>
> **RESEARCHED: CUTLASS has shipped the same bug twice, and its architecture is the answer.**
> * NVIDIA/cutlass **#3539** - "EllGemm ColumnMajor-output workspace sizing uses the unswapped problem shape
>   and **undersizes split-K** semaphores."
> * NVIDIA/cutlass **#3538** - the reduction launch "reads ptr_gemm_k_reduction from a
>   **never-initialized workspace**", while `get_workspace_size` reserved the right size.
>
> Both are the same defect this item hit: a workspace sized from shape data that does not match what
> the launcher actually uses, so the two disagree. Six threshold mismatches here, one over-allocation, and
> a false `static_assert` are the same family. That it is a known-hard area is worth knowing; it is not
> evidence that our particular instances were right.
>
> **The architectural answer is CUTLASS's own pattern and it is the one specified above.**
> `get_workspace_size()` is a method on the operation/problem descriptor and computes the size from
> the SAME shape data the launcher passes to the kernel -- one function, one source of shape truth, so
> they cannot disagree. The equivalent here is to make the partials capacity a method that receives the
> shape the launcher uses (`rows`, `k`, and the selected `Schedule`) instead of an independent
> `(k, max_tokens)` guess. That is precisely the "thread `rows` down and call the same
> `fp8_tma_split_k_plan`" refactor, and the research says the duplication risk I was worried about is
> real and is exactly what these two CUTLASS issues are.
> **THE STRUCTURAL FIX, DONE 2026-10-02.** The spec above was to thread `rows` into
> `partial_capacity_bytes` so it calls the same split-K plan the launcher calls. That was not
> implemented, because it would have introduced a second derivation of `tiles`: the plan takes
> `tiles`, and `tiles` comes from `for_each_token_slice` via `columns_per_block * kCudaGridYLimit`.
> Re-deriving that is the exact assumption the retracted `static_assert` made. It also solves the
> wrong half -- the capacity needs the ladder's SELECTION band, not the plan's engagement decision,
> which is what the existing comments in those functions say in as many words.
>
> What was done instead: every band boundary that a ladder and its sizing function **share** is now a
> named constant, read by both. Nine capacity functions across five route families shared thirty-odd
> numeric literals with the ladders beside them; none of those shared literals is a bare number now.
> **Correction:** an earlier revision of this note claimed "zero numeric band tests left in FP8 A8
> selection or sizing anywhere under `src/`". That was false — fifteen remained, all in ladder-only
> positions that no capacity function reads, so the refactor had not removed them and the claim
> overstated it. The non-monotonic windows have since been named too, because those are the ones that
> matter (see below).
>
> **Sizing semantics are deliberately unchanged.** Two of these ladders are non-monotonic --
> `n5120_k17408` selects Bulk for 385-512 and returns to Wide for 513-768, and `n16384_k5120` has its
> own 385-512 branch -- so capacity is a worst case over the bands rather than the size of the band
> `max_tokens` itself selects. Tightening that needs the arena's lifetime answered first, which is a
> separate question and was not smuggled in here.
>
> Verified: `ninfer_linear_fp8_a8_test`, `ninfer_linear_fp8_a16_test`, `ninfer_linear_add_fp8_test`
> and `ninfer_linear_swiglu_fp8_test` all green, and the full suite at 135/137 with GATE PASSED.
>
> **AND THE REFACTOR ITSELF SHIPPED A REGRESSION, which the green suite could not see.** Two of the
> ladders are non-monotonic — they return to a wide tile for a window above the point they passed it —
> and the bound on that window is now a named constant. The first version of this refactor substituted
> `kBandWideMax` (384) where the bound was 768, which made `tokens > 512 && tokens <= 384` false for
> every token count: `n5120_k17408.cu` and `fp8_linear_add_a8.cu` both sent tokens 513-768 to `Bulk`
> instead of `Wide`. Nothing failed, because both tiles are numerically correct and differ only in
> speed, and no oracle in this tree asserts *which tile* a token count selects.
>
> Two lessons, both paid for here. A green suite is evidence about values, not about selection, and a
> rename that appears semantically neutral is where a wrong constant hides. The check that catches it
> is to resolve every branch of every ladder to its numeric meaning and compare that against the
> pre-refactor source — and that check has to have a control, because the first version of it
> substituted `tokens` for `0` before evaluating, so every condition read `0 <= 64`, the first branch
> always won, and it reported the buggy tree as identical to the good one.


> **THE STRUCTURAL FIX, SPECIFIED RATHER THAN DONE.** Every threshold problem in this item has one
> cause: `partial_capacity_bytes(max_tokens)` is a static per-shape GUESS about whether a split will
> engage, while the decision is made at run time by `fp8_tma_split_k_plan`. That mismatch produced six
> under-allocations (a launch-time throw) and one over-allocation (21.3 MiB wasted), and it is the
> reason `n34816` cannot be ungated safely today.
>
> **The fix is feasible and the data is already at the boundary.**
> `attn_input_proj_workspace_capacity_bytes(parent_qtype, parent_rows, input_rows, policy, min, max)
> already receives `parent_rows` and `input_rows` - which is exactly what the plan needs, since
> `blocks = rows / BlockRows * div_up(count, BlockTokens)`. Threading `rows` down to
> `partial_capacity_bytes` and calling the *same* `fp8_tma_split_k_plan` the launcher calls would make the
> two agree by construction rather than by six hand-matched thresholds.
>
> **Superseded 2026-10-02 — and done differently.** The structural fix landed, but not in the shape
> this paragraph describes: rather than threading `rows` down to re-derive `tiles` (which would
> have introduced a second derivation of the thing the retracted `static_assert` got wrong), every
> band boundary is now a named constant read by both the ladder that selects and the function that
> sizes. Sizing semantics are unchanged. `tools/release/check_fp8_band_ladders.py` enforces it, with
> a self-test. The block below is the original decision and is kept as the record of why that
> shape was rejected.
>
> **Not done here, deliberately.** It changes the `Fp8LinearShape` signature and every plan function, and it
> is a new piece of architecture rather than a bug fix - it deserves its own verification cycle, not a
> rushed one at the end of a release. The current thresholds are CORRECT for all seven ungated routes,
> and the one route they are not correct for is held. That is a known, bounded, recorded gap rather
> than a silent one.
> **UNGATING BLOCKER FOUND: three routes would inherit a latent partials defect.** The partials-capacity
> thresholds sit *above* the token counts at which the ladder selects a split-K tile, so a split tile runs
> with `partials == nullptr` and the launcher throws:
>
> | route | split tile selected at | partials allocated above | gap |
> |---|---|---|---|
> | linear_swiglu | Bulk at >=193 | >256 | **193-256** |
> | attn_input_proj | Bulk at >=289 | >384 | **289-384** |
> | n14336_k5120 | Bulk at >=289 | >384 | **289-384** |
> | gdn_input_proj | Bulk at >=193 | >256 | **193-256** |
> | n16384_k5120 | Bulk at >=193 | >256 | **193-256** |
> | n34816_k5120 | Bulk at >=193 | >256 | **193-256** |
> | n5120_k17408 | Small<=128, Mid<=256, Wide<=384, Bulk rest | >64/>128/>256/>384 | none, thresholds match |
> | n5120_k6144 / linear_add | MidBulk >=193 / Bulk >=769 | >192 / >768 | none, thresholds match |
>
> **The first version of this table was WRONG and said gdn_input_proj was clean.** It compared each
> capacity threshold only against the *explicit* split rung and missed the **fall-through**, which is
> exactly where `Bulk` is selected on four of these ladders. Re-reading every ladder line by line put the
> real count at **six of nine routes**, not three.
>
> **All six are fixed**: each capacity threshold now equals its own ladder's selection point -- 192 where
> `Bulk` is the fall-through, 288 where it follows `Tma96x256`. `n5120_k17408`, `n5120_k6144` and
> `linear_add` were already correct and are unchanged.
> has simply been masking it. Ungating any of those three routes would inherit it, and the failure would look
> like a mysterious per-shape crash partway through ungating rather than a threshold mismatch.
>
> The *mechanism* is measured, not hypothesised: it is exactly what the forced `midbulk128` tile did --
> `FP8 TMA split-K requires aligned caller partials`, thrown from `launch_fp8_a8_mma` when
> `plan.split_ctas` is set and the buffer is null. The token-count ranges above are derived from the
> capacity functions and ladder thresholds read in each file, and are labelled derived for that reason.
>
> **So the order is: fix the thresholds, then ungate.** Lowering each capacity threshold to match its own
> ladder is the direct fix, at the cost of allocating `Bulk::kPartialBytes` across the gap band. The
> alternative -- rejecting a split tile when no partials exist -- would silently pick a different schedule
> from the one the ladder names, which is the class of defect this whole item exists to remove.


> **THE n34816 "NEVER SPLITS" CLAIM WAS WRONG, and the suite caught it.** An intermediate commit
> asserted that Fp8N34816K5120 could never engage split-K -- `blocks = 34816/256 = 136`, `tail = 136 % 170 = 136 > 85` -- and
> guarded it with a `static_assert(!kSplits)`, which I described as making the reasoning a build-time
> property. It was a guard over an ASSUMPTION. `for_each_token_slice` does not cap a slice at
> `kBlockTokens`; it uses `columns_per_block * kCudaGridYLimit`, so `blocks` grows with the token
> count. At T=257 that is 136 * 3 = 408 CTAs, tail = 68, which is inside the split range.
> `ninfer_linear_fp8_a8_test` threw "FP8 TMA split-K requires aligned caller partials" at T = 257, 511,
> 512, 1024 and 1025.
>
> **An assertion over an assumption cannot detect the assumption being wrong.** The static_assert
> compiled cleanly because it was checking my arithmetic against itself, which is exactly why it read
> as a proof. It was removed rather than repaired, and the error is recorded in the code.
>
> **What survives is the real finding: there is no compile-time shortcut here.** Split engagement
> depends on the token count, so every shape must size partials for the worst case across its whole
> selection band. That is what all six now do, and the structural refactor specified below -- threading
> `rows` down so the capacity calls the same plan function the launcher calls -- remains the only fix for
> the class rather than each instance.
>
> **The suite run that caught this was the SECOND one.** The first was reported green and discarded
> because I had rebuilt while it was executing; had I trusted it, a broken release would have shipped.
> **RESEARCHED: which SM120 TMA failures are real, and which one this tree had.** Third-party
> reports of "TMA warp-specialized GEMM produces garbage output on SM120" are real, and there are
> two distinct root causes behind that one symptom. Both are now identified against this tree:
>
> | reported SM120 failure | source | status here |
> |---|---|---|
> | **TMA descriptor alignment** - `misaligned address`, on RTX 5090 / Windows / CUDA 13.1 | PyGPUkit #107 | **The defect item 12 fixed.** That report reached the same conclusion after misdiagnosing it twice (first LDSM alignment, then an SM90-vs-SM120 kernel-selection error) and landing on TMA descriptor alignment. `Fp8TmaDescriptors` is `alignas(64)` and the encode destination is `alignas(64) CUtensorMap`, which is what `cuda.h:3749` asks for. |
> | **SMEM overflow in grouped GEMM** - every TMA-WS tactic fails to initialise, garbage output or crash | CUTLASS #3096 | **Already guarded, statically.** `static_assert(kSharedBytes <= 99 * 1024)` at four sites in `fp8_schedule.cuh`; the largest gated tile footprint is 98,304 B against the 101,376 B SM120 limit. |
> | `compute_120a` vs `compute_120f` arch suffix | CUTLASS #3096 | **Not ours.** A FlashInfer/CUTLASS JIT codegen constraint. This tree compiles natively with `sm_120a` and its TMA route is oracle-checked, so the suffix question does not arise. |
>
> **Effect on the gate decision.** This removes the reason to treat the SM120 TMA reports as a
> reason to distrust our measurement: they are different mechanisms, one of which we had and fixed, one
> of which we already prevent by construction. It confirms the discipline rather than relaxing it - a
> platform whose documented TMA failure mode is *silent garbage output* is exactly the one where a
> throughput number without an oracle is worthless, and where "it measured faster" must never be the
> argument for ungating by itself. The `garbage output` symptom is why the per-route oracle run comes
> BEFORE the gate decision, not after.
> **A harness trap worth recording.** Restoring the perturbed file with `Copy-Item` preserved the
> backup's older mtime, so ninja judged the source unchanged and **skipped the rebuild** — the test
> then reported the perturbation's failure against restored source, which read as "the fix didn't work".
> Restoring any probed source needs its mtime bumped forward, or the next build silently tests the old
> binary.
>
> **Next, in order:** the negative control, then the split-engagement measurement, then re-measure
> 384 and 1025 specifically since both are now suspect readings, then decide the gates per path rather
> than globally.

>
> Note also that the FP8 peak convention is unsettled (419 TFLOPS at FP32 accumulate vs 838 that
> circulate for the same part), so any percentage-of-peak framing for FP8 on this card is fragile.
> Throughput against the DRAM constant is the more robust denominator.

>
> **[SUPERSEDED] The standalone reproducer's finding was a harness bug, not a platform fact.**
> `tools/scripts/probe_sm120_tma_load.{cu,cmd}` reported that a bare `cp.async.bulk.tensor.2d` "does not
> run on sm_120a on this machine", and that ruled out four candidate causes. That conclusion was wrong.
> Its bounded wait mis-numbered the inline-asm operands for `mbarrier.try_wait.parity`: the barrier
> *address* was taken from the predicate-result register and the *phase* from the address, so it
> dereferenced a kernel parameter as a shared address and faulted **before any TMA instruction ran**.
> `compute-sanitizer` names `SYNCS.PHASECHK.TRANS64.TRYWAIT` at the fault offset, with `UTMALDG.2D`
> nearby and not faulting; a kernel containing no tensor map and no TMA instruction reproduces the same
> fault; and with the operands corrected, TMA runs in every legal arm — rank 2 and 3, swizzle 128B and
> NONE, by value and by pointer. So none of the four causes that reproducer claimed to eliminate was
> ever under test. `docs/research/why-no-others-hit-sm120-tma-evidence.md` has the measurement.
>
> **This does not touch the fix.** The fix was validated by a different loop that drives the project's
> own kernel — flip the gates, build `ninfer_linear_fp8_a8_test`, run it — and that one reproduced
> `cudaErrorIllegalInstruction` at `[14336,5120] T=129` and went green after `a5077adf`. This tree's
> only `try_wait` is `src/ops/common/mbarrier.cuh:21`, which is correctly numbered (no output operand,
> so `%0` is the barrier address), so the project does not share the reproducer's defect. The
> reproducer has been deleted rather than corrected: the real loop supersedes it.

**Attempted 2026-09-29, first aborted, then merged the same day. Suite green with it: 135 tests,
133 passed, the two by-construction `dflash_real` and `moe_real`; `check_test_baseline.py` GATE PASSED.**
*(A dated record of that day's run, not the current size: the suite is 137 — see the reconciliation
below the merge narrative. The 135 and 133/135 figures are what the gate read on 2026-09-29.)*

`upstream/dev` is at `d44ab584` and we were 14 behind, 426 ahead. The merge touches 171 files. It is
**five conflicts, all documentation** — every kernel, program file and test auto-merges, and the merge
does not touch the profile table, the launchers or the packager, so the shipped configuration is
unaffected. Resolutions, to reuse rather than re-derive:

- **Keep our deletion** of `docs/performance/qwen3.8-27b.md` and
  `model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md`. Upstream modified both, but their figures are pinned
  to *their* revisions (`f08597d`, `32c9881`) and *their* config (INT8 group-64 KV, 1,024-token prefill
  chunk, prefix reuse disabled). Ours is FP8 KV, a different chunk, prefix reuse on, different artifacts,
  so taking their version republishes their measurements as ours. Deleted by `1b53a301`.
- **Ours** for `README.md` and the three hunks of `docs/performance.md`: upstream's side links a dozen
  pages this port deleted and carries their AIME/GPQA/ERQA scores.
- **Neither side**, for one hunk of `tools/bench/README.md`, and this is a real three-way. The merged
  code is `RUN_SCHEMA_VERSION = 8` (upstream's bump) and `SERVER_LOG_SCHEMA_VERSION = 22` (this port's
  bump; upstream is on 21), so the correct text is **v8 and v22 plus the KV dtype** upstream added.
  Asking the merged code beat picking a side.

**What the merge needed from this port, in two parts.**

1. `src/ops/linear/fp8/fp8_a8_tma_mma.cuh` does not compile on MSVC — `error C2719`, a by-value
   `__grid_constant__ alignas(128)` TMA descriptor that the MSVC ABI cannot lay out. Fixed with the
   pattern `723c1290` established for the NVFP4 route: a macro that is a pointer on `_WIN32` and the
   by-value parameter elsewhere, a `descriptor_block` local, and an RAII device copy whose allocation,
   copy and free are all ordered on the consuming stream — a NULL-stream free is ordered against
   nothing on a non-blocking stream and the pool can recycle the block under the TMA unit's read,
   which is the 786,432-token prefill live-lock that shape already caused here once.

   **The macro this cites no longer exists, and the NVFP4 route no longer uses this fix.**
   `NINFER_NVFP4_TMA_DESCRIPTOR_PARAM` is gone from the tree — `1218d574` replaced it with a by-value
   parameter, because a device buffer filled by `cudaMemcpyAsync` reads a caller stack frame that is
   gone by the time a CUDA Graph replay runs. So the two routes now resolve the same C2719 two
   different ways, and the FP8 header's claim that it was the "same fix, and same reason" was wrong
   on both counts. Both comments are corrected; the divergence is deliberate, not drift.

   **Correction to this item's own recorded reason, 2026-10-01 — the mechanism is not what the comments
   said, and the comments have since been corrected to say so.** The comments in three kernel headers
   justified the fix as inheriting `TENSOR_MAP_ALIGN = 64` under MSVC. That is false: on this build
   `CUtensorMap` inherits **8**, because `__cplusplus` is `199711` without `/Zc:__cplusplus`, so
   `cuda.h`'s `alignas` never fires and the header's `_MSC_VER` branch is dead code. **The shipped W4A4
   route is correct by layout accident, not by the mechanism those comments recorded** — descriptors
   first at 128 bytes each give offsets 0/128/256/384, all 16-aligned, which is why it works. Anyone
   reasoning from the old comment would draw the wrong conclusion about which property is load-bearing.
   A sweep of this toolchain also shows `alignas` 8/16/32/64 accepted and **128 and 256 rejected with
   four C2719 sites each**, all in nvcc's generated host stub.

   **The correction is in the tree, in `38cdc8e0`** ("correct three TMA comments that named the wrong
   mechanism, and land the probe that measures it"). `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14-34`
   now states the opposite in its own words — "the reason is **NOT** the one an earlier revision of this
   comment gave … The attribute is not applied at all" — and quotes the probe's output
   (`alignof(CUtensorMap) = 8, TENSOR_MAP_ALIGN = 64, __cplusplus = 199711, attribute applied: NO`).
   `src/ops/common/mbarrier.cuh` and the two other headers were corrected with it. The measurement
   above is unchanged and is the reason the comments now say what they say; what changed is that the
   tree no longer contradicts it.

   **The deferral risk is live, and the premise about who carries the attribute is wrong.** Upstream
   `Neroued/ninfer` still carries `alignas(128)` on all three descriptor structs on **both `dev` and
   `master`** today (`75a89050`), so a merge reintroduces an unrecoverable C2719 — that part of the
   deferral stands. But **CUTLASS does not carry it** at any tag from `v3.8.0` to `main`: it declares
   `TmaDescriptor = CUtensorMap` outright with an `alignas(64)` fallback, and FlashAttention has zero
   `alignas` in any TMA file and inherits that alias. So the exposure is *this fork's divergence from
   upstream*, not a CUTLASS/FlashAttention attribute, and it is **not tracked** in any CUTLASS,
   FlashAttention or NVIDIA tracker reached — the same diagnosis appears only in three third-party
   Windows ports (ONNX Runtime, vllm-windows, llmjob). **No upstream fix exists to wait for**, so the
   local divergence is not temporary and should not be written up as pending one.

2. **Upstream routed every FP8 A8 path to that TMA kernel, and it faults here at execution** with
   `cudaErrorIllegalInstruction`, which poisons the context so the test aborts `0xc0000409`. There is no
   single seam: a TMA schedule and an MMA schedule are different tile shapes, so the *selection* has to
   differ. All **nine** call sites are gated on Windows — the five `linear/fp8` shape files plus
   `attn_input_proj`, `gdn_input_proj`, `linear_add` and `linear_swiglu` — each restored to its
   **pre-merge dispatch verbatim**, because those are the bodies that passed. Upstream's routes stay in
   the tree and stay selected on other platforms.

   The A/B that established the route rather than the workaround as at fault: gating only the
   `tokens <= 192` branch of `n14336_k5120` made T=129 pass and moved the failure to T=385, the next TMA
   branch. So the pre-merge route runs and the TMA route does not. One limit worth keeping: this shows
   the TMA route *as it must be built on MSVC* does not run, not that it is broken independent of the
   descriptor workaround, because the by-value form cannot be compiled here at all. `compute-sanitizer`
   under memcheck ran 30 minutes on the smallest failing test without naming an instruction and then lost
   the context, the watchdog hazard `tools/scripts/test_v3_compute_sanitizer.cmd` already documents here.

   **[SUPERSEDED] 2026-10-01: a 2-second reproducer, and four causes ruled out — but no fix.**
   `tools/scripts/probe_sm120_tma_load.cmd` — **since deleted; the resolution block at the top of this
   item records why, and the "four causes ruled out" conclusion below is false.** It reproduced this
   fault class in a standalone ~150-line program with no project code in the path, replacing the
   30-minute sanitizer run. It built a descriptor with `fp8_tma_map`'s own proven shape — rank 2,
   dimensions `{k, rows}` with K first, one `globalStrides` entry, `elementStrides {1,1}`,
   `L2_PROMOTION_NONE` — which encoded successfully, was 64-byte aligned, and then died at execution
   with "an illegal memory access was encountered". Its fault was in its own `mbarrier.try_wait.parity`
   operand numbering, so it faulted before any TMA instruction ran and none of the arms below were
   under test. **As written at the time, the table read:**

   | varied | result |
   |---|---|
   | descriptor passed **by value** (the BF16/NVFP4 form, ungated on Windows) | faults |
   | descriptor passed **by pointer** (the form this port's MSVC workaround forces) | faults |
   | swizzle **128B** vs **NONE** | both fault |
   | `fence.proxy.acquire.tensormap::generic` present vs absent; `expect_tx` before vs after the copy | faults either way |

   **So the conclusion drawn at the time — that it is *not* the by-pointer descriptor form, *not* the
   swizzle mode, *not* the missing tensormap proxy fence, and *not* the mbarrier transaction ordering —
   is withdrawn.** None of the four was under test, because the reproducer faulted in its own wait
   before issuing a TMA instruction. With the operands corrected, TMA runs in every legal arm: rank 2
   and rank 3, swizzle 128B and NONE, by value and by pointer. See the resolution block.

   **What the platform evidence says, from primary sources** (`docs/research/sm120-tma-illegal-instruction-evidence.md`):
   sm_120 **does** support TMA — `cp.async.bulk.tensor` requires sm_90 or higher and the TMA unit is
   listed for CC 9.0 through 12.x — so this is *not* "TMA is unavailable on consumer Blackwell". sm_120
   supports **neither** `wgmma` (sm_90a only) nor `tcgen05` (sm_100a/101a only), and this tree uses
   warp-level `mma.sync`, which is what sm_120 uses instead, so the MMA is not at fault either. And
   working SM120 TMA kernels do exist: CUTLASS ships `MainloopSm120TmaWarpSpecializedBlockwiseScaling`,
   vLLM ships a dense-FP8 SM120 CUTLASS GEMM, and TensorRT-LLM ships TMA FMHA as its default sm_120
   prefill. **CUTLASS #2728 reports illegal instruction inside `cp.async.bulk.tensor` on an RTX 5090** —
   the same instruction on the same GPU class, on Ubuntu rather than Windows.

   **One documented requirement is genuinely unmet, and it is labelled inferred rather than
   established.** PTX ISA §8.8.4 says the tensormap proxy "is not acquired from generic-proxy at CUDA
   Kernel start and must therefore be acquired explicitly using `fence.proxy.tensormap::generic.acquire`
   when needed", and `git grep` shows **none** of this tree's three TMA kernels emit that fence while
   the Windows build places the descriptor in a `.global` buffer rather than `.param`. Adding the fence
   is correct and cheap. **It did not fix the reproducer**, so it is not the whole cause, and no
   document maps a TMA-constraint violation to illegal-instruction or to an illegal memory access.

   **So the deferral stands, and this is not a close.** What changed is that the next attempt starts
   from a two-second reproducer instead of a 30-minute sanitizer run, and that the honest answer to
   "is TMA available on this GPU" is yes — so the divergence from upstream should not be written up as
   a hardware limitation, because the evidence does not support that. **The "four ruled-out causes" this
   paragraph originally claimed are not on that list; they were never tested, and the next attempt does
   not start from them.** (The deferral itself was later overtaken anyway — see the resolution block:
   the cause was the `alignas(128)` descriptor, and `a5077adf` fixed it.)

   **2026-10-01, later: TMA availability is now MEASURED, not inferred, and the answer is yes.**
   `bench/ops/linear_bench.cu` run at `--qtype BF16 --n 14336 --k 5120 --t 128` selects
   `launch_bf16_tma_mma` — `src/ops/linear/bf16/shapes/n14336_k5120.cu:24-25` routes `tokens <= 128`
   to it (line 24 is the `if`, line 25 the `return launch_bf16_tma_mma<…Bf16A16TmaR64T128K64S2…>`;
   the `:27` this used to cite is the `tokens <= 192` branch, one below it),
   and unlike the FP8 route it carries **no `_WIN32` gate**. It completes and exits 0 on this RTX 5090.
   (The bench times rather than validating against an oracle, so exit 0 proves the kernel *ran without
   faulting* — which is exactly what the control needed, and is not a statement about its numerics.)

   **That control is what makes the next hypothesis specific, and it is the strongest lead this item
   has: the working route and the failing route differ in the TMA variant they issue.**

   | | working BF16 route | failing FP8 route |
   |---|---|---|
   | descriptor | `bf16_tma_map` — **rank 3**, K factored into 128-byte sectors | `fp8_tma_map` — **rank 2** |
   | copy instruction | `cp.async.bulk.tensor.3d` | `cp.async.bulk.tensor.2d` |

   **[SUPERSEDED] Not yet tested**, and it was the obvious next step: that
   `cp.async.bulk.tensor.2d` is what fails on sm_120a while `.3d` works. If so the fix is not exotic —
   it is to give `fp8_tma_map` the rank-3, sector-factored shape `bf16_tma_map` already uses, which is
   a descriptor change inside one function. The standalone reproducer already has a rank-2/2d arm that
   reproduces the fault in two seconds; adding a rank-3/3d arm is the measurement, and it has not been
   run. **Do not record this as the cause until that arm passes.** — It was never the cause, and the
   arm is moot: the reproducer's rank-2/2d fault was its own `try_wait` bug, corrected TMA runs in
   *both* ranks, and the resolution block records the actual cause (`alignas(128)`) and the fix.

   **2026-10-01, later still: an independent Windows port runs this exact route, and we do two things
   it does not.** `Wallawalla47/ninfer-custom` has the same `fp8_a8_tma_mma.cuh`, and read at its
   `master` it matches ours line for line in every respect that has mattered so far — `struct
   alignas(128) Fp8TmaDescriptors`, the `_WIN32` pointer-passed descriptor macro, a **rank-2**
   descriptor with `dimensions{k, …}` and a single `strides{k}` entry, and **`cp.async.bulk.tensor.2d`**.
   Its comment names the same defect in the same terms: *"MSVC cannot pass the over-aligned
   (alignas(128)) CUtensorMap struct by value as a `__grid_constant__` parameter (C2719), so on Windows
   the descriptors are pointer-passed"*. So the C2719 and the pointer workaround are **not this port's
   invention** — another Windows port hit them independently and resolved them the same way.

   **It also does not gate the route off.** Its launch site calls the TMA kernel on `_WIN32`; ours
   dispatches the pre-merge body instead. Two concrete differences, both of which we lack:

   1. **It emits the tensormap proxy acquire that we established our tree is missing.** Inside the
      kernel, under `_WIN32`, before the first copy: *"A staged tensor map was written through the
      generic proxy; each 128-byte map needs its own acquire for the TMA (tensormap) proxy before its
      first use"*, then `acquire_staged_tensor_map(&descriptors.activation)` and the same for
      `.weight`. That is precisely the `fence.proxy.acquire.tensormap::generic` requirement from PTX
      ISA §8.8.4, applied once per 128-byte map.
   2. **Its descriptor has a persistent home.** It uses `core/tma_descriptor_staging.cuh` — a
      `TmaDescriptorStaging<Fp8TmaDescriptors>` singleton with a device buffer reused across launches —
      and the launch comment reads *"Staged once: every token-slice launch below reads the same copy,
      in stream order."* Ours `cudaMallocAsync`es and `cudaFreeAsync`es a block **inside every
      launch**, relying on both being stream-ordered to be safe.

   **This was the strongest lead this item had, and it partly answers "why does nobody else hit this":
   another Windows port runs the route, with the same descriptor shape and the same MSVC workaround, and
   it does the two things this tree does not.** It also weakens the 2d-versus-3d hypothesis above,
   because that port uses `.2d` and is not reported as disabled — so `.2d` is **not** shown to be the
   problem, and that hypothesis should be demoted rather than pursued first.
   **Both of this item's live leads are now closed as wrong.** The 2d-versus-3d theory was withdrawn
   outright, and neither this port's proxy-acquire fence nor its persistent descriptor staging was the
   cause — `a5077adf` deleted both the staging buffer and the pointer-passed descriptor. What this
   comparison did establish, and what survives, is that the `alignas(128)` attribute is *this* fork's
   divergence from upstream, carried identically by two independent Windows ports, and that no
   upstream or CUTLASS fix exists to wait for.

   **What is not established:** that their port *runs* this route successfully. I read their source, not
   their CI, benchmarks or issues — nothing here is a measurement that their FP8 TMA route executes on
   Windows. And the acquire fence alone did **not** fix our standalone reproducer, so it is not
   sufficient by itself; the staging lifetime may be the part that matters, or the pair may be.
   *(The reproducer's "did not fix" reading is itself void — it faulted in its own wait before running
   any TMA instruction. Neither the fence nor the staging is load-bearing, which is now a measured
   result rather than an open question.)*

**The suite grew 133 to 135 from this merge**, which `tools/release/test_baseline.json` now records, and
`ninfer_qwen3_5_dflash_prefill_real_test` was added to `required_tests`: it reads `NINFER_TEST_ARTIFACT`
and skipped without it, and it passed against this product's artifact in 6.73 s, so the evidence the
gate exists to demand is available here. `ninfer_bench_fixtures_test` needs no artifact and so belongs to
neither list.

**The count-derivation method, corrected, because as previously written it does not reproduce the
number.** The earlier text here said the count came from "enumerating both registration macros across
all thirteen registration sites under `tests/`". Run against the current tree that yields **135, not
137**, across **eleven** sites, so the method as stated was wrong twice. Enumerated by hand on
2026-10-01: `ninfer_add_test` / `ninfer_add_op_test` are called from **eleven** files under `tests/`
(`artifact/`, the five `cmake/` files except `NinferTests.cmake` which only defines the macros,
`models/qwen3_5/`, `ops/tests.cmake` and its four `ops/linear*/` includes) for **96** literal names;
`ops/tests.cmake`'s `ninfer_op_tests` list adds **28**; `CoreTests.cmake`'s `sync_modes` loop adds
**4**; and **10** further registrations are bare `add_test(NAME …)` calls that use neither macro
(`ninfer_public_api_test`, `ninfer_device_sync_invalid_test`, `ninfer_device_sync_empty_test`,
`ninfer_chat_templates_test`, `ninfer_artifact_writer_interop_test`,
`ninfer_context_cost_measure_test`, `ninfer_kv_cache_append_nvfp4_test`,
`ninfer_kv_cache_append_k8v4_test`, `ninfer_release_launcher_generation_test`, and the fourth
`sync_modes` entry). 96 + 28 + 4 + 10 = **137**, which `ctest --test-dir build-test -N` reports as
"Total Tests: 137" and `tools/release/test_baseline.json` records as `suite_size`. The two macros
alone miss the two loop bodies (a regex over them matches the templates `ninfer_${op}_test` and
`ninfer_device_sync_${mode}_test`, not their expansions) and all ten bare `add_test` calls — which is
exactly the 135-vs-137 gap, and why the earlier text was confident and wrong. Deriving a count from
recorded numbers alone is what produced two wrong counts in this file before, so the enumeration is
worth keeping; it has to be complete to be worth anything.

**The 135 above is this merge's figure and it was correct on the day. The suite is now 137** — the
`topk_logprobs` Op and its test took it to 136 on 2026-09-29, and `ninfer_topk_record_test` took it
to 137 on 2026-09-30, both after this was written. The intermediate "136" this paragraph used to
quote was already stale on the day it was written: it named only the first of the two additions.
`tools/release/test_baseline.json` is the authority and records 137, corroborated by
`ctest --test-dir build-test -N`, so a reader comparing this file against the baseline should expect the
difference rather than treat either as wrong. Annotated rather than edited in place: a dated record of
what a merge produced should not be rewritten to match a later state.

**Not established (as of the merge; superseded since):** why the FP8 TMA kernel faults. It may be an
sm_120a limitation or a defect in upstream's kernel, and telling upstream it faults on a consumer
Blackwell target is worth doing either way. **This is answered now and neither answer is what it
predicted:** the cause was neither, and the resolution block at the top of this item records it — the
`alignas(128)` descriptor, fixed by `a5077adf`. **RESOLVED 2026-10-02: all nine `PORT-DISPATCH`
> sites are ungated**, each on its own per-route oracle, with the transport measurement and its
> methodology recorded above. Nothing from this item remains open.

---
---

### 13. Test-suite review: one real coverage gap found and closed, and two of my own claims were wrong - **CLOSED 2026-09-29**
**2026-09-29, after the `d44ab584` merge. The suite is green at 133/135; the two failures are
`dflash_real` and `moe_real`, which fail by construction because this product ships no `dflash`
component and no 35B-A3B MoE checkpoint. `check_test_baseline.py` GATE PASSED.**
*(A dated record of that day's run, not the current size: the suite is **139** now —
`tools/release/test_baseline.json` `suite_size`, corroborated by `ctest --test-dir build-test -N`.
The current known failures are the same two.)*

**The gap: 265 lines of shipped kernel with no unit coverage.** `candidate_selector_path` dispatches
on `predecessor_codebook.qtype` alone, so the NVFP4 route runs whenever the predecessor is NVFP4. It
is a separate kernel set -- a direct walk and a lattice pair -- and `test_candidate_selector.cpp`
built only `QType::BF16`, so none of it was reachable from a test. It now sweeps kSteps 1..15, which
crosses the route boundary at kSteps<=4 (direct walk) and above it (lattice pair).

The oracle is independent where independence is possible. E2M1 and E4M3FN are decoded in the test
from the formats' own definitions rather than by calling the device codec, so this is a third
implementation agreeing with the kernel and with `tools/artifact/formats.py` on the converter side.
The scale plane is addressed with the canonical `nvfp4_scale_byte_offset` expression, while the kernel
carries its own hand-written copy in `codebook_scale_offset`; three copies agreeing is evidence, and
the limit is stated in the test rather than glossed, because all three implement one convention and a
wrong convention agreed upon would still pass. One asymmetry is modelled rather than smoothed over:
the kernel rounds the successor row to bf16 and leaves the predecessor in fp32, so a test treating
both alike would fail on rounding noise rather than on a defect.

**The assertion was falsified before it was trusted, and the first attempt was invalid.** Perturbing
the fixture's scale-word table changed nothing, because that table feeds both the bytes the kernel
reads and the oracle's expectation, so both moved together -- self-consistency, not an oracle.
Perturbing only the oracle's E2M1 magnitude (1.5 -> 1.6) fails as it should, with a localised
diagnostic on the lattice route. A test that cannot fail is worse than no test.

**The new coverage found a real defect, which is the point of having it.** Each codebook is validated
independently by `require_codebook` and the route is chosen from the predecessor alone, so an NVFP4
predecessor with a BF16 successor passed validation and reached the NVFP4 kernel, which dereferenced
the successor's null `scales` pointer: an illegal access at 0x38AC, which aborts the process rather
than failing the request. compute-sanitizer named the frame (`selector_walk_nvfp4_kernel` at
`src/ops/candidate_selector/nvfp4/candidate_selector_path_nvfp4.cu:148`, via `score_row` defined at
line 95, whose `a.successor_scales` read at line 108-111 is the dereference in question) and a
temporary probe printing the pointers
confirmed the main loop's own operands were correct, which is what attributed the fault to the mixed
case rather than to the new test. The wrapper now requires the two qtypes to match. Both codebooks
come from one artifact in production, so the guard costs nothing, and
`ninfer_dflash2_nvfp4_routes_test` still passes, which is the evidence that the real route is
unaffected. The test asserts the refusal, so the hole cannot reopen silently.

**A suspected second defect was reported here, and it was wrong; the correction is the point.** The two
scale-plane expressions in this kernel look different -- `codebook_scale_offset` for the predecessor is
the canonical `(token/128)*4*512 + (group/4)*512 + in_tile + group%4`, while the successor staging in
`score_row` reads four bytes at `(token/128*4 + lane)*512 + in_tile` -- and that reads as though
`lane` were standing in for `group/4`. It is not a defect and there is nothing to fix. They differ in
granularity, not meaning: substituting `group = lane*4 + j` into the canonical form gives
`(token/128)*4*512 + lane*512 + in_tile + j`, which is the successor's base plus the j-th byte of its
four-byte copy. So `lane` selects the 512-byte scale tile, and the four bytes of the copy supply that
tile's four `inner_k` values.

Established three ways rather than by re-reading, because the first reading was what produced the
error. By measurement: perturbing the successor staging to read the neighbouring scale tile makes the
new test fail outright -- `max_abs=1, actual=1 reference=0` -- and reverting it with a real recompile
makes the test pass, so the test genuinely constrains the group mapping. That coverage is what this
wave added, and it is why the question was answerable at all. By the specification: FlashInfer
documents the NVFP4 128x4 block-scale swizzle as
`m_tile_idx*k_tiles*512 + k_tile_idx*512 + outer_m*16 + inner_m*4 + inner_k`, with `outer_m = row % 32`
and `inner_m = (row % 128) // 32`, which is `nvfp4_scale_byte_offset` in this repository exactly, and
the kernel's own comment already says the staging gathers "the row's four 4-byte scale groups, one per
512-byte K16M128x4 tile". And by the algebra, as above.

One process note, because it nearly produced a false result here. After reverting the perturbation the
test still failed, and the cause was a build that did not recompile the `.cu`: the reverted source was
on disk while the perturbed object was still linked. The file matched HEAD and the tree was clean, so
every available check said the source was correct while the binary was not. Forcing the recompile made
it pass. A green source tree is not a green binary, and a test result is evidence about the artefact
that was actually run.

**The golden that had no in-tree anchor: labelled, not invented.** `test_engine_prefix_real.cpp`
asserted the thinking prompt at 58 tokens with no derivation, while the plain 18 is proved against the
reference tokenizer by an exact id vector. It cannot be promoted to the same status, and the test now
says so: the no-thinking render has an external reference because `test_frontend.cpp` pins it byte
for byte, but the thinking prompt's preamble is this port's own template addition, so the reference
tokenizer never produced that string. The totals are asserted separately so a failure localises to
the preamble rather than to an unexplained sum, and a third assertion rejects a thinking render that
is not longer than the plain one -- a renderer fault that would otherwise be misreported as a
template edit.

**The timeout exposure I reported does not exist.** I said the attention test was at 96% of a 900 s
gate timeout and quoted a justification for the 900 s value. Both were wrong: no `TIMEOUT` property is
set anywhere in the test tree, so the limit is ctest's 1500 s default, and no comment in the tree
mentions 262 s. The test measures 885.04 s, which is 59% of the default. Neither the false claim nor
its fabricated citation reached a file, so the correction is this paragraph.

**The per-storage split cost is now measured rather than unknown.** Upstream deliberately consolidated
the two per-storage ctest wrappers into one invocation taking `--kv-dtype`, because the wrappers had
to be listed in the baseline and skipped individually. Whether splitting them back would duplicate
CUDA setup was the open question. Measured, per storage: bf16 122.6 s, int8 147.6 s, nvfp4 180.7 s,
k8v4 211.1 s, fp8 222.2 s -- and the five separate runs sum to 884.2 s against the consolidated
885.04 s, a difference of 0.8 s. The shared setup is therefore immaterial, and a split would be
genuinely balanced, taking the longest entry from 885 s to 222 s. It is still not worth doing: there is
no timeout exposure at 59% of the default, the split re-creates the per-test baseline bookkeeping
upstream removed on purpose, and the measured saving is zero. Recorded so the next reader does not
re-derive it, and so nobody repeats the 900 s claim.

**Reviewed and found sound.** All fifteen shared test files that genuinely diverge from upstream were
reviewed hunk by hunk: zero hard coverage weakenings. Three soft changes are each deliberate with a
justification that checks out -- among them `test_materialization_budget.cpp`, where the inputs moved
from `80'000 ms` to `100 ms` and both clamp to the same 5 ms grant, so the renewal assertions are
untouched. Also verified field by field: the test's BF16 codebook `Weight` matches what production
materialises in `weight_view.cpp`, including the ten fields production leaves at defaults by
returning early for `Contiguous`. And the `try/catch` added to `test_engine_dflash_real.cpp` does
work -- it was an unhandled exception reaching `std::terminate`, not a `noexcept` frame, and
`terminate` calling `abort()` is what produced the `0xC0000409` that named nothing; the test now
prints the engine's own `missing component dflash`.

---

> **THE 261.7 FIGURE IS RETIRED. Measured 2026-10-02 on a serving lane, not the bench.**
> `tools/bench/first_request_lane.py` runs two arms, one fresh `ninfer-serve` each, eight identical
> requests per arm, temperature 0, with a SHA-256 of every reply recorded. On the `--no-prefix-reuse`
> arm -- the clean one, since it has no cache to be cold against -- **there is no first-request
> effect**: decode varies 0.2% across eight requests (1.2662 s against 1.2639 s), and round count and
> accepted tokens are identical every time. That is below this card's own noise floor, measured at
> 0.0-0.3% late-session and 5.8-7.6% early.
>
> **The sign is opposite.** Request 1 is the SLOWEST request, not the fastest -- on the reuse arm it
> reads 1.1104 s against 1.0642 s, with prefill 0.0537 s against 0.0260 s and TTFT 0.0877 s against
> 0.0366 s. The recorded figure had it 54% FASTER than requests 2-7.
>
> Hypothesis #80 is falsified for this workload: `t = 1 + accepted/rounds` is constant within each arm
> and identical across all eight requests, so there is no per-request variation for token-count-driven
> kernel selection to explain. The figure was already confounded on output length and warm-up state by
> vLLM #17472; it is now measured and gone.
>
> **Two things this measurement cannot say.** The arms are not comparable to each other: with reuse
> on, requests 2-8 take the `private_response_replay` path with 39 hit tokens and run 78 rounds
> against the no-reuse arm's 89, so the ~12% gap between arms is partly replay rather than a cache
> effect. And the absolute ~197 tok/s is not a lane figure -- 256 fixed output tokens is far too short
> for steady state -- so it is not comparable to the published 319.2.
>
> **A harness failure worth recording.** The first run of that harness put the entire output budget
> into thinking (`model_thinking_tokens` equal to `completion_tokens` in all sixteen requests), so
> every reply was empty and the harness still exited 0 printing a tidy table. The identity control
> could not catch it, because sixteen empty replies are sixteen identical replies. The harness now
> raises on an empty reply, which is the half that was missing.

### 14. A lane's first request returns different, shorter text — **RESOLVED 2026-10-02: not a first-request effect**
**Promoted out of item 6, where it was recorded under a DONE heading and therefore invisible to anyone
reading the list.** That is the only reason it moved; the content is item 6's, unchanged.

**What is observed.** After a server start, a client's *first* full-length request returns faster than
every later identical request and returns **different, shorter text** from the same seed. Requests 2..n
are byte-identical to each other. On the lane where item 6 caught it, that first request read 261.7 tok/s
against 169.8 for requests 2-7. It is **invisible at temperature 0** — the greedy digest is stable from
request 1 — and visible at the sampling temperature the bench uses.

**Why item 6's fix did not and could not address it.** The bench now discards one full-length warmup, and
skips it in `parse_spec_jsonl` because the engine appends to its request log for the whole session. That
is a correct fix for the *measurement*. It is explicitly not a fix for the behaviour: a real client still
gets different, shorter first-turn text. Fixing the harness and fixing the engine are two different jobs
and only the first has been done.

**MEASURED 2026-10-02: the transient as described does not reproduce, and the description was wrong
about what it compared.** At the temperature this item says it is visible at — 0.7, with the seed
pinned so that any divergence is the engine and not the sampler — eight identical requests in one lane
were byte-identical in both the prefix-reuse and the `--no-prefix-reuse` arm, and repeated across
three separate process launches with the same digests each time. The cache did engage: requests 2..8
took `private_response_replay` with 39 hit tokens, and the text was still identical, because replay
returns what request 1 computed.

The observation came from comparing **two separate bench runs**, `--warmup 0` against `--warmup 1`
(`tools/bench/first_request_loop.cmd`), not requests inside one lane. Those are two configurations,
and [ADR-0002](adr/0002-speculation-not-bit-identical.md) already records that different prefill and
kernel paths deterministically change sampled output — "a near-tie can flip and the continuation
diverges" — with the explicit consequence that "a user comparing output across profiles will see
differences, which is correct". So this is that documented behaviour read as a first-request effect,
not a first-request effect.

Measured here as a control on that reading, because it is the same mechanism from another side: the
two arms above differ only in prefix reuse and the six capacity flags the engine requires with it,
and for the **same prompt, seed and temperature** they produce different text — `d9fc0df811cf40f2`
against `df5604c53ea0ae6f` — byte-stable across three process launches. Cache-capacity configuration
is therefore another axis on ADR-0002's axis, and both request 1s are path `root` with zero hit
tokens, so the difference is in the prefill the plan chose rather than in anything cached. That also
falsifies the alternative I reached for first, which was cross-process nondeterminism: the digests
reproduced exactly every time.

**A separate finding, now diagnosed 2026-10-02: the cache is not broken, and the zeros have three
documented causes.** `tools/bench/check_shared_prefix_reuse.py` measures five conditions in one lane,
each with a control:

| condition | prompt_tok | hit | path |
|---|---:|---:|---|
| repeat (control) | 3,282 | 3,275 | `private_response_replay` |
| grow, unmarked | 3,824 | 0 | `root` |
| shuffle (negative) | 3,282 | 0 | `root` |
| grow, marked | 3,824 | **3,817** | `private_response_replay` |
| two conversations, system prefix declared | 2,773 | 0 | `root` |

A growing conversation carrying an explicit `prompt_cache_breakpoint` **is** served 3,817 of 3,824
tokens. Three mechanisms produce the zeros, all intended:

1. **`/v1/chat/completions` owns its own write policy.** `openai_common.cpp:177` clears
   `allow_engine_automatic_shared_prefixes` on every OpenAI request (reason at :175), and
   `parse_openai_prompt_cache_breakpoint` reads a content **part** — a plain string is not a part. An
   unmarked request therefore offers no shared-prefix write, and `root` is correct.
2. **Where a request extends a resident one, replay wins the valuation** at that frontier
   (`profiles.py:388-392`; ADR-0009's third arm, where the shared candidate is offered and accepted
   by the Program, and the planner still selects `private_response_replay`).
3. **The genuinely shared read did not reproduce.** ADR-0009 measured `shared prefix` at 301 of 344
   tokens. The shape here now matches its recipe — declaration on the system content part, two
   conversations diverging at the first user turn — and still returns 0, while publication works (the
   startup line reads `shared 7`). The read is refused at the shortlist key
   (`resource_manager.h:329`).

**Warm TTFT stays unmeasured, and the reason is now attributed rather than open.** The only shape that
reached a cached prefix was `private_response_replay`, which returns a stored response and never
decodes. The shape that gives a real encode-and-decode over a cached prefix — `shared_stable_prefix`
— did not reproduce, and the cache-selection counters say why:

```
shared_stable_prefix       0
shared_reuse_candidates    0
shared_reuse_declined      0
shared_reuse_key_mismatch  0
```

All four zero means **no shared entry ever reached a reuse plan** — not that one was refused at the
shortlist key, which is what `shared_reuse_key_mismatch` would have shown. So on this lane a
`SharedStablePrefix` candidate is never even proposed for reuse.

**Two corrections to what I said earlier in this same session, both from the same misreading:**

- The counters are emitted after all. `format_throughput_json` (`src/serve/request_log.cpp:606`)
  writes them on a **`throughput`** record, and I reported that no shipped surface emits them because
  my analyzer filtered the log to `request_done` and discarded that record. They need
  `--log-stats-interval-ms` set (default 5000) or the shutdown tail; without it the log holds none and
  a reader concludes the opposite. `report_serve_phases.py` now reads the record and **fails loudly**
  when it is absent, instead of reporting zeros as "not emitted".
- The startup line `private 8 | shared 7 | anchors 4` is the **configured capacity**
  (`operational_log.cpp:488-495`), not a count of published prefixes. I read it as evidence that
  "publication works, so the read is refused". It says nothing about publication. The counters are
  what measure publication, and they say it did not happen here.

Neither correction changes the conclusion that the cache is not broken: a marked growing conversation
is still served 3,817 of 3,824 tokens, and all three mechanisms producing the zeros remain intended
(OpenAI's own write policy; plain string content carrying no marker; replay outbidding shared at the
whole-prompt frontier). What is left unexplained is only why ADR-0009 measured the shared read being
served on a lane whose configuration is now verified identical to this one.

**Re-scoped 2026-10-01: this is two separate items, and the text half is answered.** The four candidate
causes listed above are not equally live. External research, and one in-tree fact, split them:

* **CORRECTED 2026-10-02: not prefix caching.** This bullet named prefix caching as the mechanism. It is
  not: measured at temperature 0.7 with a pinned seed, the cache engaged on requests 2..8
  (`private_response_replay`, 39 hit tokens) and the text was byte-identical to request 1's, because
  replay returns what request 1 computed. The real mechanism is [ADR-0002](adr/0002-speculation-not-bit-identical.md)'s —
  a different prefill or kernel path deterministically flips a sampling near-tie — and the original
  observation reached it by comparing two bench runs rather than two requests. The bullet is kept so
  the attribution that was acted on is visible alongside its refutation.
  SGLang's own FAQ states two identical requests can differ even at temperature 0 and names prefix
  caching as a distinct cause of the indeterminism. vLLM **#40896** (open) is this item's question
  verbatim — run 1 returns A, runs 2..N return B≠A, restart returns to A, and
  `--no-enable-prefix-caching` makes it deterministic — with a maintainer answering "we haven't fully
  supported determinism with prefix caching currently", and determinism roadmap **#27433** still listing
  it unticked. llama.cpp ships `cache_prompt` on by default and warns in the option's own docs that it
  "can cause nondeterministic results". This port has a context cache with incremental encode, so the
  mechanism is present by construction. The sharpest corroboration is a comment on #40896 measuring
  **logprobs rather than token ids**: max delta 4.6e-02 on H100 with **identical token ids**, bit-exact
  under `VLLM_BATCH_INVARIANT=1`. That is exactly this item's signature — greedy stable from request 1,
  sampling divergent.
* **An uninitialised RNG is ruled out, in-tree.** This port's sampler is counter-based with no mutable
  RNG state — `sampling_uniform(seed, position, purpose, sub)` — so there is no generator that could be
  cold on the first request. The logits or the logical position must differ instead.
* **The *speed* half is undocumented, and its sign is opposite to every documented case.** Nothing found
  anywhere documents a first request that is **faster**. Every documented first-request effect — JIT,
  graph capture, allocator and cuBLAS warm-up — *adds* latency, and this port does not call cuBLAS. The
  recorded 261.7 against 169.8 tok/s is not explained by anything in the literature.
* **One in-tree lead for the speed half:** upstream issue #80 documents that the token count
  `t = Σ(1 + accepted drafts)` selects the kernel and therefore the reduction order. That is consistent
  with the first request *accepting better* (64.6 % against 32.7 %) — a different `t` selects different
  arithmetic, which can be faster and can change sampling. This is a hypothesis with a mechanism, not a
  diagnosis.


> **A second hypothesis the item did not have, and it is testable with what now exists.**
> The speed asymmetry may be **arithmetic, not a kernel-selection effect**. Established above: the first
> request returns *different, shorter* text. If it also terminates after fewer rounds, then a tok/s
> computed over the whole request rises simply because prefill dominates and decode rounds were removed
> -- no kernel changed, nothing got faster. That would make the text and speed halves **one cause**,
> not two.
>
> **The discriminator is clean and needs one run.**
> * If it is arithmetic, **per-round cost is unchanged** between request 1 and request 2 and only the
>   round count differs.
> * If it is #80 kernel selection, **per-round cost differs** -- a different `t` selects a different
>   kernel, hence a different reduction order and a different cost.
>
> The bench already reports `spec_rounds` and `spec_acceptance_rate`, and item 5 added per-position
> Recall@1/Recall@16/path acceptance, so both quantities are available. The #80 hypothesis predicts a
> *fall* in per-round cost on request 1; the arithmetic hypothesis predicts no change. Do not fold the
> speed half into the text half until that run says which.
>
> **The literature is consistent with the cost being one-time, and it is the reason this is worth
> testing rather than assuming.** vLLM #23787 measured first prompt 5.69s / second 0.96s / third 0.50s and
> attributed it to CUDA **lazy module loading** (`cuModuleLoadData` on first launch of each kernel)
> graph capture and JIT -- then showed a 1-token warm-up call pays it upfront and the first "real"
> prompt drops to 0.97s. Every documented first-request effect *adds* latency, which is exactly why a
> first request that is *faster* needs a mechanism that removes work rather than adds warm-up. This
>
>
> **MEASURED 2026-10-02, and the bench CANNOT reproduce this item: request 1 and request 2 are
> identical to the digit.**
>
> `tools/bench/first_request_loop.cmd` runs the same binary, artifact, seed and corpus twice, differing
> only in `--warmup 0` (request 1) versus `--warmup 1` (request 2), and
> `tools/bench/first_request_report.py` reports the discriminator:
>
> | | tok/s | rounds | decode total | per-round | acceptance | drafted/accepted |
> |---|---|---|---|---|---|---|
> | request 1 | 172.31 | 103 | 1485.7 ms | 14.425 ms | 0.211111 | 720 / 152 |
> | request 2 | 173.65 | 103 | 1474.2 ms | 14.313 ms | 0.211111 | 720 / 152 |
>
> rounds +0.00%, per-round cost -0.77%, and acceptance identical to every digit.
>
> **This is not the ARITHMETIC verdict the reporter prints first, and the difference matters.**
> Arithmetic predicted *same work, different round count*. What was measured is *no difference at all* --
> which means the two runs are not observing different requests. The reporter says so itself in its own
> NOTE line, and that line is the finding:
>
>
> **THE PATH FORWARD IS VALIDATED, and the mechanism is now visible.** The IETF benchmarking-methodology
> draft is explicit: "When cold start performance is being measured (Model Load Time, Cold Start
> Latency), **warm-up MUST be skipped**" and cold starts should be "benchmark[ed] separately rather than
> hiding" them. So `--warmup 0` is the prescribed instrument, and it returned no difference -- which
> locates the cold cost rather than leaving it unexplained. The engine reports it:
>
> ```
> load_seconds      5.374      upload_seconds    3.304      <- cold start, ~8.7 s, reported separately
> measured request: prepare 0.0000  vision 0.0000  prefill 0.0136  decode 1.4857  total 1.5011 s
> ```
>
> The one-time cost is **8.7 s of load and upload**, not anything in a request. That is why request 1
> and request 2 are identical: by the time the first REQUEST runs, the cold work is already done and
> paid. The bench is not failing to measure the effect -- the effect is not there to measure.
>
> **So the path forward is confirmed, and it is specific:**
> 1. The 261.7 figure is a SERVING-lane observation, not a bench one. Confirming it needs a fresh server
>    per measurement, because that is the only boundary where "first request" and "cold" coincide -- the
>    engine has initialised, but the lane's caches and per-request state have not.
> 2. If it does not reproduce there, the figure is retired as unverified rather than left standing.
>    **EXECUTED 2026-10-02: it did not reproduce, and the figure is retired.** No first-request
>    effect exists on the clean arm — decode varies 0.2% across eight requests — and request 1 is
>    the SLOWEST request, the opposite sign to the recorded figure.
> 3. The part of item 14 that survives either way is the text half, which was observed on that same serving
>    path and has a named mechanism.
>
> What does NOT need revisiting: the eight ungatings, each with its own per-route oracle and a green
> suite. Their evidence never depended on item 14.
> **The bench is the wrong instrument for this item.** Whatever the first-request effect is, it is not
> visible here -- most likely because engine initialisation has already paid it before the first
> *request* runs, so `--warmup 0` measures an already-warm request.
>
> **What this does and does not settle.** It reproduces the steady-state figure (request 2 reads 173.65
> against the recorded 169.8 for requests 2-7) and it does NOT reproduce the 261.7 first-request figure at
> all. So the recorded number needs its provenance checked against the SERVING lane, where a real
> client's first request hits a cold lane -- which is a different instrument with a different reset
> boundary. Until that is done the 261.7 should be treated as unverified, not as an engine fact and not
> as disproved. **RESOLVED 2026-10-02: measured on a serving lane and retired.** The serving-lane
> measurement above supersedes this caveat; the 261.7 is disproved as a first-request effect, not
> left standing.
>
> This is the "harness error reported as a product defect" pattern the project's own review skill names.
> The first half of this item -- the text divergence -- is unaffected: it was observed on the serving
> path, it has a named mechanism (prefix caching), and item 5's instrument measures it directly.
> **Both halves of that sentence are wrong, measured 2026-10-02.** Prefix caching is not the mechanism:
> the cache engaged and the text stayed byte-identical. And item 5 measures recall/acceptance on fixed
> ids, not serving-lane output, so it never measured this. Corrected above.
> **RESEARCHED AGAIN, and the answer is that there is no documented mechanism -- which is the finding.**
> Searching specifically for a first request that is FASTER returns the standard TTFT-vs-TPOT framing and
> nothing else. Every published first-request effect -- lazy module loading, graph capture, JIT,
> allocator warm-up, PTX compilation -- ADDS latency, and vLLM #23787 measures the opposite direction
> (5.69s / 0.96s / 0.50s for the first three requests, paid down to 0.97s by a 1-token warm-up).
>
> So the warm-up family is ruled out by absence, not merely by argument, and no amount of further reading
> will supply a mechanism. **That is why the answer here is a measurement and not a citation:** the
> per-round-cost discriminator already recorded settles it in one run -- unchanged cost means the speed
> asymmetry is arithmetic (the round count differs, the per-round work does not), and changed cost
> points at the token-count-driven kernel selection in #80. `spec_rounds`, `spec_acceptance_rate` and
> item 5's per-position counters already supply both quantities, so nothing needs building first.
>
> One caution on the arithmetic hypothesis, because it is the tidier of the two and I nearly wrote it down
> as settled: a SHORTER output does not obviously raise tokens/second. If the metric is generated
> tokens over total wall time, fewer tokens with the same prefill pushes the rate DOWN, not up. If it is
> decode-only, round count cancels out of the ratio entirely. So the arithmetic story may be wrong too, and
> that is precisely why the discriminator has to compare per-round cost rather than be argued.
> Recorded here so the next reader does not inherit it as the answer.
> port does not call cuBLAS, so the cuBLAS-init share is not available as an explanation.
**So: the text divergence is not a defect to fix, it is documented engine behaviour** — but it is still
worth stating explicitly, because the bench compensates for it silently and a reader may remove the
warmup discard as redundant. **The speed asymmetry is the real open item**, and it is the opposite
shape from every published first-request effect, which is why it deserves its own diagnosis rather than
being folded into the text half.
**Done when:** for the text half — satisfied, characterise it in `docs/serving.md` as intended
behaviour and cite the mechanism. For the speed half — identify the cause; the token-count-driven
kernel selection in #80 is the leading candidate and is checkable against `spec_rounds` /
`spec_acceptance_rate` already reported by `ninfer_bench`.
Full evidence: `docs/research/first-request-transient-and-msvc-tma-evidence.md`.

**Why it is worth an item despite looking small.** It is user-visible and it is deterministic: the same
seed gives different text on the first turn and stable text afterwards. A client that treats turn one as
representative — an eval harness, a regression test, anything comparing a first response to a reference
— is comparing against a different sample. It also means the bench's own warmup discard is load-bearing
in a way that is easy to remove as "redundant" by a later reader.

**Done when:** the cause is identified, and either the first request is brought to parity with the rest
or the divergence is characterised precisely enough to state it as intended behaviour. Either answer is
acceptable; leaving it undocumented is not, because the bench currently compensates for it silently.

### 15. Five suite failures with no cause recorded — **RESOLVED 2026-10-05: all five fixed, gate green**

**Why:** `tools/release/test_baseline.json` recorded `suite_size: 142`, `recorded: 2026-10-04` and a
`known_failures` of two, under a note reading "Green since 2026-09-19" — while the suite it describes
had five failures it did not mention. That file was last touched by `af5890b3`, which is HEAD, so the
claim was false at the commit that wrote it. Nothing had run the suite: `.github/workflows/gpu.yml`
was published and wired correctly, but **no self-hosted runner was registered**, so the GPU tier had
never executed — see item 23, which records the three blockers that had to be fixed and the first
run on 2026-10-06. The four failures had four different causes, which is why each was attributed before
it was touched rather than fixed as a group.

- `ninfer_qwen3_5_frontend_test` — **lost port work.** Upstream's `b9114396` rewrote `tokenizer.cpp`
  and the merge took upstream's version wholesale, so ADR-0010's cache layer went with it. The
  declarations survived in the port's `tokenizer.h`, so the tree compiled and linked with the cache
  dead: `splice_from_cache` never fired, `encode_cache_splices()` stayed 0, and the ids were still
  correct because the full encode still ran. Restored from `c42406ab` verbatim — 5 of 5 functions
  byte-identical once comments and blanks are dropped — and wrapped in `NINFER-PORT` markers, with
  `encode_with_boundaries_uncached` being upstream's own body renamed.
- `ninfer_request_log_test` — **a real defect plus a stale golden.** The producer emitted a `prepare`
  clause carrying `preparation.seconds`, the same figure the `prepared` clause already states, so
  every started line printed preparation twice. The golden separately expected upstream's comma where
  this port emits a pipe between the media and preparation clauses.
- `ninfer_http_error_handler_test` — **the merge added a contract without its implementation.** Two
  cases use status 400 and expect opposite results, and that pair *is* the contract: an unrecognised
  status on an Anthropic path returns `Unhandled` with the request ID established, while an OpenAI
  path still gets the generic envelope, because a client that parses every response as JSON otherwise
  crashes on httplib's plain-text page.
- `ninfer_qwen3_5_agent_continuation_real_test` — **not a bug.** Its resource-lifetime failure was a
  cascade of the frontend golden returning early.

**Resolved — a race in the test's read, not a live request and not an engine defect.** The message
reads as two live requests and is one. `running_requests` counts ANY occupied slot and
`terminal_pending_requests` counts occupied slots with `terminal_reason` set, so the counters are not
disjoint and `running=1 terminal=1` describes a single slot. The slot was real; the read was stale.
`settle_terminal_requests` runs the whole terminal sequence per request on the worker —
`program->finish`, `resources_.publish` for the cache owner, `resources_.finish` — and only then
`complete_success`, which sets `response_done` and wakes a waiting consumer. `remove_completed_slot`
and `publish_runtime_stats` come after that, and `runtime_stats()` returns the published snapshot, so
a read taken the instant the last `wait()` returns can return a snapshot from before that request
settled.

That **confirms** the architecture document rather than contradicting it: "the final response waits
for terminal resource and cache-owner settlement to complete" holds, because settlement precedes
`complete_success`. What lags is the stats publication, not the settlement.

A bounded poll then showed the counters reaching zero, which is what ruled out a leak. The poll's
iteration count is deliberately not recorded as evidence: a tight `runtime_stats()` loop contends
with the worker on the stats mutex, so it measures its own interference.

The check now yields until the counters drain, bounded, so a real leak fails rather than hangs. The
agent-continuation settlement check carries the identical latent race and passes only because it does
more work between its last `generate()` and its read, so it gets the same treatment.

**Full suite, 2026-10-05:** `99% tests passed, 2 tests failed out of 142` — `moe_real_test` and
`dflash_real_test`, both baselined — with `check_test_baseline.py` reporting `140/142 passed` and
`GATE PASSED: no regression against the recorded baseline`.

**Also recorded there:** `exercise_artifact_frontend` is upstream's and asserts upstream's artifacts
(16 thinking tokens); this port's measure 58, because their embedded template emits the
reasoning-effort instruction as a synthetic system message — exactly the 40-token delta, with the
no-thinking count matching upstream exactly at 18. The port now asserts its own goldens and names the
actual count on failure.

**Done when:** the `C=8` assertion is attributed and either fixed or recorded with its cause. —
**satisfied 2026-10-05.**

### 16. The `b9114396` merge took upstream's version of 40 files the port had work in — **COMPLETE 2026-10-06: four silent losses restored, every reproducible candidate classified**

**Why:** for every file upstream's `b9114396` changed, the merge result equals upstream's exact blob
while the port's pre-merge blob differed from upstream's pre-merge blob — 40 files, against a control
arm of 53 where the port's version was kept. An instrument that flagged everything would have put the
control near zero, so 40 is a real signal.

**The mechanical test, run.** The merge is `355dda55` ("merge: upstream b9114396, the context-cache
replacement, with the port's work preserved"), port side `c42406ab`, upstream side `a8e212ac`, merge
base `75a89050`. The question is the one this item left open: **at HEAD, is the port's change still
gone?**

**Three instrument errors, each caught by a different control, and that is the part worth keeping.**

1. The port's own change is its diff from the **merge base**, not from `b9114396~1`. Comparing
   against the latter counts every file where the port was merely *older* as a divergence — the tell
   was 28 files reported as "taken" with no port change at all.
2. The `taken`/`behind` labels were **inverted**: a file upstream had *not* touched between the base
   and `b9114396~1` is the one whose change was the port's own, and therefore the loss candidate.
3. "Upstream changed" is everything upstream changed between the base and its **tip**, not
   `b9114396`'s own diff — the merge took the tip. This one changed the set from 176 to 183 files
   and the result not at all, which is worth knowing rather than assuming.

**Result: at the merge 40 were taken; at HEAD 19 are still gone.** The other 21 were restored by the
port's own later work. Of the 19:

* **One was a real loss, and it is restored.** `src/models/qwen3_5/frontend/processor.cpp`'s
  `encode_rendered_chat` built a bare `byte_boundaries` vector and read it back with a running
  `boundary_index`, so the push order and the read order were two independent traversals of the same
  fields agreeing only by convention — a change reaching one side and missing the other mislabels
  every later frontier *silently*, and the count check saw only the cases where the totals differed.
  The port replaced that with a `FrontierEntry` ledger written and read **by label**, and the merge
  took upstream's version of it. Upstream's own `b9114396` change then added a **sixth** field to
  both traversals (`rewrite_checkpoint->recovery_offset`), which is the pattern the ledger exists to
  make safe. Restored, carrying `RewriteRecovery` as its own kind; verified by
  `ninfer_qwen3_5_frontend_test`, `ninfer_qwen3_5_prefix_real_test` and the full suite.
* **A second loss, also restored, and it is a correctness fix.** `src/runtime/engine/engine_core.h`
  had two port changes and the merge took both. The first is the other half of the `kv_capacity.h`
  deduplication — that header is still present and still included by `model_instance.h` and
  `causal_score_core.h`, but its *caller* in `memory_summary()` had been reverted to the inline
  publication it replaced, so the dedup was half-applied. The second is a settle-order fix whose
  comment states the defect: *"Free the slot and publish the post-release snapshot before waking the
  caller, so a `runtime_stats()` read after `generate()` returns reflects the released lane instead
  of the last decode boundary."* It has three coupled parts, and the third is the one that makes it
  safe: `cancel_active_requests` held `const auto& request = slots_[lane]` and `remove_completed_slot`
  resets that slot, so reordering without taking a copy is a use-after-free. The port's comment says
  exactly that. Both restored; `remove_completed_slot` → `publish_runtime_stats` → `complete_success`
  at both sites, and the deferred `if (changed) { publish_runtime_stats(); }` removed with them.
  A third settle site exists (the batch terminal path in a `catch (...)`) that upstream's newer code
  added and the port never touched; it is left as upstream has it, and named here rather than
  silently left inconsistent.
* **And the cascade, which is the part worth remembering.** This session earlier diagnosed a
  *stats-freshness race in a test* — "settlement precedes `complete_success`, `publish_runtime_stats`
  lags" — and fixed it **in the test**. With the engine fix restored, that race is fixed where it
  lived. The test-side workaround was the port working around a loss it had not yet found, which is
  what a silent merge loss looks like from downstream.
* **The patch test is not a classifier.** All 19 "do not apply", and `processor.cpp` is proof that
  this does not mean superseded: the code had moved *and* the fix was still needed. A conflict is
  evidence that the region changed, never that the port's intent is obsolete. `engine_core.h` is the
  same lesson a second time, in the same 19.
* **Two are false positives of the HEAD-equality test**: `mtp.cpp`, which this session reverted for
  being whitespace-only, and `chat_template.h`, which converged when the native renderer was
  withdrawn. Both equal upstream at HEAD for reasons that have nothing to do with this merge.

**What is left is reading, not scripting**: `docs/serving.md`, the five model-card READMEs,
`tools/bench/ttft/{README.md,profiles.py}`, and eight source files (`execution/text.{cpp,h}`,
`frontend/frontend.cpp`, `program/prefill.cpp`, `program/transactions/commit.cpp`,
`runtime/engine/{engine_core.h,model_instance.cpp}`), with `request_log.h` already classified above
as the port having been behind.

**Done when:** every one of the 40 is classified as a loss (restore it), a superseded port change
(drop it), or the port having been behind (nothing to do), with the classification recorded.
**Partially done 2026-10-06:** the instrument is corrected and recorded, the one confirmed loss is
restored and verified, and the remaining 18 are named above rather than left implied by a count.

**COMPLETE 2026-10-06, second pass: the counts in this item are wrong, and four more losses are
restored.** The instrument was re-run from the recorded geometry (`75a89050` base, `c42406ab` port,
`a8e212ac` upstream tip, `355dda55` merge) with the three corrected rules above. Reproducible figures:
**187** paths upstream changed (`183` by `git diff --name-only`), **25** where the merge took upstream's
blob and the port had work, **1** where it kept the port's, and **11** of the 25 still equal to
upstream at HEAD. Six definitions were tried -- port-work measured against the base, against
`b9114396~1`, or against upstream's tip; upstream-changed measured against the tip, against
`b9114396`, or with rename detection -- and none reproduces the recorded **40 taken / 53 kept / 19
gone**. Those three numbers are dropped rather than carried, and the 24 real candidates (25 minus
`request_log.h`, whose port blob already equalled upstream's tip at the merge) are classified here in
full.

**Restored, each verified:**

* **`src/models/qwen3_5/frontend/frontend.cpp` — prompt preparation's five host phases** (`1cd70d17`).
  The merge took upstream's frontend.cpp and dropped the `Clock` brackets for contract, convert,
  render, positions and cache prep, while `prepared_prompt.h` kept the fields; at HEAD nothing wrote
  or read them. The port had already noticed the reduced log and adapted to it in `df91de1e` -- whose
  message says *"Upstream b9114396 collapsed PromptPreparationStats to one total plus tokenize"* and
  concludes *"neither side was a lost port change, which is why this one is not a restore"*. **That
  premise is false**: neither `75a89050` nor `a8e212ac` contains `render_seconds`, so there was no
  upstream split to collapse -- the split was this port's (`1cd70d17`, author headpiece747, not an
  ancestor of `upstream/dev`). Restored with the public fields in `include/ninfer/types.h`, the
  `contract`/`convert`/`render`/`positions`/`cache prep` clauses in `src/serve/operational_log.cpp`
  (the false comment replaced by what the phases mean), the golden and the media-less assertion in
  `tests/test_request_log.cpp`, and a producer assertion in `ninfer_qwen3_5_frontend_test` that was
  **falsified in both directions** -- with the render bracket disabled it fails with "text preparation
  did not report the render phase", with it restored it passes.
* **`src/models/qwen3_5/execution/{text.h,text.cpp}` + `program/{prefill.cpp,execution_context.h}` +
  `program/speculative/mtp.cpp` — the `TextCallConfig` refactor** (`df2004da`). The per-call inputs
  (sampling, the Linear Attention state slots, the MTP proposal extent) arrive at construction, so the
  three-setters-after-construction sequence is unrepresentable, and the slot validation moves into the
  constructor where it cannot be skipped. `set_gdn_state_action` stays a setter and says why in place.
* **`tools/bench/ttft/profiles.py` — `cache-reporter-concurrency3-scaled`**, which
  `tools/bench/ttft/cases.py`'s `reporter-concurrency3-growth` case referenced: the merge dropped the
  profile, so that case could not run. Translated onto the post-`b9114396` CLI, which no longer has
  `--host-state-slots`, `--host-kv-mib`, `--max-shared-prefixes`, `--max-private-continuations` or
  `--max-long-anchors-per-continuation` (host state and KV are one `--host-context-mib` quota now and
  the descriptor counts are derived); the reporter's values keep the same 40% scaling.
  **Run end to end for the first time on 2026-10-06**, and getting there needed three more fixes the
  case could not have revealed before it ran: the campaign controller staged its artifact under
  `/dev/shm` with `os.statvfs` and copied it with `dd` (all POSIX-only, so the case could not run on
  this machine at all), and the case called `_assistant()` with one argument where it takes two.
  Result: **360/360 requests succeeded, `constructed=true`, `admission_fallback_reason: none` and
  `revoked_checkpoints: 0` on every one, `cached_tokens` a median 98.9% of the prompt** (max 99.8%) —
  the first measurement of this profile under the new context cache, and it says the cache serves a
  growing three-lane conversation there. It is **not** the #251 re-test: that defect needs independent
  conversations saturating the pool, and this run has reuse working throughout.
* **The five upstream model cards.** `1b53a301` deleted them deliberately ("this port ships none of
  those checkpoints") and the merge restored upstream's `README.md` into each of the five directories
  -- because upstream changed that one file in the merge range while the card's other five files did
  not -- leaving five stray cards with no LICENSE, SHA256SUMS or manifest while `docs/README.md`
  points at the publishing repositories. Deleted again.
* **`README.md` -- the port's product document, which the file-level instrument could not see.** The
  merge took upstream's skeleton for the top-level sections and kept the port's content only inside
  `## Performance` and `## Windows`, so at HEAD the front page said *"NInfer requires 64-bit Linux …
  git clone https://github.com/Neroued/ninfer.git"*, documented a Docker image, carried upstream's
  `## Evaluation` table over five checkpoints this port does not ship, and offered upstream's `## Support`
  -- while the port's own title, `## Install a release (recommended)`, `## Client compatibility and
  known API limitations` and `## OpenCode Desktop integration` were gone, and the page contradicted
  itself ("no WSL2 or Docker" beside a Docker section). `31ab6dfe` had removed exactly those sections
  ("document this port's product, not upstream's"), so this is the same silent take at section
  granularity: `merge_blob == upstream_blob` is a file-level test and the file still differed from
  upstream because of the port's own sections. Restored from the port's head block plus HEAD's
  already-updated `Performance`/`Windows`, with two facts corrected in place -- constrained decoding
  exists now (upstream `eb6696ae`), and the engine preempts under resource pressure -- and the
  duplicated `ninfer::Uint128` line (a merge remnant) dropped. `check_doc_links.py` is what caught
  the consequence, with five dead card links from upstream's `## Evaluation`. **The port had already
  decided this once**: the `d44ab584` merge's resolutions (recorded above, `:1209`) say "**Ours** for
  `README.md`: upstream's side links a dozen pages this port deleted and carries their
  AIME/GPQA/ERQA scores", and the b9114396 merge undid that decision without recording one.

**Left as upstream has it, with the reason:**

* `src/models/qwen3_5/program/transactions/commit.cpp` and `tests/test_resource_manager.cpp` --
  superseded. The port's change was `ProgramImpl::can_release_continuation`, the non-consuming
  counterpart to `release_continuation` that its reclaim-at-admission fix (`d05ee90a`, for upstream
  #251) needed because moving a handle the Program would refuse corrupts a live catalog entry.
  Upstream's context-cache replacement removed both `can_release_continuation` and
  `release_shared_prefix` from `Program`, and at HEAD the port's `resource_manager.h` is upstream's
  plus the `Uint128` fix. **This is the one loss this pass could not close and does not claim to:**
  `d05ee90a` measured the #251 cliff gone (12/12 reuse at `--device-state-slots 1`), that reclaim no
  longer exists, and `tools/release/profiles.py` still ships `--device-state-slots 1` for every profile
  on the strength of it. Upstream #251 is still OPEN, and upstream's own answer to this family is the
  rewrite itself: **#366**, the maintainer's announcement of `b9114396`, lists #177, #179, #229 and
  #251 among the reports it responds to and asks reporters to re-test their cases.
  **That re-test was run on 2026-10-06 and the cliff is back** (`repro_251.py`, unchanged, against the
  new cache; `docs/research/prefix-state-eviction.md`'s last section): with `--device-state-slots 1`
  and no host backing reuse holds for **1** conversation and then stops permanently; with a ~2-image
  host quota it holds for **2** and then stops; with the shipped 8192 MiB quota it holds for all six,
  which is capacity and not reclamation -- the old finding, reproduced on the new code. So the
  shipped `--device-state-slots 1` is an unverified setting whose cliff is now bounded by
  `--host-context-mib`, and the cause is located rather than suspected: probes on that run (temporary,
  reverted) show the shortage carrying the state slot, `plan_reclaim` offering one demotion and one
  release, and **every action rejected by the value gate at
  `src/runtime/engine/context_cache/resource_manager.h:1282`** (`victim.reused=1 admission.reused=0`)
  -- a reused endpoint may not be displaced by an admission that has demonstrated no reuse. So the
  replacement evicts by policy and the port's `d05ee90a` evicted by LRU; the cliff is that policy's
  consequence for a stream of distinct conversations, and what is left is a decision (size the pools
  for this port's workload, or ask for the LRU behaviour back), not a diagnosis. Both are written up
  in `docs/research/prefix-state-eviction.md`.
  **Decided by measurement, 2026-10-06: option 1 holds and option 2 is not needed.** The shipped
  acceptance shape (12 conversations x ~20,000 tokens on the QUASAR DFlash2 lane,
  `--device-state-slots 1 --host-context-mib 8192 --kv-capacity auto`) reuses at **99.8% for all
  twelve** -- the result the retired reclaim achieved -- and the interleaved shape that a session plus
  subagents produces holds to **52 conversations** and fails at 56, which is the host quota in state
  images (~55 x 147 MiB = 8 GiB). This port's workload is 8. So the shipped setting is adequate for
  what it ships, the bound is `--host-context-mib`, and the item closes with the numbers rather than
  with a claim.
* `src/runtime/engine/context_cache/{materialization_budget.h,materialization_planner.h}` and
  `tests/test_materialization_budget.cpp` -- superseded. The port's 250 ms grant ceiling (`e92cd9d7`)
  patched a mechanism upstream's rewrite deleted; `MaterializationSearchBudget` has zero occurrences
  at HEAD. Upstream issue #229 is still OPEN and its reporter is deploying the same widening on
  current master, so the port's evidence belongs on that issue rather than in a restore.
* `src/runtime/engine/model_instance.cpp` -- superseded. The port forced `cache.policy = Default` when
  the cache is disabled and defaulted `max_shared_prefixes` from
  `kMaximumPreparedPromptCacheCandidatesPerRequest`; upstream's new `ContextCacheOptions` has neither
  field, and `--max-shared-prefixes` is no longer accepted.
* `tests/test_context_cost.cpp` -- superseded by a move: upstream's rewrite placed it at
  `tests/runtime/test_context_cost.cpp` and the port's `_getpid` portability fix is present there.
* `src/models/qwen3_5/program/context.h` -- superseded by a move: upstream deleted it and its
  `configure_text_card` declaration now lives in `program/execution_context.h`, where the
  `TextCallConfig` restore carries the port's signature change.
* `tools/bench/ttft/README.md` -- superseded: upstream's rewrite replaced the document and the media
  case table the port's one-line correction edited; the case name does not appear at HEAD.

* **`docs/serving.md`** -- restored: the `check_request_logs.py` validation paragraph and the tool-call
  fallback's `rejected_tool_name` / `rejected_tool_near_match` / `rejected_tool_name_length` subject
  fields. Both describe surfaces that exist at HEAD and were documented nowhere after the merge.

**Classified as already accounted for:**

* `src/models/qwen3_5/frontend/processor.cpp` (frontier ledger) and `runtime/engine/engine_core.h`
  (kv_capacity dedup half + settle order) -- restored in `393d1f64`.
* `src/models/qwen3_5/frontend/tokenizer.cpp` -- restored 2026-10-05 (ADR-0010 incremental encode);
  the ratchet entry carries the reason.
* `tests/models/qwen3_5/test_engine_prefix_real.cpp` -- port edits present (artifact goldens, guarded
  main, the model's default stops where a tool is declared).
* `src/serve/request_log.h` -- the port having been behind by the letter of the instrument and
  converged in fact: its blob at the merge already equalled upstream's tip (both schema 23); HEAD is at
  25.

**Verified:** `build-test` 637/637 targets and `build` 587/587; the full suite is **150/152** with
only the two baselined failures (`ninfer_qwen3_5_dflash_real_test` and `ninfer_qwen3_5_moe_real_test`,
each reporting its own cause) and `ninfer_qwen3_5_loading_real_test` skipped by construction;
`check_test_baseline.py` reports **GATE PASSED: no regression against the recorded baseline**;
`ninfer_request_log_test`, `ninfer_qwen3_5_frontend_test` and `ninfer_pretty_logging_test` pass, with
the new phase assertion falsified in both directions (the render bracket disabled makes it fail with
its own message); the ratchet is green at 268 paths with the 13 new entries carrying reasons and
verdicts; `check_profile_consistency.py` reports 0 disagreements; ruff and mypy are clean. The GPU
tier's first execution and the reporter case's first end-to-end run are item 23 and the bullet above.
End to end on a serving lane started with the restored reporter profile (`qwen3_8_27b_nvfp4qat`,
`--max-context 104857 --kv-capacity 117964 --max-concurrency 3 --device-state-slots 2
--host-context-mib 12179.05078125`): the engine pinned 11.9 GiB of host context, answered `/health`,
completed a request (`prompt 58 | output 24 | TTFT 74.8 ms`), and printed

    req#1 started | openai-chat non-stream | 1 message | max output 24 | thinking template default |
    prepared 240 us, contract 0 us, convert 1 us, render 196 us, tokenize 39 us, positions 0 us,
    cache prep 0 us | preserve thinking

which is the phase split the restore put back, produced by the restored `frontend.cpp` and consumed
by the restored `operational_log.cpp`, on a card built through the restored `TextCallConfig`.


### 17. The suite runs 3.0x faster, and the remaining 1.6x is a coverage decision — **DONE 2026-10-05; the second half is open**

**Why:** the suite took 1430.9 s and two tests were 60% of it — `ninfer_softmax_attention_test` at
584.8 s and `ninfer_context_kv_materialize_test` at 265.1 s. The dominant one was **not using the
GPU**: median utilisation 5% across 60 s of it, so its 584 s was the naive host oracle on one core
while twenty-three idled. And the suite ran strictly serially, `ctest` with no `-j`.

**Done, both measured:**

- `softmax_attention`'s causal sweep is five independent KV formats in one loop, and the binary
  already took `--kv-dtype` as a filter over the same `run_storage_cases` calls, each keeping its own
  criterion. Registered as six ctest entries, plus `--no-causal` for the two scenarios `--kv-dtype`
  cannot reach because it sets `causal_only`. 584.8 s as one entry against 145.5 s as six in parallel,
  and coverage is unchanged structurally — `selected` is only a `continue`.
- `-j 8`. It needed no new protection, and that is worth recording because I nearly added one: the
  real-model tests already carry `RUN_SERIAL TRUE`, which is stronger than a resource lock, so ctest
  runs nothing alongside them and the ~16 GiB artifact each loads cannot collide.

1430.9 s → 471.0 s with the artifact set and only the two baselined failures. Five repetitions at
different `--schedule-random` orders: 2391.53 s total, only the baselined failures, no flake.

**One regression, found by the first parallel run.**
`ninfer_qwen3_5_issue5_race_test` reproduces a cache-hitting continuation submitted while another lane
is generating — its subject is a race — and run alongside other tests its timing windows move, so it
failed. It is registered in its own block and was never added to `ninfer_qwen3_5_real_tests`, so it
carried neither `RUN_SERIAL` nor a label while every other real-model test did. 142 serial runs never
showed it, which is the argument for the repeat-run rather than a reason to skip it.

**Open, and it is a coverage decision.** `context_kv_materialize` is host-bound too (median 6%) and
its four groups measure 175 / 49 / 24 / 4 s, so splitting it would take the suite to about 300 s. It
cannot be split as it stands: its workspace-interval check compares the computed capacity for each
batch against `fixture.observed_peak[batch]`, which is last-write-wins, so the check is a whole-sweep
postcondition over the exact sequence. Relocating it changed what it asserted and broke the bare
binary — `failures=1` with no arguments, where the same binary had passed — which is why the attempt
was reverted rather than adjusted. Making the check per-group, each group validating the peaks it
recorded, would be more checks but a **different assertion** than today's. That trade is the owner's.

### 18. Upstream's five commits, merged — **MERGED 2026-10-05; the speculative path converged on upstream**

**Why:** upstream's `eb6696ae`, `08902f73`, `a643abcd`, `911d34db` and `7e2973ee` are 190 files and
+48,687/−960: JSON-mode and schema-constrained decoding, GBNF constrained decoding across the
generation backends, the vendored XGrammar CPU core, a runtime fix preserving cache sources across
admission waits, and a restructured `AGENTS.md`. Seven conflicts, and the interesting part is that
**none of them was a disagreement about behaviour** — six were the port holding upstream's *own
older* shape while upstream refactored past it, and the seventh was a document.

**What the primary sources said.** `08902f73`'s message is the key: *"Unify speculative execution
around Forward and Finish graphs for MTP, DFlash, and DFlash2."* Checking each symbol against the
merge base rather than against the conflict sizes separated the two cases cleanly:
`select_graph_profile` and `target_verify_accept` are in the base and absent from `upstream/dev`, so
upstream *removed* them; `kDrafterCandidatesPerPosition` and `copy_tokens` are absent from the base,
so the port *added* them. The conflict sizes said nothing about which was which — `decode.cpp`'s 56
lines were upstream's three helpers that upstream had deleted, and the port's 72-line recall block
was outside the hunk entirely.

**Two errors that only the compiler caught, both mine.** Taking upstream's side of `decode.cpp`'s
hunk deleted `kDrafterCandidatesPerPosition`, which was defined *inside* the namespace the hunk
replaced while the port's recall block that uses it sits outside — the file looked resolved and did
not compile. And the two bench/test conflicts where each side supplied a body for one function slot
sharing a trailing `return failures; }` produced a nested function. Neither is visible by reading
the conflict; both are visible immediately in a build.

**The withdrawal, and why it was not a close call.** `copy_tokens`/`copy_extents` came from
`70bf39f5` (n-gram copy drafting), whose core item 9 already records as replaced with its surface
withdrawn — these two were left behind, and the reasons they looked harmless were each false. They
did not preserve the capability's call sites: because `stream` is not last in the port's signature,
upstream's positional 7-argument calls landed the stream on `copy_tokens` and failed to compile, so
the pair forced *every* call site to diverge. Nothing had ever set them: every call site passed
`nullptr, nullptr`, and the only other reference was a host array in `round_buffers.h` never wrapped
into a `Tensor`. And the Op-level test that covered them covered the removed capability, not shipped
behaviour. Withdrawing them converges six files on upstream exactly — `port_delta_baseline` 255 → 249
— and the defaulted-stream pin drops from 27 sites across 8 files to 26 across 7, because the
default was carried on that declaration.

**Windows necessity.** The vendored XGrammar does not compile under MSVC.
`XGRAMMAR_UNREACHABLE()` expands to nothing on MSVC, so a value-returning lambda whose last branch
is unreachable fails with C4716 — `grammar_compiler.cc:1593`, the only error in the vendored core,
whose `LowestBit` and `PopCount` already carry MSVC fallbacks. An `_MSC_VER` branch using
`__assume(false)` fixes it at the macro, which is where it covers every use site; it emits nothing,
so reachable behaviour is unchanged. Filed in the ratchet as `windows` with its reason, because a
vendor update will drop it.

**The ratchet's reference was wrong, and its failure mode was a misleading number.** It defaulted to
`upstream/master`, which was right while the port sat at master. `dev` is where upstream works and
what a merge takes, so once dev ran ahead, comparing against master reported upstream's own newer
files as the port's divergence: **695 paths where the truth was 541**. The default is now
`upstream/dev`, and `--check` fails when the recorded reference is not the one being compared — a
stale reference now names itself instead of producing a path count that invites a re-record.

**Also:** `AGENTS.md` keeps both intents (upstream's restructured skeleton, the port's directive,
tables, 64 rules and Windows practices); `README.md`'s capability list gains the two constraint
capabilities upstream documents in `docs/cli.md`, `docs/serving.md` and
`docs/maintainer/constrained-decoding.md`; `suite_size` moves 145 → 149, which is exactly the four
ctest entries upstream adds (`ninfer_grammar_test`, `ninfer_json_schema_test`,
`ninfer_json_schema_oracle_test`, `ninfer_qwen3_5_grammar_real_test`) and no removals.

**An environment defect the merge exposed, in three layers.** `ninfer_json_schema_oracle_test` —
upstream's own new oracle — failed, and each layer masked the next. It needed `jsonschema`, declared
in `tests/text/requirements.txt` and documented in `tests/README.md`, but **no CI tier installed it**:
upstream has no CI at all (only `FUNDING.yml` and a PR template), so the test has only ever run on the
maintainer's machine. Installing it was not enough, because `tests/CMakeLists.txt` does a plain
`find_package(Python3 REQUIRED COMPONENTS Interpreter)`, which here resolves to `C:/Python314/python.exe`
— not the project's selected `C:\vllm-env\Scripts\python.exe` where the dependency belongs. And with
the interpreter corrected, the test failed a third time for a real reason: its `subprocess` calls use
`text=True` with no encoding, so Python uses the **locale** encoding — UTF-8 on Linux, **cp1252** on
Windows — and the payload deliberately carries CJK and emoji (`ensure_ascii=False`), so the first case
raised `UnicodeEncodeError` and the probe never ran. Fixed in three places: both suite recipes pass
`-DPython3_EXECUTABLE` (honouring `NINFER_PYTHON`), `gpu.yml` installs `-r tests/text/requirements.txt`,
and the four call sites state `encoding="utf-8"` — the last is a port divergence in an upstream file,
recorded in the ratchet as `windows` with its reason.

**And a latent CI defect found beside it.** `gpu.yml` set `NINFER_TEST_ARTIFACT` on the *gate* step
but not on the *suite* step, so the four required real-model tests would skip and the gate would fail
on missing coverage rather than on a regression — the exact trap the AGENTS table documents, sitting
unexercised because no self-hosted runner is registered. The artifact now sits on both steps.

**Done when:** the tree builds, the suite is green against the baseline, the ratchet and the
pre-commit gates pass, and nothing the port owns was silently dropped. **Done 2026-10-05:** build
clean; **149 tests, 99% passed, 2 failed** — both the recorded known failures — with the baseline
gate green; the mutation gate green; the ratchet green at 250 recorded paths; all twelve pre-commit
gates pass; and the port's own recipe passes end to end.

### 19. Ten live references to six flags that no longer exist — **FIXED 2026-10-06**

**Why:** upstream's `b9114396` replaced the context cache and collapsed five bounds into one shared
Host quota — `--host-state-slots`, `--host-kv-mib`, `--max-shared-prefixes`,
`--max-private-continuations` and `--max-long-anchors-per-continuation` became `--host-context-mib`,
and `--context-cache-policy` went with them because the new `ContextCacheOptions` has no policy
field. `profiles.py` recorded that in the same commit, and `README.md` and
`docs/maintainer/README.md` describe it — but ten live references kept the old names, and
`serve_options.cpp` throws `unknown argument` for anything it does not parse, so the class is not
cosmetic.

**The worst was a tool that could never run.** `tools/release/check_host_kv.py` appended
`--host-state-slots <n> --host-kv-mib <n>` to a shipped launcher's flags, so every invocation died at
startup and the harness had no way to report anything but failure. It now varies the one quota.

**The one worth reading twice** is `tools/bench/first_request_lane.py`, which dropped the capacity
flags for its `--no-prefix-reuse` arm to pre-empt a refusal ("cannot be combined with context-cache
capacity options") that no longer exists in `serve_options.cpp`. Its own docstring says the arms
differ in exactly one flag and that a second difference is the confound every version of that
question has had — so the workaround had become the defect it warns about. Removed: the arms differ
in one flag again.

**Two checks had gone vacuous**, both guarding on a dead name:
`check_profile_consistency.py`'s "does not hand-write the cache bounds" looked for
`--max-shared-prefixes`, and its server-only list named `--host-kv-mib`; `test_launcher_generation.py`'s
`server_only` set named two dead flags. All three now name live ones, and `--device-state-slots` was
added where the set was incomplete. `check_cache_capacity.py` printed `None` for every lane's bounds
and now prints the quota.

**Also fixed:** the pre-commit hook skipped the port-delta gate on `upstream/master` while the gate
now compares `upstream/dev`, so the skip condition guarded a different ref than the gate reads.

**Docs:** `CONTEXT.md`'s three-budget model, `docs/maintainer/artifact-conventions.md`'s
`INVARIANT_FLAGS` list, and `docs/opencode-settings.md`'s explanation of why a lane retains state.
The historical records (`docs/adr/0007`, `docs/research/*`, `docs/v2-v3-flag-diff.md`) keep the old
names, because they are measurements taken on the model that had them.

**Done when:** no live reference to a superseded flag remains, and the repaired tool starts a lane.
**Done 2026-10-06:** the survey is clean (only comments naming them as superseded remain); all twelve
pre-commit gates pass; the hook's pytest scope passes (108); ruff and mypy are clean;
`tests/release/test_launcher_generation.py` passes as a script with the live flag set; and
`check_host_kv.py` starts a lane on `--host-context-mib 8192` and reports **5/5 cache hits at 98.7%**
— which also confirms the merged engine retains prefixes at the shipped quota.

### 20. 250 divergences, and 245 of them said nothing about why — **RECORDED 2026-10-06; 121 need a decision**

**Why:** `port_delta_baseline.json` is the port's list of every upstream-owned file it diverges from,
and the ratchet fails when one appears or disappears. That makes the *list* safe and leaves the
*account* empty: five entries carried a reason, so the next session reading it had 245 paths and no
statement of what any of them was. The policy the ratchet exists to enforce — a divergence is a
Windows necessity or a measurably better implementation, or it reverts — cannot be applied to a list
that does not say which paths are which.

**The method, and what it deliberately is not.** A 250-path review cannot be 250 readings, and it
must not become 250 plausible sentences. Most divergences already say why they exist, in the diff:
a comment naming a `docs/active-work.md` item, an ADR, a document, an upstream commit, an `UNGATED`
marker, an MSVC diagnostic, a measurement. So the reason field is filled with a **citation** —
"Cited in the diff — item: 12" — which says what is there and leaves the reader to open it. A tool
that wrote "keeps because the port needs it" would assert a review that had not happened, which is
the failure mode this whole file exists to avoid. Files with no citation say exactly that, so the
queue is visible in the file rather than implied by an empty string.

**State: complete — 249 of 249 accounted for, and one divergence reverted.**

| kind | count | what it means |
|---|---|---|
| citation | 100 | the diff names an item, ADR, doc, upstream commit, Windows construct, MSVC diagnostic or measurement |
| removed | 24 | the port does not ship an upstream file; the reason says to check for references first |
| annotated | 125 | established by reading the diff, grouped by what it does |
| **unreviewed** | **0** | — |

**How the 125 were established, and how deep.** Depth followed blast radius rather than line count.
The numerical and model layers were read: `route_catalog.h` and `candidate_selector.cpp` were opened
in full through codegraph, which is where the two findings below came from. The FP8 family was
checked against the code (`alignas(64)` against upstream's `alignas(128)`). The rest were
characterised by what their diffs do — the deduplications, the NVFP4 drafter route, the Win32 I/O and
its Python half, the instruments, the contracts, the fixes, and the test coverage for each. That is
the honest description: a path whose diff is a type annotation carries a reason saying so, not a
reason implying a numerical review.

**One divergence was removed rather than justified.**
`src/models/qwen3_5/program/speculative/mtp.cpp` diverged by **whitespace only** — a stray six-space
indent on `configure_text_card` and one line losing a space — and the port's version was the worse
one. `git diff --ignore-all-space` is the exact test, and a sweep of all 226 modified paths found
this was the only one in the tree, so it is a one-off rather than a habit. Reverted to upstream;
verified by a clean rebuild and the speculative and real-model tests, whose only failures are the two
recorded known ones. That is the ratchet's `refactor` disposition ("revert to upstream's version --
restructuring his code buys nothing") applied for the first time, and the baseline is 249 entries for
it.

**Two findings the review produced, both from codegraph's verbatim source rather than from a diff.**

* `route_catalog.h` — its own comment records what it replaced: **nine files each defined `kAnyCols`
  as the same expression**, and each carried a closure predicate in one of four signature shapes,
  all of them the same proof. The shared form also drops a conjunct that evaluated `routes.back()`,
  which is undefined for an empty array. A deduplication with a fix in it, with five callers.
* `candidate_selector.cpp` — its mixed-format guard's comment records the measurement: each codebook
  was validated alone and the route is chosen from the predecessor, so an NVFP4 predecessor with a
  BF16 successor reached the NVFP4 kernel holding the successor's null scales pointer — **an illegal
  access `0x38AC` into unmapped memory that aborts the process** rather than failing the request. The
  reverse reads packed E2M1 as bf16 and returns wrong drafts.

**Two pattern gaps were found by running the recorder, both the mirror of the failure it exists to
prevent.** The first pass listed `_MSC_VER` and `C2719` but not **`_WIN32`**, so
`src/artifact/file_io.cpp` — whose entire divergence is `#ifdef _WIN32` with `CreateFileW` and
`FILE_FLAG_NO_BUFFERING` replacing POSIX `open`/`O_DIRECT` — read as unjustified; adding it converted
12 paths, and `sys.platform == "win32"` converted 2 more. Each pattern was verified against a diff
before being added, because a loose pattern hides a divergence instead of surfacing it.

**Done when:** every path's reason names its justification, and the unreviewed group is empty.
**Done 2026-10-06:** 249 of 249 accounted for; the ratchet green; the reverted file verified by a
clean rebuild and the affected tests.

**And the verdict is recorded rather than derived.** The report used to group by the marker
heuristic, which left twelve files the review KEPT labelled `refactor` — a disposition the tool's own
docstring defines as *"revert to upstream's version"*, so the report invited exactly the change the
review had decided against, and 131 more read as `unknown`, meaning "not yet reviewed", for paths
that had been. A baseline entry now carries a `review` field beside `reason`, carried forward on
re-record the same way, and the report prefers it: **windows 67, product 158, deleted 24, and no
`refactor` and no `unknown`**. Falsified in both directions before being trusted — 249 of 249
verdicts and reasons survive two consecutive re-records with none lost or changed, and `--check`
still gates on the path set. The heuristic still runs and is still written as `disposition`, because
it is a triage hint worth keeping; it is simply no longer mistaken for a verdict.

**Two things this deliberately does not do.** It does not decide the 121, because deciding them from
a keyword would be the transcription this file warns about. And it is not a gate: the reason field is
prose a human wrote or a tool cited, and a gate over prose is the class `AGENTS.md` records four
separate times.

**Done when:** every path's reason names its justification, and the unreviewed group is empty.
**Measured 2026-10-06:** 129 accounted for as above; the ratchet green at 250; and the two groups
verified against the code rather than the record — the FP8 family's `alignas(64)` against upstream's
`alignas(128)` (`src/ops/linear/fp8/fp8_a8_tma_mma.cuh:26`, with a `static_assert` enforcing it) and
the deduplication against `src/ops/common/validation.h`.

### 21. Upstream's eighteen commits, merged — **MERGED 2026-10-07; and a gate earned by the second instance of one defect**

**Why:** upstream's `070fa61a` through `41e50d0d` are 90 files, +3,478/−608 — a family of "support and
tune `<format>` linear `<shape>`" Op work (19 new shape translation units), a bench cold-quantile fix,
and `feat: add constrained tool calling`, which brings `tool_contract.{cpp,h}`, `tool_grammar.{cpp,h}`
and three ctest entries.

**The merge was taken with item 20's table as the checklist.** Thirteen of the incoming files are paths
the port diverges in, each already carrying a reason and a reviewed verdict, so the port's intent was
known *before* the merge touched it rather than reconstructed after — which is what that review was
for. Two conflicts, both the shape the table predicted (`tests.cmake`, where both sides added a
registration; and `test_tool_call_parser.cpp`, where each side supplied a body for one function slot
sharing a trailing `return failures; }` — the same trap as item 18's merge).

**And one auto-merge defect, which is the class that matters.** `tool_call_parser.cpp` merged cleanly
and did not compile: the port's change gave `QwenToolRegionParser::parse` a second parameter (the
rejected name, for diagnostics) while upstream's new `initialize_continuation` calls it with one. The
compiler named it. A file that merges without a conflict is not a file that merges correctly.

**The second instance of one defect earned a gate.** Upstream's new `test_tool_schema.py` wrote
`subprocess.run(..., text=True, ...)` with no encoding — the identical defect, in the identical shape,
as `tests/text/test_json_schema.py` two commits earlier: Python takes the *locale* encoding, UTF-8 on
the maintainer's host and cp1252 on Windows, and both payloads are deliberately non-ASCII
(`ensure_ascii=False`), so both raised `UnicodeEncodeError` before their probe ran. Upstream has no CI,
so neither had been run anywhere else. Two instances is this repo's threshold for structure rather
than another paragraph, so `tools/release/check_subprocess_encoding.py` now fails on any text-mode
subprocess spawn that does not state its encoding. It is **AST-based, not a text pattern**, because a
regex over a call's arguments cannot tell an argument of this call from a keyword in the next one —
and it was falsified in both directions before being trusted. It found **19 more instances** across
`tools/` and `tests/`, all latent (their subprocesses emit ASCII today), all now stating `utf-8`.

**Three of my own errors, each caught by a different control.**

* `TEMPLATE = r"""..."""` to silence a `SyntaxWarning` **changed the generated launchers by 20
  bytes** — making the literal raw turns the template's *valid* escapes literal too.
  `check_profile_consistency.py` asserts the shipped launchers are byte-identical to the render and
  caught it immediately. The warning was about an *invalid* escape (`\S` in `HKLM\SYSTEM\...`), which
  stays literal either way; doubling those backslashes fixes the warning and leaves the render
  byte-identical, which that same gate now confirms.
* A `git stash` control **failed silently** — it refused because the tree was mid-merge, so the
  "without my edit" run also had the edit and controlled nothing. It is why I first read the launcher
  failure as pre-existing. A control that cannot run is not a control that agreed.
* The ratchet's reference needed **two** corrections, not one: the merge base rather than the branch
  tip (a fetch of these 18 commits turned a green ratchet red because upstream's own new work read as
  the port's divergence), and `MERGE_HEAD` rather than `HEAD` while a merge is in progress (HEAD is
  still the port's tip until the merge commit lands — exactly the commit the hook guards).

**Done when:** the tree builds, the suite is green against the baseline, and the ratchet accounts for
every divergence. **Done 2026-10-07:** builds clean; suite green at 152 tests with only the two
recorded known failures; baseline gate passed; ratchet green at 252 with no unreviewed entry; the two
new tool tests pass.

### 22. Upstream's eighteen commits, merged — **MERGED 2026-10-07; and a product capability withdrawn with it**

**Why:** upstream's `070fa61a` through `41e50d0d` are 90 files, +3,478/−608 — a family of "support and
tune `<format>` linear `<shape>`" Op work (19 new shape translation units), a bench cold-quantile fix,
and `feat: add constrained tool calling`, which brings `tool_contract.{cpp,h}`, `tool_grammar.{cpp,h}`
and three ctest entries.

**Two conflicts**, both the shape item 20's table predicted: `tests/models/qwen3_5/tests.cmake`, where
both sides added a registration (kept both), and `tests/test_tool_call_parser.cpp`, where each side
supplied a body for one function slot sharing a trailing `return failures; }` — the same trap as item
18's merge. **One auto-merge defect**: `tool_call_parser.cpp` merged cleanly and did not compile,
because the port's change gave `QwenToolRegionParser::parse` a second parameter (the rejected name)
while upstream's new `initialize_continuation` calls it with one. A file that merges without a
conflict is not a file that merges correctly.

**The finding, corrected by measurement.** The first account here said upstream's
`initialize_continuation` rejects a continuation whose prefix already contains a completed call, and
that the port's tool-prefix reuse was built on it. That was wrong, and how it was wrong is worth
keeping: the crash dump named the *frames*, and the throw site was *inferred* from the code around
them — because nothing printed the message.

With `main` guarded so a scenario reports its failure instead of aborting, all thirteen scenarios were
run on 2026-10-07. Six fail, every one with the same message: `constraints require default EOS, text
output, no custom stops, and one output language`. That message is
`src/models/qwen3_5/frontend/frontend.cpp:915`, not the parser. Upstream's `41e50d0d` made a request
that *declares tools* take a constrained contract, and that guard rejects
`stop.include_model_defaults = false`. Upstream's own scenarios set exactly that, and `41e50d0d` did
not touch the test file; upstream has no CI, so nothing caught it. **The control is inside the data**:
`shared-rewrite-materialization` also declares tools and passes, because it never disabled the default
stops.

**The diagnosis, by `diagnosing-bugs`.** Phase 1's loop was the test binary itself (red 3/3,
deterministic, ~7 s). Phase 2 minimised it to scenarios via `NINFER_PREFIX_REAL_SCENARIO`: **six red,
seven green, and every red one a tool case** — which killed my first theory (that the crash was in the
tool-prefix case I had already edited). Phase 3's hypotheses were falsified one at a time: no
`noexcept` on the constructor that holds the call, and `catch (...)` changed nothing and its probe
never printed. Phase 4 rebuilt with `/Zi`, because a fail-fast bypasses a live debugger but a dump does not — what
the dump lacked was symbols. It named the frames outright:

```
EngineCore<ModelInstance>::submit  ->  Engine::submit  ->  Engine::generate
  ->  `exercise_agent_continuation'::`2'::<lambda_2>::operator()  ->  terminate
```

A frame is not a cause: it says a `RequestError` escaped `generate`, not which of the three throws
inside it fired. The message became available only once `main` was guarded, which is the fix this item
carries.

**Nothing was withdrawn, and the earlier revision of this item was wrong.** The six are upstream's own
scenarios — `exercise_explicit_prefix`, `exercise_nested_tool_markers`,
`exercise_anthropic_prefix_regression`, `exercise_rewrite_checkpoints`, `exercise_agent_continuation`
and `exercise_rewrite_branch` — present in `upstream/dev`, with `agent_continuation_real_test`
registered by upstream's own `tests.cmake`. Deleting five of them plus `exercise_rewrite_branch`'s call
from `all` deleted upstream's coverage and *hid* the defect above. All six are restored, and the six
sites that disabled the default stops now keep them — the change upstream would make, with every
assertion unchanged and the sweep re-run to confirm it.

**Also in this commit:** `check_subprocess_encoding.py` (AST-based, its own test asserting both
directions, the hook's thirteenth gate) and the 19 further instances it found; the ratchet's reference
corrected twice — the merge base rather than the branch tip, and `MERGE_HEAD` rather than `HEAD` while
a merge is in progress; `suite_size` 149 → 152 for upstream's three new entries; and a
`SyntaxWarning` in `tools/release/make_launchers_v3.py` fixed by doubling an invalid escape rather
than making the template raw, which `check_profile_consistency.py` caught changing the generated
launchers by 20 bytes.

**And the structural fixes these failures earned**, each replacing a rule with a mechanism.
`check_test_baseline.py`'s cache key now covers the generated `CTestTestfile.cmake` files as well as
the executables — it had reused a 145-test verdict against a 149-test tree and read green, because a
configure that adds a registration relinks nothing — and it compares the tree's identity before and
after a run, so a build that races the suite fails the gate instead of certifying two trees as one;
`tests/release/test_test_baseline_cache.py` pins both, with the old executable-only key reproduced as
the control. The `claims-gate` plugin denies a shell command that hands an interpreter an inline script
with a backslash-escaped quote (PowerShell truncates the argument there; measured) and a `cdb -c`
command list; its harness grew to 32 assertions and now runs in the hook, because a deny rule exercised
only by hand is a rule nobody has seen refuse anything. And `AGENTS.md` gains the crash-dump row
(`/Zi`, then `cdb -z <dump> -cf <file>` — a fail-fast bypasses a live debugger), the merge-stash rule,
and the contention rule's missing half: a test executable cannot be relinked while the suite is
running.

**One process note, and its remedy:** the merge's *shape* was lost and then restored. A `git stash` of
the in-progress merge preserved all 111 files of content but dropped `MERGE_HEAD`, which would have
recorded a single-parent commit *and* broken the ratchet: `check_port_delta.py` falls back to
`merge-base(HEAD, upstream/dev)` — the *old* base — so it read the merge's own 90 upstream files as the
port's divergence. Writing `MERGE_HEAD` back (`git rev-parse upstream/dev > .git/MERGE_HEAD`) restored
both. Commit a resolution before switching context, or re-create the state file.

**Done when:** the tree builds, the suite is green against the baseline, and the ratchet accounts for
every divergence. **Measured 2026-10-07** on `C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer`, running
the thirteen scenarios through `NINFER_PREFIX_REAL_SCENARIO`: all thirteen pass with the six sites
keeping their default stops, and every assertion in them still holds — `all` included, so
`exercise_rewrite_branch`'s `Checkpoint` reuse is confirmed under the new contract. The baseline gate
then ran the full suite: **150/152, the only two failures the recorded ones**, and the ratchet is green
at 255 paths with no unreviewed entry.

**Filed locally, not upstream, 2026-10-06:** the guard-versus-scenarios finding is written up in
[upstream-constrained-tools-guard.md](research/upstream-constrained-tools-guard.md) — the guard at
`src/models/qwen3_5/frontend/frontend.cpp:919`, the six scenarios it rejects, the control that passes
because it never disabled the default stops, the reproduction, and the searches showing that nothing
upstream reports it and that no upstream commit has touched either file since `41e50d0d`. It is kept
there rather than posted, because the port does not file on the upstream tracker without the owner's
decision; it is written to be pasted as an issue as-is.

### 23. The GPU tier had never executed — **RESOLVED 2026-10-06: three blockers, and the first run passed**

**Why:** `gpu.yml` was recorded as "published and wired correctly, but no self-hosted runner is
registered" (`:1871`), and the same sentence sat beside the artifact-on-both-steps fix (`:2253`). Both
were true and both were incomplete: the missing runner was one of **three** independent blockers, and
none of them had been checked against a primary source.

**The three, each measured:**

1. **No runner.** `gh api .../actions/runners` read `total_count: 0`. Registered `5090-box` on this
   machine (labels `self-hosted`, `Windows`, `X64`, `gpu-5090`) and it is online. The header's
   registration commands carried a download URL that 404s — `releases/latest/download/…` — because
   the released asset is versioned (`actions-runner-win-x64-2.338.0.zip`); the header now names the
   versioned URL and the command that finds the current one.
2. **The file could not be triggered at all.** `schedule` and `workflow_dispatch` only fire when the
   workflow file exists on the **default branch** — GitHub's own note, and the direct evidence was
   `gh workflow run gpu.yml` answering `HTTP 404: workflow gpu.yml not found on the default branch`.
   The default branch was `main`: a v1.1.0-era release cut 1774 commits behind, carrying no
   `.github/workflows/` at all, so a scheduled run would also have tested that stale tree rather than
   the one this job exists for. The default branch is now `dev`, and `gh workflow list` shows
   `gpu-suite active` where it was previously invisible.
3. **A fresh checkout could not build.** `ffmpeg/` is gitignored (`.gitignore:74`) and the Windows
   CMake layer locates FFmpeg through that tree instead of pkg-config, so the configure fails without
   it — and `actions/checkout` runs `git clean -ffdx`, which deletes a pre-staged copy inside the
   workspace. The workflow now stages `C:\AI\ffmpeg` into the checkout before configuring and fails
   with a message naming the prerequisite; the directory is in place as a junction to a working
   checkout's `ffmpeg/`.

**The first execution, and it passed.** Dispatched on `dev` (run `37541172269`), every step green in
**17m18s**: checkout, the FFmpeg staging step, the suite (cold configure + build + ctest), the
baseline gate, and the dispatch-only compute-sanitizer subset. One annotation is left standing rather
than hidden: `actions/checkout@v4` targets Node 20, which the runner forces onto Node 24 — a
deprecation warning, not a failure.

**One inefficiency was found and fixed in the same pass:** the job ran the suite twice — `test_v3.cmd`
runs ctest, then the gate found no `.gate-cache.json` in a fresh workspace and ran it again (~7 minutes
of the card, and a second chance for a flake to fail the gate). The suite step now captures its own run
(`cmd /c tools\scripts\test_v3.cmd > "%TEMP%\test_v3.log" 2>&1` and `type`s it, so the log stays
visible) and the gate reads it with `--from-log`. Verified by running exactly that pair here:
`150/152` with the two baselined failures and the mutation gate green in the capture, then
**GATE PASSED: no regression against the recorded baseline** from the gate.

**Done when:** the tier runs. **Done:** the run above, every step green.

## Gates added while this list was open

**Text encoding, 2026-10-02 — `tools/release/check_text_encoding.py`.** 181 lines of this file
and 7 of `docs/maintainer/qwen3.8-27b-artifact.md` had been re-encoded rather than preserved, by
`Get-Content` followed by `WriteAllLines`: that decodes every byte at or above 0x80 through the
console codepage and writes the result back as UTF-8, which adds a generation of encoding. Nothing
failed — the files still parsed, still rendered, and every gate still passed, because the damage
was confined to the prose. The `qwen3.8-27b-artifact.md` copy dates from `13f5f219` (2026-09-24); the
copy here came from `6d9ae366` and was three generations deep. One byte-order mark in
`src/models/qwen3_5/program_sources.cmake` went with it.

The corruption is not lossy and is exactly reversible, so the gate also repairs. Two things about the
repair are worth recording, because each looked like success before it was:

- **A bounded loop reported a success it had not established.** The marker count reached zero on
  pass seven of an eight-pass loop, so the loop ended without ever testing convergence — and pass
  eight then turned a legitimate `§` into a decode error. The repair now stops when the markers
  are gone, because the inverse is only defined on mojibake and one more pass over already-correct
  text destroys it.
- **A whole-string inverse cannot repair a mixed file.** `U+00A7` reverses to the bare byte `0xa7`,
  which is not valid UTF-8 on its own, and any line holding a character outside cp1252 cannot be
  encoded at all — which is what stranded 181 of these lines on the first attempt. Decoding
  tolerantly, preferring the longest valid UTF-8 sequence at each byte, handles both, because the
  mojibake recombines into valid UTF-8 and a lone section sign does not.

Verified by the check itself rather than by inspection: after repair, zero of the 1222 non-blank
lines of the last clean ancestor (`c414c6a5`) are missing, and 167 lines shared by both carry real
em and en dashes. The diff is encoding only: every changed line differs by its punctuation code
points and nothing else, with a possessive apostrophe going from a three-character sequence back to
one. No prose was edited.

The C1-control rule is skipped under `bench/fixtures/` and `examples/`, and prints how many files
that covers on every run, because an exclusion that cannot be seen is one that cannot be reviewed.
Those roots held generated prompts carrying `U+0097`, which an earlier revision of this gate
exempted rather than flagged. **That exemption is gone and the defect is fixed** — see below.



**The U+0097 in the TTFT corpus — FIXED 2026-10-02, and the exemption removed with it.** Ten files
carried `U+0097` where an em dash was meant, all of them the same Lewis Carroll passage in the NIAH
source (`Lisp—in fact, the defining quality of Lisp—is that it can be written in itself`). An earlier
revision of the encoding gate exempted `bench/fixtures/` and `examples/` from the C1 rule to cope
with it; both the defect and the exemption are now gone.

The cause was not what it first looked like. There is no `0x97` literal in any generator — `git
grep` finds none — so this was not a Python escape. `build_fixtures.py` seeds from
`examples/cli/messages/long_niah_64k.json`, encodes it, and decodes a *truncated* token prefix, and
the seed itself carried the character; the derived fixtures inherited it. Nine of the ten are
reachable as corpus shapes, and one of those, `long-64k-32`, points at the seed file directly.

Fixing it was not a ten-file find-and-replace, because an em dash and `U+0097` are different tokens
and `prompt_tokens` is an exact-fit contract. The sequence that made it safe:

1. **Prove the local tokenizer is the qualified one.** A regeneration with *no* seed edit had to
   reproduce every text fixture byte-for-byte, and it did. The 56 other changed files were PNGs
   re-encoded by a different media writer on this machine — unrelated to tokens, and reverted. Had the
   text not reproduced, the only correct action would have been to record the defect as blocked
   rather than regenerate against a different tokenizer.
2. **Fix the seed, regenerate, and let the fit absorb it.** 27 characters across the three
   `examples/cli/messages` seeds, then a full rebuild.
3. **Re-pin only what must move.** 20 shape digests and 3 shared digests changed. `media` and
   `tokenizer.local_qualification_path_name` were reverted, the latter because the rebuild ran against
   a Windows path and would otherwise have committed a machine-local directory name into the corpus
   manifest.

**`prompt_tokens` is identical for all 26 shapes**, and `marked_frontier_tokens` and
`example_prompt_tokens` are identical for all three shared prompts. That is the property every
published TTFT figure rests on, and it is why this could be done without re-measuring anything.
`build_fixtures.py --check` passes.

**The native chat renderer has been silently inactive outside a CRLF worktree — FOUND 2026-10-02.**
(The renderer itself was withdrawn 2026-10-05 — see ADR-0012 — so this block is the record of the
defect and its correction, not a description of the tree.)
`native_render.cpp` identifies the shipped `qwen3_8.jinja` by digest, and its own comment says a
line-ending change retires the fast path. It needed recomputing, because the constant was wrong rather
than stale: it hashed the file's CRLF form (`01befcc8`) while the committed blob is LF (`951dee26`),
`core.autocrlf=true` having handed the file CRLF bytes when the constant was written. So the native
fast path was active ONLY on a CRLF worktree and fell back to Jinja on every LF checkout — Linux, CI,
any fresh clone — while `ninfer_qwen3_5_frontend_test` passed here, because here is where the CRLF
came from. The line-ending normalisation did not cause this and did not corrupt anything; it made a
defect visible that a byte-exact gate exists to catch.

**Correction, and it reverses the consequence I first recorded here.** I wrote that every published
prompt-render and TTFT figure had been taken with the Jinja fallback. **That is false.** The
deduction: `test_registered_template_digest_matches_file` asserts `native_render_supported(sha256(source))`
over the template *as read*, and the suite passed at 135/137 before the fix — so that assertion held,
so the file as read hashed to the registered digest, which is the CRLF one. The worktree file was CRLF,
so **the native renderer was active on this machine**, which is where every published figure was
measured. `test_registered_template_media` corroborates it: it asserts native expansion and passed.

So what this fix changes is **portability, not performance here**. The constant matched only a CRLF
worktree, so the fast path was off on Linux, CI and any fresh clone, and any figure taken there carried
Jinja's cost — ADR-0012's 48.3 ms render instead of 5.99 ms. No published figure moves, and **no
re-measurement is called for**: the measured configuration did not change on the machine that
measured.

**That projection has since been measured, and it is wrong — see the 2026-10-02 block in
[ADR-0012](adr/0012-native-render-for-the-registered-template.md).** `prepared` is 12.21 ms at 42,415
prompt tokens, above the projected ~8 ms for a conversation four times that size, because the
projection treated a 42 ms render saving as the substance of `prepared` when the render is ~1.4 ms of
that 12.21 and tokenize, layout and copies dominate. Warm TTFT stays untested, because the lane has no
cached-prefix-plus-real-decode path to measure: a growing conversation with a genuinely shared prefix
returned `prefix_cache_hit_tokens = 0`, and the only rows that hit took `private_response_replay`,
which never decodes.


> **THE MEDIANS' CONFIGURATION — one authority for five files that quote them.** `fp8_attn_input_a8.cu`,
> `fp8_gdn_input_a8.cu`, `n14336_k5120.cu`, `fp8_linear_add_a8.cu` and `fp8_linear_swiglu_a8.cu` each
> quote a subset of these figures. None of them carries the configuration, and the harness that took
> them — `tools/bench/tma_ab.cmd` — was withdrawn with the gates it flipped, so nothing in the repository
> can regenerate or refute them today. They are recorded here instead, once, with everything needed to
> judge them.
>
> | | |
> |---|---|
> | Machine | RTX 5090, sm_120a, CUDA 13.3, MSVC 14.51 |
> | Quantity | median over token bands of (MMA arm time / TMA arm time). **Above 1.0 means TMA is faster.** |
> | Method | both arms pinned to the *same* tile via `NINFER_FP8_TMA_ARM=tma\|mma`, one binary, arms interleaved |
> | Why interleaved | this card's clocks drift up to ~9% between windows, so measuring A then B measures the window |
> | Oracle | each tile qualified on both arms against the route's own naive-FP32 oracle, at the tokens it selects |
> | Negative control | a token count where the split-K arm is not reached, to prove the arm is genuinely taken |
> | Harness | **withdrawn** with `bec8951e` — it toggled nine `PORT-DISPATCH` gates and there are none left |
>
> | tile | median | note |
> |---|--:|---|
> | 64x128 Stages=2 | **1.107** | |
> | 64x256 | **1.270** | MMA wins the 768 band |
> | 96x256 | **1.288** | the strongest of the five |
> | 192x128 | **1.241** | |
> | split 128x128 | **1.047** | the thinnest margin of the five; MMA wins 448 and 512 |
> | split 128x256 | **1.270** | |
> | 32x64 | 1.074 | |
>
> Ungating rests on **correctness plus the median, never on "TMA is faster"**: TMA leads the median on
> all five measured tiles and loses three bands (448 and 512 on the split 128x128, 768 on 64x128). The
> margin is largest where the machine is underfilled, which is prefill. Anyone re-measuring should
> reproduce the method row, not just the number.

**Port delta, 2026-10-05 — `tools/release/check_port_delta.py`.** The port modifies 230
upstream-owned files and deletes 24, and a merge can take upstream's version of one and drop the
port's with no error and no conflict — which is exactly what happened to ADR-0010's encode cache. The
gate records every divergent upstream-owned path in `port_delta_baseline.json` with a disposition and
a reason, and `--check` fails in **both** directions: a path that diverges without being recorded, and
a recorded path that no longer diverges. Both directions were falsified before it was wired in, and
`--write-baseline` is byte-reproducible.

Two defects in it were found by using it rather than reading it. It diffed `upstream/master..HEAD`,
but the pre-commit hook runs *before* the commit exists, so a divergence introduced by the commit
being guarded was reported **one commit late** — it now compares the working tree, falsified with an
uncommitted one-line change. And `--write-baseline` cleared every `reason`, which would have rotted
the file on each re-record; it now carries them forward. It has since blocked a real commit: editing
`tests/models/qwen3_5/test_engine_prefix_real.cpp` needed an entry with a reason before it could land.

**Skill discovery, 2026-10-05 — `tools/release/check_skills.py`.** 34 of 59 skills set
`disable-model-invocation: true`, which the V2 skills page says hides a skill from the model's
available list, and this desktop build advertises them anyway — `principle-prove-it-works` carries the
flag and is in the model's list with its description. The config declared a gate that was not being
applied, and a build that honours the flag would silently hide 34 skills. The flags are removed and
the gate keeps it that way.

It fails on the two invisibility modes a commit can fix: a skill with no `description`, which is
registered and never advertised, and a markdown file named after its own directory, where
`skills/foo/foo.md` reads as an entry point and is never discovered. It only *reports* a visibility
flag or a duplicate ID, both of which can be deliberate. Scope is split for the same reason: the
project root is gated, while the global and compatibility roots are reported only, because they are
outside the repository, absent on a CI runner, and a commit cannot fix them.

Its first version flagged all 25 supporting markdown files beside `SKILL.md` in this repository —
`ncu-report/reference/*.md` and the rest — which the V2 page explicitly blesses, and would have been
red on every commit.

## Closed — do not reopen

Each of these was investigated and settled. They look like open work and are not.

| Item | Why it is closed |
|---|---|
| **Rebuild the four models** | The merge touched no artifact, layout, binding or converter file, and no container version. Weights, layout and bindings are byte-compatible. What was stale is the *measurements* — item 3. |
| **Store the DFlash2 drafter at Q4** | Already measured here, per target. `tools/convert/official_recipes.py:303-312` records that the NVFP4 draft rule "was tried here and *lost* 3.2 acceptance points on the DFlash2 lane (57.7% against 60.9%)", which is why the Swift line keeps its draft at Q8, and states the rule: "a draft encoding is measured per target, and this target's hidden states are not the stock ones." Lines 40-51 already assign `Q8` to drafter parameters (the `mtp/`, `dflash/`, `dflash2/` branch of `_optional`, with the router/score/conv/hidden-projection suffixes skipped at 41-50). **This text also cited "and 57-58", which does not assign Q8**: lines 57-58 are the `share()` loop that aliases each drafter layer's `context_key`/`context_value` onto its `key`/`value` — a different mechanism, and the Q8 claim needs only 40-51. The published Q4 result is on a different model. |
| **Fix C2719 by passing the descriptor by reference** | It compiles and then faults: nvcc's host stub passes the host address as a device pointer. A compile-only check would pass it. The correct fix is to drop `alignas(128)`, which this port already did in `1218d574`. |
| **The drafter's block uses a causal-over-block mask** | Refuted. The reference is non-causal (DFlash paper section 4.2; the published checkpoint sets `is_causal: false`), and `context_query.cuh:275-283` already gives every query row `valid_keys = valid` with no causal predicate. |
| **Re-measure at T=1.0 to match the model card** | Backwards. DFlash2's selector measures 4.61 at T=0 against 4.25 at T=1; greedy is the *favourable* side. Re-measuring at T=1 would widen the gap. |
| **The planner's `chunked_target` topology class causes the acceptance cliff** | Refuted by measurement. Acceptance is bit-identical before and after `a012e2bc` removed the predicate. The class selected which shared `cudaGraphExec_t` a profile reused, not which kernel ran. |
| **Acceptance is low because our drafter is mismatched to the target** | True of `qwen3_8_27b_nvfp4.v3.ninfer` (unsloth quantization plus official drafter — upstream issue 298 section 3 documents 3.3-5.1% for exactly that pairing), but that artifact **is not a shipping lane**. The four shipping DFlash2 lanes accept **48.7%–63.0%** (`tools/release/profiles.py`; this row previously read 52-69%, and item 9 read 52-67% for the same measurement — neither matched the recorded per-lane figures). |
| **Take the full 16-commit upstream merge** | The 7 fp8 TMA commits needed an `alignas(64)` descriptor, which landed in `a5077adf`; they compile and run on MSVC now. The nine dispatch gates that hid them are the only thing still holding the route back. Three one-line `alignas` reapplications, already validated here. |
| **The NVFP4 block scale should be searched, not taken from the block max** | Measured on the weights that are actually re-encoded, and it loses: 4.925917 against 4.915181 paired in one window, while weight reconstruction error fell 40-66 %. It cannot be said about `NVFP4MSECalibrator` at all: that calibrator's 193 sites are the MLP and `lm_head`, which this recipe imports unencoded, while the re-encoded attention sites are FP8 `MaxCalibrator` in the source. See item 2 and `docs/perplexity-baseline.md`. Lower weight error is not better output quality. |
| **Lower quantization error implies better perplexity** | The same measurement, stated as the general form. A searched scale that clips a block's largest value reduces squared error on that block and costs output quality, because the large value is carrying signal. Qualify a converter change on perplexity, never on reconstruction error. |
| **The E2M1 block scale should be 4, not 6** | Measured 2026-09-29 on the unsloth lane, both arms freshly converted, the control reproducing the shipped `nvfp4full` artifact to every per-domain digit. Overall 4.998419 → 5.016626, **+0.36 % worse**, so NVIDIA's 6 stands and the encoder is unchanged. The per-domain spread is the durable part: `english_reference` −2.82 % against `chinese_reference` +3.56 %, opposite signs that largely cancel. `ninfer_code` moved least at +0.27 %. See [nvfp4-block-scale-4-vs-6.md](research/nvfp4-block-scale-4-vs-6.md). Distinct from the scale-*search* row above: that one varied the divisor per block, this one varied the format's own maximum. |
| **A DFlash2 lane that is refused at startup is a defect, not a slow lane** | Found and fixed 2026-09-30. `profile` mode measured `start_swift_v3_dflash2_vision` and `start_ninfer_v3_dflash2_vision` **REFUSED** through their own launchers, at the 262,144 both tables carried. Runtime grew from 10.6/10.7 GiB on 2026-09-24 to 11.5/11.6 GiB on 2026-09-30 and those two lanes had under a GiB of margin. **All eight lanes now serve the native 262,144**, each fixed at the artifact rather than at the context: the Swift lane by encoding its DFlash2 draft to NVFP4, the NVFP4-full DFlash2 lane by encoding its nine BF16 exception parents to NVFP4. The second is route-specific — that encoding is worth +11.6 acceptance points to DFlash2 and −21.4 to the MTP head on the same weights, so the line is split across two images and the MTP lane keeps the BF16 exceptions. The refusal's own arithmetic is 308 MiB short; see `docs/research/swift15-lane-measurement.md`. |
| **Runtime growth can be repaired at the artifact rather than by giving up context** | Same session, the general form. Both broken lanes were short by well under a gigabyte, and in both cases a smaller artifact cleared it — 0.77 GiB for the Swift draft, 0.7 GiB for the NVFP4-full exceptions. The reservation growth itself is upstream's: `e621c7d6` ("derive launch plans from device sm count") is on `upstream/dev` and raises the split count with the SM count, trading memory for parallelism by design. Reversing it would trade throughput back on every lane at once, which is a larger decision than repairing the two lanes that were actually refused. **Both halves of that row were later measured and one is wrong** — see the next row; no revert is needed and none was made. |
| **The runtime growth should be reverted, since it bought nothing** | Half right, and the half that was wrong is now measured. Nothing was claimed about this before it was measured: the growth was first attributed to `e621c7d6` from its commit message, and the component breakdown then showed it is not that commit's graph allowance at all but the unified workspace, +0.892 GiB, whose *peak* grew by the same amount. What it bought is **prefill, not decode** — a 2.3% prefill loss at `--prefill-chunk 4096` (13,077 → 12,770 tok/s at depth 19.2k, three interleaved rounds per arm, clusters disjoint) with decode identical at 232 tok/s. So a short-context decode comparison cannot see it, which is why the first reading of this was "it cost nothing". `--prefill-chunk` was then tried as the fix -- it is the width the reservation is maximised over, so 4096 recovers 1.116 GiB for -2.3% prefill -- and **rejected**: 262,144 is already served at 8192 on all eight lanes, so it bought margin nothing was short of, at a measured throughput cost. The lever worth changing was the split rule, and it was not the one first named. FlashAttention's `num_splits_heuristic` returns 1 split once query tiles fill 80% of the SMs, precisely to avoid over-splitting's HBM traffic; `mxfp8_tiled_partition` here already minimised waves but had **no term for the traffic it causes**, so it walked to its cap of 8 at every step — holding 1.62 GiB of FP32 partials for work that was never short of occupancy (1536 tiles against a threshold of 136). **Fixed and landed** (2026-09-30): 1.3 GiB of free VRAM per lane, +3.4% prefill at depth ~45k with disjoint interleaved arms and −0.3% at ~131k, byte-identical digests on all eight lanes, all five KV storage types against the FP64 oracle. The first attribution, to `causal_partition_target`'s flat 2*SM budget, was wrong and is withdrawn: that governs small-width families whose partials are tens of MiB. Two lessons ride with it — a single un-interleaved sample said the guard cost 3.6-12.4% prefill when it costs nothing, and both arms had to live in one binary because rebuilding per arm cannot be interleaved on this card. |
| **Upstream #217's concurrency ceiling should be ported — a ~70% throughput win upstream accepted** | **Settled 2026-09-30: it is a product decision not to, and the arithmetic agrees.** `--max-concurrency` is 1 on every lane by decision: these are single-request latency configurations, and a second in-flight sequence divides one device between two requests rather than speeding up either. Two independent findings agree. The gain is bounded and then reverses — vLLM on this class of card goes 101 tok/s at c=1 to 2,907 at c=50, saturates, and *regresses* at c=128; raising `max_num_seqs` 64→128 was measured to hurt both latency and throughput through scheduling contention. And at the native context it is arithmetically unavailable: vLLM V1 reserves KV per in-flight slot in proportion to `max_model_len`, one 262,144-token slot costs 8.681 GiB of sequence arena against ~10.5 GiB available after weights, workspace and graphs, so C=2 is short by ~6.9 GiB and would cap context near 158k. `kMaximumConcurrency` stays 8. A concurrency-2 startup was not attempted — the decision settles it, and measuring an unshipped configuration is measuring something already decided. |
| **A draft encoding measured on one target transfers to the next** | Refuted twice in opposite directions, which is the useful form. On Swift 1.0, encoding the DFlash2 draft to NVFP4 **lost** 3.2 acceptance points and the recipe kept it at Q8. On Swift 1.5 the same encoding **gains** 13.7 points on code, loses 4.6 on chinese, and is 0.77 GiB smaller — and that size is what puts the native 262,144 back within reach on the DFlash2 + Vision lane. `official_recipes.py`'s rule stands and is now demonstrated in both directions. |
| **The calibration corpus should be coding-first** | Declined on 2026-09-29 as policy: this port always calibrates with the original creator's corpus. Enforced by `tools/release/check_calibration_corpus.py` in the pre-commit hook, which pins the corpus bytes and checks that all three `full_range` carriers agree. The supporting facts are in [artifact-conventions.md](maintainer/artifact-conventions.md), and note the asymmetry that was not a reason to do it anyway: the corpus reaches only the sites needing a measured divisor, because an already-NVFP4 site derives its divisor from the checkpoint's own stored `input_scale`. |

## Also worth doing, small

- ~~`--ngram chain` is accepted, validated against the backend, carried into `Program`, and produces
  nothing~~ — **WITHDRAWN 2026-10-03.** The whole surface is gone: `NgramDraftMode`, `NgramOptions`,
  `SpeculativeOptions::ngram`, the two `ngram_*_tokens` counters that were declared and written
  nowhere, `kNgramMaximumDraftTokens`, both CLI parsers (`apps/cli/options.cpp` and the bench
  harness's separately-named `--ngram-mode|-max-drafts|-min-drafts`), the startup branch that
  rejected the flag on a non-masked-draft backend, the `ProgramImpl` members, and
  `prompt_lookup.{h,cpp}` with `ninfer_prompt_lookup_test`. No shipped launcher passed the flag, so
  nothing in this product's own configuration depended on it. What made it worth removing rather
  than leaving: `--ngram chain` was *worse* than inert — it validated, it constrained the backend, and
  it entered the Program, so a user setting it got a real restriction and no drafting. Its own
  validation comment said "accepting the flag would advertise a combination that does nothing", and
  the code did exactly that. `ninfer_ngram_selection_test` and `ninfer_ngram_graph_planning_test` are
  untouched: the masked-draft copy *selection* decision is live and has its own consumer.
- ~~`tools/release/check_doc_links.py` does not skip fenced code blocks~~ — done in `46ec0c5b`, with
  seven tests, six of which fail against the previous body.
- DFlash **v1**'s published config has no `is_causal` key, so the reference gives it five causal and
  one non-causal draft layer. Our converter's `_fixed` check only raises when the key is *present*
  and the runtime never reads it, so a converted v1 drafter silently got a uniformly non-causal
  block. **FIXED 2026-10-03.** `_fixed` grew a `required` flag and the draft config now demands
  `is_causal` be stated: the runtime applies one causality to the whole draft stack, so a config
  that omits the key leaves the converter assuming a uniformity it cannot verify, and a config this
  layer cannot represent is refused rather than converted. Both directions are pinned —
  `is_causal: true` and an absent key each raise. Three existing fixtures omitted the key and were
  the evidence that nothing covered this: the converter suite was green against configs that would
  now be refused. Not a shipping lane; the defect was that it was silent.
- ~~`C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer` sits beside the four shipping artifacts, is not a
  shipping lane, and reads acceptably by filename. It cost a full session of benchmarking before
  `profiles.py` was checked.~~ — **MOVED 2026-10-03.** It is *not* a leftover: `v3_profile_matrix.py`
  probes it deliberately for the ceiling figures and `verify_shipping_artifacts.py` carries the
  recorded perplexity against it (4.901690), so deleting it would have broken both. The hazard was
  never the file, it was the filename. It now lives at
  `C:\AI\models\_superseded\qwen3_8_27b_nvfp4.v3.ninfer`, beside the nine artifacts already retired
  the same way — including `qwen3_8_27b_nvfp4swift.v3.ninfer`, whose own comment says "a superseded
  build under a live filename is a measurement waiting to go wrong". `C:\AI\models` now holds only
  the five shipping lanes.
