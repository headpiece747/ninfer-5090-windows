# Active work — 2026-09-28

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

## Open, in order

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
already `recipe.separate()`s at `official_recipes.py:521-523`.

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
(`qwen3_8_27b_nvfp4full_noex`, `qwen3_8_27b_nvfp4swift15`), assigned at
`official_recipes.py:512` and `:518-519` — so the *format* is right, but the *justification* was not.
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

### 5. Recall@1 / Recall@16 / path-acceptance split
**Why:** the only diagnostic that discriminates three different root causes, and it needs no new
kernel. The drafter emits `frame.candidate_ids` and `scores` at `draft.cpp:351-355`, shape `[16,K,B]`.
Decompose per position:
- Recall@1 — the drafter's unary top pick
- Recall@16 — the target argmax anywhere in the 16
- path acceptance — what the selector actually commits

**Triaged 2026-10-01: KEEP — warranted, and now the cheapest item on this list.** No external evidence
is needed or wanted here: the instrument reads data the drafter already emits, so this is an engineering
task rather than a question. It is also the only item that discriminates between three *different* root
causes of the position-1 acceptance collapse rather than tuning one of them, which is why it should not
be reordered behind the quantization work.
**Done when:** all three are reported per position on a shipping lane, and they point at one of:
healthy Recall@16 with collapsing path acceptance (selector); Recall@16 itself collapsing after
position 1 (backbone or conditioning); or Recall@1 low at position 1 (head or conditioning weak from
the first column).

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

**Still open, and it is a real one:** the transient is a genuine first-request behaviour, not a
measurement artefact — a client's *first* request after a lane starts gets different, shorter text
from the same seed. It is invisible at temperature 0 (the greedy digest is stable from request 1) and
visible at the sampling temperature the bench uses. That is a serving behaviour question, not a
benchmark one, and it is not diagnosed.

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
The engine refuses to start above 5. `startup.cpp:785` raises "MTP draft window must be in [1,5]"
against `kMaximumMtpDraftTokens = 5` (`program/internal.h:11`), so windows 8 and 10 cannot be
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

Retained only because `PromptLookup` (`c0da270e`) is in the tree with no caller, and
`prompt_lookup.h:33` says so in as many words — `NOT YET WIRED`. That dead surface is now a
**withdrawal candidate on its own terms**: it is a ported component whose consumer will not exist.
Do not read this row as an argument for finishing it.

**Why it was once open:** `PromptLookup` is ported and tested (`c0da270e`); the integration is not built. It needs
`candidate_selector_tree`, `speculative_accept_tree_drafts`, `speculative_compact_columns`, tree-aware
GDN replay and tree-aware target attention, each with a host oracle.
**Expected value is low, and that is the finding, not a reason to skip it:** the selector's value is
inversely proportional to drafter strength, and our shipping DFlash2 lanes accept 52-67%. The +55%
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
   `amax / 448` at an FP8 site and `amax / 2688` at an NVFP4 one (ModelOpt `config.py:684`). Our
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

### 12. Upstream's 14 commits are merged; the FP8 A8 TMA route is held back on Windows only
**Attempted 2026-09-29, first aborted, then merged the same day. Suite green with it: 135 tests,
133 passed, the two by-construction `dflash_real` and `moe_real`; `check_test_baseline.py` GATE PASSED.**

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
   say.** The in-tree comments justify the fix as inheriting `TENSOR_MAP_ALIGN = 64` under MSVC. On this
   build it inherits **8**: `__cplusplus` is `199711` without `/Zc:__cplusplus`, so `cuda.h`'s `alignas`
   never fires and the header's `_MSC_VER` branch is dead code. **The shipped W4A4 route is correct by
   layout accident, not by the mechanism recorded** — descriptors first at 128 bytes each give offsets
   0/128/256/384, all 16-aligned, which is why it works. Anyone reasoning from the comment would draw
   the wrong conclusion about which property is load-bearing. A sweep of this toolchain also shows
   `alignas` 8/16/32/64 accepted and **128 and 256 rejected with four C2719 sites each**, all in nvcc's
   generated host stub.

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
   differ. All **six** call sites are gated on Windows — the five `linear/fp8` shape files plus
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

   **2026-10-01: a 2-second reproducer, and four causes ruled out — but no fix.**
   `tools/scripts/probe_sm120_tma_load.cmd` reproduces this fault class in a standalone ~150-line
   program with no project code in the path, replacing the 30-minute sanitizer run. It builds a
   descriptor with `fp8_tma_map`'s own proven shape — rank 2, dimensions `{k, rows}` with K first, one
   `globalStrides` entry, `elementStrides {1,1}`, `L2_PROMOTION_NONE` — which encodes successfully, is
   64-byte aligned, and then dies at execution with "an illegal memory access was encountered".
   **Every arm faults, which rules out four candidate causes:**

   | varied | result |
   |---|---|
   | descriptor passed **by value** (the BF16/NVFP4 form, ungated on Windows) | faults |
   | descriptor passed **by pointer** (the form this port's MSVC workaround forces) | faults |
   | swizzle **128B** vs **NONE** | both fault |
   | `fence.proxy.acquire.tensormap::generic` present vs absent; `expect_tx` before vs after the copy | faults either way |

   So it is **not** the by-pointer descriptor form, **not** the swizzle mode, **not** the missing
   tensormap proxy fence, and **not** the mbarrier transaction ordering. That last one matters because
   it was the most likely single bug and the fix for it is correct regardless.

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
   from four ruled-out causes and a two-second reproducer instead of a 30-minute sanitizer run, and
   that the honest answer to "is TMA available on this GPU" is yes — so the divergence from upstream
   should not be written up as a hardware limitation, because the evidence does not support that.

**The suite grew 133 to 135 from this merge**, which `tools/release/test_baseline.json` now records, and
`ninfer_qwen3_5_dflash_prefill_real_test` was added to `required_tests`: it reads `NINFER_TEST_ARTIFACT`
and skipped without it, and it passed against this product's artifact in 6.73 s, so the evidence the
gate exists to demand is available here. `ninfer_bench_fixtures_test` needs no artifact and so belongs to
neither list. The count was derived by enumerating both registration macros across all thirteen
registration sites under `tests/` and differencing that set against `HEAD` — deriving it from recorded
numbers alone is what produced two wrong counts in this file before, and a first pass that scanned two
files and one macro found the net as zero and missed both additions.

**The 135 above is this merge's figure and it was correct on the day. The suite is now 136** — the
`topk_logprobs` Op and its test landed 2026-09-29, after this was written. `tools/release/test_baseline.json`
is the authority and records 136, so a reader comparing this file against the baseline should expect the
difference rather than treat either as wrong. Annotated rather than edited in place: a dated record of
what a merge produced should not be rewritten to match a later state. The count-derivation method above
is the part worth reusing, and it still holds.

**Not established:** why the FP8 TMA kernel faults. It may be an sm_120a limitation or a defect in
upstream's kernel, and telling upstream it faults on a consumer Blackwell target is worth doing either
way.

---
---

### 13. Test-suite review: one real coverage gap found and closed, and two of my own claims were wrong
**2026-09-29, after the `d44ab584` merge. The suite is green at 133/135; the two failures are
`dflash_real` and `moe_real`, which fail by construction because this product ships no `dflash`
component and no 35B-A3B MoE checkpoint. `check_test_baseline.py` GATE PASSED.**

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
than failing the request. compute-sanitizer named the frame (`selector_walk_nvfp4_kernel` via
`score_row` at `candidate_selector_path_nvfp4.cu:108`) and a temporary probe printing the pointers
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

### 14. A lane's first request returns different, shorter text — a serving behaviour, undiagnosed
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

**Re-scoped 2026-10-01: this is two separate items, and the text half is answered.** The four candidate
causes listed above are not equally live. External research, and one in-tree fact, split them:

* **The *text* difference is a known, engine-wide behaviour with a named mechanism: prefix caching.**
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

## Closed — do not reopen

Each of these was investigated and settled. They look like open work and are not.

| Item | Why it is closed |
|---|---|
| **Rebuild the four models** | The merge touched no artifact, layout, binding or converter file, and no container version. Weights, layout and bindings are byte-compatible. What was stale is the *measurements* — item 3. |
| **Store the DFlash2 drafter at Q4** | Already measured here, per target. `tools/convert/official_recipes.py:303-312` records that the NVFP4 draft rule "was tried here and *lost* 3.2 acceptance points on the DFlash2 lane (57.7% against 60.9%)", which is why the Swift line keeps its draft at Q8, and states the rule: "a draft encoding is measured per target, and this target's hidden states are not the stock ones." Lines 40-51 and 57-58 already assign `Q8` to drafter parameters. The published Q4 result is on a different model. |
| **Fix C2719 by passing the descriptor by reference** | It compiles and then faults: nvcc's host stub passes the host address as a device pointer. A compile-only check would pass it. The correct fix is to drop `alignas(128)`, which this port already did in `1218d574`. |
| **The drafter's block uses a causal-over-block mask** | Refuted. The reference is non-causal (DFlash paper section 4.2; the published checkpoint sets `is_causal: false`), and `context_query.cuh:275-283` already gives every query row `valid_keys = valid` with no causal predicate. |
| **Re-measure at T=1.0 to match the model card** | Backwards. DFlash2's selector measures 4.61 at T=0 against 4.25 at T=1; greedy is the *favourable* side. Re-measuring at T=1 would widen the gap. |
| **The planner's `chunked_target` topology class causes the acceptance cliff** | Refuted by measurement. Acceptance is bit-identical before and after `a012e2bc` removed the predicate. The class selected which shared `cudaGraphExec_t` a profile reused, not which kernel ran. |
| **Acceptance is low because our drafter is mismatched to the target** | True of `qwen3_8_27b_nvfp4.v3.ninfer` (unsloth quantization plus official drafter — upstream issue 298 section 3 documents 3.3-5.1% for exactly that pairing), but that artifact **is not a shipping lane**. The four shipping lanes accept 52-69%. |
| **Take the full 16-commit upstream merge** | The 7 fp8 TMA commits fail on MSVC (`error C2719`) and are deferred, not forgotten. Three one-line `alignas` reapplications, already validated here. |
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

- `--ngram chain` is accepted, validated against the backend, carried into `Program`, and produces
  nothing: `ngram_drafted_tokens` and `ngram_accepted_tokens` are declared at `types.h:761-762` and
  written nowhere, and `impl->ngram` (`startup.cpp:819`) is never read. **It does not reserve
  288 MiB** — that allowance went with `ngram_policy.h` in `8c7242e6`, and `startup.cpp:884` now says a
  copy round runs at the round's own width and provisions nothing extra. **This is now a withdrawal
  item, not a build item.** An earlier revision of this file pointed at item 9 for the fix ("the flag
  is the switch `PromptLookup` needs"); item 9 is closed, so that reasoning no longer holds. What
  remains is that a shipped flag validates, enters the program and produces nothing, and
  `prompt_lookup.h` is a tested component with no caller. Either is a surface this project does not
  otherwise keep.
- ~~`tools/release/check_doc_links.py` does not skip fenced code blocks~~ — done in `46ec0c5b`, with
  seven tests, six of which fail against the previous body.
- DFlash **v1**'s published config has no `is_causal` key, so the reference gives it five causal and
  one non-causal draft layer. Our converter's `_fixed` check only raises when the key is *present*
  and the runtime never reads it, so a converted v1 drafter silently gets a uniformly non-causal
  block. Not a shipping lane; a real defect.
- `C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer` sits beside the four shipping artifacts, is not a
  shipping lane, and reads acceptably by filename. It cost a full session of benchmarking before
  `profiles.py` was checked.
