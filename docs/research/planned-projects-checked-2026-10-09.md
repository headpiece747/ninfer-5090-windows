# The seven planned projects, checked against the field

**Why this exists.** Four measurements (NIAH, task accuracy, spec correctness at temperature > 0, soak) and
three engine directions (the drafter's precision, the q/k-vs-v/o isolation, MLP A8→A16) were on the list by
name, each priced in card hours. Before spending them, each was checked against what the field publishes.
**Three change, two are reframed, two are confirmed**, and one finding is a direct warning about a
configuration this port already ships.

## 1. NIAH — change it

**Our plan:** a needle-in-a-haystack config, 200 samples or 66 `native_long`.

**The field has moved off literal retrieval.** RULER's own framing: *"this simple retrieval-based test is
indicative of only a superficial form of long-context understanding"*. NoLiMa (Adobe, ICML 2025): *"models can
exploit existing literal matches between the needle and haystack"* — and with minimal lexical overlap, *"at
32K, 11 of 13 models drop below 50 % of their strong short-length baselines"*. NeedleBench: *"retrieval remains
a largely solved problem for modern LLMs"*. And the audit in arXiv 2605.23170 finds that across four flagship
2026 releases, the seven long-context benchmark families including NIAH and RULER appear in **zero of 28 main
result-table cells** — the vendor tables lead with SWE-Bench, Terminal-Bench and BrowseComp instead.

**What the family actually measures**, per the comparison that prompted this:
MRCR (near-identical needles → disambiguation), RULER (multi-needle, tracing, aggregation → degradation with
length), LongBench v2 (503 realistic MC tasks, including **code-repository understanding**), NoLiMa (latent
association, no lexical overlap).

**Verdict.** Keep the long-context axis, replace the instrument. A literal single-needle test measures the one
capability this product's lanes do not need and that the field considers solved. Take: **a multi-needle or
aggregation task** (RULER-shaped) plus **a code-repository task with no verbatim overlap** (NoLiMa's idea on
this port's own corpus), and report **effective length** — the longest context where accuracy stays above
~85 % of the short-context baseline (NoLiMa's and RULER's own rule) — rather than a raw pass rate.

## 2. Task accuracy — keep, but size it by power, not by habit

**Our plan:** the generation grader in `eval/`, 7–11 GPU-hours per build for 258 paired questions.

**The field's instruments for code** are executable: LiveCodeBench and HumanEval+ style unit-test checks,
which are discriminative per item and cheap per item, unlike scored-likelihood endpoints. The port's existing
AIME endpoint is *task-adjacent* by its own note (it scores the reference answer's likelihood, not whether the
model produces it).

**Verdict.** Keep the paired design — it is the right shape — and add (a) a **power calculation** that sets
the sample count from the effect worth detecting (the depth work this session showed 3 % effects at 0.3–2.3 %
spreads are resolvable with the right protocol, so 258 is a habit rather than a number), and (b) at least one
**executable** code task, which is the sharpest discriminator for code-shaped traffic.

## 3. Spec correctness at temperature > 0 — replace the plan with a named test

**Our plan:** a "correctness at temperature > 0" measurement, unspecified.

**The field states the criterion exactly** (arXiv 2510.22876): *"Under greedy decoding (temperature = 0),
speculative decoding must yield outputs identical to standard autoregressive generation, and any token-level
divergence indicates an implementation error. For temperature > 0 … speculative decoding must preserve the
output distribution of the target model."*

**And vLLM implements the test**: a **chi-squared goodness-of-fit** on sampled tokens against the target
distribution, at temperatures **0.6 and 1.0**, K = 1 and 3, with bins below 5 expected counts merged and a
threshold of `df + 10·sqrt(2·df)`; plus **greedy equality** (speculative == standard at temp 0), which this
port already has as request digests.

**Its own stated gap is our opportunity**: vLLM's maintainer notes *"one testing gap … verifying losslessness
when sampling parameters are used, such as freq_penalty/topk/topp"* — and this port's lanes run top-k, top-p,
min-p and thinking budgets. Their audit also records a real bug of exactly this repo's #349 class: a rank
comparison with `>` in one path and `>=` in the other, fixed in #7899.

**And the engine-side test that design implies already exists here.** `tests/ops/test_speculative_round.cpp`
builds a target distribution from scratch (`sparse_target_distribution`) applying temperature, top_k, min_p,
top_p and both presence and frequency penalties, with cases named `accept_distribution_isolation_case` and
`greedy_penalty_case`. The accept rule is therefore already checked against an independent oracle that honours
the full sampling configuration — precisely the "proposal and target perturbed identically" property. What
that cannot show is the *assembled route*, which is why the missing check is end-to-end rather than a unit.

**Verdict.** Do not invent a threshold test. Port vLLM's two tests, and extend to the sampling parameters
their own issue admits are untested. Audit the three failure modes the batch-spec paper names: **bonus token
sampled from the draft instead of the target** (it reports DSD and Meta's work both wrong this way), rejected
tokens left in the KV cache, and position-ID desynchronisation.

## 4. Soak — keep, with instrumentation

**Our plan:** soak, motivated by #208's intermittent `cudaErrorIllegalAddress`.

**Verdict.** Keep. Add what the field checks over long runs that a crash-only soak misses: host RSS and VRAM
drift, KV page counts, and acceptance drift over hours, so a slow leak is a result rather than a mystery. The
reproduction recipe already exists (`tools/bench/chunked_context_soak.py`).

## 5. The drafter's precision — confirmed, and it is the best-supported direction

**Our plan:** vary the drafter's own precision while holding the target fixed.

**The field agrees, and more strongly than expected.** SGLang ships
`--speculative-draft-model-quantization`, default **"same as target"** — quantise the draft like the target,
which is what this port's NVFP4 draft on NVFP4 targets already is. The port's own datum (draft NVFP4 vs Q8:
acceptance 54.8 % → 58.0 %) is a case of the same rule. A consumer-card study (16 GB, Q4/Q5/Q8 targets ×
EAGLE-3/DFlash/DSpark) finds *"acceptance is robust to target quantization"* and *"a Q4 draft is both faster
and ~1 GiB cheaper than an F16 draft, α differs by only 0.010"*.

**And a negative result that validates this port's conversion**: the same study reports **DFlash on Qwen
failing catastrophically — α ≈ 0.009, "essentially never"**, degenerating to 0.63× — attributed to a draft
*"distributionally incompatible with the target's sampling … vocabulary or positional mismatch"*, using a GGUF
draft. This port measures DFlash2 on Qwen at **0.44–0.63 acceptance** because its draft is built by its own
converter from the same source checkpoint as the target. The alignment is the asset.

**Also corroborated: the n-gram supplement.** SGLang ships n-gram drafting with a trie depth of 18 and a
10 M-entry capacity, and NVIDIA NIM uses n-gram as the model-free fallback behind MTP and EAGLE3. Upstream's
#234 proposal and the fork's +8–9 % of output are the same axis.

**Verdict.** Do it, and add the n-gram supplement to the same experiment. One instrument worth borrowing: the
**break-even acceptance** fit (`TPS = a + β·α`, `α_be = (TPS_base − a)/β`), which prices an acceptance
regression in tok/s directly — the number every acceptance finding this session lacked. On a consumer card the
reported break-even is 10–40 % at k = 5–10, well below this port's measured 43–63 %.

## 6. q/k vs v/o isolation — reframed; the field has answered most of it

**Our plan:** isolate q/k from v/o precision.

**The field's answer is K ≫ V, with a theorem and a mechanism.** "Quantize What Counts" (arXiv 2502.15075)
proves key projections carry larger spectral and Frobenius norms, hence strictly higher sensitivity: **K4V2
recovers ~98 % of K4V4** while **K2V4 loses up to 30 points**, and *"key-first allocations never underperform
value-first in any configuration"*. KIVI (ICML 2024) measured the mechanism: keys have persistent outlier
channels that must be quantised per-channel (~5× attention-score error otherwise) while values prefer
per-token (~15× smaller error), and a later paper traced the key asymmetry to **RoPE**.

**And this port already ships the recommendation**: `--kv-dtype k8v4` is 8-bit keys with 4-bit values — the
field's K-prioritised split, as a lane flag.

**Verdict.** The planned isolation is largely answered, and the interesting remainder is different: (a) verify
`k8v4` against `fp8` on served-length domains — cheap, and it is the format the field says should win on
quality per byte; (b) per-layer and per-head precision (KVTuner: *"retrieval heads"* are sensitive,
*"streaming heads"* robust) is a project, not an isolation, and it needs a converter change rather than a lane
flag. The speed-and-acceptance half of (a) is being measured; it is not recorded here until it returns.

## 7. MLP A8→A16 — reframed toward what this port already owns

**Our plan:** raise MLP activations from A8 to A16.

**The field's framing**: activation quantisation is hard because of **outliers in fixed channels** ~100× the
typical magnitude (SmoothQuant), and the remedies are *outlier migration* — equivalent transforms that smooth
the activation and push the difficulty into the weights (SmoothQuant's α = 0.4–0.6), or rotations (QuaRot).
Note also a direct warning about this port's own configuration from arXiv 2505.22179: with **4-bit weights**,
tree-style draft verification loses its memory-access advantage (`T_v(n)/T_t` reaches 1.8 at tree size 60
against <1.2 for FP16/W8A8) — *"fewer draft forward passes"* is what it recommends, and this port's chain-style
block drafting is the sequence-shaped verification its hierarchical fix builds toward.

**Verdict.** A8→A16 is the a8policy direction restricted to MLP, and the a8policy measurement already prices
it: −0.54 % perplexity at 2.2× the linear cost, refused.

**And the reframing this document first proposed for it was wrong.** It said to sweep the converter's
`activation_input_divisor` auxiliary as if it were a free knob. `docs/research/quantization-coverage-evidence.md`
premise (C) had already audited exactly that: *"`6/input_scale` on an FP8 site equals `2688/amax` exactly —
the port's own `FULL_RANGE/peak` formula, verified against NVIDIA's real tensor bytes at 24 sites"*. The
divisor is not an independent axis; it *is* the port's max-abs rule, recovered from the producer's
calibration. What survives is the narrower and real concern: it is *max*-derived, which is what NVIDIA's own
documentation says leaves no headroom, and NVIDIA's replacement for it
(`NVFP4ActHeadroomCalibrator`) is opt-in, recent, **unmeasured on any checkpoint, and not used by NVIDIA's own
Qwen3.8-27B recipe**. So the open lever is headroom calibration — a conversion-side candidate with a named
implementation to read, **not** a sweep of a value that is already determined.

## What this changes, in one line each

- **NIAH** → a multi-needle or code-repository task without verbatim overlap, reported as effective length.
- **Task accuracy** → keep the paired design, size it by power, add one executable code task.
- **Spec correctness** → port vLLM's chi-squared and greedy-equality tests, and cover the sampling parameters
  their own issue says are untested.
- **Soak** → keep, and instrument leak and drift so a slow failure is a result.
- **Drafter precision** → confirmed; add the n-gram supplement and the break-even instrument.
- **q/k vs v/o** → mostly answered (K ≫ V, and `k8v4` is already the answer); verify k8v4 on served domains.
- **MLP A8→A16** → reframe to sweeping the `activation_input_divisor` the converter already exposes.

## Sources

RULER (arXiv 2404.06654); NoLiMa (ICML 2025, PMLR v267); LongBench v2 (ACL 2025); NeedleBench
(arXiv 2407.11963); "Positional Failures in Long-Context LLMs" (arXiv 2605.23170); vLLM speculative-decoding
docs and `tests/v1/spec_decode/test_probabilistic_rejection_sampler_utils.py`; vLLM issue #7627 (the
maintainer's three-layer losslessness note and the stated sampling-parameter gap); "Batch Speculative Decoding
Done Right" (arXiv 2510.22876); KIVI (ICML 2024, arXiv 2402.02750); "Quantize What Counts: More for Keys, Less
for Values" (arXiv 2502.15075); KVTuner (ICML 2025, arXiv 2502.04420); SmoothQuant (ICML 2023); "Speculative
Decoding Meets Quantization" (arXiv 2505.22179); `reyden009/speculative-decoding-lab` (consumer-GPU
quantisation × drafter study, incl. the DFlash-on-Qwen negative result); SGLang speculative-decoding docs;
NVIDIA NIM speculative-decoding docs; Quasar (arXiv 2603.01399).
