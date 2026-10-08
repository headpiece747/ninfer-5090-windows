# Task-accuracy protocol: what a 27B model on one 5090 can and cannot resolve

Research note. Written 2026-10-07. Every claim carries its source and one of these labels:

| label | meaning |
|---|---|
| **[measured]** | a run produced the number; the configuration travels with it |
| **[read in source]** | read from primary code, config or documentation |
| **[third party]** | an assertion or secondary account, quoted as such |
| **[derived]** | arithmetic performed here from the sources shown, so it can be checked |

**[no evidence found]** marks a searched absence. Nothing here was measured on this product except
where it says so.

**The question.** Importing the producer's FP8 attention and linear-attention projections instead of
re-encoding them to NVFP4 is worth **1.41 % / 2.35 % perplexity** and **+29 % / +36.7 % relative
DFlash2 acceptance** on two independent sources, at **−14.2 % / −13.4 % prefill** and **−28.5 %**
implied KV context (`docs/research/attention-topology-2026-10-07.md`; [measured, in-tree]). The
published record says quantization damage shows on *task* accuracy where perplexity hides it — the
QHR study's W4A4KV4 32B model loses **3.9 pt on AIME-120, 1.2 on MATH-500 and 0.0 on GSM8K**
(arXiv:2504.04823v2 Table 1; [read in source]). This note specifies the protocol that would test
whether the FP8-attention gain survives there, and what it can resolve.

**The answer in one line.** [derived] A 30-question AIME sitting cannot detect a 4-point difference:
its per-run standard error is **4.0 pt** and the difference of two sittings **5.6 pt**; at the
discordance this artifact pair already showed (6 of 60 questions), a paired design needs **~490
questions** for 80 % power at 4 pt. The smallest design worth running is
**GPQA-Diamond (198) + AIME25/26 (60) = 258 paired questions**, which detect ~4–5.5 pt at 80 %
power for **7–11 GPU-hours per build**; a 1-point difference is out of reach, and the 258-item run
carries a pre-registered extension for the 4-pt target. AIME alone is a tie-breaker, not a
measurement.

## 1. What protocol distinguishes a 1–4 point difference?

### The arithmetic, shown

Binary scores, two models, α = 0.05 two-sided, 80 % power (`z` = 1.96 and 0.8416, sum squared =
7.849).

- **One AIME sitting (n = 30, p ≈ 0.95).** [derived] Per-run sd = √(0.95·0.05/30) = **4.0 pt**;
  95 % CI on the score ±7.8 pt. Two independent sittings differ with sd √(2·0.0475/30) = **5.6 pt**,
  so a 4-pt gap is 0.7σ. Fisher's exact test on 29/30 vs 28/30 gives **p ≈ 1.0**.
- **Unpaired two-proportion n for 4 pt** at p₁ = 0.95, p₂ = 0.91: [derived]
  n/arm = 7.849·(0.0475 + 0.0819)/0.04² = **635**, i.e. 21 AIME sittings per arm.
- **Paired McNemar-Connor** (arXiv:2605.30315 Eq. 6): N* = 7.849·σ_D²/δ², with
  σ_D² = p_Aq_A + p_Bq_B − 2ρ√(p_Aq_Ap_Bq_B). At the port's own observed discordance q ≈ 0.1
  (σ_D² ≈ q for small δ): [derived] **4 pt → ~490 questions, 2 pt → ~1,960, 1 pt → ~7,850**.
  At ρ = 0.7 (σ_D² ≈ 0.054): 4 pt → ~265. Discordance, not sample size alone, is the planning input.
- **Minimum detectable effect at fixed N**, same formula inverted: [derived] N = 60 → **11.4 pt**;
  N = 198 → **8.4 pt** at q = 0.18 (independent GPQA) or **4.6 pt** at q = 0.054 (ρ = 0.7);
  N = 258 (GPQA + AIME) → **5.5 pt** at q = 0.1, **4.0 pt** at q = 0.054; N = 1,000 → **2.8 pt**.
- **Continuous per-question scores are the upper bound on efficiency.** Miller, *Adding Error Bars
  to Evals* (arXiv:2411.00640 §3.2): next-token probabilities "reduce the variance of the estimator
  by 2/3 (the upper limit achievable via resampling) compared to grading a single sample from each
  question"; his paired sample-size formula is
  n = (z_{α/2}+z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ² [read in source]. The port's own 60-problem
  paired per-problem PPL test (t = 1.49, p = 0.14; `eval/README.md`) implies an effect size of
  d_z ≈ 0.19, so **~213 questions** reach 80 % power on that instrument — at ~0.05 s per question
  per artifact (§3). It is a sensitivity measurement, not an accuracy one.

### What published comparisons use

| suite | n | decoding | sampling | thinking budget | extraction / grader |
|---|---|---|---|---|---|
| AIME 2024/2025, single sitting | 30 | — | pass@1 and cons@64 (o1 blog) | not stated | not stated [third party] |
| AIME-120 (2022–2025), QHR | 120 | T 0.6, top-p 0.95 | 3 seeds, averaged; ± is across seeds | 32,768 gen tokens | lighteval + vLLM [read in source] |
| AIME24/25, lm-eval-harness | 30 | **greedy** (`do_sample: false`, T 0.0) | 1 sample | 32,768 gen tokens | `\boxed{}`/`$…$` + `strip_string` string normalization; **no sympy** [read in source] |
| AIME24/25/26, EvalScope 1.10.0 | 30 | caller/job-config; port: T 1.0, top-p 0.95, top-k 20, seed 42 | 1 sample | caller/job-config; port: 122,880 | `\boxed{}` → `normalize_answer` → OpenAI PRM800K sympy `grade_answer`; optional LLM judge [read in source] |
| AIME24/25, lighteval | 30 | job-config | pass@1, **avg@64**, g-pass@16 of 48 variants | `generation_size=None` (task) / 32,768 (QHR run) | `math_scorer` (math-verify) [read in source] |
| GPQA-Diamond, paper baseline | 198 | not stated | few-shot CoT, GPT-4 38.8 % | not stated | not stated [read in source] |
| GPQA, simple-evals | 198 | 0-shot CoT | **n_repeats = 4** (792 samples), option permutation per repeat | not stated | regex `Answer: [A-D]` [read in source] |
| GPQA-Diamond, EvalScope / port | 198 | T 1.0, top-p 0.95, top-k 20, seed 42 | 1 sample | 245,760 gen tokens | CoT prompt, shuffled choices, rule letter match [read in source, in-tree] |
| MMLU-Pro, lm-eval-harness | 12,032 (10 options) | **greedy**, 5-shot CoT | 1 sample | 2,048 gen tokens | regex `answer is \(?([ABCDEFGHIJ])\)?` [read in source] |
| MMLU-Pro, d0xin subset | 280 | not stated | 1 sample | not stated | 225/280 vs 224/280 — a 1-question gap [third party, in-tree note] |

**A study that says what a given N can separate.** [read in source] Kotawala, *Resolution Diagnostics
for Paired LLM Evaluation* (arXiv:2605.30315): across 40 Open LLM Leaderboard v1 pairs, **11 are
unresolved** at (α, 1−β) = (0.05, 0.8); every pair with |δ| ≤ 2 % is unresolved and the resolution
boundary sits near |δ| ≈ 5 %. On MMLU-Pro's 12,032 items, 4 of 9 top-10 adjacent pairs are still
unresolved. Paired McNemar required-N is a median **2.15×** smaller than Miller's unpaired formula
(IQR 1.60–2.75) — and the common "unpaired Cohen-h then ×(1−ρ)" shortcut underestimates N* by
about **2×** in exactly the close-comparison regime.

**Small N needs non-CLT error bars.** [read in source] Bowyer, Aitchison & Ivanova, *Position: Don't
Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints* (arXiv:2503.01747, ICML 2025
Spotlight): at small N the CLT "usually dramatically under-estimat[es] uncertainty (i.e. producing
error bars that are too small)". At 30–258 items, use exact/McNemar or a paired bootstrap, not the
normal approximation alone (arXiv:2605.30315 §5: at n = 500 the asymptotic trio is calibrated
within ~1 pt, the exact variant is ~3 pt conservative).

**Contamination is a first-order confound for AIME.** [read in source] Balunović et al., *MathArena*
(arXiv:2505.23281): "we find strong signs of contamination in AIME 2024". The port already runs
AIME 2026 (post-cutoff) — keep it; treat AIME 2024 as unusable for this comparison. GPQA carries a
canary string and asks that examples not be republished (arXiv:2311.12022 §1). [third party]
Secondary summaries of MathArena report a ~14 % average 2024-vs-2025 inflation across the top 12
models; Vals AI likewise notes models score higher on 2024 than on the freshly released 2025 set.

## 2. What the quantized-model comparisons actually do

| source | suites | n / repeats | decoding | thinking budget | baseline arm |
|---|---|---|---|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` card | GPQA-D, Terminal-Bench, AA-LCR, MMMU-Pro, SciCode, IFBench | **not stated** | "All evaluations used temperature=1.0, top_p=0.95, and a vLLM context limit of 262,144 tokens"; `max_new_tokens` 65,536 | not stated | BF16 only; no FP8 arm |
| `unsloth/Qwen3.8-27B-NVFP4` card | none | — | recommends T 1.0 / top-p 0.95 / top-k 20; `reasoning_effort` xhigh default | xhigh | docs publish KLD + top-1 ("92–97 % accuracy recovery") and a Qwen3.6-era task figure; no protocol [read in source] |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` card | GPQA-D + AIME'26 | **GPQA-D: 2 runs, n = 396; AIME'26: 3 repeats, n = 90** | **not stated** | not stated | BF16 + two NVFP4 builds; no FP8 arm |
| NVFP4 shootout, Qwen discussion #192 | PPL (323 windows), KLD (95 prompts), MBPP 257, HumanEval 164, GSM8K 500, tool-use 20 | full suites | T 0 for all generation tiers, T 1.0 for the agentic tier; MTP 5 | thinking off for code/math, on (xhigh) for tool-use | BF16 and FP8 included |
| QHR (arXiv:2504.04823) | AIME-120, MATH-500, GSM8K, GPQA-D, LiveCodeBench | full suites, 3 seeds | T 0.6, top-p 0.95, 32,768 gen tokens | not stated | BF16 |

What the three vendor cards do **not** publish: sample counts (NVIDIA), decoding parameters
(QUASAR), confidence intervals (all), a reasoning budget (NVIDIA, QUASAR), or an FP8 arm (NVIDIA,
QUASAR; the shootout and QHR do include one). The QUASAR card's headline deltas —
BF16 91.41 → QUASAR 90.91 GPQA-D (−0.5), 100.0 → 100.0 AIME'26 — are at n = 396 and n = 90 with no
CI; the AIME'26 arm is saturated at both ends. [read in source]

**The one quantized comparison that did reach a task difference** is QHR's W4A4KV4 row: AIME-120
−3.9 pt, MATH-500 −1.2, GSM8K 0.0 for the 32B distilled model, at T 0.6 over 3 seeds. The in-tree
note records 4.17/1.40/0.00 for the same row (`docs/research/quantization-measurement-methods.md`
§1); the v2 table reads 61.7±1.7 → 57.8±4.3. The two readings differ by 0.3 pt and the note does not
record a revision — quote whichever is used with its source. [read in source vs in-tree] The
readings differ by ~0.3 pt on AIME-120 and ~0.2 on MATH-500.

**[no evidence found]** for a published task-accuracy protocol that pins a thinking budget for any
NVFP4 build of this model, for a confidence interval on any vendor delta in the table above, or for
a per-question AIME variance figure on a 27B Qwen. QHR's three-seed spread (AIME-120 61.7 ± 1.7 for
the 32B model) is the closest published variance number; everything else is a point estimate.

## 3. The cheapest protocol that is still evidence

Three method families, ranked by discriminating power per GPU-hour:

1. **A fixed-seed subset with a CI** is the weakest: it spends the between-question variance
   (ω² in Miller's formula) that pairing removes, and at n = 30–60 the 95 % CI on a difference is
   ±11 pt — wider than the effect being chased (§1).
2. **A paired design on shared questions** is the workhorse. Miller: use question-level paired
   differences; arXiv:2605.30315 measures the gain at 2.15× median over unpaired and gives the
   McNemar required-N used above. It needs no repeated sampling; one generation per question per
   build suffices.
3. **A continuous per-question endpoint is the highest-power instrument per token.** Miller's
   next-token-probability result (variance −2/3 versus one graded sample) is exactly what the port's
   per-problem instrument already computes: score each question's statement + gold answer and use
   the paired difference of log-probabilities. It resolves *that* two builds differ, not accuracy;
   the port's own 60-question run showed the two builds disagreeing in sign between token-weighted
   and question-weighted aggregation (`eval/README.md`), so report both weightings.

**Recommended smallest design, with the arithmetic.** [derived] Item set: **GPQA-Diamond 198 +
AIME 2025 30 + AIME 2026 30 = 258 paired questions**, one generation each per build, fixed order,
seed 42, thinking on, vendor sampling (T 1.0 / top-p 0.95 / top-k 20), budget 122,880 output tokens
at concurrency 2 (the AIME budget the port already ran with zero truncations; re-run any sample that
hits the cap at concurrency 1 / 245,760 and never score a truncation as wrong). 258 items detect
4.0–5.5 pt at 80 % power **only if** the pair's discordance is 5.4–10 %; at the 10 % this pair
already showed, a 4-pt target needs ~490 items, so pre-register a ~250-item MMLU-Pro extension to
be taken if the first ~100 paired items project N* > 258.

- **Tokens per answer**: [measured, in-tree] AIME25 486,049 reasoning tokens / 30 = **16.2k**, plus
  ~0.4k text; AIME26 15.4k reasoning. GPQA has **no retained timing** in the tree; the only recorded
  datum is a 154,453-token completed tail (`eval/README.md`). Planning estimate **20–50k tokens per
  GPQA answer**, to be measured before the run.
- **Seconds per token**: [measured, in-tree] 30 AIME questions in 1,853 s and 30 in 1,725 s at
  concurrency 2 → ~**270 tok/s aggregate**, ~135 tok/s per stream, 3.7 ms/token.
- **GPU-hours per build**: [derived] AIME 60 questions ≈ **1.0 h**; GPQA at 30k tokens/answer and
  C = 2 ≈ **6 h** (at C = 1 ≈ 12 h; at 50k/C = 2 ≈ 10 h). Total ≈ **7–11 h per build, 14–22 h for
  both** — plus the continuous endpoint at ~13 s per build.
- **What it detects**: **4.0–5.5 pt** at 80 % power at 258 items (discordance 5.4–10 %), **4.0 pt**
  at ~490 items and the observed 10 % discordance, and a reportable CI of roughly ±4–6 pt otherwise.
  **What it cannot detect**: 1–2 pt; and it cannot *rule out* a 1-pt difference. To detect 2 pt,
  budget ~2,000 paired questions (MMLU-Pro's 12,032-item bank is the cheapest source; EvalScope
  1.10.0 ships an `mmlu_pro` adapter, so it is a config entry), at a few times the hours above —
  MMLU-Pro's answer length under thinking must be measured in the pilot.
- **Two-stage discipline**: run ~100 paired items first, measure the observed discordance, recompute
  N* (the required N depends on it 2× between ρ = 0 and ρ = 0.7), then extend once. Pre-specify the
  extension rule, or use the anytime-valid e-process of arXiv:2605.30315 §6.5, whose threshold
  inflates the z-budget ~2× — otherwise extending on a peeked result inflates α.

## 4. Grading and failure modes

**How each harness extracts a final answer.** AIME: `\boxed{}` is the convention everywhere, but the
grader differs — lm-eval does `last_boxed_only_string` then string normalization (`strip_string`,
from `hendrycks_math`; no sympy), EvalScope 1.10.0 does `extract_answer` → `normalize_answer` →
PRM800K sympy `grade_answer` (integer-strict), lighteval uses `math_scorer` (math-verify), and
simple-evals has no AIME task at all (its MATH grader asks GPT-4o whether two expressions are
equivalent). GPQA: simple-evals/lm-eval/lighteval require `Answer: X`-style lines (`ANSWER_PATTERN_MULTICHOICE`,
`The answer is …`, `Answer: $LETTER`), EvalScope matches `ANSWER: [LETTER]` and, if the model omits
it, **falls back to the last uppercase letter in the completion** (`multi_choices.py`,
`_fallback_parse_answer`). [read in source]

| failure mode | evidence | direction of the bias |
|---|---|---|
| Generation truncated at the budget, then scored as wrong | [measured, in-tree] at a 65,536 budget: 4/60 and 2/60 AIME generations truncated, scored as misses → 28/30 and 27/30; at the documented 122,880: 0/60 → 29/30 | understates the build that truncates more; if symmetric, dilutes toward zero |
| Thinking budget / effort not pinned | [read in source] simple-evals: same model, GPQA 83.4 (o3-high) vs 78.6 (o3-low), 4.8 pt; o3-mini 77.2 vs 67.6, 9.6 pt | can create or hide an effect **larger** than the one being measured |
| Budget forcing changes the answer | [read in source] s1 (arXiv:2501.19393): terminating/extending thinking moves AIME24 50 % → 57 % | same as above |
| Answer-format sensitivity | [read in source] *Let Me Speak Freely* (arXiv:2408.02442): format restrictions degrade reasoning, stricter constraints worse; each harness's regex accepts only its own format | mostly toward zero; the "last uppercase letter" fallback can score either way |
| Option order / position bias | [read in source] EvalScope 1.10.0 shuffles choices with an unseeded `random.shuffle` (main fixed it by hashing the question: `--rerun-review` once "paired cached predictions with a freshly shuffled answer key and accuracy collapsed to chance") | noise; dilutes the difference; pin or record the permutation |
| Pooling K samples per question as independent | [read in source] Miller §3.1: "computing a pooled standard error across all K·N answers will be inconsistent" | inflates n, narrows CIs falsely |
| Contamination | [read in source] MathArena: "strong signs of contamination in AIME 2024"; GPQA ships a canary string | inflates both builds and compresses the gap at the ceiling |
| Post-hoc suite or subset selection | [read in source] arXiv:2605.30315 §6.3; Miller §5 (power analysis) | inflates α; fix the item set and the analysis first |
| Ceiling | [measured, in-tree] the port's AIME25/26 score is 96.67 % (29/30); one question is 3.3 pt | a suite at the ceiling cannot show a 4-pt effect at n = 30 |

## 5. What already exists in this tree and elsewhere

**In this tree.** `eval/` is a repository-local coordinator whose first backend is **EvalScope
1.10.0** (`eval/requirements.txt`), with AIME25/AIME26/GPQA-Diamond jobs, rule scoring
(`judge_strategy: rule`), the vendor sampling settings (T 1.0 / top-p 0.95 / top-k 20 / seed 42 /
`enable_thinking`), 122,880-token AIME and 245,760-token GPQA budgets, and recorded runs
(`eval/configs/qwen3_8_27b_nvfp4_reasoning.yaml`, `eval/runs/20260924T…`; [read in source, in-tree]).
The AIME pair campaign config was deleted in `a51fc963` ("drop AIME, keep what it was standing in
for") precisely because 60 questions cannot resolve a two-sample difference; the adapter, the GPQA
job and the run records remain. **No MMLU job exists.** So the work this note specifies is a
protocol and a config, not a new runner — the per-sample `reasoning_tokens` and durations needed for
§3's arithmetic are already in the review JSONL and `summary.json`. One integration limit matters:
the serving API accepts `logprobs: false` / `top_logprobs: 0` only (`docs/serving.md`), so the
continuous per-question endpoint runs through `ninfer-perplexity`/`score_topk` offline, not through
the HTTP API.

**Elsewhere**, for the same question, against a local OpenAI-compatible endpoint:

| harness | what it needs | offline during evaluation? | what the grader does |
|---|---|---|---|
| lm-evaluation-harness | `--model local-chat-completions --model_args base_url=…`; `think_end_token` for reasoning models (the server must return the delimiter in `content`); GPQA's HF dataset is gated | yes, after dataset/model cache; no network calls in the grader | AIME: boxed + string normalization; GPQA: two regex filters (`The answer is …`, `\([A-Z]\)`); MMLU-Pro: 5-shot CoT + `answer is (X)` [read in source] |
| lighteval | vLLM or an endpoint backend; HF datasets (`HuggingFaceH4/aime_2024`, `Idavidrein/gpqa`) | yes, after cache | `math_scorer` (math-verify) for AIME; `choice()` + `gpqa_instruct_pass_at_k` for GPQA; `generation_size = 32768` for reasoning models [read in source] |
| simple-evals | OpenAI/Claude API samplers only | **no** — the math equality grader is an LLM call; deprecated July 2025 | `check_equality` via GPT-4o; GPQA/MATH regexes; use as a reference implementation, not a runner [read in source] |
| EvalScope (in-tree) | ModelScope dataset cache + the coordinator | yes, after cache | as §4 [read in source] |

The IRT/subsampling line (tinyBenchmarks, arXiv:2402.14992: 100 MMLU items estimate a score) is for
**point estimates**, not for separating two close models — arXiv:2605.30315 §2 draws that line
explicitly. [read in source]

## 6. Recommended protocol

1. **Suites and items.** GPQA-Diamond (198) + AIME 2025 (30) + AIME 2026 (30), fixed item list,
   same order and same option permutations for both builds; pre-register a ~250-item MMLU-Pro
   extension (5-shot CoT as lm-eval defines it, or the same thinking regime) to be taken if the
   first ~100 paired items project N* > 258. Do not include AIME 2024.
2. **Decoding.** Thinking on, `reasoning_effort` pinned to the template default (xhigh), T 1.0 /
   top-p 0.95 / top-k 20, seed 42, same chat template and same engine build flags for both arms;
   concurrency 2; budget 122,880 output tokens with a documented re-run rule for any truncated
   sample (re-run at concurrency 1 / 245,760; never score a truncation as wrong).
3. **Graders.** EvalScope 1.10.0 with `judge_strategy: rule` — sympy PRM800K for AIME, rule letter
   match for GPQA — and the choice permutation pinned (1.10.0 shuffles unseeded). Record the grader
   and its version with the scores; do not compare across harnesses.
4. **Analysis.** Per suite and combined: McNemar exact/mid-p on paired correctness, paired-bootstrap
   95 % CI, discordant counts, and the resolution ratio q = N/N* with the MDE. Report the per-suite
   breakdown; state the pooling assumption if a single combined p-value is given.
5. **Secondary endpoints, same run.** (a) Per-question gold-answer log-probability via
   `ninfer-perplexity` (paired t + sign test; report token-weighted and question-weighted
   separately); (b) truncation count, reasoning-token distribution and repetition/loop flag from the
   same generations.
6. **What it establishes.** [derived] A significant 4.0–5.5 pt difference at 80 % power at 258
   items (4.0 pt with the extension), or a defensible null bounded by a ±4–6 pt CI. It cannot detect
   1–2 pt, and it must say so rather than report the point estimate as the result. Cost: ~7–11
   GPU-hours per build, ~14–22 h for the pair, plus seconds for the continuous endpoint.

## Sources

| source | what was taken from it |
|---|---|
| arXiv:2411.00640 (Miller, *Adding Error Bars to Evals*) | paired-difference variance, sample-size/MDE formulas, next-token-probability variance bound, no-temperature-tuning advice |
| arXiv:2605.30315 (Kotawala, *Resolution Diagnostics*) | McNemar-Connor required-N, paired-vs-unpaired 2.15× median gain, resolution ratios on OLL v1/MMLU-Pro, exact-vs-asymptotic calibration, multiplicity/sequential caveats |
| arXiv:2503.01747 (Bowyer et al., *Don't Use the CLT*) | CLT error bars too small below a few hundred datapoints |
| arXiv:2504.04823 (Liu et al., *Quantization Hurts Reasoning?*) | AIME-120 definition and protocol, per-task damage ordering, seed spread |
| arXiv:2505.23281 (Balunović et al., *MathArena*) | AIME 2024 contamination |
| arXiv:2311.12022 (Rein et al., GPQA) | 198-question diamond split, baselines, canary request |
| arXiv:2406.01574 (Wang et al., MMLU-Pro) | 12,032-item bank, 10 options, CoT benefit, 2 % prompt sensitivity |
| arXiv:2402.14992 (Polo et al., tinyBenchmarks) | IRT/subsampling is point-estimate efficiency, not comparison power |
| arXiv:2408.02442 (*Let Me Speak Freely*) | format restrictions degrade reasoning |
| arXiv:2501.19393 (Muennighoff et al., s1) | budget forcing moves AIME24 50 → 57 % |
| lm-eval-harness `aime/`, `gpqa/generative/`, `mmlu_pro/` task configs and `docs/interface.md` | suite sizes, greedy decoding, budgets, filters, graders, `local-chat-completions` + `think_end_token` |
| lighteval `tasks/aime.py`, `tasks/gpqa.py` | prompts, pass@1/avg@64/g-pass variants, `math_scorer`, generation size |
| EvalScope 1.10.0 and `main` `aime/`, `gpqa/`, `utils/multi_choices.py` | prompts, sympy grader, MCQ regex + last-uppercase fallback, shuffle behaviour |
| `nvidia/Qwen3.8-27B-NVFP4`, `unsloth/Qwen3.8-27B-NVFP4`, `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` cards; Unsloth Qwen3.8 docs; Qwen discussion #192 | §2's comparison methodologies and settings |
| simple-evals README + `math_eval.py`, `gpqa_eval.py`, `common.py` | reasoning-effort spread, repeats/permutation, LLM equality grader, API-only samplers |
| this tree | `eval/README.md`, `eval/configs/qwen3_8_27b_nvfp4_reasoning.yaml`, `eval/requirements.txt`, `eval/runs/20260924T*`, `docs/serving.md`, `docs/research/attention-topology-2026-10-07.md`, `docs/research/quantization-measurement-methods.md`, `git show a51fc963` |
