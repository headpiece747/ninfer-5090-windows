# Evidence: the first-request serving transient, and the MSVC C2719 TMA-descriptor defect

Two independent questions, researched 2026-10-01 against primary sources only: the port's own tree and
tracker, upstream `Neroued/ninfer`, CUTLASS and FlashAttention source, the CUDA and Microsoft
documentation, and the first-party issue trackers of vLLM, SGLang, llama.cpp, ONNX Runtime and CUTLASS.

Every claim below is labelled **[measured here]**, **[read in source]**, **[vendor-claimed]** or
**[not found]**. Nothing in this note recommends an action; it reports what the sources say and, where
a source is silent, what was searched.

The short answers:

- **Question 1.** The *output-difference* half is a documented, engine-wide behaviour with a named
  mechanism in three separate projects' own trackers and docs: a cold first request and a
  cache-warmed later request take different computation paths, and those paths are not
  bit-identical. vLLM, SGLang and llama.cpp all document or file it. The *speed* half -- the first
  request being **faster** than later ones -- is the opposite of the documented direction everywhere
  else, and I found no first-party account of it anywhere.
- **Question 2.** The risk is **live**: upstream `dev` and `master` still carry `alignas(128)` on all
  three descriptor structs today. The C2719 defect is **not tracked** in CUTLASS, FlashAttention or any
  NVIDIA tracker I could reach. No portable upstream fix exists. The header's own `_MSC_VER` branch is
  the standing answer, and there is one third-party precedent for an additional workaround.

---

# Question 1 -- the first-request transient

## 1.1 What the port observes, as recorded in-tree

`docs/active-work.md:480-510` (item 14) and `docs/adr/0005-per-profile-flags-are-measured.md:14-17`
record the behaviour and its consequences:

- After a server start, a client's first full-length request "returns faster than every later
  identical request and returns **different, shorter text** from the same seed. Requests 2..n are
  byte-identical to each other."
- Measured on one lane: 261.7 tok/s on the first request against 169.8 for requests 2-7.
- Speculative acceptance on the first request was 64.6% against 32.7% for later requests; the log
  reads 35.3% over all records and 32.7% with the first discarded.
- **Invisible at temperature 0**: "the greedy digest is stable from request 1".
- The 16-token probe `measure_decode` warmed up with "does not reach the state the transient affects",
  so the transient was averaged into the first measured run. It was corrected by discarding one
  full-length warmup (`docs/active-work.md:118-121`).

Two properties of that record constrain any explanation, and both are worth keeping separate:

1. **The divergence is in the sampled path, not the argmax.** Greedy output is byte-stable from
   request 1. So whatever differs between request 1 and request 2 perturbs the logits by less than the
   margin that separates the argmax, but enough to move a multinomial draw. That is a very small
   numerical difference, not a different model or a different prompt.
2. **The first request accepts drafts better, not worse.** A speculative drafter on a *cold* prefix
   accepting 64.6% where the warm path accepts 32.7% is the opposite of "the drafter has no context
   yet". It is the signature of the target's logits being computed by a *different, more favourable*
   arithmetic route on request 1 -- or of the draft/verify round width `t` differing, which selects
   different kernels (see 1.5).

## 1.2 The documented engine-wide behaviour: cold path vs warm path are not bit-identical

This is the finding that most changes the framing. It is not this port's defect; it is a known
property of prefix caching that three projects state in their own words.

**SGLang, in its own FAQ** ([docs.sglang.io/docs/references/faq](https://docs.sglang.io/docs/references/faq),
fetched 2026-10-01):

> "You may notice that when you send the same request twice, the results from the engine will be
> slightly different, even when the temperature is set to 0. From our initial investigation, this
> indeterminism arises from two factors: **dynamic batching and prefix caching**. Roughly speaking,
> dynamic batching accounts for about 95% of the indeterminism, while **prefix caching accounts for
> the remaining portion**. ... Different batch sizes can cause PyTorch/CuBLAS to dispatch to
> different CUDA kernels, which can lead to slight numerical differences. ... Similarly, when prefix
> caching is enabled, it can also dispatch to different kernels. Even when the computations are
> mathematically equivalent, small numerical differences from different kernel implementations lead
> to the final nondeterministic outputs."

**[vendor-claimed]** -- this is SGLang's own characterisation of its own engine, and it names
prefix caching as a distinct, smaller cause. It also states the consequence plainly: at temperature 0,
identical requests can differ.

**SGLang's deterministic-inference page** states the same root cause and adds that it survives
radix-cache use ([Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference),
fetched 2026-10-01): "The main source is varying batch sizes. Different batch sizes cause GPU kernels
to split reduction operations differently, leading to different addition orders. Due to
floating-point non-associativity ... this produces different results even for identical inputs." The
page's own verification suite includes a mode aimed squarely at this port's question --
`python3 -m sglang.test.test_deterministic --test-mode radix_cache`, described in
`python/sglang/test/test_deterministic.py` as "test radix cache determinism (**cached vs uncached
prefill**)" -- with the expected result "Unique samples: 1". **[read in source]**

That test's existence is the strongest single piece of evidence for Question 1: SGLang considers
"cached vs uncached prefill must produce identical output" a property worth a dedicated regression
mode, and its own FAQ says it does not always hold.

**vLLM, filed and open.** [Issue #40896, "[Bug]: vLLM v1 with prefix caching: first request differs
from subsequent identical requests at temperature=0"](https://github.com/vllm-project/vllm/issues/40896)
(open, opened 2026-04-26, Qwen3-8B on H100, vLLM 0.19.0). The report is the same shape as this port's,
with temperature 0 rather than a sampling temperature:

> "Run 1 on a freshly started server returns output A. Runs 2..N return output B != A, but stable
> across runs. Restarting the server returns the first request to A. **Disabling prefix caching with
> `--no-enable-prefix-caching` makes the output deterministic across all runs.**"

The reporter's own summary line is "This bug can be simply circumvented by just having a few warm up
rounds, but I still report it here for the record". **[vendor-claimed]** -- filed by a vLLM MEMBER
against vLLM, and still open.

A vLLM maintainer replied in that thread ([`yewentao256`, 2026-05-08](https://github.com/vllm-project/vllm/issues/40896)):

> "Yeah, we haven't fully supported deternism with prefix caching currently, feel free to explore more
> and have a pr towards this."

**[vendor-claimed]** -- this is the clearest possible statement that the behaviour is *known* and
*unsupported*, not merely unnoticed. It is the direct answer to "is this documented anywhere": the
maintainer says it is not supported.

A second commenter attached a reduced bundle to the same issue
([`SAKETH11111`, 2026-05-05](https://github.com/vllm-project/vllm/issues/40896)), reporting at
temperature 0 with an explicit `seed=42`:

```
no prefix cache:   '7577'
prefix cache cold: '7577'
prefix cache warmed: '5777'
```

with controls: reproduced on a second H100, not on A100, not on Qwen3-1.7B.

A third commenter measured the mechanism directly
([`brianosaurus`, 2026-08-17](https://github.com/vllm-project/vllm/issues/40896)), comparing **logprobs
rather than token ids**, which is what makes a sub-argmax perturbation visible at all:

| engine | GPU | config | token ids | max logprob delta |
|---|---|---|---|---|
| vLLM `017e9f4448` | H100 (sm_90) | default | identical | 4.613e-02 |
| vLLM `017e9f4448` | H100 | `VLLM_BATCH_INVARIANT=1` | identical | 0, bit exact |
| vLLM `017e9f4448` | RTX 4090 (sm_89) | default | identical | 3.426e-03 |
| vLLM `017e9f4448` | RTX 4090 | `VLLM_BATCH_INVARIANT=1` | identical | 0, bit exact |
| SGLang 0.5.9 | RTX 4090 | default | identical | 4.964e-02 |
| SGLang 0.5.9 | RTX 4090 | `--enable-deterministic-inference` | identical | 0, bit exact |

Their method note is the part that matters for this port, because it is the control this port's own
observation lacks:

> "every case flushes the cache, runs twice cold, then runs warm. **Cold vs cold was bit exact in every
> row, so a warm difference is attributable to prefix reuse rather than generic nondeterminism.** Cache
> reuse was confirmed by `vllm:prefix_cache_hits_total` moving rather than assumed. I compared logprobs,
> not only token ids, which is what makes the drift visible at all."

And their point 3 is exactly this port's situation:

> "It did not cross an argmax boundary here, so token ids matched in every run. That is the difference
> between this and the original report, and it is presumably why the effect looks intermittent and
> prompt dependent."

**[third-party measurement, reported in a first-party tracker]** -- I have not reproduced these numbers
and cannot vouch for the versions; the caveat the reporter attaches ("everything above is
`017e9f4448`, which postdates the 0.19.0 this was filed against") is theirs.

**vLLM, ROCm, closed as not planned.** [Issue #33123, "[Bug][ROCm]: Prefix caching produces different
output on first request (cache miss) vs subsequent requests (cache hit)"](https://github.com/vllm-project/vllm/issues/33123)
(MI355X/gfx950, Qwen3-0.6B, vLLM 0.14.0rc2.dev). The title is the port's question verbatim. The table:

| Request | Cache Status | Computation Path | Output |
|---|---|---|---|
| Run 1 | cache miss | Full prefill | `'The European Union consists of 27 member states'` |
| Run 2+ | cache hit | Partial prefill + cached KV | `'The European Union (EU) consists of **2'` |

The author's edit is worth quoting because it narrows the scope honestly: "this does not happen on
CUDA, even under the same attention backend." **[vendor-claimed]** -- and note the reporter's
correction: it is *not* a CUDA behaviour on that build.

**llama.cpp, in its own server documentation**
([`tools/server/README.md:587`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
`main`, fetched 2026-10-01):

> `cache_prompt`: Re-use KV cache from a previous request if possible. This way the common prefix does
> not have to be re-processed, only the suffix that differs between the requests. Because (depending on
> the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes
> (prompt processing vs. token generation) **enabling this option can cause nondeterministic results**.
> Default: `true`

**[vendor-claimed]** -- llama.cpp ships `cache_prompt` on by default and warns in the option's own
description that it can cause nondeterministic results. This is the closest analogue to what the port
does: an engine whose *default* configuration makes request 1 differ from request 2+.

**llama.cpp, seeded-determinism discussion.** [ggml-org/llama.cpp#8593, "Random seed possible
problems"](https://github.com/ggml-org/llama.cpp/issues/8593) (2024, stale). A user reports that an
explicitly seeded run reproduces, but a run whose seed was auto-assigned does not; the resolution is
that the printed seed and the interpreted seed differed. The maintainer explanation offered is
adjacent to this port's question: "if you use >1 slots or **prompt caching** on the server then the
input parameters can vary and thus the outputs will vary too." **[vendor-claimed]**, and the issue is
stale, so treat it as weak.

## 1.3 Does anyone seed generators per request identically?

Yes -- and this rules out one of the port's own candidate explanations by comparison.

**vLLM** creates a fresh per-request generator from the request's seed
([`vllm/v1/worker/gpu_model_runner.py:1272-1279`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py),
`main`, fetched 2026-10-01):

```python
if (
    sampling_params
    and sampling_params.sampling_type == SamplingType.RANDOM_SEED
):
    generator = torch.Generator(device=self.device)
    generator.manual_seed(sampling_params.seed)
else:
    generator = None
```

and the sampler threads those generators through
([`vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)):

```python
for i, generator in generators.items():
    q[i].exponential_(generator=generator)
```

**[read in source]** -- vLLM's per-request seeding is stateless with respect to request order: a fresh
`torch.Generator` seeded from `sampling_params.seed`. Identical seed and identical request order give
identical draws, whatever the server did before.

**SGLang** defaults to a fixed sampling seed and documents reproducibility as unconditional
([Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference)):
"By default, SGLang uses a sampling seed of 42 for reproducible sampling ... This will always produce
the same response across runs", and "The same seed will always produce the same response across
different runs." **[vendor-claimed]** -- and, read against SGLang's own FAQ, that promise holds for the
*seed*, not for the *logits*: the same seed draws from a distribution that the FAQ says moves.

**This port** uses a counter-based RNG keyed on `(seed, logical_positions[b], purpose)`
([`include/ninfer/ops/sampling.h:68-74`](../../include/ninfer/ops/sampling.h)):

> "Row b uses counter-based RNG key `(configs[b].seed,logical_positions[b],purpose)`, **without mutable
> RNG state or dependence on the compact row index**."

**[read in source]** -- and the kernel agrees
([`src/ops/kernel/sampling_device.cuh:92-101`](../../src/ops/kernel/sampling_device.cuh)):

```cpp
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                  int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key = sampling_splitmix64(key ^ (position * 0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(key ^ (purpose << 21) ^ (sub * 0x2545F4914F6CDD1Dull));
```

**[inferred from source]** -- because the draw is a pure function of `(seed, position, purpose)` and
carries no mutable state, this port has no RNG warm-up state that could behave differently on the
first request. A first-request difference therefore cannot come from an uninitialised or
differently-advanced generator; it has to come from the *inputs* to the draw, i.e. the logits or the
logical position. The `purpose` subkey split (`kSamplePurposePrefill` = 0, `kSamplePurposeDecode` = 1,
`kSamplePurposeSpeculativeAccept` = 2, `kSamplePurposeSpeculativeCorrection` = 3,
`kSamplePurposeSpeculativeBonus` = 4, `kSamplePurposeDFlash2Proposal` = 5,
[`include/ninfer/ops/sampling.h:14-21`](../../include/ninfer/ops/sampling.h)) is what keeps a prefill
draw from colliding with a decode draw at the same position -- which is precisely the mechanism that
would produce "same seed, different text" if the *purpose* or *position* differed between two
otherwise-identical requests. **[inferred]**

## 1.4 Lazy initialisation, JIT, cuBLAS workspace, autotuning

Searched, and the honest answer is that these are all documented as *latency* effects, not as
*output* effects.

- **JIT/warm-up raises first-request latency.** vllm-ascend [#7193, "[Bug]: High TTFT for the first
  request. Can vllm-serve warmup itself after startup?"](https://github.com/vllm-project/vllm-ascend/issues/7193)
  (open, 2026-03-12): "After bootstrapping the vllm serve ... it will do some compiling (using bisheng
  and clang-17) when receive the first request. That will obviously increase the TTFT." The proposed
  fix is an automatic post-startup warm-up. **[vendor-claimed]** -- and note the direction: first
  request **slower**.
- **SGLang documents first-use graph compilation as a latency spike**, including per-configuration
  first-use ("The first request using a new configuration (such as the first high-temp with top-k path,
  or the first batch size after scaling up load) may trigger graph recompilation") -- Intel Gaudi vLLM
  plugin warm-up docs. **[vendor-claimed]**
- **vLLM runs a startup profile/warm-up**: `vllm/v1/engine/core.py:364` logs "init engine (profile,
  create kv cache, warmup model) took ...". **[read in source]**
- **ONNX Runtime documents the same shape generically**: "the first inference run is expected to be a
  lot slower than the second run as the first run is where most CUDA memory allocations happen (this
  is costly) and cached in the memory pool for subsequent runs"
  ([microsoft/onnxruntime#19177](https://github.com/microsoft/onnxruntime/issues/19177)). **[vendor-claimed]**
- **cuBLAS's own reproducibility statement does not carve out a first call.** From the cuBLAS
  documentation, §2.1.4 *Results Reproducibility*
  ([docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/index.html), v13.4 page, fetched
  2026-10-01): "By design, all cuBLAS API routines from a given toolkit version, generate the same
  bit-wise results at every run when executed on GPUs with the same architecture and the same number of
  SMs." The stated exceptions are multiple concurrent streams, fixed-point emulation with insufficient
  workspace, and `cublasSetAtomicsMode()` -- **not** a first-call effect. **[vendor-claimed]**

  This matters for Question 1: it is evidence *against* a cuBLAS-workspace or heuristic-cache
  explanation, because NVIDIA's own guarantee is unconditional on call count, and because **this port
  does not call cuBLAS at all** -- it has its own Op kernels ([`AGENTS.md`](../../AGENTS.md): "Ops own
  closed mathematical and state-transition implementations"; "Do not introduce ... runtime weight
  repacking"). The lazy-init hypothesis has no cuBLAS surface to act through here.
- **Triton autotuning changing numerics was a real vLLM bug, and it was fixed.**
  [vllm-project/vllm#25194, "Bug: vLLM produces non-deterministic output due to Triton
  autotuner"](https://github.com/vllm-project/vllm/issues/25194) (2025-09-18, **closed as completed**,
  fixed by PR #25197). Nondeterminism at temperature 0 with no batching, no prefix caching and no
  chunked prefill, traced to the autotuner. **[vendor-claimed]**. This is the one documented case where
  a *first-use* mechanism (autotuning) changed *output*, and it was closed. It is also a poor analogue
  for this port: no Triton, no autotuner, and a persistent engine that does not JIT.

**[not found]** No first-party source -- docs, issue tracker, or release notes -- from vLLM, SGLang,
TensorRT-LLM, llama.cpp or NVIDIA describes a first request that is **faster** than later identical
requests. Every documented first-request effect I found increases latency (JIT compile, graph capture,
memory-pool population, cuBLAS/allocator warm-up) or changes output without changing speed. Searched:
vLLM issues for "first request", "warmup", "warm up", "first token differs"; SGLang issues likewise;
TensorRT-LLM issues for "first request", "lazy initialization", "warmup"; the CUDA and cuBLAS release
notes for first-call numerics; the NVIDIA developer forums' search API for `C2719`,
`C2719 grid_constant`, `CUtensorMap alignas MSVC`, `tensor descriptor MSVC C2719`,
`alignas 128 tensor map` (the last three rate-limited to HTTP 429 before returning, so that part of
the sweep is incomplete).

## 1.5 What the port's own upstream tracker says

This is the most directly relevant corpus, and it is where the sharpest in-tree corroboration lives.

**Issue #80, "Design: batch-invariant greedy decode (opt-in fixed decode width)"**
([Neroued/ninfer#80](https://github.com/Neroued/ninfer/issues/80), closed). The body is a map of *where
the token count `t` selects arithmetic* in this very engine:

> "Greedy decoding is not bitwise-reproducible across batch compositions. Every token-batched decode op
> selects its kernel -- and often its internal schedule -- from the step's token count
> `t = Σ_lanes (1 + accepted drafts)`. Variants differ in reduction order (split-K partition count,
> warp-level K partitioning, tile width), so **logits differ by sub-ulp amounts between compositions;
> on near-ties the argmax flips and long greedy continuations diverge**."

and it measures the class boundaries (`t=17` for W8 [1024x5120] and [6144x5120], "368 of 1024
elements, max |Δ| 2"). **[vendor-claimed]** -- filed by a third party against upstream and closed.

Two things follow, and they line up with the port's observation precisely:

1. The mechanism is **sub-ulp logit differences from a token-count-dependent kernel selection**. That is
   exactly the magnitude that leaves a greedy argmax intact while moving a multinomial draw.
2. `t = Σ_lanes (1 + accepted drafts)` -- so the *draft acceptance* feeds back into the arithmetic that
   produces the *next* acceptance. A first request whose acceptance is 64.6% against a warm 32.7% is
   running a different `t` sequence, and therefore a different kernel sequence, from request 1 onward.

**Issue #105, "Host checkpoint restore changes greedy output"**
([Neroued/ninfer#105](https://github.com/Neroued/ninfer/issues/105), closed as completed 2026-08-29).
A prefix-real test failed on a clean tree: `restored=64,1248, baseline=64,56127`. The maintainer's
closing comment:

> "Thanks for your report, this was not caused by prefix caching, but by **numerical variation from
> different prefill chunk splits**."

**[vendor-claimed]** -- and note what it rules out. This is upstream saying, about this engine, that
*different prefill chunk splits change the numerics*. A cold first request prefills the whole prompt in
`prefill_chunk`-sized pieces from `root`; a warm request replays a checkpoint and prefills only the
suffix ([`src/models/qwen3_5/program/transactions/commit.cpp:338-384`](../../src/models/qwen3_5/program/transactions/commit.cpp),
**[read in source]**, the `cursor`/`count` loop over `prefill_chunk`). Different splits, different
arithmetic, per the maintainer.

**Issue #322** ([Neroued/ninfer#322](https://github.com/Neroued/ninfer/issues/322), open) is a
third-party fork's report that upstream's prefix caching is "too complex and fragile", with an 80% TTFT
reduction and >90% hit rate claimed for a replacement. Not evidence about the transient, but it is
first-party evidence that **upstream's prefix-cache design is contested on its own tracker**. It also
notes the fork "includes changes allowing it to be built and run on Windows".

**[not found]** No issue in `Neroued/ninfer` mentions the first-request transient. Searches run against
the tracker with `--state all` for: `first request`, `warmup`, `seed determinism`, `deterministic seed`,
`sampler rng`, `output differs`, `reproducible`, `lazy init`, `first token differs`, `C2719`,
`alignas`, `MSVC TMA`, `tensor map`, `CUtensorMap`. The closest hits are #80 and #105 above, plus
#251 (prefix reuse stops once the checkpoint budget is exercised), #270 (`max_shared_prefixes` smaller
than one request's own ceiling, "silently killing prefix-cache hits at low concurrency") and #333
(cudaErrorLaunchTimeout during DFlash2 prefill immediately after ~98%-cache-replay prefill).

The port's own documentation does not record a determinism contract for this. `docs/serving.md` has no
"Determinism" heading; a case-insensitive search of `docs/` for `determinis|bitwise|bit-for-bit` returns
only unrelated hits. **[measured here]** -- grep over `docs/serving.md` for `Determinism`: no matches;
for `(?i)determin|greedy`: four matches, of which one is the `--greedy` flag row and one is the word
"deterministic" in a prefix-reuse sentence. Upstream's `docs/serving.md` likewise has no Determinism
section (its headings are listed in full in the search above). **[measured here]**

## 1.6 Speculative decoding on an empty prefix

**[not found]** as a documented engine-wide effect. Searches of the vLLM, SGLang and TensorRT-LLM
trackers and docs for speculative decoding behaving differently on the first step or on an empty prefix
returned no first-party report of that shape. What the literature and vendor docs do contain:

- The mechanics of the first forward pass are described consistently -- the drafter conditions on the
  verified prefix, and at `r = 0` "immediate rejection, where no draft token is accepted and ŷ is
  defined as an empty prefix" ([FlexDraft, arXiv:2605.20022](https://arxiv.org/pdf/2605.20022)) --
  but as algorithm description, not as an observed first-request anomaly. **[third-party]**
- vLLM's own speculative-decoding docs list the generic numerical caveats ("Floating-Point Precision:
  Differences in hardware numerical precision may lead to slight discrepancies in the output
  distribution"; "Batch Size and Numerical Stability") without a first-request claim.
  **[vendor-claimed]**

**[inferred]** The port's `t`-feedback loop (1.5) is the part that is actually load-bearing here. A
speculative round's token count is `1 + accepted drafts`, and #80 establishes that `t` selects the
kernel and therefore the reduction order. The port's own measurement -- first request accepting *better*
-- is consistent with the first request running a `t` sequence that lands in a different arithmetic
class, and with that arithmetic difference being what changes the sampled tokens. This is an inference
from two in-tree facts, not a measurement, and it is stated here as the shape of a candidate
explanation rather than as a finding.

## 1.7 Verdict on Question 1

| Sub-question | Answer | Basis |
|---|---|---|
| Is "request 1 differs from request 2+ at temperature 0" a known engine-wide behaviour? | **Yes.** Named in SGLang's FAQ, filed open in vLLM (#40896, "we haven't fully supported determinism with prefix caching"), filed and closed in vLLM for ROCm (#33123), and warned about in llama.cpp's own `--cache-prompt` documentation | vendor-claimed, four independent first-party sources |
| Is the cause identified upstream? | **Partly.** "Cold prefill and cached-KV replay dispatch different kernels and are not bit-identical" is the stated mechanism in SGLang's FAQ and is consistent with the vLLM #40896 thread. No upstream project has fixed it; vLLM's determinism roadmap (#27433) still lists "Prefix caching support" as an unchecked nice-to-have | vendor-claimed |
| Does anyone seed per-request identically? | **Yes** -- vLLM builds a fresh `torch.Generator` per request from `sampling_params.seed`; SGLang defaults to seed 42 and promises reproducibility for the draw. Neither promise covers the logits | read in source + vendor-claimed |
| Is there a documented first-request effect that changes *output* through lazy init / JIT / cuBLAS / autotuning? | **One closed case** (vLLM #25194, Triton autotuner, temperature 0, fixed by PR #25197). Everything else documented is latency-only. cuBLAS's bit-wise guarantee is unconditional on call count, and this port does not call cuBLAS | vendor-claimed + measured here (no cuBLAS in the tree) |
| Is a first request **faster** than later ones documented anywhere? | **No.** Every first-request effect found increases latency. This port's direction is the opposite | not found, with the search list in 1.4 |
| Is this port's behaviour specific to it? | **The output-difference half is not.** The speed half is. Nothing in the four projects' trackers describes a first request that outruns its successors | not found |

**What this changes about whether it needs fixing.** The output-difference half is not a defect of this
port: it is the documented, currently-unsupported behaviour of every comparable engine, arising from
prefix reuse, and its magnitude is a sub-argmax logit perturbation that only a sampling temperature can
expose. That is context for how it is treated, not a decision -- the port's own contract is still its
own, and no determinism contract was found in `docs/serving.md` either here or upstream. The
**faster** half is the part with no precedent: nothing in any of the four projects' trackers or docs
describes it, and the usual warm-up mechanisms all push the other way.

---

# Question 2 -- the MSVC C2719 TMA-descriptor defect

## 2.1 Is it a known, reported issue?

**Not in CUTLASS, FlashAttention, or any NVIDIA tracker I could reach.** **[not found]**, with the
search recorded:

- CUTLASS issues, `--state all`: `C2719` (0 results); `TENSOR_MAP_ALIGN` (0); `alignas(128)
  grid_constant` (0); `MSVC tensor map descriptor` and `alignas TMA MSVC` both return the same single
  issue, #2906, which is about a *device-side* misaligned shared address, not a host compile error.
- GitHub code search, `alignas(128) TmaDescriptor repo:NVIDIA/cutlass`: **0 results**.
- GitHub code search, `TENSOR_MAP_ALIGN repo:NVIDIA/cutlass`: **0 results**.
- GitHub code search across all of GitHub for `TENSOR_MAP_ALIGN`: 13 hits, none of them NVIDIA's
  `cuda.h`, and none in CUTLASS or FlashAttention. (Listed in 2.6.)
- `Neroued/ninfer` issues, `--state all`: `C2719` (0), `alignas` (0), `CUtensorMap` (0).
- NVIDIA developer forums, search API: `"C2719"` returns 3 topics, none about tensor maps
  (`float4 alignment inconsistency`, `Issues compiling Win-32 project with CUDA 5.5 64 bit`,
  `Alignment of builtin vector types .cu and .cpp`). The four follow-up queries
  (`C2719 grid_constant`, `CUtensorMap alignas MSVC`, `tensor descriptor MSVC C2719`,
  `alignas 128 tensor map`) returned 0, 0, HTTP 429, HTTP 429 -- so **the forum sweep is partial**,
  and I am not claiming the forums are clean.

**GitHub-wide issue search** for `C2719 alignas tensor map` returns exactly **2** results, both
third-party Windows ports, and both corroborate the defect:

- [SystemPanic/vllm-windows#41, "[Installation]: compile error C2719: 'unnamed-parameter': formal
  parameter with requested alignment of 128 won't be aligned"](https://github.com/SystemPanic/vllm-windows/issues/41)
  (opened 2026-03-04, closed 2026-03-12). CUDA 13.1, torch 2.10.0+cu130, RTX 5070 Ti, vLLM on Windows,
  building CUTLASS-derived sources. **[third-party]**
- [super3/llmjob#242, "Release v0.5.9: fix Blackwell mining on Windows; CUDA 13 core for RTX 50-series
  on driver 580+"](https://github.com/super3/llmjob/issues/242). This one is the most informative
  third-party account I found, and it reaches the *same* mechanism from the other direction:

  > "**The cause.** `cuda.h` aligns `CUtensorMap` (the TMA descriptor type) only under
  > `#if __cplusplus >= 201103L`. MSVC reports `199711L` unless given `/Zc:__cplusplus`, and our Windows
  > build doesn't pass it. So on Windows the sm_120 tall fold's two TMA descriptors sat at param offsets
  > `0x98`/`0x118`, 8-byte aligned. Linux has them at `0x100`/`0x180`. The first
  > `cp.async.bulk.tensor` then faults."

  and, on the fix:

  > "The descriptor is now wrapped in our own `PearlTensorMap`, which is `alignas(128)` in the **device**
  > pass. `alignas` is a keyword, not a macro, so MSVC honours it. The device pass is what lays out the
  > kernel parameters, and the launch copies to those offsets. **The host pass sees a plain 128-byte
  > struct.** nvcc's host stub takes kernel parameters by value, and MSVC refuses an over-aligned
  > by-value parameter (C2719). The first attempt hit exactly that, and CI caught it."

  **[third-party]**, and note the shape of their solution: they *need* the device-side 128 alignment,
  and they get it by making the type a plain 128-byte struct to the host. This is a different trade from
  this port's, because they hit the alignment requirement at runtime while this port does not.

**ONNX Runtime, independently, with the same diagnosis.** Their `main` branch carries both the original
commit's description and a refined workaround.

`cmake/onnxruntime_providers_cuda.cmake`, `main`, **[read in source]**:

```cmake
if(NOT MSVC)
  # The native SM90 (Hopper) TMA/WGMMA launchers pass CUTLASS TMA descriptor types through
  # NVCC-generated host stubs. With CUDA 13 + MSVC those stubs contain 128-byte over-aligned
  # by-value formal parameters, which triggers MSVC C2719 ("formal parameter with requested
  # alignment of 128 won't be aligned"). Disable the native SM90 fpA_intB (COMPILE_HOPPER_TMA_GEMMS)
  # and grouped MoE (COMPILE_HOPPER_TMA_GROUPED_GEMMS) TMA kernels on MSVC; ...
  target_compile_definitions(${target} PRIVATE COMPILE_HOPPER_TMA_GEMMS)
  target_compile_definitions(${target} PRIVATE COMPILE_HOPPER_TMA_GROUPED_GEMMS)
endif()
```

and, further down, a **second and more interesting workaround**:

```cmake
# CUDA 13 gives CUtensorMap 128-byte host alignment when MSVC reports an accurate
# __cplusplus value. The target-specific host flag below avoids C2719 for FP4 QMoE.
if(onnxruntime_cuda_sm120_tma_srcs AND
   (NOT MSVC OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0 OR onnxruntime_USE_FP4_QMOE))
  ...
  if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    # Keep CUDA's CUtensorMap payload unchanged while avoiding an
    # alignas(128) by-value parameter that MSVC cannot represent in
    # NVCC-generated host stubs (C2719).
    target_compile_options(onnxruntime_providers_cuda_sm120_tma PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus->")
  endif()
```

**[read in source]** -- `/Zc:__cplusplus->` (note the trailing minus: *disable* the accurate
`__cplusplus`) makes MSVC report `199711L`, which takes `cuda.h`'s `#if __cplusplus >= 201103L` branch
as false, which removes the `alignas` from `CUtensorMap` entirely, which removes C2719. This is a
**build-flag-only** workaround that does not touch a line of kernel source. It is the only portable fix
I found anywhere, and it is third-party rather than NVIDIA-sanctioned.

The original commit is
[microsoft/onnxruntime@291311e7, "Fix CUDA 13 nuget packaging pipeline failures and enable CUDA 13.3
builds (#28736)"](https://github.com/microsoft/onnxruntime/commit/291311e7d8d4ac17672e6bdaffaa4a8539115e87),
whose message says:

> "**CUDA 13.2/13.3 compiler incompatibilities**: NVCC 13.x generates host stubs that pass TMA descriptor
> parameters (with `alignas(128)`) by value. MSVC rejects this with error C2719."

**[vendor-claimed by a third party]** -- quoted from the commit message, as the existing
`docs/research/grid-constant-tma-descriptor-msvc.md` already records.

## 2.2 Is the reintroduction risk live? -- Yes, measured today

`git grep -n "alignas(128)" upstream/dev -- src` on `upstream/dev` at **`75a89050`** (fetched
2026-10-01) returns six hits, all still present:

```
upstream/dev:src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14:struct alignas(128) Bf16TmaDescriptors {
upstream/dev:src/ops/linear/fp8/fp8_a8_tma_mma.cuh:19:struct alignas(128) Fp8TmaDescriptors {
upstream/dev:src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24:struct alignas(128) Nvfp4A4TmaDescriptors {
upstream/dev:src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:85:    alignas(128)
upstream/dev:src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:87:    alignas(128)
upstream/dev:src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:96:union alignas(128) Nvfp4A4TmaScratch {
```

`upstream/master` returns the identical six. **[measured here]**

Only the first three are the C2719 sites. Lines 85, 87 and 96 are `__shared__` staging buffers inside
`Nvfp4A4TmaTensorStorage` and `Nvfp4A4TmaScratch`, which live in dynamic shared memory and are
reinterpret_cast from an `extern __shared__ __align__(128)` block
([`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:86-116`](../../src/ops/linear/nvfp4/nvfp4_a4_tma.cuh),
**[read in source]**) -- they are not kernel parameters and cannot trigger C2719. A merge that
reintroduces the defect reintroduces exactly those three struct declarations.

The most recent upstream commit touching `nvfp4_a4_tma.cuh` is `fc3993d8` ("perf(ops): unify nvfp4
templates and add a16 mma routes"); `bf16_a16_tma_mma.cuh`'s is `ecbc3357`. Neither mentions alignment.
**[measured here]**

## 2.3 Does upstream CUTLASS or FlashAttention still carry the attribute?

**No -- neither does.** This corrects the premise in the brief, and it is the most consequential finding
in Question 2.

**CUTLASS `main`.** `include/cute/arch/copy_sm90_desc.hpp` lines 290-296, **[read in source]**, fetched
2026-10-01:

```c++
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
```

Two properties matter. CUTLASS **aliases the descriptor to `CUtensorMap` outright**, inheriting
`TENSOR_MAP_ALIGN` and therefore the per-compiler 64/128 decision; and its own fallback type for a
128-byte descriptor payload is `alignas(64)`, not `alignas(128)`. NVIDIA's own library never asks a
user-declared type for 128.

I checked whether that changed recently by fetching the same file at five tags
(`v3.8.0`, `v3.9.2`, `v4.0.0`, `v4.2.0`, `v4.3.0`) plus `main`. Every one returns
`TmaDescriptor = CUtensorMap | struct alignas(64) { char bytes[128]; }` -- identical across the whole
range. **[measured here]** There has never been an `alignas(128)` on CUTLASS's TMA descriptor type.

The remaining `alignas(128)` occurrences in CUTLASS `main` -- 23 files per GitHub code search -- are all
on **shared-memory or workspace** types, not on kernel parameters. Verified by reading the four files
that matter:

| file | what carries `alignas(128)` | kernel parameter? |
|---|---|---|
| `include/cute/atom/copy_traits_sm90_tma.hpp` | 0 occurrences of `alignas` in 67,768 bytes; `TmaDescriptor tma_desc_;` is a plain member of `Copy_Traits` | no |
| `include/cute/arch/copy_sm90_tma.hpp` | 0 occurrences in 57,107 bytes | no |
| `include/cutlass/gemm/collective/builders/sm90_common.inl` | 0 occurrences in 20,335 bytes | no |
| `include/cute/atom/copy_traits_sm100_tma.hpp` | 0 occurrences in 33,826 bytes | no |
| `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp` | 0 occurrences in 4,289 bytes | no |
| `include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp` | `alignas(128) cute::ArrayEngine<...> smem_A/B/E;` inside `SharedStorage` | no -- shared memory |
| `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp` | `struct TensorMapStorage : cute::aligned_struct<128,_1>` and `alignas(128) EpilogueTensorMapStorage epilogue;` inside `SharedStorage` | no -- and `SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);` at line 574 confirms it is dynamic shared memory reached by pointer |
| `include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp` | same `TensorMapStorage` shape | no |

The one `alignas(128)` in `include/cute/container/alignment.hpp` is the `#define CUTE_ALIGNAS(n) alignas(n)`
macro definition. **[read in source]**

And the aggregate still travels by value under `CUTLASS_GRID_CONSTANT`
([`include/cutlass/device_kernel.h`](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/device_kernel.h)):

```c++
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
```

**[read in source]** -- the by-value `__grid_constant__` shape, with no `alignas` added. So CUTLASS's
answer to this exact problem is: inherit the compiler's alignment, pass by value, never name 128 in user
code. Which is what this port does.

**FlashAttention `main`.** Checked the three files that matter, all fetched 2026-10-01:

| file | size | `alignas` | `__grid_constant__` |
|---|---|---|---|
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | 106,810 B | **0** | 0 |
| `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` | 66,268 B | **0** | 0 |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | 76,953 B | **0** | 0 |

**[read in source]** The only match in the fwd mainloop is a comment at line 570, "/// Issue Tma
Descriptor Prefetch". FlashAttention's descriptors are `TMA_Q`/`TMA_K`/`TMA_V` types produced by
`make_tma_copy_A_sm90` / `make_tma_copy_B_sm90` (lines 248-262), i.e. CUTLASS `Copy_Traits`, and they
travel inside `struct Params` (line 435-440) reached through `cutlass::device_kernel`, which
`hopper/flash_fwd_launch_template.h` includes by name with the comment `#include "cutlass/device_kernel.h"  // For device_kernel`.
**[read in source]** -- so FlashAttention inherits CUTLASS's alias and adds nothing.

**Conclusion: the attribute is not "still carried" by CUTLASS or FlashAttention.** The three structs that
carry it are this project's own, in `Neroued/ninfer`. The reintroduction risk is entirely a
merge-from-upstream risk, not a vendored-library risk.

## 2.4 The mechanism, measured on this machine

Everything above is documentary. This section is **[measured here]**, on this port's toolchain:
CUDA 13.3 (`V13.3.73`), MSVC 14.51.36231 (Visual Studio 18 Community, `vcvars64.bat`), `sm_120a`.

**(a) The threshold is exactly 64, and 128 is a hard reject.** Sweeping `alignas(N)` on a by-value
`__grid_constant__` aggregate of two `CUtensorMap`, launched from host so the stub is generated:

| `alignas(N)` | `alignof(Wrapper)` | result |
|---|---|---|
| 0 (none) | 8 | accepted |
| 8 | 8 | accepted |
| 16 | 16 | accepted |
| 32 | 32 | accepted |
| 64 | 64 | accepted, `no error` |
| **128** | 128 | **`error C2719`, four sites** |
| **256** | 256 | **`error C2719`, four sites** |

The four C2719 sites for `N=128`, verbatim:

```
tma_align_sweep.cu(19): error C2719: 'm': formal parameter with requested alignment of 128 won't be aligned
...\tma_align_sweep.compute_120a.cudafe1.stub.c(17): error C2719: 'unnamed-parameter': formal parameter with requested alignment of 128 won't be aligned
...\tma_align_sweep.compute_120a.cudafe1.stub.c(19): error C2719: '__cuda_0': formal parameter with requested alignment of 128 won't be aligned
...\tma_align_sweep.compute_120a.cudafe1.stub.c(6): error C2719: 'unnamed-parameter': formal parameter with requested alignment of 128 won't be aligned
```

Note *where* it lands: `cudafe1.stub.c` -- nvcc's generated **host** file, handed to `cl`. The device
compilation succeeded.

**(b) `cuda.h` already encodes the MSVC answer, and it is not being used here.** `cuda.h:3747-3758` in the
installed 13.3:

```c
#if defined(_MSC_VER)
  #define TENSOR_MAP_ALIGN 64
#else
  #define TENSOR_MAP_ALIGN 128
#endif
```

**[measured here]**, read from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\cuda.h`.

But a probe compiled with **this port's flags** prints:

```
__cplusplus=199711 TENSOR_MAP_ALIGN=64 alignof(CUtensorMap)=8 sizeof=128
```

`TENSOR_MAP_ALIGN` is 64 and `alignof(CUtensorMap)` is **8**. The header's own alignas is *not* being
applied, because `cuda.h` guards it:

```c
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    alignas(TENSOR_MAP_ALIGN)
#elif __STDC_VERSION__ >= 201112L
    _Alignas(TENSOR_MAP_ALIGN)
#endif
```

and nvcc's host pass on MSVC reports `__cplusplus = 199711L` unless `/Zc:__cplusplus` is passed. This
port does **not** pass it: `CMakeLists.txt:33-40` sets `/Zc:preprocessor`, `/FS`, `/utf-8` and nothing
else. **[measured here]**, grep for `__cplusplus|Zc:` over `CMakeLists.txt` and `cmake/` returns only
the three `/Zc:preprocessor` lines.

With `-Xcompiler=/Zc:__cplusplus` added:

```
__cplusplus=201703 TENSOR_MAP_ALIGN=64 alignof(CUtensorMap)=64 sizeof=128
alignof(FourMaps)=64 offsetof(d)=384
```

**[measured here]** So the `_MSC_VER → 64` branch, which NVIDIA wrote for exactly this compiler, is
**dead code on this port's build**, and the port's descriptors are laid out at alignment 8.

**(c) What that means for the port's shipped kernel.** The port's NVFP4 W4A4 kernel takes the
descriptors as its first parameter
([`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:145-150`](../../src/ops/linear/nvfp4/nvfp4_a4_tma.cuh)):

```cpp
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a4_tma_kernel(
    const __grid_constant__ Nvfp4A4TmaDescriptors descriptors, float alpha,
```

so the four descriptors occupy parameter offsets 0, 128, 256, 384 -- every one of them 16-byte aligned
regardless of the type's 8-byte alignment, because each is 128 bytes and they start at offset 0. A
runtime probe confirms it:

```
PortShape: alignof=8 off(a)=0 off(b)=128 off(d)=384
port  &d=0000000F00610500  &b=0000000F00610400
```

**[measured here]** -- `&b` at `...400` and `&d` at `...500`, both 256-byte aligned in practice. The
`cuobjdump -ptx` of the port's actual built object confirms the parameter is declared
`.param .align 8 .b8 ..._param_0[512]` -- alignment 8, size 512, at offset 0.

So the port is correct **by layout accident, not by construction**: it works because the descriptors
happen to be first and happen to be 128 bytes each. `llmjob`'s failure was the same type at offsets
`0x98`/`0x118` -- 8-mod-16, because 20 bytes of scalars preceded them. A runtime probe reproduces that
arithmetic exactly:

```
ScalarFirst: alignof=8 off(a)=24 off(b)=152  (a%16=8 b%16=8)
scalar &b=0000000F00610418  &a=0000000F00610398
```

**[measured here]** -- `&a` at `...398` and `&b` at `...418`, both 8-mod-16. That is `llmjob`'s
`0x98`/`0x118` shape, reproduced.

**[measured here]** And the PTX `.param .align 8` is not cosmetic: with `/Zc:__cplusplus` the same
by-value parameter compiles as `.param .align 128`, and with an explicit `alignas(128)` wrapper as
`.param .align 128` too. The declared alignment tracks `alignof`, which tracks `__cplusplus`.

This is a **finding about the port's current state, not a defect report**: the shipped W4A4 route lays
its descriptors out at 16-byte-aligned offsets and runs, which is why `1218d574`'s test passes and the
engine serves. But it does mean the port's stated reason for its own fix is not the operative reason.
`docs/research/grid-constant-tma-descriptor-msvc.md:99-106` and the comment at
`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14-19` both say the wrapper "inherits
`alignas(TENSOR_MAP_ALIGN)`, which is 64 under MSVC". **[measured here]** On this build it inherits
**8**, because `__cplusplus` is 199711 and the header's `alignas` never fires. The comment describes a
mechanism that is not executing.

**(d) The `_MSC_VER → 64` branch is recent.** Two independent third-party sources date it:

- CUTLASS [#2906](https://github.com/NVIDIA/cutlass/issues/2906), comment from `lsyyy666`, 2026-01-08:
  "CUDA toolkit 13.1 is automatically aligning it to 128B which has known issues with MSVC. CUDA
  toolkit 13.2 will revert the alignment of tensor descriptors back to 64B by default. Before 13.2, you
  can workaround this issue with `alignas(64)`." A CUTLASS contributor (`alexngUNC`, 2026-02) replied:
  "I still see that the Tensor map descriptor alignment is 128B in CUDA 13.2 and 13.3, which still
  causes issues with MSVC (and therefore PyTorch on Windows). Are there still plans to change it to
  64B?" **[third-party, and the two comments disagree]**
- [SystemPanic/vllm-windows README](https://github.com/SystemPanic/vllm-windows/blob/main/README.md):
  "IMPORTANT FOR CUDA 13.0 TO CUDA 13.2 BUILDS: CUDA 13.0 to CUDA 13.2 cuda.h currently has 128 byte
  alignment. MSVC does not support yet passing over-aligned types like alignas(128) by value as function
  parameters. **CUDA 13.3 will revert back to 64 byte alignment**, but if you have installed a CUDA 13
  version before 13.3, you need to patch it." **[third-party]**

The installed 13.3 header on this machine has the `_MSC_VER` branch, which is consistent with "13.3
reverted it" and inconsistent with `alexngUNC`'s "still 128B in 13.3". I did not install 13.1 or 13.2
to check their headers, so I report the disagreement rather than resolve it. **[measured here]** for
13.3 only.

**NVIDIA release notes: nothing.** I searched the CUDA Toolkit release notes for 13.0.0, 13.1.0, 13.1.2,
13.2.0, 13.3.0, 12.6.0, 12.8.0 and 12.9.0 for `tensor map`, `TENSOR_MAP`, `alignas`, `over-aligned`,
`C2719`, `descriptor align`. Every one returns 0 hits except **CUDA 13.0.0**, which carries a *related
but distinct* resolved compiler issue:

> "Under certain scenarios involving an over-aligned type and the MSVC `#pragma pack` directive, NVCC
> incorrectly computed the size, alignment, and other layout characteristics of the type. This issue can
> occur when using an over-aligned type as the embedded object of `std::optional<T>` from the MSVC STL.
> **It only affects Windows when NVCC is used with an MSVC host compiler.** ... The issue was
> introduced in CUDA 12.9. It is fixed in CUDA 13.0."

**[vendor-claimed]** -- an nvcc/MSVC over-aligned-type layout bug, on Windows, fixed in 13.0. It is not
this defect (different trigger, different symptom), but it establishes that over-aligned types crossing
the nvcc/MSVC boundary are a known problem class NVIDIA has fixed bugs in. No release note in 13.1, 13.2
or 13.3 mentions `TENSOR_MAP_ALIGN` or C2719 at all.

## 2.5 Is there a portable upstream-acceptable fix?

**In NVIDIA's own code: the header's `_MSC_VER` branch, and nothing else.** The
[CUDA Driver API reference for `CUtensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html)
states the requirement as a compiler requirement -- "Tensor map descriptor. **Requires compiler support
for aligning to 128 bytes**" -- and the [Tensor Map Object
page](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) states the
*runtime* requirement as the weaker one: "tensorMap address must be aligned to **64 bytes**."

**[vendor-claimed]** Those two sentences together are the whole story: the type wants 128, the address
needs 64, and MSVC's x64 convention caps a by-value parameter at 16. The Programming Guide's
recommended form is by-value `__grid_constant__` (§4.12.2.2, "Using TMA to transfer multi-dimensional
arrays"), so the recommended form and the 128-byte type are in tension on MSVC by construction, and
NVIDIA resolves it in the header.

**Does upstream still carry the attribute?** Yes -- in `Neroued/ninfer`, on all three structs, on both
`dev` and `master`, today (§2.2). No, in CUTLASS and FlashAttention, at every tag I checked (§2.3).

**Would an upstream fix make the local divergence unnecessary?** No such fix exists. Of the four
candidate resolutions found:

| resolution | who | upstream-acceptable? | keeps graph-capture safety? |
|---|---|---|---|
| remove the `alignas(128)` override, inherit `TENSOR_MAP_ALIGN` | this port (`1218d574`) | **Yes** -- matches CUTLASS's own `using TmaDescriptor = CUtensorMap;` | Yes -- bytes in the kernel node |
| `-Xcompiler /Zc:__cplusplus->` (build flag only) | ONNX Runtime `main` | **Yes**, and requires no source change | Yes |
| pass by pointer via a device buffer | ONNX Runtime (original), upstream PR #233 | Yes, but this port measured it as graph-capture-unsafe | **No** -- `1218d574`'s whole finding |
| wrapper that is `alignas(128)` only in the device pass | llmjob | yes, but only where runtime alignment is actually required | Yes |

The last row is interesting and is the one closest to "the portable fix": **`alignas` is a keyword, not
a macro, so MSVC honours it in the device pass and nvcc's host stub is what breaks.** llmjob's fix is to
make the type a plain 128-byte struct as far as the host is concerned while keeping the device-side
alignment. No NVIDIA or CUTLASS source does this, and I did not find it proposed anywhere else.

## 2.6 What I searched for Question 2, exhaustively

Primary sources read directly:

- `upstream/dev` (`75a89050`) and `upstream/master`: `git grep alignas(128)`, `git log` on the three
  files.
- CUTLASS `main` via `raw.githubusercontent.com`: `include/cute/arch/copy_sm90_desc.hpp`,
  `copy_sm90_tma.hpp`, `copy_sm100_tma.hpp`, `copy_traits_sm90_tma.hpp`, `copy_traits_sm100_tma.hpp`,
  `copy_traits_sm90_tma_swizzle.hpp`, `copy_traits_sm90_im2col.hpp`,
  `include/cute/container/alignment.hpp`, `include/cutlass/device_kernel.h`,
  `include/cutlass/gemm/collective/builders/sm90_common.inl`,
  `include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp`,
  `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp`,
  `sm103_blockscaled_gemm_array_tma_warpspecialized.hpp`,
  `include/cute/tutorial/hopper/wgmma_tma_sm90.cu` and the four Blackwell CuTe tutorials,
  `examples/93_blackwell_low_latency_gqa/tgv_gqa.cuh`. Plus the full 8,552-entry `main` file tree, to
  enumerate every TMA-related header.
- CUTLASS tags `v3.8.0`, `v3.9.2`, `v4.0.0`, `v4.2.0`, `v4.3.0` for `TmaDescriptor`.
- FlashAttention `main`: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`,
  `mainloop_bwd_sm90_tma_gmma_ws.hpp`, `csrc/flash_attn/src/flash_fwd_kernel.h`,
  `hopper/flash_fwd_kernel_sm90.h`, `hopper/flash_fwd_launch_template.h`, plus the full `hopper/`
  directory listing (64 files).
- This port's `CMakeLists.txt`, `cmake/`, and the three descriptor declarations, plus `cuobjdump
  -ptx` and `cuobjdump -res-usage` on the built `nvfp4_w4a4_tma.cu.obj` in `build/`.
- The installed `cuda.h` at `v13.3\include\cuda.h:3747-3758`.

Trackers and docs: CUTLASS issues (`C2719`, `TENSOR_MAP_ALIGN`, `alignas(128) grid_constant`,
`alignas TMA MSVC`, `MSVC tensor map descriptor`, all `--state all`); `Neroued/ninfer` issues and PRs
(§1.5, plus PR #233 "Windows: native MSVC build with vcpkg-managed dependencies", open, whose body
says "MSVC rejects over-aligned (alignas 128) kernel parameters (**C2711**)" -- the wrong error number,
and whose fix is the device-pointer form this port rejected); GitHub code search for `alignas(128)
repo:NVIDIA/cutlass` (23 files, all classified in §2.3), `alignas(128) TmaDescriptor repo:NVIDIA/cutlass`
(0), `TENSOR_MAP_ALIGN repo:NVIDIA/cutlass` (0), `alignas(128) path:include/cute` (1),
`TENSOR_MAP_ALIGN` globally (13 hits: `NVIDIA/cudnn-frontend`, `NVIDIA/CompileIQ`,
`headpiece747/ninfer-5090-windows`'s own `docs/upstream-reports/`, and five unrelated repos); GitHub
issue search for `C2719 alignas tensor map` (2 hits, both in §2.1); CUDA Toolkit release notes for
12.6/12.8/12.9/13.0/13.1.0/13.1.2/13.2/13.3 (§2.4); the NVIDIA developer forums search API (partial, as
noted). ONNX Runtime's `main` CMake and commit `291311e7`.

## 2.7 Verdict on Question 2

| Question | Answer | Basis |
|---|---|---|
| Is the C2719 TMA-descriptor defect a known CUTLASS / FlashAttention / NVIDIA issue? | **No.** Not in CUTLASS's tracker, not in FlashAttention's, not found on the NVIDIA forums (partial sweep). It appears only in third-party Windows ports: ONNX Runtime, SystemPanic/vllm-windows, super3/llmjob -- all three with the same diagnosis | not found upstream; three third-party corroborations |
| Does upstream still carry `alignas(128)`? | **`Neroued/ninfer`: yes**, all three descriptor structs, on `dev` and `master`, at `75a89050`. **CUTLASS: no**, at every tag from `v3.8.0` to `main`. **FlashAttention: no**, zero `alignas` in any TMA file | measured here, today |
| Is a portable upstream-acceptable fix available? | **The header's own `_MSC_VER → TENSOR_MAP_ALIGN 64` branch**, which is what CUTLASS relies on by aliasing `TmaDescriptor = CUtensorMap`. Plus one third-party build-flag workaround (`/Zc:__cplusplus->`, ONNX Runtime). No source-level fix has landed anywhere | read in source |
| Would an upstream fix make the local divergence unnecessary? | **No upstream fix exists to wait for.** And the divergence cannot be adopted upstream as-is, because upstream's own Windows PR #233 takes the device-pointer route that `1218d574` measured as graph-capture-unsafe | measured here + read in source |
| Is the reintroduction risk live? | **Yes.** A merge from `upstream/dev` reintroduces three `alignas(128)` declarations, each of which is an unrecoverable C2719 on this toolchain | measured here, reproduced four times |

**One finding that bears on the port's own record rather than on upstream.** The port's comments and
research note justify the fix as "inheriting `TENSOR_MAP_ALIGN`, which is 64 under MSVC"
([`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14-19`](../../src/ops/linear/bf16/bf16_a16_tma_mma.cuh),
[`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24-26`](../../src/ops/linear/nvfp4/nvfp4_a4_tma.cuh),
`docs/research/grid-constant-tma-descriptor-msvc.md:99-112`). **[measured here]** On this port's build
flags the wrapper inherits alignment **8**, not 64, because nvcc's host pass reports
`__cplusplus = 199711L` without `/Zc:__cplusplus` and `cuda.h`'s `alignas` is inside a
`__cplusplus >= 201103L` guard. The shipped W4A4 route is nonetheless correctly laid out, because its
descriptors are the first parameter and each is 128 bytes -- offsets 0/128/256/384, all 16-byte aligned.
That is a correct result reached for a different reason than the one recorded, and the reason recorded
would not survive a signature change that put a scalar before the descriptors. `llmjob`'s `0x98`/`0x118`
failure is that exact case, reproduced here at offsets 24 and 152.

---

# Provenance

Sources are primary throughout: the two projects' own source and issue trackers, the two NVIDIA
libraries' source at named refs, Microsoft's and NVIDIA's documentation, and local compilation on this
port's toolchain. Two figures in §1.2 are third-party measurements reported inside first-party trackers;
they are labelled as such and were not reproduced here. The NVIDIA developer forums sweep is partial
(two of six queries rate-limited), and no claim in this note rests on it.

Existing in-tree research this note extends rather than repeats:
`docs/research/grid-constant-tma-descriptor-msvc.md` (the C2719 mechanism and the by-value
`__grid_constant__` recommendation), `docs/research/windows-port-deferred-items.md` (which records the
`alignas` fix as "the recommended fix, not a workaround"), and `docs/active-work.md` item 14 (the
first-request transient itself).
