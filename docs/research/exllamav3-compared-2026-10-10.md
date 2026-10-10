# exllamav3, read against this port (2026-10-10)

Read because [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) is the closest thing in the
field to this project's stated goal -- maximum single-GPU inference on consumer hardware -- and the useful
question is not "which is better" but "which decisions differ, and which of theirs would transfer".

Every claim about exllamav3 below is read from its repository on 2026-10-10 (README, file tree, GitHub API
metadata); **no head-to-head measurement was taken**, so nothing here is a speed comparison. Their repository
at that date: 1,634 stars, 233 forks, 98 open issues, MIT, created 2025-04-06, last push 2026-10-10T10:54Z.

## The two shapes

| | exllamav3 | this port |
|---|---|---|
| Product | a general library: any HF model, many quantization formats, an ecosystem of prequantized models | one model family (Qwen3.8-27B), five artifacts, eight measured launchers |
| Weight formats | EXL3 (a QTIP-derived trellis format, 2-8 bpw) plus 2-8 bit cache quantization | NVFP4/FP8 recipes per artifact; engine kernels for bf16, fp8, nvfp4 and q4/q5/q6/q8 weights |
| Stack | Python 4.28 MB, CUDA 1.69 MB, C++ 0.59 MB by bytes; attention, cache and recurrent kernels are Triton; the extension compiles at first import, and the prebuilt wheels are `linux_x86_64` | C++/CUDA only: 225 `.cu`, 228 `.cpp`, 292 headers under `src/`; MSVC + CUDA 13.3; no Python in the inference path |
| Parallelism | tensor-parallel and expert-parallel inference, CPU offload with AVX2/AVX512 | one GPU, one resident model, one to eight lanes fixed at startup |
| Batching | continuous dynamic batching, a paged cache, CPU-side cache spilling (`generator/job.py`, `pagetable.py`, `cpu_cache.py`) | fixed lanes, no dynamic batching -- a stated product boundary here, not a kernel gap |
| Speculation | n-gram drafting plus draft models plus confidence tracking (`ngram.py`, `draft_confidence.py`) | DFlash2 and the model's native MTP, depth fixed per lane, acceptance measured per lane |
| Serving | not in the repository; TabbyAPI is the recommended backend server | in-engine: OpenAI, Anthropic and Responses APIs, request log, prefix cache, response replay |
| Quantization work | the product's core: a one-step converter (Hessians on the fly, fused Viterbi kernel) and self-calibration scripts (`sc_*.py`, `doc/optimize.md`) | per-artifact recipes over pinned source checkpoints, built by this port's converter, each artifact measured and gated |
| Tests | `tests/` (a generator stress test, kernel sweeps, `conftest.py`), `eval/`, `science/` | 212 test files, 14 commit-time gates, lane fingerprints carrying artifact and binary provenance, ADRs |

## What would not transfer, and why

Each of these is a product decision here rather than an oversight:

- **Multi-GPU parallelism and CPU offload.** The product is one resident 27B model on one 5090; it fits, and
  splitting it across consumer GPUs or streaming experts from host memory buys nothing this product needs.
- **Continuous batching.** This port's boundary is one to eight fixed lanes; large-scale continuous batching
  is an explicit product change, and adopting it would re-cut the scheduler, the context cache and the lanes.
- **Triton kernels.** Their Windows install *requires* `triton-windows` because the attention, cache and
  recurrent kernels are Triton. This port's kernels are hand-written CUDA with per-shape selection ladders
  (`src/ops/linear/<fmt>/<fmt>_dispatch.cpp`), and the evidence apparatus -- lane fingerprints, shape
  ladders, the Op contracts -- is built around them.

## What is worth reading, in descending order of fit

1. **A lower bitrate.** EXL3 reaches 2-8 bpw from an HF model plus a target bitrate, in one step. This port's
   artifacts are ~4.5 bpw (NVFP4) or 8 (FP8); for a model that already fits, the value is headroom, not
   capability. The *converter* design is the part to read before any "ship a 3 bpw line" decision.
2. **Adaptive speculation.** `draft_confidence.py` implies a drafting depth that follows observed confidence,
   where this port fixes depth per lane from measurements. The lane table's acceptance (45.9-64.0%) says depth
   is a real lever, and the instruments to test a confidence-driven depth already exist here
   (`tools/bench/realtext_acceptance.py`, and `accepted_per_position` in the request log).
3. **A combined drafter.** Their generator carries both an n-gram path and draft-model paths. This session
   measured n-gram *alone* and rejected it (91-98% of rounds rejected) while recording that TRT-LLM, vLLM and
   FastDeploy ship *selection* between neural and n-gram drafting; their tree is a second instance of that
   pattern.
4. **KV cache granularity.** Their cache quantization spans 2-8 bits; this port ships three named dtypes
   (bf16, fp8, k8v4) chosen per lane under a maximin bound. Their finer granularity is a reminder that the
   cache dtype is a per-workload lever rather than a global default -- which is what the k8v4 adoption
   already concluded for five of the eight lanes.

## What would settle "which is faster"

A shared-model A/B on this card. Their architecture table lists `Qwen3_5ForConditionalGeneration` and
`Qwen3_5MoeForConditionalGeneration` with multimodal support, so the same Qwen3.8-27B base could be converted
to EXL3 and run against a shipped lane on identical prompts and sampling, with the drafter held constant
(their speculation versus DFlash2 is itself a variable). That experiment needs the card for hours; until it
exists, a speed claim in either direction is unmeasured.
