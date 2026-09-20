# NInfer 5090 Windows

> Native Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance.

This repository is the native Windows port of [Neroued/ninfer](https://github.com/Neroued/ninfer):
the same C++/CUDA engine, built with MSVC and CUDA 13.3 on Windows 11 x64, with no WSL2 and no
Docker. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible
HTTP APIs. The runtime is deliberately specialized: one GPU, one resident model, and a
startup-fixed capacity of one to eight active requests.

Release v1.1.0 moves to the **v3 artifact line**: four measured launchers over two artifacts, both
vision-capable at the full 262,144-token context, with DFlash2 or MTP speculative decoding. The
engine rejects v2 artifacts outright, so a v1.0.x user must download a v3 artifact or
[upgrade the one they have](docs/weight-conversion.md#upgrade-an-existing-v2-artifact). Details in
[RELEASE_NOTES.md](RELEASE_NOTES.md).

## Install a release (recommended)

1. Download the archive attached to the release page and check it against the SHA-256 listed there.
2. Extract it anywhere. The executables, the launchers and `download_model.bat` sit in the root.
3. Run `download_model.bat`. It offers the two shipped artifacts, downloads the one you choose, and
   verifies its size and SHA-256 against the pin in `download_model.py`.
4. Double-click a launcher. Each one checks that the engine and the artifact exist before starting,
   and leaves the failure on screen if they do not.

The server then answers on `http://127.0.0.1:<port>/v1` under the model id in that launcher's
header. `GET /health` answers once the model is loaded. The four launchers and their measured
figures are under [Profiles and launchers](#profiles-and-launchers); building from source is under
[Windows](#windows).

Five engine artifacts are available. The shipped launchers use the first two.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4qat.v3.ninfer` | [QUASAR QAT](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer), 17.36 GiB, `8b86901a…` |
| Qwen3.8-27B | `nvfp4full` | `qwen3_8_27b_nvfp4full.v3.ninfer` | [NVFP4-full](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer), 18.07 GiB, `ac98cd39…` |

The two shipped artifacts are the QUASAR QAT image and cometkim's fuller-NVFP4 image. Their sizes
and full SHA-256 digests are pinned in `download_model.py`, which verifies every download against
them; a republish upstream means updating that pin.

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start (building the engine on Linux)

This section is upstream's, and it builds the engine on Linux from source. For the native Windows
build in this repository, see [Windows](#windows) below.

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

Upstream targets Linux and ships no packaged binary distribution; it runs from its source build
tree. This Windows port is different: see the Windows section below, which ships a packaged
archive with the engine, its runtime DLLs and one launcher per profile.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Windows

Upstream builds on 64-bit Linux only. This tree carries a Windows layer that compiles the
same engine natively on Windows 11 x64 with MSVC and CUDA, with no WSL2 or Docker.

### Requirements

- Windows 11 x64 and an NVIDIA GeForce RTX 5090 (`sm_120a`);
- a CUDA 13.3-compatible NVIDIA driver and the CUDA Toolkit (13.3 verified);
- Visual Studio with the **Desktop development with C++** workload — the build enters the
  MSVC x64 environment, and the VS installer ships the CMake and Ninja it uses;
- the FFmpeg shared development tree staged in `ffmpeg/` at the repository root
  (`include/` and `lib/`, from an `ffmpeg-master-latest-win64-gpl-shared` build).

### Build

```bat
call "<VisualStudio>\VC\Auxiliary\Build\vcvars64.bat" x64
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DNINFER_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

producing `build/apps/ninfer-serve.exe`.

Two build notes specific to Windows:

- FFmpeg is located through the local `ffmpeg/` tree instead of `pkg-config`, and is exposed
  to consumers as the **`PkgConfig::FFMPEG`** target, so the upstream targets that link it
  need no changes.
- The FFmpeg runtime DLLs must sit beside the executables. Copy `ffmpeg\bin\*.dll` into
  `build\apps\` after building; without them the process exits immediately with
  `0xC0000135` (`STATUS_DLL_NOT_FOUND`) and prints nothing.

### Runtime notes

- Artifact I/O uses Win32 memory-mapped and unbuffered positional reads
  (`CreateFileW` with `FILE_FLAG_NO_BUFFERING`), the counterpart of POSIX `O_DIRECT`/`pread`.
- The Blackwell NVFP4 TMA kernels pass their descriptor block by **device pointer**, because
  MSVC cannot pass an `alignas(128)` struct by value as a kernel parameter (`C2719`).
- MSVC has no `__int128`; the runtime contract's 128-bit cost arithmetic goes through
  `ninfer::Uint128` (`src/core/uint128.h`).

### Profiles and launchers

Four launchers ship for the RTX 5090, one per measured-optimal profile. Every number below was
measured on this machine with the exact argument set the launcher uses, in one interleaved pass --
absolute decode varies by up to ~9% between sessions on a card whose clocks are not pinned, so
compare lanes to each other and expect your own absolute figures to differ. Both artifacts are
vision-only here because Vision measured free on both at 262,144; the with/without comparison is
recorded in `docs/adr/0004`. No degraded text-only variant ships, and every profile reaches the
full native context.

| Launcher | Artifact | Spec | Vision | Context | Decode | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| `start_quasar_v3_dflash2_vision.bat` | QUASAR | DFlash2 (7) | yes | 262,144 | **343 tok/s** | 62.5% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR | MTP (4) | yes | 262,144 | 220 tok/s | 58.3% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (7) | yes | 262,144 | **345 tok/s** | 63.7% |
| `start_ninfer_v3_mtp5_vision.bat` | NVFP4-full | MTP (5) | yes | 262,144 | 254 tok/s | 64.2% |

Context ceilings are measured, not assumed. The engine refuses a profile whose minimum Engine
runtime reservation plus its 1 GiB automatic headroom does not fit in what remains after
weights, and it reports the byte counts when it refuses. That ceiling is a property of the artifact's weight size:
QUASAR carries 16.1 GiB and NVFP4-full 17.0 GiB of device weights, both of which leave room
for the full KV pool. Our earlier NVFP4 image carried 19.7 GiB and could not, which is why it
was replaced.

`--lm-head-draft` is set per profile because its value is not uniform; the measured gains behind
each choice are recorded in `docs/adr/0005`:

- QUASAR: on for both routes;
- NVFP4 MTP: on at depth 5;
- NVFP4-full DFlash2: **on**, but marginally, and it costs headroom, so it is the first thing to
  turn off if a profile ever refuses to start.

MTP depth is chosen per artifact from measurement rather than convention: depth 4 is fastest on
QUASAR, depth 5 on NVFP4. Acceptance rate does not predict throughput, because tokens committed
per round matters more than the proportion accepted, so depth is selected on measured decode
rate.

### Two behaviours to know before relying on them

- **Speculative decoding is not bit-identical to plain decoding.** Greedy output differs
  between no-spec, every MTP depth and DFlash2, deterministically and reproducibly. This is
  documented engine behaviour rather than a porting defect: acceptance compares a proposal
  token against the target argmax for its verify column, and the maintainer notes state that
  speculation "does not impose token or logits equality between different quantization, prefill
  or kernel paths" — the batched verify kernel is not the single-token decode path, so a
  near-tie can flip and the continuation diverges. Speculation measured 3-4x faster
  (67-83 tok/s without it against 239-343 with it).
- **Vision is free on both shipped artifacts.** The with/without comparison at 262,144 is recorded
  in `docs/adr/0004`, so every profile here carries Vision. The retired NVFP4 image did cost 16,384-27,008 tokens of context, which is
  part of why it was replaced. The Vision runtime still has its own input envelope of 32,768
  merged tokens (131,072 raw patches) per request.

### Context-cache bounds are set deliberately, and they matter

The launchers pass `--max-shared-prefixes 7 --max-private-continuations 8
--max-long-anchors-per-continuation 4`. At `--max-concurrency 1` the defaults are
`max(1,4)` shared, 2 private and 2 anchors, and that is quietly expensive: measured on five
distinct ~530-token prompts sent and then resent, the defaults gave **1/5 round-2 cache hits
at a 19.8% token-level hit rate**, with four of the five prompts re-prefilling in full on
every call and no error or degradation signal anywhere. Raising the shared-prefix bound
alone changed nothing; the anchor and private-continuation bounds were the binding
constraints. All three together gave **5/5 hits at 99.1%**.

The bounds cost nothing measurable in the profile they were measured on (QUASAR DFlash2 with Vision at 262,144). `--kv-capacity auto` sizes each pool from the VRAM left after weights, so every profile's capacity is its own measured ceiling, not a shared number.
The failure mode is silent, so it is worth setting these even when a single repeated prompt
appears to cache perfectly — a lone resident prefix masks it.

## Client compatibility and known API limitations

The server implements OpenAI Chat Completions and Responses, the Anthropic Messages API, and model
listing. It has no legacy completions endpoint and no embeddings endpoint, and it validates request
bodies strictly: a field it does not implement is refused with a named error instead of being
ignored.

| Route | Status |
|---|---|
| `GET /health` | readiness |
| `POST /cancel` | cancel an in-flight response |
| `GET /v1/models`, `GET /v1/models/{id}` | listing and metadata |
| `POST /v1/chat/completions` | OpenAI chat, streaming and non-streaming |
| `POST /v1/responses`, `/compact`, `/input_tokens` | OpenAI Responses API |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| `POST /v1/completions` | not implemented (404) |
| `POST /v1/embeddings` | not implemented (404) |

Refused request fields, each with its own error code:

| Field or feature | Error code | Workaround |
|---|---|---|
| `response_format` other than text | `response_format_not_supported` | ask for JSON in the prompt |
| `grammar`, `guided_*`, `structured_outputs` | `constrained_decoding_not_supported` | prompt-level formatting |
| `logprobs`, `top_logprobs` | `logprobs_not_supported` | none |
| `logit_bias` | `logit_bias_not_supported` | none |
| `n > 1` | `n_not_supported` | send the request again |
| `tool_choice` other than `auto` or `none` | `tool_choice_not_supported` | let the model choose |
| `parallel_tool_calls: false` | `parallel_tool_calls_not_supported` | allow parallel calls |
| legacy `functions` / `function_call` | `legacy_tools_not_supported` | use `tools` |
| `strict` tool schemas | `strict_tools_not_supported` | drop `strict` |
| Anthropic server tools | `server_tools_not_supported` | run tools in the client |
| Anthropic `document` blocks (PDF) | `documents_not_supported` | extract text first |
| `store`, `background`, `service_tier` | `*_not_supported` | omit them |
| audio or file inputs | `audio_inputs_not_supported`, `file_inputs_not_supported` | text and images only |
| `repetition_penalty` other than 1.0 | `repetition_penalty_not_supported` | leave it neutral |

Two things the server does offer, with defaults worth knowing:

- Responses state is process-local, bounded to 1024 records and 256 MiB
  (`--response-store-max-records`, `--response-store-max-mib`).
- `--cors` is off by default and no launcher passes it. With no `--api-key` set, turning CORS on
  lets any page in a browser on this machine drive the model.

## OpenCode Desktop integration

Point an OpenAI-compatible provider at a running launcher, using the model id from that launcher's
header. [docs/opencode-settings.md](docs/opencode-settings.md) carries the `opencode.json`, the four
model entries, the compaction settings, and the concurrency answer: the launchers run
`--max-concurrency 1` deliberately, and that only works because eight conversation states are
retained in host RAM, which covers a main session plus parallel subagents.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.

<!-- ninfer:features:start -->
| feat branch | stacked on | status | squashed on dev as |
|---|---|---|---|
| [`feat/msvc-test-constexpr`](docs/features/msvc-test-constexpr.md) | `master` | C++20/MSVC test fixes for sqrt constant expressions and explicit array headers. | `squash(feat/msvc-test-constexpr)` |
| [`feat/nvfp4-dflash2`](docs/features/nvfp4-dflash2.md) | `master` | Standalone 34-object weight-only NVFP4 DFlash2 execution, including subview-scale and codebook binding corrections. | — |
<!-- ninfer:features:end -->
