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

Two v3 artifacts ship with this port. The four launchers use these two, and `download_model.bat`
offers exactly them:

| Model | Weights | Artifact | Download |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4qat.v3.ninfer` | [QUASAR QAT](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer), 17.36 GiB, `8b86901a…` |
| Qwen3.8-27B | `nvfp4full` | `qwen3_8_27b_nvfp4full.v3.ninfer` | [NVFP4-full](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer), 18.07 GiB, `ac98cd39…` |

Both are Qwen3.8-27B. Their sizes and full SHA-256 digests are pinned in `download_model.py`, which
verifies every download against them; a republish upstream means updating that pin.

Upstream publishes artifacts for other checkpoints, which this port neither ships nor measures. The
engine requires a v3 container, and a copy fetched before the republish must be
[upgraded](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) first: the engine rejects a v2
container outright.

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The four launchers' figures are in
[Profiles and launchers](#profiles-and-launchers), each at the exact argument set its launcher
starts. The [performance index](docs/performance.md) and its
[measurement rules](docs/performance/methodology.md) hold the engine's per-model run records, which
cover checkpoints this port does not ship.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. On both shipped artifacts the DFlash2 route
accelerates generated-text decode after multimodal prefill, not Vision encode itself.

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
recorded in [ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md). No degraded text-only variant ships, and every profile reaches the
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
each choice are recorded in [ADR-0005](docs/adr/0005-per-profile-flags-are-measured.md):

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
  near-tie can flip and the continuation diverges. Speculation measured 3-4x faster; the lane
  table above carries each launcher's own measured decode figure.
- **Vision is free on both shipped artifacts.** The with/without comparison at 262,144 is recorded
  in [ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md), so every profile here carries Vision. The retired NVFP4 image did cost 16,384-27,008 tokens of context, which is
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
constraints. All three together gave **5/5 hits at 99.1%** on a host State pool with headroom, and that
qualifier is load-bearing: those five prefixes and the private continuations they leave behind compete
for the same pool. Re-measured with the shipped DFlash2 bounds (`--host-state-slots 8
--max-private-continuations 8`) the same five-prompt resend gives **3/5 hits at a 59.1% token-level
rate** -- the last two prefixes are evicted to hold the five continuations -- and with
`--max-private-continuations 2` it returns to **5/5 at 98.6%**. Retention policy makes no difference to
either figure.

A second retention limit sits above those bounds. Once the State pools are full, publishing the
conversation's newest checkpoint means replacing a resident, and by default that requires the capture
to be demanded by two matching reuse domains or to carry explicit evidence. With no session key every
request is its own reuse domain, so one append-only conversation never clears that bar: measured here,
its reusable frontier froze at 52,723 tokens and time to first token grew with the whole prompt --
2.6 s at 65k context, 15.1 s at 117k. `--context-cache-policy rolling` supplies that standing from the
conversation's own proven lineage, the residents its request matched exactly at their frontier, and the
frontier then tracks the conversation: 104,283 of 117,180 tokens cached and 4.1 s to first token on the
same workload, with one Device checkpoint slot rather than the eight that used to be needed. It is
opt-in because it will evict a prefix that another conversation sharing it still wants.

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

## Capabilities and limits

The engine provides the following capabilities, with optional components enabled at startup:

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

Both shipped artifacts carry the DFlash2 companion weights and support
`--spec dflash2 --draft-tokens 7` on the same Text/Vision Engine path, with draft counts 1..15 and
either full or optimized proposal heads.

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
- [Upstream's README](https://github.com/Neroued/ninfer#readme) — the engine's own Linux build,
  Docker image and full artifact lineup. This port does not ship or verify those routes.

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

This port's own modifications, and the third-party software it carries, are attributed in
[NOTICE](NOTICE): the upstream engine and the Windows-port lineage this fork builds on, the Windows
changes this fork owns, and the bundled libraries. Apache-2.0 section 4 asks that the notice travel
with the distribution, so the release archive ships it beside this file.

Both shipped artifacts are published under the `cometkim` Hugging Face account. Both derive from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and carry the DFlash2 companion from
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The QUASAR QAT image
additionally uses
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
for its quantisation-aware-trained weights; the NVFP4-full image additionally uses the fixed mixed
FP8/NVFP4 weights from [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4).
These source repositories are distributed under Apache-2.0. Vendored dependencies retain their own
license files under `third_party/`.
