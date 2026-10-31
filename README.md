# NInfer 5090 Windows

> Native Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance.

This repository is the native Windows port of [Neroued/ninfer](https://github.com/Neroued/ninfer):
the same C++/CUDA engine, built with MSVC and CUDA 13.3 on Windows 11 x64, with no WSL2 and no
Docker. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible
HTTP APIs. The runtime is deliberately specialized: one GPU, one resident model, and a
startup-fixed capacity of one to eight active requests.

Release v1.2.3 ships **eight measured launchers over four Qwen3.8-27B artifact lines**, all
vision-capable at the full 262,144-token context, with DFlash2 or MTP speculative decoding. The
engine rejects v2 artifacts outright, so a v1.0.x user must download a v3 artifact or
[upgrade the one they have](docs/weight-conversion.md#upgrade-an-existing-v2-artifact). Details in
[RELEASE_NOTES.md](RELEASE_NOTES.md).

## Install a release (recommended)

1. Download the archive attached to the release page and check it against the SHA-256 listed there.
2. Extract it anywhere. The executables, the launchers and `download_model.bat` sit in the root.
3. Run `download_model.bat`. It offers the shipped artifacts, fetches the source checkpoints the one
   you choose is built from at pinned revisions, and builds it with the converter the archive ships.
   No `.ninfer` file is downloaded prebuilt, so there is no artifact digest pin to go stale.
4. Double-click a launcher. Each one checks that the engine and the artifact exist before starting,
   and leaves the failure on screen if they do not.

The server then answers on `http://127.0.0.1:<port>/v1` under the model id in that launcher's
header. `GET /health` answers once the model is loaded. The eight launchers and their measured
figures are under [Profiles and launchers](#profiles-and-launchers); building from source is under
[Windows](#windows).

Five v3 artifact files define this port's shipped surface. The eight launchers use these five: the
unsloth line ships twice, because its DFlash2 lane runs the no-exception image and its MTP lane the
BF16-exception one.

| Model | Weights | Artifact | Launcher pairs |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4qat.v3.ninfer` | DFlash2 + MTP4 |
| Qwen3.8-27B | `nvfp4nvidia` (ModelOpt) | `qwen3_8_27b_nvfp4nvidia.v3.ninfer` | DFlash2 + MTP4 |
| Qwen3.8-27B | `nvfp4full` | `qwen3_8_27b_nvfp4full.v3.ninfer` | MTP4 |
| Qwen3.8-27B | `nvfp4full_noex` | `qwen3_8_27b_nvfp4full_noex.v3.ninfer` | DFlash2 |
| Qwen3.8-27B | `nvfp4swift15` (Swift 1.5) | `qwen3_8_27b_nvfp4swift15.v3.ninfer` | DFlash2 + MTP4 |

All four lines are Qwen3.8-27B, and every shipped artifact is built locally by this port's converter
from the source checkpoints `download_model.py` fetches at pinned revisions; no `.ninfer` file is
downloaded prebuilt, so there is no artifact digest pin to go stale.

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

See [Resource scheduling and context cache](docs/maintainer/context-management.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The eight launchers' figures are in
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

You can run `build_windows.bat` from the repository root for an automated one-click build (which automatically downloads the matching FFmpeg LGPL shared dev files if missing, configures CMake with Ninja, compiles in Release mode, and stages the FFmpeg DLLs).

Alternatively, compile manually from an x64 Native Tools Command Prompt:

```bat
call "<VisualStudio>\VC\Auxiliary\Build\vcvars64.bat" x64
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DNINFER_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_MEDIA_ACQUIRE=ON
cmake --build build --config Release -j
```

producing `build/apps/ninfer-serve.exe`, `ninfer.exe`, and `ninfer-perplexity.exe`.

Two build notes specific to Windows:

- FFmpeg is located through the local `ffmpeg/` tree instead of `pkg-config`, and is exposed
  to consumers as the **`PkgConfig::FFMPEG`** target, so the upstream targets that link it
  need no changes.
- The FFmpeg runtime DLLs must sit beside the executables. Copy `ffmpeg\bin\*.dll` into
  `build\apps\` after building (done automatically by `build_windows.bat`); without them the process exits immediately with
  `0xC0000135` (`STATUS_DLL_NOT_FOUND`) and prints nothing.

### Runtime notes

- Artifact I/O uses Win32 unbuffered positional reads (`CreateFileW` with
  `FILE_FLAG_NO_BUFFERING`), the counterpart of POSIX `O_DIRECT`/`pread`. Nothing is mapped: the
  payload is read, never mapped into the address space.
- The Blackwell TMA kernels pass their descriptor block **by value** in parameter space
  (`const __grid_constant__`, `alignas(64)`), as upstream does. The port once passed it by device
  pointer because MSVC rejects a by-value `alignas(128)` parameter with `C2719`; that form broke
  CUDA Graph capture and was removed.
- MSVC has no `__int128`; the runtime contract's 128-bit cost arithmetic goes through
  `ninfer::Uint128` (`src/core/uint128.h`).

### Profiles and launchers

Eight launchers ship for the RTX 5090 -- the four measured-optimal artifacts, each in both
its DFlash2 and MTP routes (eight profiles total). Every number below was measured on this machine
with the exact argument set the launcher uses, in one interleaved pass -- absolute decode varies by
up to ~9% between sessions on a card whose clocks are not pinned, so compare lanes to each other
and expect your own absolute figures to differ. All four lines are vision-only here because Vision
measured free on every one at 262,144; the with/without comparison is recorded in
[ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md). No degraded text-only variant
ships, and every profile reaches the full native context. **Three lanes ship `--kv-dtype k8v4`**
(QUASAR DFlash2, Swift15 MTP, NVFP4-nvidia MTP); the rest ship `fp8`.

| Launcher | Artifact | Spec | Vision | Context | Decode | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| `start_quasar_v3_dflash2_vision.bat` | QUASAR | DFlash2 (7) | yes | 262,144 | **262 tok/s** | 45.5% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR | MTP (4) | yes | 262,144 | 188 tok/s | 57.1% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (9) | yes | 262,144 | **314 tok/s** | 47.3% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP (4) | yes | 262,144 | 163 tok/s | 45.6% |
| `start_swift_v3_dflash2_vision.bat` | Swift15 | DFlash2 (7) | yes | 262,144 | **310 tok/s** | 56.7% |
| `start_swift_v3_mtp4_vision.bat` | Swift15 | MTP (5) | yes | 262,144 | 180 tok/s | 51.7% |
| `start_nvidia_v3_dflash2_vision.bat` | NVFP4-nvidia | DFlash2 (9) | yes | 262,144 | **292 tok/s** | 45.6% |
| `start_nvidia_v3_mtp4_vision.bat` | NVFP4-nvidia | MTP (4) | yes | 262,144 | 165 tok/s | 46.7% |

Context ceilings are measured, not assumed. The engine refuses a profile whose minimum Engine
runtime reservation plus its 1 GiB automatic headroom does not fit in what remains after
weights, and it reports the byte counts when it refuses. That ceiling is a property of the artifact's weight size:
QUASAR carries 16.1 GiB and NVFP4-full 17.0 GiB of device weights, both of which leave room
for the full KV pool. Our earlier NVFP4 image carried 19.7 GiB and could not, which is why it
was replaced.

The optimized proposal head (`--proposal-head optimized`, formerly `--lm-head-draft`) is set per profile because its value is not uniform; the measured gains behind
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
- **Vision is free on every shipped artifact.** The with/without comparison at 262,144 is recorded
  in [ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md), so every profile here carries Vision. The retired NVFP4 image did cost 16,384-27,008 tokens of context, which is
  part of why it was replaced. The Vision runtime still has its own input envelope of 32,768
  merged tokens (131,072 raw patches) per request.

### Context-cache bounds are set deliberately, and they matter

The launchers pass `--host-cache-capacity 8192MiB`: one shared Host quota covering StateImages, Main and
backend KV, pause snapshots and in-flight transfer destinations. Upstream `b9114396` replaced the
five separate budgets this paragraph used to name (`--host-state-slots`, `--host-kv-mib`,
`--max-shared-prefixes`, `--max-private-continuations`, `--max-long-anchors-per-continuation`), and
none of them exists now. The quota is what bounds prefix retention, and the shortfall it prevents is
silent.

That silence is why it is set rather than left at the default. On the superseded cache, measured on
five distinct ~530-token prompts sent and then resent, the engine's defaults gave **1/5 round-2 cache
hits at a 19.8% token-level hit rate**, with four of the five re-prefilling in full on every call and
no error or degradation signal anywhere; the bounds gave **5/5 at 99.1%**. Those figures describe the
pre-`b9114396` cache and are kept as the record of why a bounded quota is set at all — the current
cache has its own accounting and no equivalent re-derivation has been published.

The quota costs nothing measurable in the profile it was measured on (QUASAR DFlash2 with Vision at 262,144). `--context-capacity auto` sizes each pool from the VRAM left after weights, so every profile's capacity is its own measured ceiling, not a shared number.
The failure mode is silent, so it is worth setting even when a single repeated prompt
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

Qwen3.8-Flash-Next uses CPU/GPU Main experts, disk-backed NF4 PLE and segmented CUDA Graph.
Its native MTP uses the full shared head; DFlash/DFlash2 and optimized proposal heads are not
supported for this architecture. See the [CLI commands](docs/cli.md),
[conversion recipe](docs/weight-conversion.md) and
[current support reference](docs/maintainer/qwen4_exp-model.md).

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- one to eight resident execution lanes with bounded FIFO ingress;
- resource-pressure preemption with snapshot or token-replay recovery;
- no priority/QoS, active-request swapping, weight offload, multi-GPU, or distributed serving;
- startup-fixed typed KV pools shared by active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--context-capacity` (formerly `--kv-capacity`) sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights. Both explicit and automatic modes use 1 GiB of
startup planning headroom. Explicit capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/context-management.md)
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
