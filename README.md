# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and one to eight execution lanes fixed at startup.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

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

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two execution lanes and prefix caching:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
resident requests and retained prefixes. Requests acquire KV pages as execution advances; under
pressure, the scheduler can pause a request and resume it later. The profile provides two extra
Device StateImages and the default shared pinned Host budget: 8 GiB plus eight model StateImages,
used for retained state, KV and pause snapshots.

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

A reusable checkpoint combines KV with the complete continuation state at an exact token frontier.
The engine retains completed conversation endpoints and stable input boundaries for multi-turn and
agent reuse. Inactive checkpoints share Device and pinned Host capacity; pressure reclaims retained
resources before pausing resident requests. Paused requests resume from a snapshot or rebuild their
state by replaying already committed tokens.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The eight launchers' figures are in
[Profiles and launchers](#profiles-and-launchers), each at the exact argument set its launcher
starts. Upstream's own per-model run records live in its `docs/performance/` pages, which cover
checkpoints and platforms this port does not ship and are therefore not reproduced here.

### Profiles and launchers

Eight launchers ship for the RTX 5090, one per measured-optimal profile. Every number below was
measured on this machine with the exact argument set the launcher uses, in one interleaved window
against the artifacts this release ships. **The window is dated 2026-09-30, not 2026-09-28 as this
previously said**: `tools/release/profiles.py` records the current figures as re-measured 2026-09-30
across all eight lanes and names the 2026-09-24 set they replaced, and the 2026-09-28 date belongs to
the interleaved depth sweep further down this section, not to this table. Absolute decode varies by up
to ~9% *between* sessions on a card whose clocks are not pinned, so compare lanes to each other and
expect your own absolute figures to differ; within one interleaved window the same lane repeats to
1.5% or less. Acceptance is the column to trust in a comparison.

The earlier revision of this table was measured with a harness defect: the first full-length decode
after a server start is a transient that returns faster than every later identical request and
different, shorter text, and the 16-token warmup did not reach the state it affects, so that request
was averaged into the figures. On the NVIDIA MTP5 lane it read 261.7 tok/s against 169.8 for requests
two onward, which is most of the difference between that row's old 228 tok/s and its measured 166.
The harness now discards a full-length warmup, and its acceptance denominator excludes the same
request. No lane, flag, context or draft depth changed. All five
artifacts are vision-only here because Vision measured free on every one of them at 262,144; the
with/without comparison is recorded in
[ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md). No degraded text-only variant ships,
and every profile reaches the full native context.

| Launcher | Artifact | Spec | Vision | Context | Decode | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| `start_quasar_v3_dflash2_vision.bat` | QUASAR | DFlash2 (7) | yes | 262,144 | **319 tok/s** | 55.0% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR | MTP (4) | yes | 262,144 | **221 tok/s** | 59.6% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (7) | yes | 262,144 | **296 tok/s** | 48.7% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP (4) | yes | 262,144 | 190 tok/s | 51.4% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 (7) | yes | 262,144 | **362 tok/s** | 63.0% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP (4) | yes | 262,144 | 218 tok/s | 58.7% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA | DFlash2 (7) | yes | 262,144 | **338 tok/s** | 56.2% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA | MTP (4) | yes | 262,144 | 210 tok/s | 53.1% |

Context ceilings are measured, not assumed. The engine refuses a profile whose minimum Engine
runtime reservation plus its 1 GiB automatic headroom does not fit in what remains after weights, and
it reports the byte counts when it refuses. That ceiling is a property of the artifact's weight size,
measured per lane with each launcher's own flags, and every one of the five artifacts still fits the
full KV pool. The figures the tree records, per artifact, are in
[the artifact reference](docs/maintainer/qwen3.8-27b-artifact.md): the QAT line at **16.3 GiB** on its
MTP lane and **17.2 GiB** on its DFlash2 lane, the NVIDIA line at the same 16.3/17.2, the unsloth
BF16-exception image at **17.1/17.9**, and the unsloth no-exception image — the DFlash2 lane's build —
at **17.2 GiB** on that lane (`tools/release/profiles.py`, section on the refusal it cleared).
**This paragraph previously said "these four builds carry 16.3-17.1 GiB … and 17.2-18.0 GiB": the
count was one low (five artifacts ship) and the 18.0 upper bound is not a figure anything in the tree
records, so it is dropped rather than reproduced.** The retired builds it then contrasts are real and
stay: the FP8-importing Swift build this release replaced carried 18.90 GiB on its MTP lane and
stopped at 240,000, and the retired NVFP4 image carried 19.7 GiB and could not reach the full native
context at all, which is why it was replaced.

`--lm-head-draft` is set per profile because its value is not uniform; the measured gains behind
each choice are recorded in [ADR-0005](docs/adr/0005-per-profile-flags-are-measured.md):

- QUASAR: on for both routes;
- NVFP4 MTP: on at depth 4;
- NVFP4-full DFlash2: **on**, but marginally, and it costs headroom, so it is the first thing to
  turn off if a profile ever refuses to start.

MTP depth is chosen per artifact from measurement rather than convention, and every MTP lane now
ships **depth 4**. QUASAR and NVFP4-full previously shipped depth 5; both were re-swept 2026-09-30 over
depths 1-5 on five domains at three interleaved rounds each and moved to 4, because depth 5's worst
domain is -17.0% on QUASAR against depth 4's 0.0%. NVIDIA and Swift 1.5 were swept the same way and
already sat at 4. All sixteen lane-by-depth combinations were re-measured
2026-09-28 with the warmup transient excluded, interleaved two rounds. The previous choice -- d4 on
QUASAR, d5 on the other three -- came from records that mixed runs with and without that transient,
and three of the four lanes were on the wrong depth as a result; NVIDIA's was the largest, d5 reading
228.3 against d4's 223.4 there but measuring 167.8 once the transient was excluded.

Acceptance rate does not predict throughput, because tokens committed per round matters more than the
proportion accepted, so depth is selected on measured decode rate. Depth is also not monotone in either
direction: NVFP4-full's d4 is the slowest depth on any lane (172.8 against d5's 234.1) while accepting
least, and NVIDIA's d4 beats its d5 on both throughput and acceptance. Neither figure can be carried
between artifacts, which is why all four are measured.

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

**This subsection was rewritten by the 2026-10-04 upstream merge, because the flags it
documented no longer exist.** It previously described `--max-shared-prefixes`,
`--max-private-continuations`, `--max-long-anchors-per-continuation`, `--host-state-slots` and
`--context-cache-policy`, and measured what raising them bought: 1/5 round-2 cache hits at the
Engine defaults against 5/5 at 99.1% once all three bounds were raised together. That was the
evidence for shipping them, and it remains the reason to look at the host pool at all.

Upstream `b9114396` replaced the context cache and collapsed those five bounds into one shared
Host quota, `--host-context-mib`, which every launcher now ships at **8192** -- the same 8 GiB
the old `--host-state-slots 16` / `--host-kv-mib 8192` pair reserved. The per-catalog counts are
the Engine's defaults and are no longer this port's to set.

**The measurements above were taken against the previous cache and are not re-verified against
the new one.** They are kept because they are the only evidence this port has for why the host
pool is worth sizing deliberately, and because a silent cache miss is the failure mode they
describe -- a lone repeated prefix masks it. Re-running the five-prompt resend against
`--host-context-mib` is the check that would confirm or refute them, and it is **open**: the
cache rewrite invalidates the flag combination the experiment used, so it has to be re-expressed
against the new surface rather than re-run as written.
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

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Clean shutdown (SIGINT/SIGTERM) |
| `1` | Configuration, bind, warmup, or listen error |
| `3` | Engine-wide failure — restart required |

Exit code `3` is distinct from `1` so a supervisor can tell "engine fault, restart required" from
"bad configuration, do not retry". The engine latches failure permanently (`fail_all_locked` sets
`failed_` once and nothing clears it), so the serve loop checks `is_available()` after `listen()`
returns and exits with code `3` if the engine has failed.

Two build notes specific to Windows:

- FFmpeg is located through the local `ffmpeg/` tree instead of `pkg-config`, and is exposed
  to consumers as the **`PkgConfig::FFMPEG`** target, so the upstream targets that link it
  need no changes.
- The FFmpeg runtime DLLs must sit beside the executables. Copy `ffmpeg\bin\*.dll` into
  `build\apps\` after building; without them the process exits immediately with
  `0xC0000135` (`STATUS_DLL_NOT_FOUND`) and prints nothing.

### Runtime notes

- Artifact I/O uses Win32 unbuffered positional reads
  (`CreateFileW` with `FILE_FLAG_NO_BUFFERING` and `OVERLAPPED`), the counterpart of POSIX `O_DIRECT`/`pread`.
- Every TMA kernel passes its descriptor block by value as a `__grid_constant__` parameter, on every
  platform. MSVC cannot pass an `alignas(128)` struct that way (`C2719`); `alignas(64)` is accepted on
  this toolchain, and a map in parameter space is already in the proxy the TMA unit reads it through,
  which is also what makes it CUDA Graph safe. See `tools/scripts/probe_tma_align.cmd`.
- MSVC has no `__int128`, so 128-bit arithmetic goes through `ninfer::Uint128` (`src/core/uint128.h`), which is the native type on GCC/Clang and a constexpr fallback on MSVC. Two call sites depend on it: the runtime contract's cost arithmetic, and upstream's `--host-context-mib` parser, whose fractional-MiB handling divides a 128-bit numerator by a 128-bit divisor.
  `ninfer::Uint128` (`src/core/uint128.h`).


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
- GBNF, JSON-object, and JSON-schema output constraints on every decoding backend;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- one to eight resident execution lanes with bounded FIFO ingress;
- resource-pressure preemption with snapshot or token-replay recovery;
- no priority/QoS, weight offload, multi-GPU, or distributed serving;
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
