# NInfer 5090 Windows

> Native Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance.

This repository is the native Windows port of [Neroued/ninfer](https://github.com/Neroued/ninfer):
the same C++/CUDA engine, built with MSVC and CUDA 13.3 on Windows 11 x64, with no WSL2 and no
Docker. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible
HTTP APIs. The runtime is deliberately specialized: one GPU, one resident model, and a
startup-fixed capacity of one to eight active requests.

**v1.2.0 is the current release and the first since v1.1.0.** It adds a fourth and fifth artifact
(Swift 1.5 and NVIDIA ModelOpt, both built by this port), taking the product to **eight measured
launchers over five artifacts**, every one vision-capable at the full 262,144-token context with
DFlash2 or MTP speculative decoding. It also re-measures the whole lane table — three recorded lanes
had stopped reproducing — corrects two MTP draft depths, and stops the attention planner from
splitting a saturated launch, which returns about 1.3 GiB of VRAM per lane at no throughput cost.

v1.1.0 moved to the **v3 artifact line**: four measured launchers over two artifacts. The
engine rejects v2 artifacts outright, so a v1.0.x user must download a v3 artifact or
[upgrade the one they have](docs/weight-conversion.md#upgrade-an-existing-v2-artifact). Details in
[RELEASE_NOTES.md](RELEASE_NOTES.md).

## Install a release (recommended)

1. Download the archive attached to the release page and check it against the SHA-256 listed there.
2. Extract it anywhere. The executables, the launchers and `download_model.bat` sit in the root.
3. Run `download_model.bat`. It offers the five shipped images, fetches the source checkpoints that line
   needs at the revisions pinned in `download_model.py`, and builds the image locally with the converter
   in `tools/`. Nothing is downloaded prebuilt; `build_model.py --list` shows each line's recipe and
   sources without building anything.
4. Double-click a launcher. Each one checks that the engine and the artifact exist before starting,
   and leaves the failure on screen if they do not.

The server then answers on `http://127.0.0.1:<port>/v1` under the model id in that launcher's
header. `GET /health` answers once the model is loaded. The eight launchers and their measured
figures are under [Profiles and launchers](#profiles-and-launchers); building from source is under
[Windows](#windows).

Four weight **lines** ship with this port, and the unsloth line ships as **two images** because one
image cannot serve both routes: the DFlash2 lane runs the no-exception build, its MTP lane the
BF16-exception one. That is **five artifacts across eight launchers**.

**Every one of the five is built locally** by this port's converter (`converter: ninfer-v3` in each
artifact's `.conversion.json`) from the Hugging Face **source** checkpoints below. No `.ninfer` file
is downloaded prebuilt; `download_model.py` fetches sources, not artifacts.

| Model | Weights | Artifact | Built from |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4qat.v3.ninfer` | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), 17.65 GiB |
| Qwen3.8-27B | `nvfp4full` | `qwen3_8_27b_nvfp4full.v3.ninfer` | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), 18.36 GiB, `4c1616bc…` |
| Qwen3.8-27B | `nvfp4full_noex` | `qwen3_8_27b_nvfp4full_noex.v3.ninfer` | the same unsloth line, BF16 exceptions re-encoded; DFlash2 lane |
| Qwen3.8-27B | `nvfp4swift15` (Swift 1.5) | `qwen3_8_27b_nvfp4swift15.v3.ninfer` | [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) on its [BF16 finetune](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), 17.65 GiB |
| Qwen3.8-27B | `nvfp4nvidia` (NVIDIA ModelOpt) | `qwen3_8_27b_nvfp4nvidia.v3.ninfer` | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), 17.65 GiB |

All eight lanes additionally embed [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
on the BF16 base [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

All are Qwen3.8-27B, and all five images are this port's own builds: `download_model.py` fetches the
source checkpoints above at pinned revisions, and the converter turns them into the `.ninfer` files the
launchers run. So the sizes above and the measured lane figures under
[Profiles and launchers](#profiles-and-launchers) describe the same files, and each image's own
`.conversion.json` names the sources it consumed. The
[artifact reference](docs/maintainer/qwen3.8-27b-artifact.md) records each image's identity, sources and
production command.

Upstream publishes artifacts for other checkpoints, which this port neither ships nor measures. The
engine requires a v3 container, and a copy fetched before the republish must be
[upgraded](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) first: the engine rejects a v2
container outright.

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

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
| `start_quasar_v3_dflash2_vision.bat` | QUASAR | DFlash2 (9) | yes | 262,144 | **368 tok/s** | 58.0% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR | MTP (4) | yes | 262,144 | **217 tok/s** | 64.0% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (9) | yes | 262,144 | **380 tok/s** | 56.7% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP (4) | yes | 262,144 | 190 tok/s | 52.7% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 (7) | yes | 262,144 | **281 tok/s** | 51.8% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP (5) | yes | 262,144 | 195 tok/s | 50.9% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA | DFlash2 (9) | yes | 262,144 | **323 tok/s** | 45.9% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA | MTP (4) | yes | 262,144 | 210 tok/s | 57.3% |

**Five lanes ship `--kv-dtype k8v4`** — `start_quasar_v3_dflash2_vision.bat`,
`start_quasar_v3_mtp4_vision.bat`, `start_ninfer_v3_mtp4_vision.bat`,
`start_swift_v3_dflash2_vision.bat` and `start_swift_v3_mtp4_vision.bat`; the other three ship `fp8`,
because the format's measured effect is artifact-dependent and each adoption had to clear the
code-weighted bound ([evidence](docs/research/kv-dtype-evidence.md)). Every figure in the table is its lane's last
measurement, carrying the identity of what measured it in
[`lane_figures.json`](tools/release/lane_figures.json), which the pre-commit hook checks.

**The 2026-09-30 measurement this table was first built from, and a 2026-10-07 re-measurement of the
same profiles on the same protocol, disagreed sharply**: the re-measurement read 10.3-48.7% lower on
seven of them, with the run-to-run spread widened from within +/-1% to up to +/-30% and the
speculative **round cost** -- normalised for acceptance, so the generated content cannot explain it --
up 22-60%. The numbers, their runs, the ruled-out causes and the candidate set are in
[the lane regression record](docs/research/lane-regression-2026-10-07.md); the change is unattributed
and open.

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
- The serve app raises the process timer resolution (`timeBeginPeriod(1)`, `apps/serve/main.cpp`). The
  engine's scheduling loop polls with a 1 ms timed wait (`queue_cv_.wait_for`, `engine_core.h`), and a
  timed wait in a process that has not raised the resolution resolves on the 15.625 ms system timer:
  measured before the change, `initial_binding` was bimodal at 0.9 ms or 16.0-17.0 ms and a request
  paid 18-45 ms of `queue_wait` + `initial_binding` + loop slack, against a flat 2.0-2.6 ms after.
  Windows 10 2004 and later apply the request per-process, so only this server's timer changes; the
  CLI and perplexity apps run the same engine loop and do not raise it.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. On all five shipped artifacts the DFlash2 route
accelerates generated-text decode after multimodal prefill, not Vision encode itself. **(This said
"four shipped artifacts"; five ship, and every one of the eight launchers carries `--vision`.)**

## Client compatibility and known API limitations

The server implements OpenAI Chat Completions and Responses, the Anthropic Messages API, and model
listing. It has no legacy completions endpoint and no embeddings endpoint, and it validates request
bodies strictly: a field it does not implement is refused with a named error instead of being
ignored.

| Route | Status |
|---|---|
| `GET /health` | readiness |
| `GET /v1/models`, `GET /v1/models/{id}` | listing and metadata |
| `POST /v1/chat/completions` | OpenAI chat, streaming and non-streaming |
| `GET /metrics` | Prometheus counters, gauges and latency histograms |
| `POST /v1/responses`, `/input_tokens`, `/compact` | OpenAI Responses API |
| `GET`/`DELETE /v1/responses/{id}`, `/input_items` | locally stored Response state |
| `POST /v1/responses/{id}/cancel` | cancel an in-flight Response |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| `POST /v1/completions` | not implemented (404) |
| `POST /v1/embeddings` | not implemented (404) |

Refused request fields, each with its own error code:

| Field or feature | Error code | Workaround |
|---|---|---|
| `grammar`, `guided_*` aliases | `invalid_request_error` | `response_format` or `structured_outputs.grammar` |
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
`--max-concurrency 1` deliberately, and that only works because the launchers' 8 GiB host quota
retains conversation state — measured 2026-10-06 with `repro_251.py`: interleaved across 8, 16, 32, 48
and 52 conversations every one reused its later turns at ~96%, 56 does not, and the shipped acceptance
shape (12 conversations of ~20,000 tokens) reuses at 99.8%. That covers a main session plus subagents
with a wide margin.

## Capabilities and limits

The engine provides the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- GBNF, JSON-object, and JSON-schema output constraints on every decoding backend;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

Every shipped artifact carries the DFlash2 companion weights and supports
`--spec dflash2 --draft-tokens 7` on the same Text/Vision Engine path, with draft counts 1..15 and
either full or optimized proposal heads.

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
- [Upstream's README](https://github.com/Neroued/ninfer#readme) — the engine's own Linux build,
  Docker image and full artifact lineup. This port does not ship or verify those routes.

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

This port's own modifications, and the third-party software it carries, are attributed in
[NOTICE](NOTICE): the upstream engine and the Windows-port lineage this fork builds on, the Windows
changes this fork owns, and the bundled libraries. Apache-2.0 section 4 asks that the notice travel
with the distribution, so the release archive ships it beside this file.

Four lines ship, all derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and all
carrying the DFlash2 companion from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).
The QUASAR QAT image uses
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
for its quantisation-aware-trained weights; the NVFP4-full image uses the mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); the NVIDIA image uses
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4); and Swift is
[ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) and its
[NVFP4 re-encoding](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4), whose sources and whose
non-Apache licence `NOTICE` records. All five images are built by this port; none is fetched prebuilt.
Source
repositories other than Swift's are distributed under Apache-2.0, as `NOTICE` records. Vendored
dependencies retain their own license files under `third_party/`.
