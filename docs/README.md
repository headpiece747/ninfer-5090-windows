# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [Startup configuration](startup-configuration.md) | shared CLI/serve TOML schema, defaults, overrides, resource controls and startup display |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [opencode settings](opencode-settings.md) | the eight shipped model entries, compaction, and the concurrency answer for OpenCode Desktop |
| [v2 to v3 flag diff](v2-v3-flag-diff.md) | what changed between the retired v2 launchers and the shipped v3 ones, flag by flag |
| [FP8 ungated bands](maintainer/fp8-ungated-bands.md) | the matched-tile measurement, its control, and the two structural limits behind the ungated FP8 A8 TMA bands |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

This port ships four weight lines, as **five images** — the unsloth line ships twice, because the
DFlash2 lane runs the no-exception build and its MTP lane the BF16-exception one. All five are built by
this port from the sources below, fetched at pinned revisions:

| Model | Weights | Built from | Versioned model card source |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | [model card](../model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4full` | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | — |
| Qwen3.8-27B | `nvfp4full_noex` | the same unsloth line, BF16 exceptions re-encoded; DFlash2 lane | — |
| Qwen3.8-27B | `nvfp4swift15` (Swift 1.5) | [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) on its [BF16 finetune](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | — |
| Qwen3.8-27B | `nvfp4nvidia` (NVIDIA ModelOpt) | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | — |

All five shipped artifacts are built locally by this port's converter from these sources; none is
fetched prebuilt. `download_model.py` fetches sources.

Upstream publishes artifacts for other checkpoints, each with its own model card on its own
repository. This port ships none of them and keeps no copy of their pages:
[upstream's README](https://github.com/Neroued/ninfer#readme) is their authority.

The storage contract the shipped artifacts implement is the
[Qwen3.8-27B artifact reference](maintainer/qwen3.8-27b-artifact.md): identity, object inventory,
formats, layouts, aliases and source transforms, with a section per line in Sections 14 to 19.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Open work

[Open issues](issue-checklist.md) keeps pending problems and MLA performance work.
Qwen4-27B's [architecture and quantization](designs/qwen4-27b-rtx5090.md) and
[Op requirements](designs/qwen4-27b-op-checklist.md) are a separate adaptation plan.

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

**Twelve of them are written in Chinese by upstream.**
[Maintainer documents: English index](maintainer/README.md) summarises in English what each one
decides and links to the original, which is the source of truth. The originals are not translated or
modified: a translation is a permanent divergence, and every future upstream merge would then conflict
in every translated file.

[`adr/`](adr/) holds the decisions behind the shipped profiles and artifact rules, each with the
measurement that produced it: why a v3 container is required, why speculation is not bit-identical,
the draft-binding contract, why the profiles are vision-only, why every profile flag is
measured rather than chosen, and the NVFP4 context-materialize V-cache stride / runtime DLL staging fix (ADR-0013).

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | ownership, startup budget, request and execution transactions, commit and retire |
| [Distributed execution](maintainer/distributed-execution.md) | centralized scheduling, rank preparation, control and data communication, complete completion and failure |
| [Context management](maintainer/context-management.md) | KV/State unified capacity, identity, reservation, prefix policy, ADP and recovery |
| [Frontend](maintainer/frontend.md) | model policy and shared text/media foundations, input identity, output and grammar transactions |
| [Weights and loading](maintainer/weights-and-loading.md) | representation, binding, conversion, value-preserving sharded materialization and backing |
| [Qwen3.5](maintainer/qwen3_5-model.md) | Dense/MoE, Main/MTP/DFlash, Vision and state composition |
| [Qwen4Exp](maintainer/qwen4_exp-model.md) | Flash Next, HC/QSA/PLE, quantization, TP/EP and execution composition |
| [GLM5Next](maintainer/glm5next-model.md) | KDA/MLA/mHC, quantization, ADP, MTP/Vision and Graph |
| [Speculative execution](maintainer/speculative-decoding.md) | MTP alignment, ReplaySSM, DFlash distributions and accepted prefix |
| [Expert cache](maintainer/expert-weight-cache.md) | fixed and streaming cache, CPU/GPU split, upload and consumption lifetime |
| [Observability](maintainer/observability.md) | time and resource accounting, measurement windows, rank statistics and log boundaries |
| [Op development](maintainer/op-development.md) | semantics, implementation organization, numerical validation and tuning method |
| [Build system](maintainer/build-system.md) | targets, compilation and linking, source and dependency ownership |
| [Artifact conventions](maintainer/artifact-conventions.md) | what a new conversion must match to join the shipped lanes: all-NVFP4 text, the weight envelope, the lane shape and the measurement gates |
| [Qwen3.8-27B artifact reference](maintainer/qwen3.8-27b-artifact.md) | identity, object inventory, formats, layouts, aliases and source transforms of the shipped artifacts |
| [FP8 ungated bands](maintainer/fp8-ungated-bands.md) | the matched-tile measurement, its control, and the structural limits behind the ungated A8 TMA bands |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

## Dated records

Three directories hold the working record behind the decisions above. They are not user guides and not
contracts; they are what a later reader needs in order to check a claim instead of taking it on trust.

| Directory | What it holds |
|---|---|
| [`features/`](features/) | design notes for runtime work this port carries, including the weight-only NVFP4 DFlash2 runtime and the MSVC test-portability change |
| [`upstream-reports/`](upstream-reports/) | what this port reported or considered reporting upstream, and the fork-divergence record |

Qwen3.8-Flash-Next is the Qwen4Exp instance; Qwen3.8-27B uses the Qwen3.5 architecture.
The model references own mathematical and stored-representation facts, and the execution reference
owns integrated support and its limits. Concrete Op contracts live in `include/ninfer/ops/`.
