# Single-GPU serving performance

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.

This port ships four launcher profiles. Their measured figures, the exact argument set each one
starts, and the interleaving rule behind the numbers are in the
[README's Profiles and launchers section](../README.md#profiles-and-launchers). That table is the
port's published performance claim, and `tools/release/profiles.py` is its only copy.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## What this port does not publish

Upstream publishes per-checkpoint result pages and model cards for the artifacts it ships:
Qwen3.6-27B, Qwen3.6-35B-A3B, and the Qwen3.8-27B `groupwise-int` and `nvfp4` images. This port
ships none of those checkpoints, measures none of them, and keeps no copy of their pages or cards.
[Upstream's README](https://github.com/Neroued/ninfer#readme) is their authority.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external streaming
TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs can generate
different token totals even with the same prompts and seeds. Output-limit and repetition samples
remain labeled in the measured corpus; throughput alone does not establish successful task
completion.

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow.
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.
