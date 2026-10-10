# Infernix, read against this port (2026-10-10)

Unlike exllamav3, this is not a distant cousin: **Infernix is a fork of the same upstream this port
follows.** Its description says so -- "C++/CUDA single-GPU inference engine for Qwen3.8 (incl.
Qwen3.8-Flash-Next with offloaded experts) on the RTX 5090; grew from NInfer" -- it reads NInfer's
`.ninfer` artifacts, and its `src/` tree is module-for-module the same shape (`artifact`, `core`,
`media`, `models`, `ops`, `product`, `runtime`, `serve`, `text`). It was created 2026-10-08, two days
before this reading; 80 stars, 9 forks, Apache-2.0, C++, 5 open issues.

It also carries an explicit AI disclaimer: everything added since the fork, including most of its
README, was written with AI (Claude Opus 5.5, Qwen3.8-27B running *on Infernix itself*, plus others),
and "is likely to be neither complete nor entirely accurate. This is hobby development."

Everything below is read from its repository on 2026-10-10 (README, `docs/`, `docs/maintainer/`,
`src/` listing, GitHub API metadata). **Nothing here was reproduced on this machine.**

## The bets that differ

- **A second model family that does not fit.** Qwen3.8-Flash-Next has 24,576 routed experts (63 GiB at
  NVFP4). Infernix keeps the experts in pinned system RAM, caches the hot ones in VRAM, computes part
  of each layer's cache misses on the CPU while the rest cross PCIe, streams a 52 GB n-gram embedding
  table from NVMe, and adds an SSD expert tier. That is the opposite of this port's boundary (one
  resident model, everything in VRAM), and it is what its headline benchmark measures.
- **A hybrid prefix cache** (`docs/maintainer/hybrid-prefix-cache-spec.md`, 117 KB): KV cached by
  content in 64-token blocks, snapshots of the recurrent state at reuse points, and three tiers --
  GPU, host RAM, and a file that survives restarts. Aimed at the same linear-attention artifacts this
  port serves, and the same problem this session kept meeting (incremental encode, splice counts,
  shared-prefix identity).
- **Seven KV formats** -- `int8`, `bf16`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`, the last two holding
  three to four times INT8's context -- against this port's three (`bf16`, `fp8`, `k8v4`). Plus YaRN
  to 1M context.
- **n-gram copy drafting as a supplement, not a replacement** (`docs/ngram.md`, 19 KB): position-indexed
  lookups (16-, 8- and 4-token suffixes, exact comparison, backward extension), wide copy rounds (up to
  63 drafts at concurrency 1, 15 above), capture families at 7/15/31 drafts, the proposal represented
  as a one-hot distribution inside DFlash2's existing sparse verifier, tool-result de-numbering
  (numbered source listings), and a RAM session archive keyed by client identity headers. Five
  real-artifact tests guard it, all gated on an env-var weight path.
- **Agent-facing depth**: tolerant recovery of broken tool calls, a reasoning-loop guard, `/v1/decide`,
  constrained decoding (JSON schema, GBNF, regex) on every mode (`docs/maintainer/constrained-decoding.md`,
  66 KB), preemption under memory pressure, and request options matched to Claude Code, Qwen Code,
  Codex, Zed and GitHub Copilot.
- **Its own artifact container** (`.infernix`) alongside reading `.ninfer`, its own conversion recipes
  (Q4-Q8, FP8, NVFP4, mixed) and five published Hugging Face conversions, and a native MSVC *and* GCC
  build including WSL2 -- so the fork grew a Linux path while this fork is the Windows port.

## What it claims against NInfer

Its 27B benchmark is "Infernix vs NInfer (`master` at `68c54356` with the Windows port)", same artifact,
same flags, with Infernix additionally using n-gram drafting. Its agentic replay (three seeds, 130
requests each) reports: average TTFT 5.7 s -> **2.7 s**; prompt tokens served from cache 83.7% ->
**90.4%**; prompt tokens prefilled 982K -> **569K**; prefill tok/s on no-cache-hit requests 6,127 ->
**8,254**; decode 199 -> **235** tok/s single-request and 249 -> **282** at its own batching; workload
wall time 16.9 -> **12.5 min**. Single long-prompt prefill is 1.34-1.51x faster from 128K tokens;
decode without speculation is "level"; quality is "level" (perplexity 4.468 against 4.479 on its own
1.04M-token corpus at 16K context).

Conditions, stated so the numbers are not read as facts about this port:

- self-published, AI-written, on their machine (a PCIe Gen5 **x8** link, which they note would read
  higher at x16), through their own `bench/agentic_ab` harness;
- their NInfer arm is upstream master **plus the Windows port**, not this port's shipped lanes (which
  run k8v4 KV on five of eight lanes, per-lane depths, chunk 8192, and this port's tuning);
- the Infernix arm uses flags this port does not ship by default and n-gram drafting the NInfer arm
  did not have;
- the fork is two days old.

## What is worth reading here, and what is not

Worth reading, in order:

1. **`docs/ngram.md`.** This session measured n-gram drafting *alone* and rejected it (91-98% of rounds
   rejected), and recorded that TRT-LLM, vLLM and FastDeploy ship *selection* between neural and n-gram
   drafting. This is a third instance, and the only one with a full design spec: wide copy rounds, a
   one-hot proposal into the existing verifier, tool-result de-numbering, and an evaluation section
   that insists a useful test demonstrate accepted n-gram proposals rather than a flag that turned on.
2. **`docs/maintainer/hybrid-prefix-cache-spec.md`.** Recurrent-state snapshots plus a persistent file
   tier is a direct answer to the cache problem this port's own research notes kept circling.
3. **The KV formats.** `vq2`/`k4v2` claim three to four times INT8's context; this port's k8v4 choice
   was made per lane under a maximin bound, so a fourth and fifth format is a real question.
4. **The Flash-Next offload architecture** -- only if a model that does not fit ever becomes a product
   goal here, which is an explicit product change, not a kernel.

Not applicable: the offload design (this port's boundary is one resident model), the `.infernix`
container (this port ships `.ninfer` v3), and the second build platform (this fork is the Windows one).

## The experiment that would settle it

Build Infernix on this machine and run *this port's* instruments against it -- `lane_fingerprint.py`,
`probe_prefill_floor.py`, the profile matrix, the coding suites -- on the same artifact, same flags,
same prompts, with and without its n-gram drafting. That is a card-hours job, and it is the only way to
convert their published comparison into a measurement of this port's standing.
