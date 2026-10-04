# Maintainer documents: English index

Seven maintainer documents in this directory are written in **Chinese by upstream**. This page
summarises what each one decides, in English, and links to the original for the detail. **The
Chinese originals are the source of truth and are not modified.**

## Why they are not translated

Upstream maintains these documents in Chinese, so a translation is a permanent divergence: every
future upstream merge conflicts in every translated file, and a reviewer reading the merge cannot
tell a real behavioural change from a wording change. Measured on 2026-10-04, the seven documents
hold roughly 29,000 CJK characters.

The trade is accepted deliberately: an English reader gets each document's decisions and its scope
here, and follows the link for the mechanism. That is enough to review a change against the design,
which is what these documents are for.

## What this is not

- **Not a substitute.** If a change touches a rule stated in one of these documents, read the
  original before approving it. The summary below is deliberately a map, not a specification.
- **Not current for `docs/maintainer/` as a whole.** Six further documents in this directory are
  already in English, including `op-development.md`, `resource-scheduling-and-context-cache.md`'s
  English predecessors and the compiler/toolchain notes. Only the seven listed here are Chinese.
- **Not a claim about upstream's intent.** Where a document is ambiguous, the summary says what the
  text says and links out.

## The documents

### 1. [engine-architecture.md](engine-architecture.md) — execution ownership and request lifecycle

Defines the model instance, who owns execution, the request lifecycle, and the commit relationships
across modules. Its two companion documents define the cache and preemption policy and the physical
page and replica contracts, so read it as the top of a three-document set.

Sections: product execution model (architecture, instance, weights); ownership and call boundaries
(Gateway, Frontend, EngineCore, Scheduler, ResourceManager, Model, Program, instance lifetime and
fixed execution); request lifecycle including pause and resume.

### 2. [resource-scheduling-and-context-cache.md](resource-scheduling-and-context-cache.md) — cache policy and preemption

The policy layer: what a cache record *means*, when work is saved, when it is reclaimed, and when a
request is paused. This is the document behind the flags the launchers pass.

The load-bearing idea is that cache management uses **three different units**, and conflating them is
the usual source of confusion:

- a **continuation record** owns one history's resume point, named entry points and genuine reuse
  eligibility;
- a **checkpoint** expresses an exactly resumable position with its StateImage and per-backend KV
  coverage;
- **physical objects and replicas** carry the actual occupancy, sharing, transfer and release.

Ordinary execution reserves incrementally, one unit ahead. When the working set grows past what can
be co-resident, the engine pauses younger requests and advances older ones first; a resume takes a
fresh full permit covering the old frontier plus the first new unit, and returns the balance once
there is progress. A request's committed history and output semantics survive the pause; device
bindings are rebuilt.

Two consequences worth knowing before reading: a **prefill chunk only splits scheduled work and does
not by itself save state** — save opportunities come from input semantics and request lifecycle; and
**a cache record's retention grade is not the same as pinning the whole history**.

Capacity table (§3.1) — the flags this port ships:

| Option | Meaning |
|---|---|
| `max_context` | logical context ceiling for one request |
| `kv_capacity` | Main KV pool capacity in token equivalents, page-rounded; may also be solved from the startup VRAM budget |
| `max_concurrency` | ceiling `C` on simultaneous execution bindings, **1–8** |
| `context_cache.device_state_slots` | extra StateImage slots beyond the `C` base, default `C` |
| `context_cache.host_capacity_bytes` | pinned Host bytes shared by StateImages, Main/backend KV, pause snapshots and transfer destinations |

Note the last two: upstream `b9114396` (2026-10-04) collapsed five former bounds
(`--host-state-slots`, `--host-kv-mib`, `--max-shared-prefixes`, `--max-private-continuations`,
`--max-long-anchors-per-continuation`) into `--host-context-mib`. Main KV pages are 64 tokens.

Also covers: ownership table (Frontend, Engine request record, Scheduler, ResourceManager, Native
Program, in-Program stores), transaction and binding, source lookup, retention and reclamation (real
demand, two-stage retention, limited physical actions), scheduling and preemption including
Snapshot/Replay recovery, transfer and cancellation, and how the rules apply along a request chain.

### 3. [paged-kv-cache.md](paged-kv-cache.md) — physical store contract

The physical layer, complementing §2: pages, replicas, address space and consumer contracts. This is
what a reader implementing or debugging the store needs.

Covers the physical model; three independent granularities; the typed pool set and its capacities
(Main, automatic, Backend, Host); page groups and physical layout including the grouping invariant
and the closed Device plane orders; logical pages and Device/Host replicas with transfer and
descriptor lifetime; KV history and address space with execution rows; and bounded execution units
with bind, reserve/materialise, commit/rollback, pause/finish/release and stable boundaries.

### 4. [artifact-container.md](artifact-container.md) — the `.ninfer` format

The normative format description: contract and notation, file set and address space, binary framing
(entry header, continuation-volume header, file directory) and the JSON master directory. A reader and
a writer can be implemented independently from this document; the binder for a given architecture then
interprets logical parameters and configuration.

A v3 artifact has a fixed entry point and one master directory: a single `.ninfer` file when small, or
an entry plus continuation volumes when large. Includes the default 32 GB volume cap.

### 5. [replayssm-gdn.md](replayssm-gdn.md) — ReplaySSM for Gated DeltaNet

How GDN speculative verify avoids storing full recurrent state per verify position: the target
verify keeps the ordinary per-token recurrence but records only the raw inputs that drive the state
transition, and once the accepted length is known, replays from the committed checkpoint.

Covers the problem (GDN state size, the snapshot baseline, raw-input ReplaySSM), the recurrence and
what the raw inputs are sufficient for (limited-precision ordering, grouped q/k heads, sufficiency of
the transition), and accepted-prefix replay including verify inputs and outputs.

### 6. [linear-benchmark.md](linear-benchmark.md) — the authority for `linear_bench.cu`

Defines the commands, timing contract, metrics, preset suites and extension rules for the pure-Linear
benchmark. It deliberately **does not** change the production Linear route and does not re-select any
route winner.

Use cases: single-point performance, NCU on a single point, small-T sweeps, and typical model suites
(the default T sets, `qwen3_6_27b`, `qwen3_6_35b_a3b`).

### 7. [examples/q4-linear.md](examples/q4-linear.md) — measured Linear Op performance

Re-measured 2026-09-26. Records the full Linear Op performance for this shape as a reference for
evaluation and later tuning. Every measured call has a single Graph kernel node and zero external
workspace, so the scope is the pure Linear Op. Covers what was measured and under what conditions,
the final timing curves, logical bandwidth and Tensor Core utilisation, how to read and verify the
curves, and reproduction.

It notes that the bench log's `cuda_runtime` value comes from a compile-time macro and that the
actually-loaded runtime version must be queried separately — a figure whose configuration differs from
its source.

## Corpora and test data that are Chinese by design

Not documentation, and deliberately not translated:

- `eval/corpora/perplexity-1m/data/zhwiki/*.txt` — Chinese Wikipedia, the perplexity corpus.
  Translating it would change what perplexity measures, and Chinese is one of the acceptance domains
  the DFlash2 draft-width sweep is chosen on.
- The `chinese` domain prompts in `tools/bench/ttft/cases.py` and related tests.

## Provenance

The seven Chinese documents are upstream's. This port does not modify them: `git diff upstream/master
-- <path>` is empty for six of the seven, and `paged-kv-cache.md` differs only in the two anchors
repointed when upstream renamed two sections in its own Chinese text.
