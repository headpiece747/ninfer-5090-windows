# Maintainer documents: English index

Twelve maintainer documents in this directory are written in **Chinese by upstream**. This page
summarises what each one decides, in English, and links to the original for the detail. **The
Chinese originals are the source of truth and are not modified.**

## Why they are not translated

Upstream maintains these documents in Chinese, so a translation is a permanent divergence: every
future upstream merge conflicts in every translated file, and a reviewer reading the merge cannot
tell a real behavioural change from a wording change. Measured on 2026-10-29, after upstream's
consolidation, the twelve documents hold roughly 15,650 CJK characters.

The trade is accepted deliberately: an English reader gets each document's decisions and its scope
here, and follows the link for the mechanism. That is enough to review a change against the design,
which is what these documents are for.

## What this is not

- **Not a substitute.** If a change touches a rule stated in one of these documents, read the
  original before approving it. The summary below is deliberately a map, not a specification.
- **Not current for `docs/maintainer/` as a whole.** Five further documents in this directory are in
  English and maintained by this port: `artifact-conventions.md`, `build-system.md`,
  `fp8-ungated-bands.md`, `project-map.md` and `qwen3.8-27b-artifact.md`.
  Only the twelve listed here are Chinese.
- **Not a claim about upstream's intent.** Where a document is ambiguous, the summary says what the
  text says and links out.

## The documents

### 1. [engine-architecture.md](engine-architecture.md) — execution ownership and request lifecycle

The top-level reference. Sections: ownership (Gateway, Frontend, EngineCore, Scheduler,
ResourceManager, Model, Program, instance lifetime); startup and resource preparation; request and
execution; commit, publication and retirement; device execution and Graph; maintenance entry points.

### 2. [distributed-execution.md](distributed-execution.md) — centralized scheduling and ranks

Sections: processes and plans; startup and failure; commands and completion; communication and
graphs. Covers rank preparation, control/data transport, and complete completion or failure.

### 3. [context-management.md](context-management.md) — KV/State capacity and cache policy

Succeeds the former cache-policy and physical-store pair. Sections: layers and capacity; identity,
content and address; execution addressing; reservation and continuous progress; save, provenance
and reclamation; Attention DP; migration and finish.

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
bindings are rebuilt. Capacity flags and defaults are documented in
[`docs/startup-configuration.md`](../startup-configuration.md) and the executable `--help`.

### 4. [frontend.md](frontend.md) — input identity, output and grammar

Sections: input and model policy; input identity and cache boundary; output transactions and tools;
constrained decoding; bounded media preparation.

### 5. [weights-and-loading.md](weights-and-loading.md) — representation, binding and materialization

Sections: source weights to execution parameters; objects, bindings and uses; numerical
representation and physical layout; merge and sharded materialization; backing and transfer. This is
the successor to the former artifact-container, tensor-format and storage-layout references.

### 6. [qwen3_5-model.md](qwen3_5-model.md) — Qwen3.5 Dense/MoE

Sections: computation structure; representation and execution; state and history; speculative
backend; Vision and frontend.

### 7. [qwen4_exp-model.md](qwen4_exp-model.md) — Qwen4Exp (Flash Next)

Sections: trunk and state; weight selection and merging; PLE; parallelism and placement; MTP, Vision
and Graph.

### 8. [glm5next-model.md](glm5next-model.md) — GLM5Next

Sections: trunk; KDA and ReplaySSM; Kpool Sparse MLA; representation selection; parallelism and
placement; MTP and Graph; Vision and frontend.

### 9. [speculative-decoding.md](speculative-decoding.md) — MTP, ReplaySSM and DFlash

Sections: acceptance and commitment; MTP's stable frontier; ReplaySSM; DFlash and DFlash2; Graph and
lifecycle. Replaces the former standalone DFlash reference.

### 10. [expert-weight-cache.md](expert-weight-cache.md) — streaming experts and mixed execution

Sections: weights and computation boundary; fixed region and streaming window; upload, borrow and
reuse; CPU/GPU handoff; budget and observation.

### 11. [observability.md](observability.md) — timing, resources and logs

Sections: ownership; time boundaries; resource boundaries; measurement windows; log presentation.
Replaces the former standalone logging reference.

### 12. [op-development.md](op-development.md) — Op contracts, validation and tuning

Sections: semantic boundaries; code organization; headers and resources; numerical validation;
tuning and measurement. Its §6.4 (perturbation coverage) is the one English section, maintained by
this port, because the mutation seam and `tests/ops/mutations.json` are port work; the heading is
kept so `#64-perturbation-coverage` resolves.

## Corpora and test data that are Chinese by design

Not documentation, and deliberately not translated:

- `eval/corpora/perplexity-1m/data/zhwiki/*.txt` — Chinese Wikipedia, the perplexity corpus.
  Translating it would change what perplexity measures, and Chinese is one of the acceptance domains
  the DFlash2 draft-width sweep is chosen on.
- The `chinese` domain prompts in `tools/bench/ttft/cases.py` and related tests.

## Provenance

The twelve Chinese documents are upstream's, except op-development.md's English §6.4, which this
port adds. For the other eleven, on 2026-10-29 `git diff upstream/dev -- docs/maintainer/<path>` was
empty.
