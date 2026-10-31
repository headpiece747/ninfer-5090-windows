# NInfer Windows v1.2.3 (RTX 5090)

## What changed in 1.2.3

A major upstream synchronization and stability release following `v1.2.2`, incorporating 35 upstream commits from `Neroued/ninfer@dev` and resolving critical runtime and grammar constraints on Windows:

- **Upstream Synchronization (35 commits)**: Merged `Neroued/ninfer@dev` (`996d9214..458863be2`):
  - **Unified Context Cache Pools**: Merged upstream PR #366 (`bea5a04ec`), unifying KV cache pages and State records into a single consolidated pool, eliminating dual allocation contention and superseding legacy separate materialization budgeting.
  - **Updated FP8 Cache Representation**: Adopted Group-128 UE8M0 cache encoding (`824bd3a41`) across all supported lanes.
  - **Split Runtime Startup Observability**: Streamlined startup logging across dedicated `capacity`, `context`, and `rank` lines (`e5e18b974`), providing transparent token-capacity estimates.
  - **Modularized Causal Attention**: Upstream split monolithic softmax attention test runners into modular compilation units under `tests/ops/softmax_attention/causal/` (`f30e661f8`).
- **Deterministic Reasoning Grammar Delimiter**: Fixed an ambiguous Earley NFA regression introduced upstream in `1407c62f6` where reasoning delimiter edges leaked into user grammar rules. Restored deterministic exact-byte DFA for reasoning prefixes in `src/text/grammar.cpp`, ensuring strict delimiter termination and passing `ninfer_qwen3_5_grammar_real_test` 100%.
- **Context Cache Checkpoint Queue & Leak Remediation**: Implemented a deferred `pending_release_` queue in `ResourceManager` (`src/runtime/engine/context_cache/resource_manager.h`) to retry checkpoints whose pins were temporarily held during in-flight transactions, permanently resolving Native checkpoint leaks and physical page exhaustion.
- **End-to-End Hardware Verification**: Verified all 8 shipping profile launchers end-to-end on physical RTX 5090 hardware (`verify_launchers_v3.py`): all 8 servers boot in ~7.5s, commit 323k–440k KV tokens (well above the 262,144 ceiling), and pass both chat text generation and multi-modal image evaluation.
- **Cleaned Orphaned Source & Documentation Drift**: Removed orphaned 2,765-line file (the uncompiled pre-split causal cache test) per Rule 41, updated divergence documentation, and fixed broken documentation anchors.

---

## What changed in 1.2.2

A stability and feature synchronization release following `v1.2.1`, integrating upstream updates and completing decoupled speculative decoding dataflow:

- **Upstream Synchronization (3 commits)**: Merged upstream `Neroued/ninfer@dev` (`1b6c7232`, `d4a78e03`, `996d9214`):
  - **Decoupled Speculative Accept Dataflow**: Speculative acceptance kernels (`speculative_accept_sparse_drafts` and `speculative_accept_dense_drafts`) now write out decoupled `licensed_tokens` and `licensed_counts` ($A + 1$) without mutating `token_counts` or context state in-place, giving model callers (`qwen3_5`, `qwen4_exp`, `glm5next`) explicit control over sequence state commit and rollback.
  - **Paired BF16 Linear Projection Kernel**: Added fused dual-projection kernel (`src/ops/linear/pair/bf16/launch.cu`) optimizing adjacent linear projections under BF16.
  - **Unified State Fields Indexing**: Added zero-copy `gather_state_fields` and `scatter_state_fields` CUDA kernels (`src/ops/indexing/state_fields.cu`) for sequence state boundary transitions.
  - **Distributed Rank Telemetry**: Added host-side high-resolution timing markers and rank boundary synchronization measurement in `src/runtime/executor/measurement.cpp`.
- **Universal Windows 1ms Timer Resolution**: Implemented RAII `WindowsTimerResolution` scope (`timeBeginPeriod(1)`) in `Engine::Impl`, eliminating Windows default 15.6ms timer quantums during streaming token generation and background task polling.
- **Tokenizer Cache Splice Recovery**: Enhanced prefix caching recovery in `tests/text/test_tokenizer_cache_splice.cpp` to correctly handle prompt splice boundaries.
- **MSVC 14.51 / NVCC Portability**: Resolved C++ uniform initialization narrowing conversion errors in QSA `tail_extract.cu` under MSVC 19.44 / NVCC 13.3.
- **Release Audit & Packaging**: All 14 fast pre-commit gates, documentation links, citation audits, and test baselines pass; zero-dependency PE header import verification enforced.

---

## What changed in 1.2.1

A quick-turnaround patch release following `v1.2.0`, incorporating full upstream synchronization and critical Windows runtime stability fixes:

- **Upstream Synchronization (39 commits)**: Merged all algorithmic and architectural updates from upstream `Neroued/ninfer@dev` (`05327b22..46ee5c3b`):
  - Unified `RequestStats` and `EngineSnapshot` runtime observability (`engine.snapshot().runtime`, `snapshot().ranks.at(0).memory`), retiring superseded `publish_kv_capacity`.
  - Upstream distributed startup, worker loops, and model preparation.
  - Clamped Expert Parallel (EP) MoE fixtures, Sigmoid routing oracles, and SwiGLU BF16 tests.
  - Upstream observability clocks, host timing refactors, and vocabulary reduction / sampling operations.
- **Native Windows Memory Planning Fix**: Resolved fatal engine startup failure (`cannot read system memory availability`) caused by upstream's Linux-only `/proc/meminfo` query by implementing native Windows `GlobalMemoryStatusEx` in `src/core/device_memory.cpp`.
- **MSVC 19.44 / 14.51 Template & CTAD Portability**: Fixed MSVC class template argument deduction (CTAD) in braced initializer lists across `vocabulary_logprobs.cpp` and `vocabulary.cpp` with explicit `std::array<int, N>` types and included `<array>`.
- **PE Import Dependency Audit**: Integrated zero-dependency PE header import analysis into release packaging (`package_release.py` and `check_archive_contents.py`) to permanently prevent missing DLL errors at packaging time.
- **Hardware Verification**: Re-verified all 8 production launchers on RTX 5090 hardware: all 8 lanes load in ~7.5s, commit 262,144 KV tokens, and pass chat completion and image multimodal inference.

---

## What changed in 1.2.0

This is the first release since `v1.1.0`. Two drafts of these notes existed — one calling this
release `1.2.0` and a later one `1.3.0` — and they are merged here into `1.2.0`, because versions are
published in order and never skipped, and `1.2.0` has never been published. Shipping it as `1.3.0`
would leave a `1.1.0 → 1.3.0` gap in the public list, which reads as a withdrawn release. Everything
below is what a user on `v1.1.0` gets.

All five artifacts are now built by this port from their published sources rather than fetched from
`cometkim`, and one draft rule replaces another across every line. All eight launchers change, and two
are new.

- **The QUASAR line is rebuilt from its QAT source.** Every text linear is imported from
  `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` with its own activation scale, so nothing is re-encoded and no
  corpus is needed. Hashing every binding against the artifact it replaces leaves 1394 of 1513
  byte-identical, so the source repository moving between revisions did not change the weights. The 96
  that differ are `gdn/a_projection` and `gdn/b_projection`, which this build takes pristine from the
  base where the fork decoded them from the source's quantized control words — and at (96, 5120) the
  NVFP4 layout cannot hold them at all.
- **The unsloth line is rebuilt**, its MLP imported and its FP8 attention, linear-attention and MLP
  56-63 encoded from the BF16 base. This source carries no activation scale for those 233 matrices —
  checked exhaustively — so the divisors are measured on this checkpoint by
  `tools/convert/calibration.py` and committed with it. Borrowing another quantization's stored scales
  instead measured 2.2%/0.45% worse: an activation maximum does not transfer between weight
  realizations. Perplexity on the full corpus at fp8 KV is **4.724219** against the published build's
  4.98768 (`docs/perplexity-baseline.md`, the shipping gate's own table, re-taken 2026-10-07 and
  reproduced to the last digit on 2026-10-10). The `--quick` pair recorded on 2026-09-24 was 4.75750
  from 4.82452; `--quick` is one stream per domain and is not comparable, which that document explains.
- **A fourth line joins the release.** `start_nvidia_v3_dflash2_vision.bat` and
  `start_nvidia_v3_mtp4_vision.bat` serve NVIDIA's own ModelOpt quantization of the base model: its
  NVFP4 MLP imported on all 64 layers, its FP8 attention encoded from the BF16 base using the
  checkpoint's per-site scales. Perplexity is **4.90168** against the official stock's 4.90169 on the
  full corpus — the same, not better — while the file is **20% smaller** (18.95 GB against 23.72 GB),
  contains no FP8 tensor where the official has 146. It reaches the full 262,144 at the `fp8` KV the
  launchers use, where the copy of the official artifact measured here stops at 240,000 on MTP and lower
  still on DFlash2; the published official does reach 262,144 at `int8` KV, so the reach is a property of
  the build and the KV dtype together rather than a wall. The artifact reference carries the exact
  ceilings.
- **The draft's projections are encoded as NVFP4 on three lines**, where upstream's recipe gives them
  Q8. Measured on the port's own bench: 58.0% against 54.8% acceptance on the QUASAR lane, and 68.8%
  against 49.4% on the unsloth line's DFlash2 lane, which is also 19% faster. It touches no text weight,
  which the MTP digests confirm is bit-identical. **It is not applied to Swift, where it lost**: 57.7%
  against 60.9% on that artifact's DFlash2 lane. The z-lab draft was trained against the stock model's
  hidden states and this finetune's are not the stock ones, so which encoding matches them is a property
  of the target and is measured per artifact rather than carried across.
- **The proposal head is now a record for all eight lanes, and no setting changed.** ADR-0005 requires
  the spec route, draft depth and proposal head to be a per-profile decision from a record; the two
  artifacts added after that ADR had none, while all eight lanes passed `--lm-head-draft`. Measured on the
  served sampling, one lane at a time: every lane confirms the shipped setting, and on several of them the
  full head accepts *more* and still decodes slower, which is why decode rate rather than acceptance is
  the criterion. One lane's greedy measurement pointed the other way, and reversed when the two arms were
  interleaved on the protocol the lanes actually serve — 329.5/323.6 tok/s with the flag against
  257.2/257.9 without. That reversal is what the record carries: taking the first number would have
  shipped a 25 % regression on that lane.
- **Two lane figures had drifted, and one artifact could not be measured at all.** Sweeping the
  *unchanged* artifacts, the acceptance figures recorded on 2026-09-17 no longer reproduce: one
  artifact's DFlash2 lane reads 45.7% against a recorded 62.5%, and its MTP d4 lane 62.4% against
  58.3%. Nothing failed in between — the release check verifies that the launchers, the tables and the
  profiles agree with each other, and they did. Every lane figure in this release is re-measured, and
  compared by interleaving the alternatives in one window, because acceptance reproduces across windows
  and throughput does not.
- **A `--quick` perplexity comparison cannot be quoted as a result.** `--quick` scores one stream per
  domain and `full` four, so one stream decides the number: the NVIDIA build's `zhwiki-00` moves 8.85%
  where the full corpus's four Chinese streams move 1.80% together. Its full-corpus figures are
  identical across two domains won and two lost, and that is what the release says they are.
- **`--quick` figures for the QUASAR artifact do not reproduce** either: 4.89741 recorded, 4.94879
  measured. Every row in `docs/perplexity-baseline.md` now carries the date it was taken.
- **A Windows timer defect added up to 16 ms to every request, and is fixed.** The engine's scheduling
  poll waits with a 1 ms timeout, which a process that never raises the system timer resolves on the
  15.625 ms tick instead. Measured on one lane, same artifact and same requests: `initial_binding` was
  bimodal at 0.9 ms or 16.0-17.0 ms, `queue_wait` bunched at 0-15 ms, and total latency averaged
  77.3 ms. `ninfer-serve` now calls `timeBeginPeriod(1)` and the same requests read a flat 2.0-2.6 ms
  `initial_binding` with 56.3 ms total. Decode and acceptance are unmoved (fingerprint `code` cell
  320.1 tok/s / 45.9%). `ninfer-cli` runs the same loop once per invocation, where it is invisible
  beside model load, and `ninfer-perplexity` does not run it at all.
- **The three apps set the console code page**, so their output renders as UTF-8 in any console rather
  than through the machine's OEM code page. Verified in both directions with a probe that reads the
  console's output code page: 437 before, 65001 after.
- **The A16 SwiGLU decomposition above T=16 is gone, and the fused route covers every extent.** Every
  A16 call with T>16 used to project gate/up into a BF16 buffer and apply the nonlinearity to that,
  which measured 8.1e-3 against the profile's own 6.3e-3 tail bound — about 29% over, and identically
  at 17, 32, 64 and 129, so it was the rounding rather than the extent. The route, its `silu_mul` pass
  and its `[34816,T]` workspace are deleted. The A16 profile in `tests/ops/linear/swiglu/cases_nvfp4.cpp`
  now runs 19 extents from 1 to 1024 and passes at every one, with no materialised buffer and no second
  pass. Measured, and it is not a uniform speedup: 3-7% slower at T=17-32, even at 64, 8.5% faster at
  129, and a much tighter tail — the old route's p95 at T=64 was nearly twice its median, where the
  fused route's is within 1%. T=16 is the control: it is the same route in both revisions, and its
  0-2.8% spread is the noise floor. Every shipped lane runs a single lane, so the change is invisible
  to them.

### Artifacts

| Weights | Artifact | Local size | SHA-256 |
|---|---|---|---|
| `nvfp4qat` (QUASAR) | `qwen3_8_27b_nvfp4qat.v3.ninfer` | 17.65 GiB | `42359a09…` |
| `nvfp4full` (unsloth) | `qwen3_8_27b_nvfp4full.v3.ninfer` | 18.36 GiB | `4c1616bc…` |
| `nvfp4full_noex` (unsloth, DFlash2 lane) | `qwen3_8_27b_nvfp4full_noex.v3.ninfer` | 17.65 GiB | `0feb3075…` |
| `nvfp4swift15` (Swift 1.5) | `qwen3_8_27b_nvfp4swift15.v3.ninfer` | 17.65 GiB | `bdb257f7…` |
| `nvfp4nvidia` (NVIDIA) | `qwen3_8_27b_nvfp4nvidia.v3.ninfer` | 17.65 GiB | `d93a60e8…` |

Five artifacts across eight launchers. The unsloth line ships twice on purpose: the DFlash2 lane runs
the **no-exception** image and its MTP lane the BF16-exception one, because re-encoding those nine
exception parents to NVFP4 is what returns the native context to the DFlash2 lane and is worth +11.6
acceptance points there, while the same re-encoding costs the MTP head 21.4 (DFlash2 d7 / MTP d5) so one image cannot serve both routes.

`download_model.py` fetches the **source** checkpoints every artifact above is built from, at pinned
revisions, and `build_model.py` converts them with the converter the archive ships under `tools/`. No
`.ninfer` file is downloaded prebuilt, so a fresh install builds the image it runs.

### Retained in 1.2.0, from the draft that carried this version

Two retention defects, and the configuration that hid them. All six launchers change, so an upgrade
behaves differently without any action on your part.

- **Two Swift lanes join the release.** `start_swift_v3_dflash2_vision.bat` and
  `start_swift_v3_mtp4_vision.bat` serve UkisAI's Swift finetune as a third artifact. Its ModelOpt
  checkpoint keeps attention and GDN in FP8; importing that left 9 GiB of 8-bit weights and capped
  both lanes below the native context. They are now encoded to NVFP4 from the finetune's BF16
  source, with both W8 endpoints Q8, so no FP8 code word reaches the artifact. Perplexity on the
  fixed corpus improves to **4.68429** from 4.84938 for the same recipe importing the FP8, and both
  lanes reach the full 262,144-token context. The lanes now serve **Swift 1.5**, and their DFlash2 draft
  is encoded to NVFP4, which is what returns the native context to the DFlash2 lane; see the DFlash2
  entry above.

- **A growing conversation stopped reusing its own prefix.** Once the State pools filled, publishing
  the newest checkpoint meant replacing a resident, and the capture could not be valued against doing
  so, so the reusable frontier pinned and every later request re-prefilled its whole tail. Measured on
  one conversation at one Device checkpoint slot: the frontier froze at 52,723 tokens with time to
  first token rising from 2.6 s at 65k context to 15.1 s at 117k. The launchers now pass
  `--context-cache-policy rolling`, which lets the portfolio fold decide whether a capture is worth more
  than the resident it would displace; the frontier then tracks the conversation (104,283 of 117,180
  tokens cached, 4.1 s).
- **Five conversations did not all stay cached**, because the host State pool could not hold the
  prefixes plus the private continuations they leave behind, and the shortfall was silent: five prompts
  sent and then resent gave 3/5 hits at a 59.1% token-level rate. `--host-state-slots` is now 16, which
  covers every configuration the other two bounds permit (7 shared prefixes plus 8 private
  continuations). The same test gives 5/5 at 98.9%.
- **`--device-state-slots 8` is no longer needed** as a workaround for the first defect. The launchers
  stay at one Device slot, which saves 1.3 GiB and a measured decode cost.
- **A release check now refuses to package** when the cache bounds stop retaining the working set, so
  this class of silent shortfall cannot reach a release again.
- **A byte-order mark in an embedded template is now harmless.** One local artifact's chat template
  carried a UTF-8 BOM, and the renderer emitted it as the first character of the prompt: a wasted
  token, and on the vision path a broken prefix reuse. Neither published artifact is affected --
  checked directly, the QUASAR and NVFP4-full templates carry no BOM -- and no launcher loads the one
  that does. The template loader strips a leading BOM now, so any artifact, or any `--chat-template`
  file, is safe.

The request log gained a `policy` field and a `captures` group (`offered`, `no_vacancy`,
`plan_refused`, `infeasible`) naming why a capture was refused.

### Retained from 1.1.0, for context

First Windows release on the **v3 artifact line**, with speculative decoding working on
upstream-shaped artifacts, four measured-optimal launchers, and two production bugs fixed
that earlier builds shipped.

## Requirements

- NVIDIA GeForce RTX 5090 (32 GB, `sm_120a`) and a recent driver.
- **No CUDA toolkit needed at runtime.** The CUDA runtime is linked statically; the engine
  loads from this folder.
- Windows 10/11 x64.

## What runs

| Launcher | Artifact | Spec | Vision | Context | Decode | Draft accept |
| --- | --- | --- | --- | --- | --- | --- |
| `start_quasar_v3_dflash2_vision.bat` | QUASAR QAT | DFlash2 (7) | yes | 262,144 | **262 tok/s** | 45.5% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR QAT | MTP (4) | yes | 262,144 | **188 tok/s** | 57.1% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (9) | yes | 262,144 | **314 tok/s** | 47.3% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP (4) | yes | 262,144 | 163 tok/s | 45.6% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 (7) | yes | 262,144 | **310 tok/s** | 56.7% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP (5) | yes | 262,144 | 180 tok/s | 51.7% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA | DFlash2 (9) | yes | 262,144 | **292 tok/s** | 45.6% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA | MTP (4) | yes | 262,144 | 165 tok/s | 46.7% |

Every number was measured on an RTX 5090 with the exact arguments the launcher passes; every context
ceiling is the highest value the engine accepts for that configuration -- the next step up is refused,
not degraded. Decode is the column with the caveat, measured on 2026-10-13:
`start_swift_v3_dflash2_vision.bat` read 261.6, 290.8, 293.8, 295.5 and 319.3 tok/s in five runs on this
machine - 22% from lowest to highest - and `start_ninfer_v3_mtp4_vision.bat` read 186.6 to 212.8. Decode
moves with the card's thermal and power state, which is machine-specific, so no temperature is a target and
a figure is a sample rather than a constant. Each row is therefore measured the same way - one lane at a
time, after the card has settled, so no row inherits the previous one's heat - and that spacing is a
fairness between rows, not a claim about yours: what your own cooling, ambient and power delivery give is
not something this table can tell you. The comparisons that chose these settings are **interleaved**, both
candidates alternating inside one session, so the state cancels and no decision rests on an absolute
figure. Compare lanes to each other and expect your own absolute figures to differ; acceptance is stable
across sessions, so it is the column to trust in a comparison. Each row carries the identity of what
measured it, that cell's own spread and the run level it was taken at in
`tools/release/lane_figures.json`, which also carries the card state where the run recorded one and which
the pre-commit hook checks; the rows were re-measured on 2026-10-13, after the 2026-09-30 pass this table
was first built from, the 2026-10-07 re-measurement that read 10.3-48.7% lower on seven of them, and the
2026-10-09 pass.
**Three lanes ship `--kv-dtype k8v4`** - `start_quasar_v3_dflash2_vision.bat`,
`start_swift_v3_mtp4_vision.bat` and `start_nvidia_v3_mtp4_vision.bat`; the other five ship `fp8`.
Four lanes moved on 2026-10-13, each on an interleaved comparison at three to six rounds, and each decided
on the **served-length** code cell - which is what the code-weighted bound selects on, because the short
cells' margins invert, so they are disclosed rather than deciding.
`start_swift_v3_dflash2_vision.bat` and `start_ninfer_v3_mtp4_vision.bat` moved k8v4 -> fp8 (long code
222.0 against 205.7, +7.3% against a 0.9% spread, and 149.5 against 139.1, +7.0% against 0.8%),
`start_quasar_v3_mtp4_vision.bat` k8v4 -> fp8 (136.2 against 132.7, +2.6% against 0.7%), and
`start_nvidia_v3_mtp4_vision.bat` fp8 -> k8v4 (145.7 against 136.3, +6.5% against 2.4%). Disclosed with
them, because the rule requires the costs recorded rather than vetoing: swift dflash2 gives up 11.0% of the
short code cell and 7.6% of long Chinese, nvidia mtp4 11.0% of the short code cell and 6.6% of long
Chinese, ninfer mtp4 6.0% of long Chinese, and quasar mtp4 gains 3.1% of long Chinese. Each earlier attempt
at this question measured a hand-rolled protocol - one prompt, temperature 0, no warmup - which ranks
configurations at a sampler nobody serves by, and a fresh fingerprint against a recorded one, which is a
cross-session comparison the interleave exists to replace.
code-weighted bound (evidence). **This block said "re-measured 2026-09-30" and dated
the table to that pass as if it were current** — the figures are now the 2026-10-09 re-measurements,
and `profiles.py` names each earlier set they replaced.

**All four lines are built here now**, so the choice is a measured trade-off rather than one of
provenance. The QUASAR line is the only one whose text weights are a quantization-aware-trained
checkpoint imported unchanged; the other three re-encode their FP8 attention from the BF16 base and
score lower full-corpus perplexity (4.98, 4.92 and 4.90 against QUASAR's 4.99). **The claim that
"the NVFP4-full line also carrying the fastest DFlash2 lane (340 tok/s against QUASAR's 311)" is
stale and is withdrawn**: in the table above the NVFP4-full DFlash2 lane reads **380 tok/s** against
QUASAR's **368**, so the fastest DFlash2 lane in this release is the NVFP4-full one. This sentence
previously read "QUASAR at 382 tok/s" — a depth-7 figure for a lane that moved to depth 9 and, on the merged engine's re-derivation, back to 7
and to `k8v4` — and before that 331 against 319, the same table at an earlier measurement. The
340/311 pair is a superseded measurement that no longer reproduces, in the same way as the withdrawn
figures `profiles.py` names in its own notes. Compare artifacts only on
the full corpus: a `--quick` figure is decided by four streams and is not comparable, as
`docs/perplexity-baseline.md` explains. ADR-0004 records why each DFlash2 lane rides a third-party draft
with no in-house fallback, and Vision is free on every shipped artifact (the with/without comparison is
in [ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md)).

## Getting a model

`download_model.bat` offers the five images, fetches the source checkpoints the one you choose needs at
the revisions pinned in `download_model.py`, and builds it with the converter this archive ships under
`tools/`. There is no prebuilt download and no SHA-256 pin to go stale: the only `.ninfer` is the one
this machine builds. `build_model.py --list` shows each line's recipe and sources without building
anything. Plan for roughly 75-80 GiB of source checkpoints (they are shared between lines) and minutes
of GPU time per image.

Put the `.ninfer` file at `C:\AI\models\` (the path `launcher_env.bat` expects), then double
click the launcher you want. Each launcher checks the engine and the artifact exist before
starting, and leaves the failure on screen if they do not.

## A pre-existing v2 copy must be upgraded first

Neither published repository ships v2 any more, so a fresh download needs no upgrade. A copy
fetched before the republish does: the v3 engine **rejects v2 containers outright** and names
the tool in the error:

```
python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer
```

That upgrade is offline and preserves the weight bytes. On Windows this now works; in
earlier builds the script called POSIX-only `os.posix_fadvise`/`os.fdatasync` and died with
`AttributeError`. The tool ships in this archive, beside the `chat_templates/` data it reads.

## Speculative decoding is not bit-identical to plain decoding

Greedy output differs between no-spec, each MTP depth and DFlash2, deterministically. This
is documented engine behaviour rather than a defect: acceptance compares a proposal token
against the target argmax for its verify column, and the maintainer notes state that
speculation "does not impose token or logits equality between different quantization,
prefill or kernel paths" — the batched verify kernel is not the single-token decode path,
so a near-tie can flip. Speculation measured 3-4x faster (67-83 tok/s without it, 239-343
with it).

## Fixed in 1.1.0

- **DFlash2 on upstream-shaped artifacts.** Our loader demanded a fused
  `dflash2/layers/*/attention/query_key_value` parameter that upstream's converter never
  emits — it groups `attention/query`, `attention/key` and `attention/value` instead. Any
  artifact not produced by our own upgrade path was refused at startup with
  `FATAL missing logical parameter`. The runtime already assembles the fused parent itself,
  so the redundant requirement is gone. The official Qwen3.8-27B and community
  fuller-NVFP4 artifacts now serve with `--spec dflash2`.
- **A graph-capture fault in the NVFP4 A4 TMA route.** The Windows port kept the tensor-map
  descriptor in a device buffer filled from a caller stack frame; under CUDA Graph capture
  that copy becomes a node whose source is gone by replay, so the TMA unit read a dead
  frame and the kernel trapped with an illegal instruction. The descriptor now travels by
  value in parameter space, as upstream does, which is also what makes it replay-safe.
- **Artifact reads blocked writers.** `FILE_SHARE_READ` alone meant Windows refused any
  write to a file the engine had open, where POSIX permits it. Reads now share
  read/write/delete.
- **Python tooling could corrupt artifacts on Windows.** `os.open` without `O_BINARY` opens
  in text mode, which translates CRLF and stops at `0x1A` inside payload bytes, and
  `os.pread`/`os.pwrite` do not exist there. Both are now portable.
- **Context-cache bounds.** The launchers set `--max-shared-prefixes 7
  --max-private-continuations 8 --max-long-anchors-per-continuation 4`. With the engine
  defaults, five distinct prompts resent gave 1/5 round-2 hits at a 19.8% token-level hit
  rate, with four of five re-prefilling in full on every call and no signal; the bounds
  give 5/5 at 99.1% and cost nothing measurable.
- **CORS removed from every launcher.** They passed `--cors` with no `--api-key`, which
  lets any browser page on the machine drive the model and read completions. opencode talks
  to these endpoints as a native HTTP client, so CORS bought nothing.

## Known limitations

- Every shipped profile reaches the full 262,144 context with Vision. Earlier builds capped the
  NVFP4 lane because that artifact carried 19.7 GiB of device weights; the builds this release
  ships carry 16.3-17.1 GiB on their MTP lanes and 17.2-18.0 GiB on their DFlash2 lanes.
- **Upstream tracks an intermittent CUDA illegal-access failure under sustained MTP use**
  ([Neroued/ninfer#208](https://github.com/Neroued/ninfer/issues/208), open). It was reported against
  an upstream commit on 2026-09-07, appears after tens of minutes of sustained agentic decode rather
  than on a single request, did not reproduce with `CUDA_LAUNCH_BLOCKING=1` or with MTP disabled, and a
  comment reports a fix in a long-context fork. It is recorded here because this release ships
  long-context MTP lanes, which is the configuration it concerns. It has not been observed on this
  port: the eight launchers were verified end to end, the 262,144 ceiling probes ran every spec, and
  the lane sweeps held MTP and DFlash2 at the full context across all four artifacts, on this engine
  revision.

## Release numbering

Versions are published in order and never skipped. `v1.0.1` and `v1.0.2` were built on the
maintainer's machine and never released — their archives are still in `C:\AI\releases` — so the
published list reads `1.0.0, 1.0.3, ...`. Nothing was withdrawn, and `1.0.0` is unaffected.

**`v1.2.0` is this release, and it is the first since `v1.1.0`.** Two drafts of these notes called it
`1.2.0` and `1.3.0`; because `1.2.0` has never been published, shipping it as `1.3.0` would leave a
`1.1.0 → 1.3.0` gap in the public list, and a gap reads as a withdrawn release. The two are merged
here under `1.2.0` and the `1.3.0` label is withdrawn. A locally built `v1.2.0` archive from 2026-09-22
predated this work and was deleted rather than left where the filename would find it.

Patch numbers carry fixes to the shipped profile set; a minor number carries a new artifact line,
which is what `1.1.0` is: the v3 container, the measured profile table, and the QUASAR lane.
