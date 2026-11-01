# Documentation audit

Status: complete (2026-11-01) — every finding fixed or carrying a dated correction
Started: 2026-10-31

## Progress

Fixed and verified; `check_profile_consistency.py` still 0 disagreements, `ruff` clean:

- `README.md:95` — GPL → LGPL FFmpeg build (matches `build_windows.bat:9`)
- `README.md:157`, `README.md:301` — 1 GiB → 512 MiB headroom (matches `include/ninfer/types.h:40`)
- `README.md:167`, `README.md:172` — NVFP4 MTP depth 5 → 4; depth 5 belongs to Swift 1.5
- `README.md:251` — `--response-store-max-mib` → `--response-store-max-size` (matches `src/product/config/schema.cpp:105`)
- `AGENTS.md:143` — "four artifacts × {MTP, DFlash2}" → five artifact images over four weight lines
- `AGENTS.md:185` — 35 skills → 41 (the directory holds 41)
- `docs/README.md:3` — "download a published artifact" → build from source checkpoints
- `docs/README.md:46` — "Sections 14 to 19" → 14 to 21 (the artifact doc runs to 21)
- `docs/README.md:69` — the "not translated or modified" rule now names its one exception
- `docs/README.md:107` — "Three directories" → two (the table lists two)
- `CONTEXT.md:73` — the quoted tok/s example now points at the record instead of restating figures
- `tools/release/profiles.py:47` — `--host-context-mib` → `--host-cache-capacity` (the flag the parser defines)
- `docs/performance/lanes.md:67` — "Prefill throughput" replaced: those were decode figures, three of them
  belonging to other lanes. The section now quotes the table's own decode cells and says prefill is not
  ranked here.
- `docs/performance/lanes.md:81` — Swift MTP +6.5% → +2.4%, QUASAR MTP +7.0% → +2.6% (from `tools/release/profiles.py`)
- `docs/performance/lanes.md:94`, `:99`, `:101` — three draft-depth histories corrected against the notes
- `docs/performance/lanes.md:103` — `kMaximumMtpDraftTokens` → `kMtpDecodeMaximumDrafts`
- `docs/perplexity-baseline.md:9` — the tokenizer-rewrite attribution replaced: the probe shows the
  tokenizer identical from 2026-08-27, and the +319 is unattributed
- `docs/perplexity-baseline.md:15` — the 3,390-nat "engine effect" replaced with the unconfounded ~150 nats
- `docs/perplexity-baseline.md:108` — the official-source gap is 1.78 % on both dates, not 1.75 % then 1.78 %
- `docs/adr/0001-v3-container-required.md:38` — the downloader no longer fetches a prebuilt artifact; correction appended
- `docs/adr/0002-speculation-not-bit-identical.md:56` — dead `src/ops/kernel/speculative_round.cuh` → `include/ninfer/ops/speculative_accept.h`
- `docs/adr/0004-vision-only-and-third-party-artifact.md:12` — four profiles → eight
- `docs/adr/0005-per-profile-flags-are-measured.md:7` — `--lm-head-draft` → `--proposal-head` (both mentions)
- `docs/adr/0007-turn-closure-retention-under-host-pressure.md:186` — `--host-context-mib` → `--host-cache-capacity`
- `docs/adr/0009-shared-prefix-served-where-declared.md:39` — log schema 23 → 31; `:115` — dated correction for the retired counters
- `docs/adr/0010-incremental-prompt-encoding.md:62` — both renamed tests repointed
- `docs/maintainer/build-system.md:15` — the `rtx5090` target now says Windows is supported (it is this port's gate)
- `docs/maintainer/build-system.md:156` — the library table gains a Kind column (`ninfer_cpu`, `ninfer_core`,
  `ninfer_cpu_executor` are STATIC) and the loading note distinguishes ELF RPATH from Windows staging
- `docs/maintainer/artifact-conventions.md:210`, `:217`, `:225` — draft "7 or 9"; `--kv-dtype` removed from the
  invariants and described as per-profile since 2026-10-09
- `docs/weight-conversion.md:266` — `fp8_block_maxabs` restored for block FP8; `:467` — `--rows-per-chunk` has no default
- `docs/v2-v3-flag-diff.md:25` — `--cors` exists in v3 (off by default); `:46` — three renamed flags marked as such
- `docs/maintainer/project-map.md` — regenerated from the index (126 modules, 0 control self-edges); the table
  now includes `src/models/common` and `src/models/qwen4_exp/program`, which the hand-aged copy omitted
- `docs/opencode-settings.md:18` — `minimal`/`high` are accepted aliases (mapped to `low`/`xhigh`), not HTTP 400s;
  the snippet at `:153` shows the current alias block and the warning at `:458` is replaced
- `docs/opencode-settings.md:51` — the 32,768-token cap belongs to the run (`--output-tokens 32768`), not the tool's
  default, which is 2048
- `docs/opencode-settings.md:168`, `:182` — `--default-thinking-budget` → `--thinking-budget`
- `docs/opencode-settings.md:201` — `--kv-dtype k8v4` for a lane that ships k8v4
- `docs/opencode-settings.md:213` — model id: 404 `model_not_found`, and only on the OpenAI routes
- `docs/opencode-settings.md:324` — the context error quoted as the engine emits it
- `docs/opencode-settings.md:528`, `:537` — `--host-cache-capacity` (the old name is retired)
- `docs/opencode-settings.md:572` — `--temperature 0`, not `--greedy`
- `README.md:221`, `:225` — the cancel route is `/v1/responses/{id}/cancel` and always 400; `/compact` is 400
- `README.md:230` — the refused-fields table rebuilt from the codes `src/serve` actually emits; the four
  fabricated codes are gone and the supported fields are named
- `README.md:330` — "published under cometkim" replaced with the build-from-sources statement, five images
- `RELEASE_NOTES.md:270` — 380/368 → 314/262 (the table's own cells)
- `RELEASE_NOTES.md:180`, `:185`, `:338` — three retired cache flags marked as retired rather than presented as current
- `RELEASE_NOTES.md:332` — the shared-read claim narrowed: the direct path still opens `FILE_SHARE_READ` (`src/core/io/file.cpp:70`)
- `docs/perplexity-baseline.md:275`, `:406` — `full` is 1,044,557 tokens, not the pre-fix 1,044,876
- **bench/README.md, all nine**: the retired `pp/tg` matrix replaced with the current one-request-set
  contract; the constraint flags described as CLI-scoped and rejected; the fabricated "schema version 24"
  report shape replaced with `report.cpp`'s actual keys; `-r 3` → `--repetitions 3` (both examples);
  prefix reuse corrected to on-for-decode; the corpus-input claim removed; `batch out t/s` replaced with
  `output_tokens / wall_seconds`; `bench/cmake/` path fixed; the missing curve tables clause dropped
- `README.md:149` — "NVFP4-full" → "NVFP4-full no-ex" for the no-exceptions lane
- `tests/README.md:137` — `tools/release/test_v3.cmd` → `tools/scripts/test_v3.cmd`
- `model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md:179` — `--max-new` → `--max-tokens` (two commands;
  the retired flag aborts at argument parsing)
- `model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md:200` — 17.35 → 17.36 GiB (the same file's table)
- `examples/configs/README.md:6` — dead `#get-weights` link replaced with the launcher table and the
  build-from-sources step
- `eval/README.md:186` — the dead model-card link replaced with the fact that neither run directory nor
  the scores are carried in this tree

Deferred with reason: `docs/issue-checklist.md:13` claims a build passed under CUDA 13.4; only v13.3 is
installed. The entry is a dated work item, so it is **annotated** rather than rewritten: a dated note
records that only 13.3 is installed, that all four build trees are configured against it, and that the
13.4 record cannot be checked on this tree or machine.

Remaining: 0. Every finding is either fixed or carries a dated correction that states what the tree
supports.

## Method

All 75 project documents (`.md`, excluding `build/`, `scratch/`, `third_party/`, corpora) were checked by two
methods:

1. **Mechanical** (`scratch/doc_claims.py`): every statement of a lane figure, grouped by lane, to find the
   same fact stated with different values. Result: the three lane tables (`README.md`, `RELEASE_NOTES.md`,
   `docs/performance/lanes.md`) agree on all eight lanes. No drift. A mojibake scan over the same set found 0
   lines.
2. **Fresh-context reviewers** (`egm-codebase-auditor`, no conversation, reading the files and the code they
   cite): six batches — root documents, rules and index, performance, ADRs and designs, maintainer and
   conversion, CLI and configuration.

Each finding carries the evidence the reviewer cited. Only those marked **verified** were re-checked with a
second instrument; the rest are the reviewer's reading and must be checked before the edit.

## What the audit says about the repository

The lane figures are the one thing that is consistent. Everything else drifts in one direction: **documents
describe flags, routes, depths and procedures the code stopped having.** The pattern is the same one that
produced the lane-figure incident: a name was true when written, the tree moved, and nothing compared the two.

## Findings

### README.md — the front door

| Line | Problem | Fix | Status |
|---|---|---|---|
| `README.md:233-241` | Five "refused request fields" rows describe behaviour the server does not have; four named error codes are emitted nowhere in `src/` | Replace with the accepted/refused split in `docs/serving.md:103-119`; drop the four codes | fixed |
| `README.md:221`, `README.md:224` | `POST /cancel` does not exist; `src/serve/http_server.cpp:493` registers `/v1/responses/{id}/cancel`, and `/compact` always returns 400 (`src/serve/openai_responses_http.cpp:616`) | Correct both rows and mark the refusals | fixed |
| `README.md:250` | `--response-store-max-mib` does not exist; the option is `--response-store-max-size` (`src/product/config/schema.cpp:105`) | Rename | **verified** |
| `README.md:156`, `README.md:301` | "1 GiB" startup headroom; the default is 512 MiB (`include/ninfer/types.h:40`, `docs/startup-configuration.md:85`) | 512 MiB in both places | fixed |
| `README.md:94` | Names the **GPL** FFmpeg build; `build_windows.bat:9` stages the **LGPL** one and says why | lgpl-shared | **verified** |
| `README.md:167`, `README.md:171` | "depth 5 on NVFP4" — both NVFP4 MTP lanes ship draft 4 (`tools/release/profiles.py:225`, `tools/release/profiles.py:339`); only Swift MTP is 5 | Correct both statements | fixed |
| `README.md:330` | "Both shipped artifacts are published under the `cometkim` account" — superseded by `docs/adr/0004-vision-only-and-third-party-artifact.md` and the sources-only downloader | Replace with the build-from-sources statement | fixed |

### RELEASE_NOTES.md

| Line | Problem | Fix | Status |
|---|---|---|---|
| `RELEASE_NOTES.md:270` | "380 tok/s against QUASAR's 368" matches no figure in the tree; the table says 314 and 262 | Correct or delete | fixed |
| `RELEASE_NOTES.md:180`, `RELEASE_NOTES.md:338` | Retired cache flags presented as current; contradicts `README.md:193` | Mark as pre-`b9114396`, name `--host-cache-capacity` | fixed |
| `RELEASE_NOTES.md:332` | "Reads now share read/write/delete" is false for the direct path (`src/core/io/file.cpp:70`) | Narrow the claim or fix the code | fixed |

### AGENTS.md, CONTEXT.md, docs/README.md

| Line | Problem | Fix | Status |
|---|---|---|---|
| `AGENTS.md:143` | "eight lanes: four artifacts × {MTP, DFlash2}" — five images, and the pairing is not a cross product | "four lines over five images" | fixed |
| `AGENTS.md:136`, `docs/issue-checklist.md:13` | Two CUDA versions for one machine (13.3 vs 13.4); only 13.3 is installed | AGENTS states 13.3; the checklist entry carries a dated note that only 13.3 is present | **fixed** |
| `AGENTS.md:185` | "35 skills" — the directory holds 41 | Drop the count or correct it | fixed |
| `docs/README.md:3` | "download a published artifact" — the README builds from sources; nothing is downloaded prebuilt | Correct the sentence | fixed |
| `docs/README.md:46` | "Sections 14 to 19" - the artifact doc has 14-21 | 14 to 21 | fixed |
| `docs/README.md:69` | "not translated or modified" — the maintainer index records one port addition | Qualify | fixed |
| `docs/README.md:107` | "Three directories" — the table lists two | Two | fixed |
| `CONTEXT.md:56` | Flag-rename chronology stated three ways; `--host-context-mib` is not a CLI flag | One primary source; call the rest harness arguments | fixed |
| `CONTEXT.md:73` | The document forbids figures, then quotes tok/s that exist nowhere else | Name the record carrying each figure | fixed |

### docs/performance

| Line | Problem | Fix | Status |
|---|---|---|---|
| `docs/performance/lanes.md:67` | The "Prefill throughput" paragraph is **decode** data relabelled; two figures belong to other lanes and 365.6 belongs to QUASAR | Delete or restate from a real prefill cell (~13,000 tok/s) | fixed |
| `docs/performance/lanes.md:81` | Two KV-format gains attributed to the wrong lanes (+7.0% is ninfer MTP's, +2.4% is Swift MTP's) | Swap to +2.6% / +2.4% | fixed |
| `docs/performance/lanes.md:94` | "QUASAR DFlash2: 7 → 9" — ships 7; the note records the revert | "7 → 9 → 7" | fixed |
| `docs/performance/lanes.md:99` | "Swift MTP: 4 → 5 → 4" — ships 5, and the revert was itself reversed | Correct | fixed |
| `docs/performance/lanes.md:101` | NVIDIA MTP depth case cites 223.4/61.4% that `tools/release/profiles.py:341` retracts | Cite the fingerprint or keep the conclusion | fixed |
| `docs/performance/lanes.md:103` | `kMaximumMtpDraftTokens` does not exist; it is `kMtpDecodeMaximumDrafts` | Rename | fixed |
| `docs/perplexity-baseline.md:9` | The tokenizer-change cause (commits and date) is contradicted by the port's own audit record; the +319 is unattributed | Keep the counts, replace the cause, mark unattributed | fixed |
| `docs/perplexity-baseline.md:15` | The 3,390-nat "engine numerics change" is confounded with tokenization; the unconfounded effect is ~150 nats | Quote ~150 nats | fixed |
| `docs/perplexity-baseline.md:273` | `full` sized at the pre-fix 1,044,876 beside a post-fix quick count | Use 1,044,557 | fixed |
| `docs/perplexity-baseline.md:108` | The "official-source gap moved 1.75% → 1.78%" is one number computed two ways | 1.78% on both dates | fixed |

### docs/adr and docs/designs

| Line | Problem | Fix | Status |
|---|---|---|---|
| `docs/adr/0001-v3-container-required.md:38` | Amendment says the downloader fetches the `.v3.ninfer` artifact; it fetches sources and pins no artifact | Mark superseded in part | fixed |
| `docs/adr/0006-context-parallelism-gated-off.md:36` | The decision argues against `cp_enabled`, which exists nowhere; its named test file does not exist | Dated correction appended: the gate and its guard are absent here, the measurements stand | **fixed** |
| `docs/adr/0004-vision-only-and-third-party-artifact.md:12` | "four vision-only profiles" — eight ship | Eight | fixed |
| `docs/adr/0009-shared-prefix-served-where-declared.md:115` | "three counters are permanent" — retired; log schema is 31, not 23 | Dated correction | fixed |
| `docs/adr/0005-per-profile-flags-are-measured.md:7` | `--lm-head-draft` does not exist; it is `--proposal-head` | Rename once, note the former name | fixed |
| `docs/adr/0002-speculation-not-bit-identical.md:56` | Cites `src/ops/kernel/speculative_round.cuh`, which does not exist; the Op is `speculative_accept.h` | Repoint; same claim in `bench/models/qwen3_5/score_token_logprobs.cpp:31` | fixed |
| `docs/adr/0010-incremental-prompt-encoding.md:62` | Both named tests were renamed | Cite `test_exact_repeat_caching`, `test_incremental_turn_splicing` | fixed |
| `docs/adr/0010-incremental-prompt-encoding.md:115` | The 3.69 ms render split is native-render-era; that renderer was withdrawn and Jinja renders at 48.3 ms | Dated correction appended: the split describes a path the tree no longer has | **fixed** |
| `docs/adr/0007-turn-closure-retention-under-host-pressure.md:186` | Cites `--host-context-mib 8192`, retired | `--host-cache-capacity 8192MiB` | fixed |
| `docs/designs/qwen4-27b-rtx5090.md:404` | Compares a ceiling against the NVFP4 trunk payload; README publishes 17.0 GiB, which reverses the sign | Row relabelled as the enabled-weight ceiling, with a note stating both comparisons and that the two bases cannot be mixed | **fixed** |

### docs/maintainer and conversion

| Line | Problem | Fix | Status |
|---|---|---|---|
| `docs/maintainer/build-system.md:15` | Says the `rtx5090` target is Linux-only; this port builds it on Windows | Add Windows, note the bypass | fixed |
| `docs/maintainer/build-system.md:158` | Lists `ninfer_cpu`, `ninfer_core`, `ninfer_cpu_executor` as shared; all three are STATIC; describes ELF RPATH, not Windows staging | Correct the table and the loading note | fixed |
| `docs/maintainer/artifact-conventions.md:217` | `--kv-dtype fp8` listed as an invariant every lane carries; three lanes ship `k8v4` | Drop from invariants | fixed |
| `docs/maintainer/artifact-conventions.md:210` | "dflash2 (draft 7)" — two DFlash2 lanes ship 9 | "7 or 9" | fixed |
| `docs/weight-conversion.md:266` | `fp8_e4m3fn_block128x128_bf16` listed as having no method; `fp8_block_maxabs` exists and is used | Correct the cell | fixed |
| `docs/weight-conversion.md:467` | "`--rows-per-chunk` defaults to 512" — it defaults to None (byte-sized batches), as `docs/weight-conversion.md:184` says | Delete | fixed |
| `docs/v2-v3-flag-diff.md:25` | `--cors` filed as "removed on purpose"; it exists (`src/product/config/schema.cpp:99`) | Reclassify | fixed |
| `docs/v2-v3-flag-diff.md:46` | Three flag names presented as shared; none exists | Mark historical | fixed |
| `docs/maintainer/project-map.md:17` | "Modules by size" omits `src/models/common` and `src/models/qwen4_exp` entirely | Regenerate or date it | fixed |

### docs/cli, startup, opencode-settings

| Line | Problem | Fix | Status |
|---|---|---|---|
| `docs/opencode-settings.md:18` | `minimal`/`high` documented as HTTP 400; the shipped template maps them | Correct the rows and the warning | fixed |
| `docs/opencode-settings.md:49` | Says the 2,048-token cap was raised to 32,768 in the tool; the default is still 2048 | Raise the default or state the flag | fixed |
| `docs/opencode-settings.md:165` | `--default-thinking-budget` does not exist; it is `--thinking-budget` | Rename | fixed |
| `docs/opencode-settings.md:525` | `--host-context-mib` and `--device-state-slots` are not engine flags | Use `--host-cache-capacity` | fixed |
| `docs/opencode-settings.md:198` | Recommends `--kv-dtype fp8` for a lane that ships `k8v4` | Correct | fixed |
| `docs/opencode-settings.md:212` | Wrong model id returns 404, not 400, and only on OpenAI routes | Correct | fixed |
| `docs/opencode-settings.md:321` | Quotes an error string that does not occur in `src/` | Quote the current message | fixed |
| `docs/opencode-settings.md:569` | `--greedy` does not exist; it is `--temperature 0` | Rename | fixed |

### docs/serving.md (fixed)

| Line | Fix | Status |
|---|---|---|
| `docs/serving.md:1092` | six Host phases (`session_call`, `executor_poll`, `executor_flush` added); scheduler yield separated | **fixed** |
| `docs/serving.md:931` | the warning carries no subject; the three fabricated fields are gone | **fixed** |
| `docs/serving.md:1031` | replaced with the four fields `ToolCallParseDiagnostics` actually has | **fixed** |
| `docs/serving.md:962` | `ninfer_device_wait_seconds_total` → `ninfer_scheduler_yield_seconds_total` | **fixed** |
| `docs/serving.md:965` | `constraint_draft_wait` → `executor_flush` | **fixed** |
| `docs/serving.md:1157` | "five phases"/"Device wait" → six phases / scheduler yield | **fixed** |
| `docs/serving.md:1220` | the path label is printed only for checkpoint reuse | **fixed** |

### docs/maintainer/qwen3.8-27b-artifact.md (newly audited)

| Line | Problem | Fix | Status |
|---|---|---|---|
| `:337` | `text/draft_head*` are not the bound names; the tree binds `proposal/head`, `proposal/token_ids` | Renamed in all 15 occurrences | **fixed** |
| `:259` | frontend resource objects omit the `resource/` namespace | Prefix added to all six and to the namespace bullet | **fixed** |
| `:352` | `mtp/layer/` → `mtp/layers/0/` | Renamed all 18 occurrences | **fixed** |
| `:21` | "only the two v3 fork artifacts" — five images ship | Now "five v3 images across eight lanes" | **fixed** |
| `:240` | "Object name" columns are binding names; physical objects are `weight/NNNNNN` | Replaced the write-order list with the three real blocks (resources, auxiliaries, weights) and stated that the names are binding names | **fixed** |
| `:221` | divisors are documented as interleaved; they form a separate auxiliary block | Corrected to "written in the auxiliary block that precedes the weights" | **fixed** |
| `:489` | the alias table omits the DFlash2 context projections | Two rows added: context key/value share `attention/key` and `attention/value` (`tools/convert/official_recipes.py:71`) | **fixed** |
| `:896` | the documented conversion entry point and flags do not exist | Both blocks now use `python3 -m tools.convert --recipe … --source NAME=PATH` (`tools/convert/__main__.py:114-149`); `qwen3_8_27b/inventory.py` and the second `-m` path retargeted; a third stale path (`nvfp4_encode.py`) found by grep and fixed | **fixed** |
| `:575` | the source revision `60e813d4` was force-pushed out | Resolved by the Hub API: `unsloth/Qwen3.8-27B-nvfp4` resolves to `Qwen3.8-27B-NVFP4` at `f0b7c9e7`; §10.1 now pins that and marks the other two superseded | **fixed** |
| `:835` | "exactly 81 tensors, no extra accepted" — the reader validates only what it reads | Now says a missing or mistyped tensor fails while an extra one is ignored (`tools/convert/sources/safetensors.py:112-121`) | **fixed** |

### bench/README.md (newly audited)

| Line | Problem | Fix | Status |
|---|---|---|---|
| `:128` | the `pp{P}`/`tg{G}` matrix no longer exists; the runner is retired | Replace with the current one-request-set contract | fixed |
| `:184` | constraint flags are `Scope::Cli` and rejected by `ninfer_bench`; `--mixed-constraints` has no implementation | Delete or extend `applies_to` | fixed |
| `:1663` | "schema version 24" and its object list are not what `report.cpp` emits | Rewrite from `report.cpp` | fixed |
| `:164` | `-r 3` fails; only `--repetitions` exists | Rename in both examples | fixed |
| `:138` | "all requests disable prefix reuse" — only prefill-only does | Correct | fixed |
| `:124` | the benchmark does not read `bench/fixtures/bench_corpus.ids` | Point at `--prompt`/`--messages` | fixed |
| `:189` | `batch out t/s` is not an output field | Correct | fixed |
| `:33` | helper is at `bench/cmake/NinferBenchmarks.cmake` | Fix the path | fixed |
| `:311` | "the final curve and metric tables" do not exist | Drop the clause | fixed |

### docs/upstream-reports, docs/features, upstream survey (newly audited)

| Line | Problem | Fix | Status |
|---|---|---|---|
| `docs/upstream-reports/fork-divergence.md:74` | four paths in the reconciliation table no longer exist (`file_io.cpp`, `linear_swiglu/…plan.cpp`, the W4A4 headers), so the documented `git diff` step returns empty | Retargeted to `src/core/io/file.cpp`, `src/ops/linear/swiglu/nvfp4/shapes/n34816_k5120/`, `nvfp4_a4_tma.cuh`, `tma_launch.h`; the `ReadFile` description corrected to `CreateFileW` | **fixed** |
| `docs/upstream-reports/ninfer-tma-descriptor-graph-capture.md:74` | presents a deleted test, header and symbol as the current explanation | Replaced with what the tree supports: the test, header and symbol were deleted with the rewrite | **fixed** |
| `docs/upstream-reports/README.md:61` | the 250 ms grant is described in the present tense after its file was retired | Now says it was not re-applied | **fixed** |
| `docs/upstream-reports/ninfer-reasoning-effort-mismatch.md:13` | quotes a `docs/serving.md` sentence that does not exist; a code line moved 840 → 780 | Citation deleted; line corrected | **fixed** |
| `docs/upstream-reports/ninfer-tma-descriptor-graph-capture.md:25` | every identifier in the reproduction was renamed (A4, not W4A4) | Test and shape paths retargeted; a rename note added before the trace | **fixed** |
| `docs/upstream-reports/README.md:37` | describes #82 as open; the survey records it closed | Corrected to "#233 (`#82` is closed and was not merged)" | **fixed** |
| `docs/upstream-survey-2026-09-18.md:119` | prints `--lm-head-draft`, which the tree no longer parses | Dated correction naming the three renames | **fixed** |
| `docs/features/msvc-test-constexpr.md:9` | points at the apps tree, which is configured with `BUILD_TESTING` off | Now names `tools/scripts/test_v3.cmd` and says why not `build/` | **fixed** |

### tools, tests, eval, model-cards, examples (newly audited)

| Line | Problem | Fix | Status |
|---|---|---|---|
| `eval/README.md:174` | "starts a fresh local server" two paragraphs after saying the harness is gone | Replaced with the two `ninfer-serve` invocations | **fixed** |
| `tools/README.md:20` | the commit-gate table lists 10 rows and calls them "the twelve"; five hook gates are missing | All 15 rows listed; the hook is named as the authority; the two conditional gates marked | **fixed** |
| `tests/README.md:140` | "builds the tests it perturbs first" — the gate dry-runs the build and refuses a stale tree | Now "dry-runs the build (`cmake --build … -- -n`)" | **fixed** |
| `eval/README.md:223` | the "agrees with 0.50%" claim has the opposite PPL sign on the same page | Now states the accuracy direction agrees and the paired-PPL direction does not | **fixed** |
| `eval/README.md:43` | "Two datasets are deliberately absent", then three are named | "Three datasets" | **fixed** |

## Coverage

All **76** project documents have been read by a fresh-context reviewer: the 75 `*.md` outside `build/`,
`scratch/`, `third_party/` and the corpora, plus `.github/pull_request_template.md` — a 76th that a plain
glob misses because it is in a hidden directory. Two files are records of that work rather than subjects of
it: this report, and `docs/designs/acc-tolerance/review-report.md`. The design those records describe,
`docs/designs/acc-tolerance/design.md`, was read by the drift check that produced the corrections in the
section below.

## Re-audit (2026-10-31)

After remediation, three fresh-context reviewers re-read the changed files with one instruction: find what
the fixes broke. They found eighteen defects in the fixes themselves, including one that would have shipped
a command that still could not run.

| Finding | Fix | Status |
|---|---|---|
| The §12.4 conversion command passed `--source base=`, which `tools/convert/__main__.py:177-178` rejects, and omitted the required `--model` | Rewritten with `--model`, `--components text,vision,mtp,dflash2` and `--proposal` | **fixed** |
| The §13 command had the same defect | Rewritten the same way | **fixed** |
| The two new alias rows named `attention/key` and `attention/value` objects that do not exist | Now name the key/value row views of `attention/query_key_value`, matching §8.1 | **fixed** |
| The rewritten weight-block order contradicted the writer: fused parents are emitted first, not model order | Corrected to `packing_groups` order, then unfused parameters, with `proposal/head` last | **fixed** |
| "Batch throughput is not a named field" — `output_tokens_per_wall_second` exists | Named the field | **fixed** |
| "JSON and CSV identify DFlash2" — the CSV has no backend column | Restricted to JSON `configuration` | **fixed** |
| The bench synopsis omitted `--spec none` | Added | **fixed** |
| `README.md:135` still described four artifacts in both routes | Rewritten for five images over eight lanes | **fixed** |
| `README.md:221` promised 400 for the cancel route, which returns 404 first for an unknown id | Both statuses stated | **fixed** |
| `README.md:230` said "anything not listed is accepted" beside a non-exhaustive list | Reduced to "common refusals" and pointed at `docs/serving.md` | **fixed** |
| `README.md:159` device weights matched no record | Now 16.3 GiB (QUASAR MTP) and 17.1 GiB (NVFP4-full), per `docs/maintainer/artifact-conventions.md` | **fixed** |
| `docs/perplexity-baseline.md:149` repeated the 1.75% → 1.78% claim three lines after its correction | Both dates now 1.78% | **fixed** |
| `docs/opencode-settings.md:21` still said the template rejects `minimal`/`high` | Corrected | **fixed** |
| `docs/v2-v3-flag-diff.md:7` still named `--host-context-mib` | Corrected | **fixed** |
| The model card still opened with `hf download cometkim/...`, the premise the audit called superseded | Replaced with `download_model.py` and `build_model.py --line nvfp4qat` | **fixed** |
| `docs/maintainer/artifact-conventions.md:204` said "two lanes over one artifact", false for the unsloth line | Corrected for the five-image split | **fixed** |
| `docs/performance/lanes.md:103` attributed a depth decision to the run the record says invalidated it | Rewritten to say the value is retained, not re-decided | **fixed** |
| `docs/maintainer/project-map.md` lists `scratch/` and `eval/` although its header says those are excluded | Added both to `SKIP_DIRS` in `tools/release/project_map.py`, regenerated: 124 modules, neither directory listed | **fixed** |

## Fix order

1. **README.md** — the front door; three verified blockers.
2. **AGENTS.md / CONTEXT.md / docs/README.md** — the rules and the index.
3. **docs/performance** — published claims about measured behaviour.
4. **ADRs** — dated records; corrections are append-only, never rewrites of the decision.
5. **maintainer / cli / opencode-settings** — reference material.
6. **RELEASE_NOTES.md** — a dated record; mark superseded, do not rewrite history.

Each edit is checked against the cited source before it is applied. A finding the code contradicts is a
suspected bug in the code, not a licence to change the document (AGENTS.md).
