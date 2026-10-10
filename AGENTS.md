# AGENTS.md

## Product and architecture

NInfer is a C++/CUDA inference engine for maximum single-GPU performance, targeting `sm_120a`
on NVIDIA GeForce RTX 5090. Choose designs for functional and numerical correctness, clear
ownership, and the performance goal within the requested scope.

It implements `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`. Checkpoints and recipe combinations
of existing representations use the same architecture, binding and execution path; do not add
checkpoint-specific execution registration. Generation uses one GPU, one resident model, and
one to eight resident execution lanes fixed at startup.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving, and
inference benchmarks use the public Engine. There is no Python model-inference route or
installed/exported C++ SDK.

This is a local, single-owner project with trusted local models, artifacts, and workflows.
Do not derive requirements from another deployment or trust model.

Model data is immutable; every Program owns its mutable state and device allocations. The loader
validates, uploads, and binds the stored representation; actual Parameters and Ops determine
execution support, without a whole-artifact capability registry.
Runtime owns execution and publication policy; product/serving own input acquisition and protocol
translation. Model code does not acquire media or own transport.

Keep architecture, binding, and execution explicit. Without a product requirement, do not introduce
generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, or runtime weight repacking. New mathematical architectures, execution platforms,
large-scale continuous batching, or priority/QoS require an explicit product change. When a task
changes product or architecture, update affected contracts and implementation together.

## Change policy

Project-owned contracts do not preserve backward compatibility. Replace behavior completely:
remove superseded aliases, fallbacks, transition branches, and their tests within the affected contract.
Advertised OpenAI and Anthropic protocols are external contracts; update affected schema tests
and serving documentation together.
Update stable requirements in their existing authoritative document; maintain one current authority.
Use Conventional Commit subjects with concise lowercase types for every commit.

## Verification and reporting

Before changing Ops, numerical or state semantics, read
[Op development](docs/maintainer/op-development.md). Qualify affected production routes directly
against the contract's independent mathematical oracle or specified exact reference. Kernel parity
and plausible model output do not establish mathematical correctness.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result,
not an end-to-end improvement. For comparisons, establish the baseline, workloads, metrics,
aggregation, and acceptance criteria before evaluating results.
Distinguish new capability, fallback replacement, and improvement to an optimized implementation.
Report the baseline, hardware/toolchain, workloads or commands, run conditions, metrics, coverage,
result distribution, worst changes, and exceptions. Include small and unexplained regressions;
do not dismiss slowdowns as noise without evidence.

### Before proposing anything (user directive, 2026-09-27)

Do the research and re-check the work *first*, then propose. On 2026-09-26/27 a single session
proposed, in order: that a wider verification round "roughly doubles the target's work" (measured
+5.7 %); that the extra graph family costs ~480 MiB (measured +288 MiB); that the feature would cost
"17-23 % of headroom" (wrong on both counts); that "no shipped engine combines a neural drafter with
n-gram drafting" (TRT-LLM and vLLM ship selection, FastDeploy ships concatenation); that byte-identity
at temperature 0 is the correct correctness gate (the neural route alone already diverges 0.25-0.75
nats from non-speculative decoding); that a host/device synchronisation sat in the middle of a round
that only needed a 512-byte field on an existing copy; and that n-gram alone is a viable drafter (it
rejects 91-98 % of rounds). Each was reasonable from the material at hand and each was wrong. The
recurring cause is proposing a conclusion before the check that would establish it, and a reviewer
should treat an unsourced number or an unexplained "this is why" as a defect in the proposal rather
than as a summary of it.

Concretely, before stating a number, a design conclusion, or a causal claim:

- **A number about cost, time or memory is measured, or labelled an estimate.** Compute what a design
  *implies*, then measure it. Do not let the arithmetic stand in for the figure; three separate cost
  claims in one session were falsified exactly that way.
- **A claim about another project is read to its mechanism, not taken from a headline.** The decisive
  fact about a design is usually in the code or the commit, and a feature request's premise can be
  wrong -- llama.cpp #23184 asked for chaining that already existed in the tree at the commit before
  it was filed.
- **A claim about this tree is re-read, not recalled.** Name the file and line. This is also why a
  proposal that quotes our own code without re-reading it is suspect.
- **Search before proposing an approach at all**, and say which sources were used. If the answer is
  that nobody has done it, that absence is the finding and it is worth stating as such.
- **A test result is only evidence if the setup could have passed for the right reason.** Confirm the
  case under test actually reached the branch it names. Twice in one session a harness omitted an
  argument -- a model path, then a prompt -- so every "refusal" was for the wrong reason, and a third
  time an index-based CSV reader turned real data into clean zeros.
- **Say which of the three it is**: measured here, read in the source, or reported by a third party.
  An estimate presented as a measurement is the failure this section exists to prevent.
- **A measurement that decides a shipped setting has to outlive the session that took it.** A decision
  resting on a sweep run from a scratch script is unverifiable the moment that script is gone, and it
  can then be neither reviewed nor withdrawn. In one session, two sweeps that reversed a shipped
  configuration were taken from temp scripts and never persisted -- they were in no document, not in the
  bench's own record file, and not under `profiles/` -- so the change they contradict could be neither
  confirmed nor reverted, and had to be recorded as unresolved instead. Land the numbers, then decide.
  `tools/release/v3_profile_matrix.py` now writes the workload and the sampling beside every figure for
  exactly this reason.
- **A control that agrees for a structural reason is not a control.** Before reporting one, name what
  would have to be false for it to fail. At `top_k = 1` the target's support is one token, so `p` is a
  point mass, so `p >= q` holds for every draft whatever `q` is and "accept always" is correct -- which
  is why the sparse accept cases agreeing there said nothing about the accept rule, and why a
  multi-token disagreement I reported was later found to be my own harness. A configuration in which
  the code under test is degenerate cannot corroborate it.
- **Attribute a disagreement before reporting it.** When a device and an oracle disagree, the third
  possibility is the harness. One session reported such a disagreement as a kernel-or-oracle
  divergence; it was the throwaway harness and the cause was never identified. Four more harness
  errors that session were caught before they reached a report -- a draft that was deterministically
  accepted so the branch under test never ran, Op-maintained token counts drifting between trials, a
  probe that passed the wrong draft and seed to the oracle, and a scratch script whose `cd` walked out
  of the repository so the suite's exit code came from a failed `call` and read as a test failure.
  "Unattributed" is a reportable result; picking the side that is easier to describe is not.

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.
Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references, `git diff --check`, and `tools/release/check_text_encoding.py` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

### Tooling, and when to reach for it

Reach for the tool that fits the question; each of these exists because a hand-rolled check missed
something, and each is named here so it gets used rather than rediscovered.

| situation | tool |
|---|---|
| every commit | `.githooks/pre-commit` — enable once with `git config core.hooksPath .githooks`. **Fourteen** gate scripts (`check_doc_links.py`, `check_doc_citations.py`, `check_text_encoding.py`, `check_fp8_band_ladders.py`, `check_profile_consistency.py`, `check_calibration_corpus.py`, `check_production_stream_defaults.py`, `check_rule_count.py`, `check_dead_types.py`, `check_duplication.py`, `check_port_delta.py`, `check_skills.py`, `check_subprocess_encoding.py`, `check_lane_figures.py`), plus the opencode plugin load check and the claims-gate logic harness when `node` is on PATH, then `pytest tests/convert` plus six named test files, then `ruff check` and `mypy`: seconds, no network. **This row previously said "Doc links, profile consistency, converter tests", which is three of the fourteen steps** — it omitted the calibration-corpus gate, the stream-default ratchet, the rule count, the citation gate, the dead-type ratchet, the duplication baseline, the port-delta ratchet, the skill-discovery gate, the subprocess-encoding gate, ruff and mypy. **Counts in this row have gone stale twice, so read them as a summary and the hook as the authority** |
| a declared type nothing can reach | `check_dead_types.py` — a `struct`/`class`/`enum` whose name occurs exactly once in the tree, at its own declaration. **Types only**: "no textual reference" stops meaning "no use" for a macro or a template instantiation, and a gate that guesses gets muted. It found the copy-drafting withdrawal, and it is a **ratchet**, not a detector: it stops the surface growing rather than proving the code reachable |
| copy-paste, and whether it is *new* | `check_duplication.py` — jscpd 5.4.0 over `src/` and `include/`, against `.jscpd-baseline.json`. **The baseline is the point**: 378 clones already exist here (3.86% of lines), so a threshold gate would fail on every commit and get muted. It fails only past 5 *new* clones, and names them. Three exits, and the middle one matters: 0 clean, 1 new duplication, 2 **jscpd missing or analyzed nothing** — a scan that did not run is not a pass. Install with `uv tool install jscpd==5.4.0`. Measured: 490 files in ~65 ms; at `--min-lines 20`, 9 clones (0.36% of C++ lines) |
| a document cites `file.ext:LINE` | `check_doc_citations.py` gates the resolvable ones; it names a citation pointing past the end of a file that exists, and only *reports* one naming no tracked file, because `docs/research/` cites other projects by construction and a gate that cannot tell an external reference from a dead one is guessing |
| a check that passes here and fails in CI | `tools/scripts/verify_as_ci.cmd` **first**, before forming any hypothesis. It reproduces the runner's conditions — Python 3.11, CI's package set, `NINFER_PYTHON` as a command name, a scratch venv. On 2026-09-25 six serious hypotheses were formed against a failing CI gate without once reproducing the runner's conditions, and five were wrong; the sixth was found in one run of this script |
| a change to a workflow file | **dispatch it** — the runner's shell is part of the change, and running the same commands locally does not reproduce it. A `run:` block on Windows is PowerShell, so cmd syntax needs `shell: cmd`; the 2026-10-06 `REM` failure was found only because the tier was dispatched, while the local run of the same commands (from a `.cmd`) passed |
| a C++ or upstream change reaching the suite | `tools/scripts/test_v3.cmd`, then `tools/release/check_test_baseline.py` — **with `NINFER_TEST_ARTIFACT` set**: without it the four required real-model tests skip and the gate fails on missing coverage rather than on a regression, which is how it was misread once |
| anything that could be order- or state-dependent | the suite recipe passes `--schedule-random`; run it twice before believing a fixed order |
| a device-side memory, race or synchronisation question | `tools/scripts/test_v3_compute_sanitizer.cmd` — memcheck on a small subset; `racecheck`/`initcheck`/`synccheck` and the wider method are in the `cuda-debugging` skill |
| Measure a serving lane's phase split and whether its prefix cache engaged | `tools/bench/report_serve_phases.py` — reads `request-log-jsonl`, never a wall clock. `prefix_cache_hit_tokens` is the field that says whether a cache effect exists to explain anything |
| a speculative acceptance or decode comparison between artifacts | `tools/bench/realtext_acceptance.py` — real text at a lane's own flags, greedy so the arms are comparable, model id read per arm from `/v1/models`. The bench corpus cannot answer this from either end of its range: one-token seed reads 0.085-0.129 and a real prompt on the tiled corpus saturates at 0.991, while real domains sit at 0.44-0.55 |
| a figure quoted from a captured request log | `tools/release/check_request_logs.py --logs "<glob>"` — validates the log against the schema `docs/serving.md` documents, and that each phase fits inside the clock it is measured in (`prefill <= ttft <= total`). Run it on the run you are about to quote: it cannot be a commit-time gate, because the logs it checks live outside the repo, which is also why it sat unwired |
| `prepared` and TTFT against conversation size | `tools/bench/warm_lane_sweep.py`. **Seed the prefix with a shorter conversation, not the same one twice** — an identical repeat takes `private_response_replay`, returns a stored response and never decodes, so its TTFT is replay latency |
| whether a first request is slower than later identical ones | `tools/bench/first_request_lane.py`. `--temperature` picks the question and `--seed` must be pinned, because an omitted seed is replaced per request with a fresh random one |
| whether a prefix cache is serving a shared prefix | `tools/bench/check_shared_prefix_reuse.py`. **Never conclude from zero hits alone**: it needs an exact repeat that hits (else it exits 2) and a shuffled conversation that does not. **Read a lane's `path`, not its hit count**: `/v1/chat/completions` reuses through `private_response_replay` whether or not a shared candidate was published, so only a `shared_stable_prefix` path settles it, and an unmarked OpenAI arm correctly reads `root`. Both boundary locations resolve on the native path as of 2026-10-03, so the marker location is no longer a candidate explanation. And the two requests must be *different conversations*: seeding with a shorter version of the same one takes `private_response_replay`, which never decodes |
| a host-side lifetime question | `tools/scripts/test_v3_asan.cmd` — ASan cannot instrument device code, which is why the two recipes are separate |
| a crash with no message (`0xC0000409`, event `BEX64`) | build the target with `/Zi` and read the dump in `%LOCALAPPDATA%\CrashDumps` with `cdb -z <dump> -cf <file>`. **A fail-fast bypasses a live debugger**: `cdb -c "sxe ..."` never fires for it, which is how one session spent hours unable to interrupt a process that a dump then explained in a single read. Pass `-cf <file>`, never `-c` — a command list is exactly what the shell splits |
| a kernel's performance | the `ncu-report` skill, records under `profiles/ncu/` and `profiles/nsys/`. **`profiles/ncu/` exists on this machine (`gdn_decode`); `profiles/nsys/` does not, and `profiles/` is gitignored in full, so its absence here is not evidence the layout is wrong — the path is unverifiable from the tree and is kept on the skill's own authority** |
| a host-side C++ question | `clang-tidy -p build src/text/jinja.cpp` — `.clang-tidy` sets a narrow check set and `build/compile_commands.json` already exists; run it from the Visual Studio environment so the MSVC headers resolve |
| Python tooling, before committing it | `ruff check tools tests` and `mypy tools/release tools/convert tests` — both clean, both enforced by the hook, so a new finding is a regression rather than a cost |
| a lingering suspicion of flakiness | `ctest --test-dir build-test --repeat until-fail:5` |
| what upstream already decided, whether a symptom is known, **or what is proposed and unmerged** | the upstream tracker, **both halves**: `gh issue list --repo Neroued/ninfer --search <term>` **and `gh pr list --repo Neroued/ninfer --state open`** — this project's reference corpus. Survey both: the open pull requests are the unmerged half, and six of one fork's sixteen engine changes (prompt-attention kernels, compact KV formats, overlapped decode, YaRN, agent-client compatibility, a cache redesign) were readable there for a month while only issues were searched, because an issue search cannot return a PR title and no artifact survey can surface an engine change |
| **an unfamiliar subsystem, a wrong belief about the code, or a claim that a fix works** | **a skill — see [Skills](#skills-and-when-to-reach-for-one) below** |

CI is installed in two tiers, split by what needs the card. `.github/workflows/ci.yml` runs
`bash .githooks/pre-commit` on `windows-latest` for every push and pull request to `dev`/`main`, so
the fast gates have one definition and CI cannot drift from the hook. `.github/workflows/gpu.yml`
runs `test_v3.cmd` and then the baseline gate on a **self-hosted** runner labelled `gpu-5090`, on a
nightly cron and on dispatch, serialised against measurements by its `concurrency` group.
GitHub-hosted runners cannot take the second tier: they have no GPU and no CUDA toolkit.
**The runner is registered and the tier has run** (2026-10-06: `5090-box`, online, dispatches green —
the gate reads the recipe's own captured log, so the suite runs once). Three preconditions are easy to
lose, and all three are now measured: the workflow must exist on the **default branch** for
`schedule`/`workflow_dispatch` to fire at all, which is why the default branch is `dev` rather than
the release-only `main`; the runner is a process, so it needs a per-user logon task
(`Register-ScheduledTask -AtLogOn`; `schtasks /create /sc onlogon` is an any-user trigger and is
denied without elevation) to survive a reboot; and **the cron can be hours late** — the 04:00 slot on
2026-10-07 ran at 10:55:59 UTC. `gpu.yml`'s header carries all of it, the FFmpeg staging the checkout
needs, and the registration commands. This paragraph previously read "Not installed deliberately: CI",
which the two-tier commit `88ee28f0` had already made false, then "no self-hosted runner is
registered … never executed", which the first dispatch falsified, and then "the schedule is not firing
at all", which the run six hours late falsified.

### Skills, and when to reach for one

**Before the first edit of a task, name the skill you loaded, or say none applied.** One line.

| situation | skill |
|---|---|
| a C++/CUDA change on its way in | `cpp-cuda-review` |
| a hard bug, or a belief about the code that might be wrong | `diagnosing-bugs` |
| before claiming a fix works | `principle-prove-it-works` |
| "how does X work", or an unfamiliar subsystem | `how` |
| `AGENTS.md` or an ADR may have gone stale after a change | `rules-check-drift` |
| a design with no precedent in this tree | `principle-exhaust-the-design-space` |
| a claim about another project | `research` |
| a multi-phase change needing an auditable trail | `show-me-your-work` |
| a claim you are about to write into a document | `claims-gate` (`.opencode/plugins/`) blocks the **edit** on a citation that is provably wrong — missing file, or line past EOF. `node --experimental-strip-types tools/opencode/test_claims_gate.mjs` (14 cases, both directions) |
| a rule you keep forgetting rather than one you keep breaking | `rules-inject` re-injects **three** rules on every model call via `session.hook("context")`. Injection is not enforcement — a gate *denies*, and forgetting is not what a denial fixes. `tools/opencode/test_rules_inject.mjs` (17 assertions, **including the ≤3-rule cap**, so "small and followed" cannot quietly become "complete and ignored") |

### Two rules for correcting a document

**A document's invariant that the code breaks is a suspected bug in the code.** Do not edit the
document to match the code; that buries the defect under a lie that now looks deliberate. Report it
and change neither side until it is settled.

**Never confirm a document claim without the evidence in hand, and say what you checked.** "Looks
right" and "I read it earlier" are not acks. Open the cited line, name what you found there, and if
the claim is behavioural rather than textual, run it.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md`; `docs/maintainer/project-map.md` for the generated module graph, blast radius and module cycles |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` — both written in Chinese upstream, so read `docs/maintainer/README.md` first for an English summary of what each decides |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| A maintainer document written in Chinese upstream | `docs/maintainer/README.md` indexes all seven and states why they are not translated |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or
ownership. [README](README.md) and executable `--help` define capabilities and exact commands.
The [documentation map](docs/README.md) routes to detailed contracts;
[Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands and execution details.

## Local environment

Use Python 3.11 via `/home/neroued/miniconda3/envs/py311/bin/python` explicitly; use `python3` only
after selecting the maintainer environment or checking its version.
The usual local model is `out/qwen3_8_27b_nvfp4.ninfer`. Select artifacts by explicit path, never glob order,
modification time, or unqualified “latest”. Source checkpoints and large artifacts are prerequisites;
download or regenerate them only when that work is in scope.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
*(This paragraph is upstream's Linux text and the Windows section below disclaims it. Two of its
claims are false on this machine and are recorded here rather than edited, because editing them would
make the upstream section silently Windows-specific: the toolchain is **CUDA 13.3**, not 13.1 —
`build/CMakeCache.txt` has `CMAKE_CUDA_COMPILER=…/CUDA/v13.3/bin/nvcc.exe` and
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` holds `v13.3` only; and of the three `profiles/`
paths only `profiles/ncu/` exists, `profiles/perplexity/` being the other populated one. `profiles/`
is gitignored in full, so the two missing directories are unverifiable from the tree rather than
known-wrong, and `profiles/bench/` is independently attested by `bench/README.md`, which writes its
CSVs and JSON there.)*
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Commit and push when it is deemed necessary and nothing in flight is disturbed: land a coherent unit
of work with its evidence, and push `dev` so the work is not one disk failure from gone. Do not commit
or push while a build, a suite run or a measurement holds the tree, and never rewrite published
history. Use Conventional Commit subjects with concise lowercase types such as `feat`, `fix`, `perf`,
`bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## Working practices (Windows port)

This fork is the Windows port. The Linux paths in Local operations above do not apply here: the
port targets MSVC 14.51 and CUDA 13.3, and the Python used for tooling is
`C:\vllm-env\Scripts\python.exe`. What ships is governed by `tools/release/profiles.py`, and the
release surface is documented in the Windows section of `README.md`. There are **four** build trees
on this machine, all gitignored by the `build-*/` rule: `build/` for the apps and the bench,
`build-test/` for the suite the release gate runs (`ctest --test-dir build-test`), `build-asan/` for
the five host-only sanitizer tests (`CMAKE_CXX_FLAGS=/fsanitize=address /Zi`; `test_v3_asan.cmd` sets
`TESTS` to five names and skips a sixth, `ninfer_context_cost_test`, as a diagnosed failure), and
`build-bench/` for
the standalone bench binaries. **This sentence previously said "two build trees" and named only the
first two.** `build-asan` is not new — the sanitizer rule below already runs against it — and
`build-bench/` exists for the bench; both are listed because a rule that names a directory should say
which of the four it means.

- **A phase field is only what its own definition covers.** `prepare_seconds` spans the whole frontend,
  from `prepared.lifetime->started` to just before `submit`, so it includes render, tokenize and
  layout. Read that before comparing it to a figure derived from one of those terms alone: ADR-0012
  projected `prepared ≈ 8 ms` by treating a 42 ms render saving as the substance of `prepared`, and
  measurement put `prepare` at 12.21 ms for a *quarter* of the projected conversation size, rising
  with the prompt. The render was ~1.4 ms of that 12.21. A projection built by subtracting one term
  from a total is a claim about the other terms, and it had never been checked against them.
- **Say which process a number came from.** The same renderer measured 5.99 ms in a fresh process and
  48.3 ms in a long-lived server on this machine, a 40x gap that is the process, not the input. Two
  lanes with identical request bodies produced different text for a reason in neither: they differed in
  cache capacity, which changes the prefill the plan chooses. A harness that does not record its own
  configuration cannot be compared with one that does, and a repeated run does not test that.

Sixty-six rules, each earned by a failure rather than chosen:

- **Run a verification recipe through the recipe.** `ctest --test-dir build-asan -R <broad regex>`
  pulls in device tests, which ASan cannot instrument and which hang: one such run burned fifty
  minutes before its timeout. `tools/scripts/test_v3_asan.cmd` names its host-only tests for
  exactly that reason and says so in its header. The recipe's scope is part of the recipe.
  **This rule said "six", then "five"; the recipe sets `TESTS` to three**
  (`ninfer_resource_manager_test`, `ninfer_artifact_reader_test`, `ninfer_kv_capacity_test`)
  **and separately skips `ninfer_context_cost_test`, as a diagnosed 2026-09-20 failure.** Three run,
  four are named. It said five until 2026-10-04, when the upstream context-cache rewrite deleted
  `ninfer_admission_policy_test` and `ninfer_materialization_budget_test` along with the subsystems
  they tested; the recipe still named them, so its `--target` list could not build.
- **Check before you package, not after.** A package built before its review has to be re-cut and
  re-packaged: this session's was, three times, because a code review and the sanitizer run both
  landed afterwards. "The archive is cheap to regenerate" is the reason to check first, not to
  skip the check.
- **`git tag -f` tags HEAD, so create a release tag while on the release branch.** Tagging from
  `dev` put `v1.1.0` on a dev commit; the trees were identical, which is why only the commit
  pointer gave it away. Likewise, do not swallow a command's output with `| Out-Null` when its
  success is the thing you are checking.

- **The checkout must not decide the bytes a comparison sees.** `.gitattributes` pins every
  tracked text file to LF except `*.bat` and `*.cmd`, which are CRLF by definition. A
  generator that digests a file must write it platform-independently (`Path.write_text`
  translates `\n` to the platform separator by default): `build_fixtures.py` once digested
  CRLF bytes on Windows, so 18 of 26 TTFT digests described bytes no LF checkout produces and
  the corpus gate passed here and failed everywhere else.
- **A test that hashes a file certifies the checkout, not the file.**
  `native_render.cpp` -- the port's second renderer, since withdrawn with ADR-0012 -- held the
  shipped chat template's digest in the CRLF form while the
  committed blob is LF, so the native fast path was active only on a CRLF worktree and fell
  back to Jinja on Linux, CI and any fresh clone -- while its test passed here, because here is
  where the CRLF came from. Hash what ships.
- **A selection ladder and the function that sizes its buffers must read the same named
  constants.** Naming the boundaries removes a drift class; it introduces one, because a bound
  substituted with the wrong constant compiles and stays correct in value. `n5120_k17408.cu`
  and `fp8_linear_add_a8.cu` shipped `tokens > 512 && tokens <= 384` -- false for every token
  count -- sending 513-768 to `Bulk` instead of `Wide`. **No oracle here asserts which tile a
  token count selects**, so the suite stayed green.
- **Reach for the indexed tool before a manual search.** `.codegraph/` exists here, so a code
  question ("where is X", "who calls X", "how does X work") goes to `codegraph_explore` before
  `grep`, `glob` or `Read`: one call returns the verbatim source, the call path and the blast
  radius that a grep-and-read loop rebuilds by hand. Before investigating or changing anything,
  also check whether a loaded skill, an MCP server or a subagent already covers it, and whether a
  primary source on the internet owns the answer. Manual search is for what codegraph does not
  index (docs, configs, logs) or to confirm one detail it did not surface. Earned by grepping an
  already-indexed codebase.
- **Never inline a script through PowerShell.** Quotes inside quotes break the argument splitting.
  That happened five times in one session and the fix was always to write a file first. The failure
  is silent in both directions: the command can *succeed* while writing an empty file, so check the
  output's size, not just its exit code. **A quoted path handed to the call operator --
  `& "C:\Program Files\..."` -- is still an inline script**: it parses differently inside a compound
  command, and it failed that way twice on 2026-10-09, once on escaped quotes (which the claims-gate
  blocked outright) and once on the call operator, which PowerShell reported as "the string is missing
  the terminator".
- **A provider entry is not visible until the opencode service restarts.** The desktop app is V2
  (2.0.16) and takes its model list from the background service (port 49374 on this machine), which
  reads `~/.config/opencode/opencode.json` once at startup: the NVIDIA lanes added at 22:36 on 2026-09-24
  were missing from the app because that service had started at 17:29 the same day. Prove an entry is
  registered with the desktop's own bundled CLI
  (`%APPDATA%\ai.opencode.desktop\cli\<version>\opencode-cli.exe models`), not by looking at the app.
  The key is `provider` (singular) and V2 honours it for V1 config compatibility -- the 1.18.4 CLI on
  PATH and the bundled 2.0.16 one list the same 47 models. Restart with the bundled binary, since the
  `opencode` on PATH is 1.18.4 and may manage a different service, and expect in-flight sessions to be
  interrupted.
- **A batch that fails at parse time fails before any of its logic, so "nothing ran" and "the step it
  needed was skipped" look identical.** `build_windows.bat` printed its header and stopped: an `echo`
  inside an `if ( ... )` block carried unescaped parentheses -- "binaries (LGPL shared)" -- so cmd
  ended the block at that `)`, and the leftover `...` became "... was unexpected at this time". Escape
  them as `^(` and `^)`. Three more cmd rules from the same session: `copy` does not expand wildcards
  inside quotes, so staging files needs `xcopy`; a `.cmd` written with bare-LF endings loses its `goto`
  labels ("cannot find the batch label"); and `timeout` needs a console stdin, so it fails under a
  redirected or detached run -- use a retry loop instead. Run a batch you write before trusting it;
  reading it reveals none of these.
- **A superseded build left in `out/` under a shipped artifact's filename is a measurement waiting to
  go wrong.** The reverted Swift draft build was still there as
  `qwen3_8_27b_nvfp4swift.v3.ninfer` -- a different artifact from the shipped file of that name, and
  the one that lost its acceptance comparison. Anything resolving artifacts from `out/` would have
  measured it. Keep one file per name in `C:\AI\models`, and delete a superseded build rather than
  leaving it where a filename finds it.
- **The engine accepts only a `.ninfer` extension, so a preserved copy needs a hardlink to be
  measurable.** A `.fetched` file is refused at startup -- "NInfer accepts only .ninfer artifacts" --
  and `ninfer-perplexity` applies the same check. `cmd /c mklink /H out\qat_fetched.ninfer
  <artifact>.fetched` makes it measurable without copying 19 GB, and the run then reports the
  artifact as `out/qat_fetched.ninfer`, which is why the published-file reports carry that name.
- **Reference this tree relatively, or a path that exists will read as missing.** Absolute paths into
  the workspace stopped resolving mid-session: `cd C:\AI\infer-v3-windows` answered "Cannot find
  path", `Test-Path` and `[System.IO.Directory]::Exists` returned false for directories `Get-Item`
  resolved and `Get-ChildItem` enumerated, `Start-Process -FilePath <absolute workspace path>` said
  "cannot find the file", and a `.cmd` whose first line was `cd /d C:\AI\infer-v3-windows` died there
  -- while relative lookups through the process cwd, and absolute paths outside the tree
  (`C:\AI\models`, `C:\vllm-env`), kept working throughout. It cost two harness attempts and a
  misreading of the first as a missing file. When a workspace path is reported missing, re-test it
  relatively before acting on the report, and write scripts that use relative paths or derive them
  from `__file__`.
- **A transient file lock can mark a measurement failed while its data is intact.** The published
  QUASAR's AIME 2025 job recorded `failed` with 0 of 30 completed, and its log carried
  `[WinError 5] Access is denied: '...\aime25\progress.json.tmp' -> '...\aime25\progress.json'` -- a
  rename that could not complete, which aborted the job at 15 samples. The samples it had run were all
  on disk, and reading `backends/<suite>/reviews/<model>/<suite>_default.jsonl` gave the score the
  wrapper never wrote, validated against a job whose known result was 27/30 and 29/30. So: read the
  score out of the backend's own review file when a job reports failure, and read nothing inside a run
  directory while it is being written -- the same fault produced "the file is being used by another
  process" for `aime_quasar.log` twice in one session.
- **Do not guard with string presence over prose.** A check for a token that also appears in a
  comment, a docstring or a filename misfires. That happened four times. Assert the specific call
  site instead.
- **Instrument the branch before changing the candidate.** Three edits were built to tell one
  unknown apart -- whether a shared capture was refused by an empty scenario list or by a refused
  plan -- and none could, because both sat on paths the freeze never reached. One temporary probe
  printed `scenarios=1 planned=0 domains=1` in a single build. When several gates can produce one
  symptom, observe the decision or read its inputs first: an edit answers "did this change the
  outcome?", never "which branch ran?". Scope a probe by reading the block it lives in; anchoring it
  on memory of the enclosing function cost a compile error. The permanent form is a counter, because
  the context cache cannot log: logging is a layer above `src/runtime`. Counters carry decisions only;
  a magnitude (a baseline, a threshold, an allowed time) needs a value-bearing probe or a diagnostics
  field. A counter can prove that a mechanism fired without saying whether the value it added was
  enough -- and the mechanism it measured was later removed as inert, which that counter could not
  have told anyone.
- **The build and the running product contend for the same files.** Linking `ninfer-serve.exe`
  failed with LNK1104 because a server started earlier still held it, and a test source was edited
  while `build-test` was reading it -- survived only because that phase had not yet begun. Stop the
  server before rebuilding it, and do not edit a source while a build is compiling it. The same
  applies to a script while it runs: cmd reads a batch file by byte offset, so an edit mid-run hands
  it shifted text, and `test_v3.cmd` was edited while its own suite was running on 2026-09-25. A test
  executable cannot be relinked while the suite is executing it either: `LNK1104` on
  `ninfer_qwen3_5_prefix_real_test.exe`, twice in one session, reads as a broken build rather than as
  a held file. Stop the suite before a rebuild, and stop the rebuild before a suite — and note that
  `check_test_baseline.py` now identifies the build tree before and after its run, so a build that
  races the suite fails the gate instead of silently certifying two trees as one.
- **A merge's shape is lost by `git stash`, and restorable.** `git stash` on an in-progress merge
  preserves every byte and drops `MERGE_HEAD`, which records a single-parent commit *and* breaks
  `check_port_delta.py`: it falls back to `merge-base(HEAD, upstream/dev)`, the OLD base, and reads the
  incoming branch's own files as the port's divergence -- 90 of them for an 18-commit merge. Writing
  the state file back (`git rev-parse <upstream-ref> > .git/MERGE_HEAD`) restores both, and the commit
  then records two parents. Commit a resolution before switching context, or re-create the state file.
- **Never edit a file from memory.** Five edits failed in one session because the anchor was
  reconstructed rather than read -- twice in the same file, and once after the same lesson had already
  been written down. Read the block, or anchor on a unique substring that omits the leading whitespace,
  which sidesteps the indentation a reconstruction gets wrong. The rule about scoping a probe by
  reading its block is this rule with a narrower scope.
- **Validate the harness before believing its numbers.** Four times in one session a measurement
  returned plausible zeros: a loop variable that collided with a read-only automatic (`$Host`, which
  also wrote three junk filenames), a log path the launcher named differently from the loop tag, a run
  too short for the request log to flush, and a server that never started. Check that the input exists
  and that the run happened before reading a single figure. A zero that cannot be explained is not a
  result. The same applies to a *failure*: four real-model tests failed in `cmd` because
  `set VAR=value && ctest` sets the value **with a trailing space**, so the artifact path became
  `...v3.ninfer ` and an extension check rejected it -- while the identical binary passed under
  PowerShell's `$env:`. Use `set "VAR=value"` in `cmd`, never `set VAR=value && ...`, and before
  diagnosing a test failure check that the harness handed it what you think it did.
- **A figure in a comment carries its configuration, and a SERIES carries the configuration it was
  measured along.** A startup line reading "pinning host state | 1.46 GiB" was recorded as the 16-slot
  cost and committed; it is the 8-slot figure, and three logs side by side give 1.46 / 2.19 / 2.92 at
  8 / 12 / 16. A number that arrives without its configuration is how a wrong figure gets committed and
  then cited as corroboration. The same applies to the axis a trend is fitted along, which is the part
  a per-figure label does not cover: on 2026-10-04 the longest GPU kernel was extrapolated from three
  `ninfer_bench` points at the artifact's default `kv_cache=bf16` to the shipping lane's
  `--kv-dtype fp8`, and the kernel the extrapolation named appears **zero times** in the serving
  route's trace. The fit was clean, the exponent was plausible, and the answer was wrong by ~8x
  (343 ms predicted, 43.10 ms measured) because the two routes never ran the same kernel. Before
  extrapolating, confirm the thing being extrapolated is present in the destination — one
  `Select-String` for the kernel name in the destination report — and state which single axis the
  series shares. A trend across points that differ in something else is two series, not one.
- **Redirect a long-running command to a file; a truncating filter kills it.** `Select-Object -First N`
  closes the pipeline and terminates the command, so a full launcher verification died half way and
  reported failure. This is the same family as swallowing output with `| Out-Null`: when the command's
  own success is the thing being checked, let it write a file and read the file.
- **Read primary sources before reasoning from this tree.** Upstream's converter, loader and
  maintainer notes are authoritative; the port is not. Reading a platform header settled in a
  minute what inference had concluded wrongly twice. The upstream tracker is the same kind of
  source, and a session that already cites an issue number should read it before opening another
  file: `gh issue view 180` named the exact site (`candidate_demand_mask` at the capture input), the
  exact inheritance source (`exact_resident_keys`, restricted to the lane's own reuse domain) and
  the measured masks (2, 4, 8, ... 256 -- not zero) -- after an hour of inference had produced a
  different site, a different source, and a wrong premise.
- **Verify a claim about a file before asserting it.** A wrong statement about which flags a
  launcher passes survived until an independent review corrected it. A negative claim needs its
  scope checked too: "this layer cannot log" came from a grep limited to `context_cache/` and
  happens to be true, but it was asserted before anything had searched the rest of `src/runtime`.
  A truncated search is not evidence of absence either: "the shared handle is never populated" came
  from `git grep .handle.emplace` piped through `Select-Object -First 20`, which cut the match list
  short, and an ADR was written and committed on it before a wider read found the assignment at
  `src/runtime/engine/context_cache/resource_manager.h:1086` and `:3307`.
- **Profile values come from `profiles.py`.** Run `tools/release/check_profile_consistency.py`
  after touching a launcher, a doc table, an opencode provider entry or a harness. Fixing tables by
  hand once touched three files and missed two.
- **Confirm a commit landed.** Two commands reported success while committing nothing this
  session: a multi-paragraph message passed with `-m` split on its own quotes and became
  pathspecs, and a file under `tools/build/` was silently ignored because `.gitignore` has
  `build/` with no leading slash, which matches at any depth. Use `git commit -F <file>` for
  anything longer than a line, and check `git log -1` or `git status` after every commit.
- **Prove an extraction by byte-identity.** Regenerating output that must not change is stronger
  evidence than re-running the behavior, because it rules out any change at all.
- **Revert every diagnostic probe before committing, and never commit its rationale.** A threshold
  changed to test a hypothesis (`kCpWarmupThreshold` -10.0 -> -13.0) was committed with a comment
  asserting it fixed a residual error, when the measurement had already shown byte-identical
  output and therefore no effect at all. The code then contradicted the ADR and commit message that
  correctly recorded the finding. A probe that disproves its hypothesis leaves the original value
  and a comment saying what was measured, or it is reverted outright.
- **A claim in a comment is a claim: verify it against the measurement that produced it.** The
  same probe's comment reasoned from `exp(threshold) * |state|` to a conclusion the test had
  already falsified. Reasoning that survives only until it meets the artifact belongs in a
  hypothesis, not in a comment that the next reader will trust. Advertised surface makes the same
  kind of claim: an option once sat in this tree that parsed, reached the manager, and lifted nothing
  it was meant to lift, with no row in `docs/serving.md` and no mention in the README -- the tree it
  shipped in would have advertised a fix on the strength of a design document. A CLI option, a README
  row or a doc paragraph lands with its measurement and its documentation, or it does not land.
- **Check whether a skill is actually loaded before relying on it, and say which one you used.**
  Five project skills added mid-session (`cpp-cuda-review`, `ncu-report`, `cuda-debugging`,
  `sanitizers`, `address-sanitizer`) were invisible to the running session, and two wrong
  hypotheses about the cause followed. The loaded list is rebuilt when the session's context is
  rebuilt, not continuously: a context built before the skill existed, or built in a worktree that
  does not carry `.opencode/`, keeps the old list. So verify the file is present in *this*
  worktree's `.opencode/skills/`, then re-check the loaded list after a context rebuild or a new
  session — a junctioned `.opencode/` made 40 skills appear mid-session with no restart. Do not
  debug the frontmatter first; the project's own review skill is `cpp-cuda-review`, not the .NET
  `code-review` import.
- **One workspace branch, one worktree.** All work lands on `dev`; `main` is only ever a squashed
  release cut and never a place to work. Do not keep a second worktree: a session then finds the
  tooling in one directory and the code in another, which is what happened when this repository was
  worked from `ninfer-quasar-5090` on one branch and `ninfer-v3-windows` on another. Push `dev` —
  unpushed work is one disk failure from gone, and `main` being three weeks stale is the same fault
  seen from the other side. Keep one *checkout* as well: three clones of this same origin and a
  redundant upstream clone had accumulated under `C:\AI`, each holding refs the working checkout
  already had. The remote is already configured, so a second clone buys nothing.
- **Three remotes exist, and only one is upstream.** `upstream` is `Neroued/ninfer`, the project this
  port follows: it is what AGENTS means by upstream, and what `gh issue list --repo Neroued/ninfer`
  queries. `origin` is `headpiece747/ninfer-5090-windows`, this port's own published repository, which is
  where `dev` goes. `cometkim/ninfer` is a third-party fork that earlier supplied the NVFP4 artifacts; it is not
  upstream, and fetching it answers no question about being current. On 2026-09-25 a "current with
  upstream" claim was checked against `cometkim` and reported as if it were the real thing.
  **As of 2026-10-04 it is not a dependency either**: every `.ninfer` artifact this port ships is built
  locally by its own converter from upstream Hugging Face source checkpoints (`converter: ninfer-v3`
  in each artifact's `.conversion.json`), and `download_model.py` fetches those sources rather than any
  prebuilt artifact. `NOTICE` records what came through that fork, which is attribution, not a
  dependency.

  Ask the question of the right remote, and ask it of each branch by name, because "nothing returned" is
  how both mistakes happened: a local ref that is not fetched, or a branch name that does not exist on
  that remote, produces an empty answer that reads exactly like "we are current".

  ```bash
  git fetch upstream --quiet
  git log --oneline HEAD..upstream/dev | wc -l     # behind, both branches, on the real upstream
  git log --oneline HEAD..upstream/master | wc -l
  ```
- **Integrate upstream by merging into `dev`.** Never park local commits on a tracking branch: that
  is how `cometkim-qat` became 25 commits ahead and 41 behind, living in another worktree.
- **Publish every version you build, in order, or do not build it.** A gap in the release list reads
  as a withdrawn release. `v1.0.1` and `v1.0.2` were built — both archives are in `C:\AI\releases` —
  and never published, so the list is `1.0.0, 1.0.3, ...` and nothing records why. The version was
  hand-typed in the packager, which is why nothing caught it.
- **Verify a provenance claim against a baseline.** Two traps cost one session. A file that is
  byte-identical across two repositories is usually *upstream's own*, unchanged by either, so compare
  it against upstream before calling it derivation. And a shared-line ratio means nothing until you
  measure the baseline for unrelated files in the same project: 33% here, 27-28% for
  `src/artifact/reader.cpp`. Measured that way, the 1.0.x Windows layer derives from
  `Don-Chad/ninfer-3090` (501 of 524 lines identical) while the v3 Windows files are this port's own
  work over upstream's base. `NOTICE` carries the result.
- **Compare alternatives by interleaving them.** This card's clocks are not pinned, so decode drifts
  by up to ~9% between windows: measure A and then B and you have measured the window. Two findings
  died that way in one session, a 14% slot-count claim and a 9% artifact claim, and both were
  committed before the interleaved run disproved them. ADR-0003 records the detail.
- **One GPU job at a time, and never a measurement beside one.** Three times in one session a job held
  the card while another was meant to be measuring: a soak that ran through a gate, and two recipe runs
  that could have interleaved with an A/B. A number taken while something else is on the card is not a
  smaller number, it is a different one -- the rule above exists because this card's behaviour moves
  with load. Serialise the card: stop the server before a sweep, finish the suite before the
  measurement, and give any automated GPU work a concurrency lock rather than a timer that can fire
  mid-measurement, which is why `.github/workflows/gpu.yml` declares one and fires on a nightly cron as well as on dispatch.
- **The release matrix kills every `ninfer-serve.exe` on the machine, including the lane a chat session
  is using.** `v3_profile_matrix.py` stops every engine before each sweep, which is right for its own
  measurement and fatal to anything else serving. On 2026-09-25 a sweep took down the desktop app's
  local lane, and its next request failed as though the provider had broken. Check what is serving
  before a sweep, and say so rather than discovering it afterwards.
- **A rejected request's reason is in the engine's log; opencode does not report it.** The desktop app
  surfaces a provider failure as `Provider request failed with HTTP 400` with a stack trace and no body,
  while the engine answers with a specific JSON error -- `image_detail_not_supported`, `invalid_media`,
  `modality_not_supported`, `image URL must use HTTP(S)`. The engine's own log carries that line, so a
  lane started detached from a console has put it nowhere: start the lane with its output captured
  before diagnosing one, or the only evidence is the one line that says nothing.
- **Do not ration work against an assumed context limit.** A session is not short: it can run long,
  and compaction exists for when it does not fit. An agent that behaves as if it is about to run out
  takes smaller changes than the job needs, defers the next step it has already identified, cuts an
  investigation short, and hands over work it could have finished -- each one a worse result that
  nobody asked for. Do the whole job now. Never cite remaining room as a reason for anything: not for
  a smaller diff, not for stopping, not for what to attempt next.
- **A numerics change re-states every recorded figure, and the tree will not tell you.** `512f5b2b`
  restored the accurate `silu` in one SwiGLU epilogue and moved the QUASAR artifact's corpus
  perplexity from 1.606336 to 1.609905, which silently made the scores quoted in ADR-0003 stale within
  hours of their being written -- nothing failed, no check moved, and an ADR audit found it only by
  reading. After any change to an Op's arithmetic, sweep the recorded measurements yourself: the
  perplexity and accuracy figures in `docs/`, the compiled tok/s in `tools/release/profiles.py`, and
  every ADR that quotes a score. A measurement is a claim about a revision, so it carries the revision
  that produced it, the way a startup figure carries its configuration.
- **A bench that does not reproduce the process can refute a true hypothesis.** `prepared - tokenize`
  was read as "the render", and a host bench was built to check that reading. The bench measured
  `CompiledChatTemplate::render` at **80 ms** for a 229-message, 690 KB conversation, which appeared
  to rule the render out, so the render was set aside and the note said so. The live server's own
  phase timer then put the same call at **2.4 s**, and replaying the server's *exact* request body
  through the bench still gave 76 ms against the server's 2,430 ms — same body, same code, same three
  render calls, same 300 checkpoints. A neutral allocation loop in both showed the server runs
  ordinary host allocation ~40x slower than a fresh process on the same machine: the difference was
  the process, not the input, and the bench had been right about its own measurement while wrong about
  what it stood in for. Two lessons: a stand-in must reproduce the *conditions* of the thing it
  replaces, not only its input; and when a bench and the live system disagree, find out which one is
  lying before acting on either. What caught it was bracketing the total — the phase fields put the
  time in the render, where the bench had said there was none.
- **Batch what goes into `include/ninfer/types.h`, because the device tree reaches it.** `src/ops/*.cu`
  and `src/core/*.cu` include `core/paged_kv_storage.h`, which includes that header, so adding one
  field to a public stats struct rebuilt every CUDA object in `build-test` — twice in one session,
  about twenty minutes each, for two fields that could have landed together. Decide the whole set of
  instrumentation fields before editing it, or keep them in a frontend-internal struct until they
  actually have to be public.
- **A helper that returns a view over its own temporary is a silent corruption generator.** Writing
  `std::string_view content = trim(render_content(...))` dangles the moment the call returns, and the
  bytes that land are whatever the allocator hands back: the text stays the *right length* and the
  corruption appears mid-string, so a byte-identity check reports a difference at some offset and the
  offset moves as unrelated code changes. Three separate survivors of this in one renderer, and the
  last one was found only because a field-by-field comparison ran; a text-only check had passed it.
  Return `std::string` from such helpers rather than a view, and when a caller only needs a view bind
  it to something with a declared lifetime.
- **Compare structures field by field, not by their largest member.** A renderer that produced
  identical bytes with different cache, message and execution boundaries passed every text-only check
  and would have handed the engine a wrong frontier. Two of that session's four defects — a
  same-length divergence and a wrong control-channel flag — were invisible to a byte comparison and
  found by comparing every field. The reverse also holds: a field-by-field comparison needs its
  control (`x` against `x`) printed first, so a broken instrument reads as broken rather than as a
  result.
- **A projection is not a measurement, and the first live run will correct it.** `prepared` and TTFT
  were projected twice from a render benchmark and both times were wrong: the second projection
  omitted a term entirely (a ~10 ms engine queue wait) and was off by 2x. What fixed it was reading
  the server's own `request-log-jsonl` and its `prepared ..., render ..., tokenize ...` breakdown,
  which already carried the answer — the projection was built from a bench because the phase split
  had not been read, not because it did not exist. Quote a figure with the scope it was taken at, and
  when a component figure exists next to an end-to-end one, prefer the end-to-end one.
- **A cache that serves an extension but not a repetition has a gap, not a design.** ADR-0010's
  incremental encode required a *strict* prefix, so a growing conversation spliced and a retried or
  duplicated one re-encoded the whole prompt - `splices=0` on every request, for as long as it went
  unnoticed. The counter that would have shown it (`encode_cache_splices`) existed and was not
  reported anywhere. When a cache has a hit counter, publish it; when it has a hit *condition*, ask
  which common cases fall outside it.
- **A server left running holds its own executable, and "I stopped it earlier" is not a check.** A
  `ninfer-serve.exe` from an instrumentation run outlived its measurement by the rest of a session,
  holding `build/apps/ninfer-serve.exe` and listening on its port. Every "no process listed = idle"
  report in between was true when it was made and stale by the end. The failure it produces next is
  LNK1104 on the following rebuild, which reads as a broken build rather than a held file. Stop the
  server before every rebuild and verify at the moment it matters: `Get-Process` *and* a port check,
  both in the same command as the thing that depends on them.
- **A scratch wrapper is part of the probe, and probes get removed.** A `.cmd` written only to launch
  an instrumented server is a live hazard, not a note: it can be re-run, it can leave a process
  behind, and its logs outlive the finding. Delete the wrapper and its logs with the probe, in the
  same turn, rather than sweeping them when the session happens to notice.
- **Read the access level off the Hub API, not off an impression.** A model was assumed gated and a
  download deferred for it; `https://huggingface.co/api/models/<repo>` returns `gated` and `private`
  directly and both read `false`. A gate seen on one repository says nothing about the next one - the
  check is one request, and asserting a *negative* about access without making it is as wrong as
  asserting one about code without reading it.
- **Parse a tool's tabular output by header name, never by column position.** A width sweep read
  `workspace_peak_bytes` as the round count and `prefill_seconds_mean` as the decode time, and
  printed a clean table of `round=0.000 ms` and `rounds=107360256` at every width - the plausible
  zeros, produced by a reader that was wrong rather than by a measurement that was. `ninfer_bench`'s
  CSV has 39 columns and one empty field, so positions are not stable. Take the header line, index by
  name, and re-derive from the saved logs rather than re-running an expensive sweep to get them.
- **A bench answers only the route it was written for, and needs its runtime beside it.**
  `ninfer_qwen3_5_dflash_round_bench` exited `0xC0000135` printing nothing until the FFmpeg DLLs were
  staged next to it - the same missing-runtime failure `test_v3.cmd`'s header documents for the test
  tree - and then reported `missing component dflash`, because it measures DFlash v1 while this
  product's artifacts ship `dflash2`. Stage the DLLs into `build\bench\` as well as
  `build-test\tests\`, and check the bench's backend against the artifact's before trusting a zero.
  `ninfer_bench --spec dflash2 --draft-tokens` is the instrument for the shipped lane; it reports
  `spec_rounds`, `spec_acceptance_rate` and `decode_seconds_mean`, so the round cost is
  `decode_seconds_mean / spec_rounds`.
- **Never rewrite code with a regex that crosses a line boundary.** Wrapping fifteen call sites with
  a match on `f\(\{` and a second pass to close the parenthesis truncated every one of them, because
  the closing pattern stopped at the first `)`. It reported fifteen successful replacements and the
  damage was visible only in the compiler output. Write the file, or make one anchored edit; a
  scripted multi-site rewrite needs a count that must match.
- **A scripted edit needs an expected occurrence count, and must be idempotent.** A removal script
  asserted each anchor appeared exactly once; `startup.h` legitimately carried the pair in two
  structs, and a second run then hit an already-applied edit and stopped. Pass the expected count and
  treat "already applied" as success, or the second run of a fix reports a false problem with the fix.
- **An assertion that cannot fail is worse than no assertion.** The guard rejecting a zero neural draft
  window was "tested" by calling the function with seven. It passed, and it was fiction; it surfaced
  only because the same test also covered the real empty-batch behaviour. When a check passes,
  confirm it would have failed without the thing it names.
- **This card's early runs are unreliable, and the spread is measurable.** Interleaved samples of one
  configuration differed by 5.8 % and 7.6 % in the first half of a session and by 0.0 % and 0.3 % in the
  second: the clocks settle. Any effect below about 8 % needs many more repetitions than a recipe's
  default and a longer warmup, or the verdict is the warmup. Interleave rather than group, and say
  which half of a session a number came from.
- **A cost you can compute is not a cost you have measured.** A wider verification round was argued to
  "roughly double the target's work" and its extra graph family estimated at ~480 MiB; measurement on
  this product gave **+5.7 %** round time from draft window 7 to 15, and a **288 MiB** graph allowance
  that did not vary with width at all. Extra columns are nearly free because the weights are read once
  per forward regardless of column count. Compute what a design *implies*, then measure it, and do not
  let the arithmetic stand in for the number: three separate cost claims in one session were falsified
  exactly this way.
- **A superseded design's surface is removed in the same wave, not left reachable.** Replacing the
  copy-drafting core meant withdrawing the archive, its option fields, `NgramSessionHints`, three
  counters, a stats struct, two tests, four CLI flags and a document, in one commit. Anything kept
  "for now" becomes advertised surface the product cannot honour, and a document naming flags the
  binary rejects is worse than no document.
- **A scripted edit's verification must use the file's real consumer, not a convenient parser.**
  Windows PowerShell 5.1's `Set-Content -Encoding utf8` writes a **BOM**, and `Get-Content` **decodes with the console codepage**, so a `Get-Content`/`WriteAllLines` round trip re-encodes every byte at or above 0x80 and the damage compounds with each pass. 181 lines of `docs/active-work.md` were lost to that and **no gate noticed**, because the corruption is confined to prose. `tools/release/check_text_encoding.py` now fails on it. A scripted edit to
  `test_baseline.json` was checked with `ConvertFrom-Json`, which accepts a BOM, and passed; the gate
  then failed on `json.load`, which does not, with "Unexpected UTF-8 BOM". Checking a file with the
  same tool family that wrote it verifies almost nothing. Write through
  `[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))`, and
  re-read the file with whatever actually consumes it. Five files carried BOMs into a commit this way
  before the audit found them; audit the *committed* files too, because `git status` no longer lists
  them.
- **Compute a tool's memory cost before running it, and refuse the run when it is over budget.** On
  2026-09-29 a new artifact verifier exhausted this 47.8 GiB machine and took it down. Two defects,
  both arithmetic that was available before the run and not done: the NVFP4 parent was assembled from
  nested Python lists at a **measured 28 bytes per element** (4.6 GiB for the lane's largest parent
  alone), and every decoded parent was cached for the whole run, so the lane wanted **688 GiB**. The
  rule is not "be careful with memory" -- it is that a per-element cost times a count is a number, and
  the number comes before the command. `tools/convert/verify_artifact.py` now prints a projected peak
  from the directory alone and **refuses `--values` before decoding anything** if it is over
  `--max-peak-gib`. The vectorized float32 form measures 1.65 GiB against a 4 GiB budget. Anything
  that walks a multi-gigabyte artifact gets the same pre-flight, because the failure mode is a dead
  machine rather than a raised exception.
- **A memory guard that reports zero is a broken guard, and must fail rather than pass.** The first
  probe for that verifier sampled the child process's working set from the parent and printed
  `0.00 GiB` for a child that finished in 0.18 s -- a plausible zero from a sampler that never caught
  the process, and a failure path that returned 0.0 on a `GetProcessMemoryInfo` error. A guard that
  passes on a non-positive reading passes exactly the case it exists to catch, so the child now reads
  its **own** `PeakWorkingSetSize` through ctypes with explicit `argtypes` (without them ctypes
  truncates the handle and the call fails), and a reading at or below zero is a hard failure.
  `Get-Process` polling is not a substitute: it samples, and a fast child is invisible to it.
- **The Python directory is not the C++ one; read the schema you are coding against.** Writing
  `tools/convert/verify_artifact.py` against `artifact::Binding` produced `binding.parts`,
  `part.object.index` and `part.begin`, none of which exist. The Python side is plain dicts --
  `{"object": id}` or `{"parts": [{"object": id, "range": [begin, end]}]}` -- with **element** ranges
  rather than byte ranges, and `uses` a tuple of dicts rather than a keyed map. The C++ structs in
  `src/artifact/schema.h` are a richer form of the same data, not the same data, so reading them is
  not reading the target. Three crashes and a rewrite came from that one assumption; the four fixes
  after it (`SafetensorsSource(path)` is a context manager and has no `.open`; a recipe needs a real
  `Recipe(model)` and not `None`; `dflash`/`dflash2` are `companions` and not `sources`; and a script
  run as `__main__` needs absolute imports) were each one more place the same habit would have caught.
- **A binding is usually a slice of a fused parent, and NVFP4 cannot decode a slice.** 409 of 1,513
  bindings cover part of a larger packed parent -- `dflash2/layers/0/attention/query` is 4096 rows of a
  6144-row object, because query, key, value and output share one parent whose 128-row tiles carry the
  block scale. Asking the source for the slice's own row count, which is the obvious first thing to
  write, asks it for 20,971,520 elements of a 31,457,280-element parent and raises. The parent is
  decoded once and the slice taken from the result, and the slice geometry is read from the binding's
  `range` rather than from the object's shape.
- **An identifier that encodes a format hint is not interchangeable with one that does not.** 558
  dflash2 sites reported `missing source tensor 'layers.0.self_attn.k_proj.weight_packed'` because the
  verifier passed `format="nvfp4"` to a factory whose source is **Q8-packed upstream** -- the hint made
  it look for packed weights that are not in that checkpoint. The recipe's own call is
  `model.source(name, store)` with no hint for a locally encoded site. Worse, the call *succeeded* and
  only `values()` raised, so a mapping that returned an object was not evidence the hint was right: a
  store is accepted only once it can also produce values. The first fix attempted for this was based
  on a wrong hypothesis and produced byte-identical output, and only reading `__main__.py` and
  reproducing the recipe's call in isolation found it.
- **A failing expectation is as much a defect as a failing tool.** Ten tests written alongside that
  verifier had three wrong expectations, and the tool was right every time: `0x3C` is 1.5 in E4M3FN
  and not 1.0 (mantissa 4), `0x7F` *raises* rather than returning NaN, and `0x7E` has high nibble 7, so
  it is (-4, +6) while `0xFE` is the negative pair. Each was caught only because the expectation was
  written from the format's definition instead of from what the code happened to produce -- a recorded
  output would have recorded the bug. The grid is a table, not a magnitude list.
- **A driver that buffers a step's output loses it when the driver dies.** A three-step chain ran for a
  minute, died inside step 2, and took everything step 2 had produced with it -- the smoke's output
  existed only in the dead process's pipe, and the chain's own log ended cleanly at step 1, which reads
  exactly like "step 2 has not finished yet". Give every step its own log file and let the driver record
  return codes only, so a chain that reports `rc=0` per step always has a file behind the claim. Earned
  2026-10-09, where the loss cost a full re-run and was found only because the user asked whether it was
  still running. **And print the step's start *before* running it**: two chains died inside a step without
  ever reporting it, and two constructed probes of that same sequence -- the smoke alone, then a
  fingerprint followed by the smoke -- both refused to reproduce it. The cause is therefore unknown and
  the failure is silent, so the defences are the per-step log file, the start line, and giving each long
  step its own background shell rather than chaining them.
