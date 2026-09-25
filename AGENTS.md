# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. The implementation targets `sm_120a` and
is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
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

Finish when the deliverable is usable, applicable contracts are satisfied, material claims have
sufficient evidence, relevant checks pass or their limitations are clear, and no known in-scope
issue blocks use. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

### Tooling, and when to reach for it

Reach for the tool that fits the question; each of these exists because a hand-rolled check missed
something, and each is named here so it gets used rather than rediscovered.

| situation | tool |
|---|---|
| every commit | `.githooks/pre-commit` — enable once with `git config core.hooksPath .githooks`. Doc links, profile consistency, converter tests: seconds, no network |
| a C++ or upstream change reaching the suite | `tools/scripts/test_v3.cmd`, then `tools/release/check_test_baseline.py` |
| anything that could be order- or state-dependent | the suite recipe passes `--schedule-random`; run it twice before believing a fixed order |
| a device-side memory, race or synchronisation question | `tools/scripts/test_v3_compute_sanitizer.cmd` — memcheck on a small subset; `racecheck`/`initcheck`/`synccheck` and the wider method are in the `cuda-debugging` skill |
| a host-side lifetime question | `tools/scripts/test_v3_asan.cmd` — ASan cannot instrument device code, which is why the two recipes are separate |
| a kernel's performance | the `ncu-report` skill, records under `profiles/ncu/` and `profiles/nsys/` |
| Python tooling, before committing it | `ruff check tools tests` (clean, so a new finding is yours); `mypy tools/release tools/convert` (adopted with 67 known errors — treat a new error as yours) |
| a lingering suspicion of flakiness | `ctest --test-dir build-test --repeat until-fail:5` |
| what upstream already decided, or whether a symptom is known | the upstream tracker: `gh issue list --repo Neroued/ninfer --search <term>` — this project's reference corpus |

Not installed deliberately: CI. GitHub-hosted runners have no GPU, so the suite needs a self-hosted
runner on this machine, and publishing a workflow needs a push.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## Working practices (Windows port)

This fork is the Windows port. The Linux paths in Local operations above do not apply here: the
port targets MSVC 14.51 and CUDA 13.3, and the Python used for tooling is
`C:\vllm-env\Scripts\python.exe`. What ships is governed by `tools/release/profiles.py`, and the
release surface is documented in the Windows section of `README.md`. There are two build trees:
`build/` for the apps, and `build-test/` for the suite the release gate runs
(`ctest --test-dir build-test`).

Thirty-five rules, each earned by a failure rather than chosen:

- **Run a verification recipe through the recipe.** `ctest --test-dir build-asan -R <broad regex>`
  pulls in device tests, which ASan cannot instrument and which hang: one such run burned fifty
  minutes before its timeout. `tools/scripts/test_v3_asan.cmd` names its six host-only tests for
  exactly that reason and says so in its header. The recipe's scope is part of the recipe.
- **Check before you package, not after.** A package built before its review has to be re-cut and
  re-packaged: this session's was, three times, because a code review and the sanitizer run both
  landed afterwards. "The archive is cheap to regenerate" is the reason to check first, not to
  skip the check.
- **`git tag -f` tags HEAD, so create a release tag while on the release branch.** Tagging from
  `dev` put `v1.1.0` on a dev commit; the trees were identical, which is why only the commit
  pointer gave it away. Likewise, do not swallow a command's output with `| Out-Null` when its
  success is the thing you are checking.

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
  output's size, not just its exit code.
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
  it shifted text, and `test_v3.cmd` was edited while its own suite was running on 2026-09-25.
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
- **A figure in a comment carries its configuration.** A startup line reading "pinning host state |
  1.46 GiB" was recorded as the 16-slot cost and committed; it is the 8-slot figure, and three logs
  side by side give 1.46 / 2.19 / 2.92 at 8 / 12 / 16. A number that arrives without its configuration
  is how a wrong figure gets committed and then cited as corroboration.
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
  `resource_manager.h:3249`.
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
