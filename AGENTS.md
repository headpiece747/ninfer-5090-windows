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
release surface is documented in the Windows section of `README.md`.

Sixteen rules, each earned by a failure rather than chosen:

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
- **Do not guard with string presence over prose.** A check for a token that also appears in a
  comment, a docstring or a filename misfires. That happened four times. Assert the specific call
  site instead.
- **Read primary sources before reasoning from this tree.** Upstream's converter, loader and
  maintainer notes are authoritative; the port is not. Reading a platform header settled in a
  minute what inference had concluded wrongly twice.
- **Verify a claim about a file before asserting it.** A wrong statement about which flags a
  launcher passes survived until an independent review corrected it.
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
  hypothesis, not in a comment that the next reader will trust.
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
