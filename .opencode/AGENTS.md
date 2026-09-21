# Skills policy (this repo)

How agent skills are chosen for NInfer. The repository `AGENTS.md` governs the product; this file
governs what lives under `.opencode/skills/`.

## Tooling discipline (check all five before investigating)

Before a code question or a change, check these in order and say which were used:

1. **Tools / MCP** - `codegraph_explore` first for any "where is / who calls / how does X work"
   question. `.codegraph/` exists here; grep, glob and Read rebuild by hand what one call returns,
   and grepping an already-indexed codebase is the failure this rule exists to stop.
2. **Skills** - is there a loaded or project skill that already covers the task (`.opencode/skills`,
   the global set)?
3. **Agents** - delegate parallel search or research (`explore`, `general`) instead of doing it
   serially in the main thread.
4. **Internet** - do upstream's tracker, maintainer notes, or a primary source own the answer?
   `gh issue view <n> --repo Neroued/ninfer` works and is authenticated.
5. **Plugins / hooks** - anything already loaded that answers this?

Manual grep/Read is for what codegraph does not index (docs, configs, logs) or to confirm one
detail it did not surface. See the same rule in the repository `AGENTS.md`.

## Added for this repo

| Skill | Origin | Notes |
|---|---|---|
| `cpp-cuda-review` | authored for this repo | The review skill here: FP32/FP64 oracle rule, ownership boundaries, state lifetime, C/C++ memory safety, CUDA. |
| `ncu-report` | vendored from `mit-han-lab/ncu-report-skill` (MIT) | Nsight Compute profiling. Audited before install (no network, process execution, deletion, or credential handling). Adapted to RTX 5090 / sm_120a and `profiles/ncu/`. |
| `cuda-debugging` | vendored from `mohitmishra786/low-level-dev-skills` | cuda-gdb / compute-sanitizer / error-code triage (700, 702). Directly relevant to open #208. Noted: Windows/MSVC, sm_120a, limited cuda-gdb on Windows. |
| `sanitizers` | vendored from `mohitmishra786/low-level-dev-skills` | ASan / UBSan / TSan / MSan / LSan chooser, flags, report reading. Noted: MSVC has ASan only, so UBSan/TSan/MSan are advisory here, and the host-side subset is `tools/scripts/test_v3_asan.cmd` — which is also where the mandatory CUDA `-Xcompiler=/fsanitize=address` and the absent LeakSanitizer are explained. Noted: that script calls `vcvars64`, so running `ctest --test-dir build-asan` from a plain shell instead needs the toolchain's `clang_rt.asan_dynamic-x86_64.dll` on `PATH`, or every test exits `0xc0000135`. |
| `address-sanitizer` | vendored from `trailofbits/skills` (`testing-handbook-skills`, MIT) | ASan deep-dive: builds, `ASAN_OPTIONS`, report reading, LeakSanitizer. Noted: clang/gcc reference form; MSVC uses `/fsanitize=address`. |

## Shared set — do not remove

`arch-check`, `code-review`, `convention-learner`, `desloppify`, `health-check`, `security-scan`,
`testing`, `verify` are .NET / Roslyn / Glider skills **used by another project**. They stay in this
tree even though NInfer is C++/CUDA. Do not delete or "correct" them here. For NInfer review use
`cpp-cuda-review`; the shared `code-review` is already scoped to ".NET projects" in its own
description.

## Not installed (dated rejections)

Evaluated in the skills.sh pass **2026-08-24** and deliberately not installed: `triage`,
`prototype`, `wizard`, `teach`, `to-questionnaire`. Reasons were not recorded in this file at the
time; re-evaluate against a current need before installing.

## Third-party skill policy

- Vendor a third-party skill only after reading it: download to a temp sandbox, scan every script
  for network access, process execution, deletion and credential handling, and read any file that a
  scanner or an auditor flags. Install only when clean.
- Record the origin (source repo / registry, license) beside the skill.
- Prefer first-party sources and skills with real audits; skills.sh shows Gen Agent Trust Hub,
  Socket and Snyk per skill.
- `deepskill.market` repackages can be incomplete — a "安全审查" label there is a content scan, not
  a packaging check. Case in point: its `c-review` shipped without the scripts and agents its
  SKILL.md depends on.
- `trailofbits/skills`' `c-review` is a Claude Code plugin (`workflows/c-review.js`,
  `allowed-tools: Agent/TaskCreate`, `${CLAUDE_PLUGIN_ROOT}`) and cannot run under opencode; its
  taxonomy is carried by `cpp-cuda-review` instead.

## Evaluation log

- **2026-09-19** — skills.sh pass. Installed after sandbox audit: `mit-han-lab/ncu-report-skill`
  (vendored as `ncu-report`), `mohitmishra786/low-level-dev-skills@cuda-debugging` and
  `@sanitizers`, and `trailofbits/skills@address-sanitizer`. `trailofbits/skills@c-review` is not
  portable under opencode, so its taxonomy was adapted into the authored `cpp-cuda-review`.
  Audit note: `address-sanitizer` is flagged MEDIUM by Runlayer ("1/1 file flagged"); the single
  file is `SKILL.md`, and reading it found only documentation URLs and standard sanitizer
  commands — no executable content in any of the three packages.
