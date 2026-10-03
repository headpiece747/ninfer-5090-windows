---
name: rules-check-drift
description: "Check whether your rules file (CLAUDE.md or AGENTS.md) still matches the codebase after recent changes, run before a merge, or fold into your code-review pass. Reports stale/now-false rules, drifted architecture-map entries, and any new invariant worth adding, each with the minimal edit. Advisory and anti-bloat: it keeps the rules file true, never longer than it needs to be."
---

# rules-check-drift: keep your rules file true, not longer

> **Project adaptation (NInfer).** The rules set is **`AGENTS.md`** at the root (the product rules:
> 64 earned rule bullets, the tooling table, the skills table, the two document-correction rules) plus
> **`.opencode/AGENTS.md`** (skills policy, the five-source tooling discipline, the third-party
> skill-vendoring policy). Both are in scope, and so is **`CONTEXT.md`** (architecture map and
> glossary — its state-shaped entries, like test counts, must match the codebase). **`docs/adr/*.md`
> is also in scope here**, which the generic instructions below exclude: an ADR is a decision record
> this repository actively cites from code comments, so a false claim in one is read as a design
> decision rather than as a record. The global `~/.config/opencode/AGENTS.md` is **out of scope** —
> it is the user's, not the project's.
>
> **The mechanical half already exists here; do not rebuild it.** `tools/release/` carries five
> gates that answer the parts of this skill a script can answer, and they run in
> `.githooks/pre-commit`:
> `check_doc_links.py` (links and anchors), **`check_doc_citations.py` (every `file.ext:LINE`
> citation still points inside the file it names — added 2026-10-03 after two stale citations sent
> a reader to the wrong place in one session)**, `check_text_encoding.py`,
> `check_profile_consistency.py`, and `check_rule_count.py`. Run them first; this skill covers what
> they cannot: whether a rule is *still true*, whether a *consequence* changed, and whether a
> *decision's rationale* has been overtaken by the code.
>
> Diff range: `git diff HEAD` for uncommitted work, otherwise `git log --oneline HEAD..upstream/dev`
> to see what upstream moved — this repository merges `Neroued/ninfer` into `dev` and its ADRs are
> rewritten when a decision is superseded, never edited in place without saying so.

Your rules file, **`CLAUDE.md`** or **`AGENTS.md`**, is a **steering document, not documentation**: your ground
rules, your conventions, and a current **map of where things live**. Its only failure mode that matters is being
**wrong**: a stale rule or a drifted map actively misleads the agent on every future run. This skill checks the
rules file against what just changed and proposes the **smallest** edit that keeps it true.

> **Wrong rules are worse than missing rules. A longer rules file is worse than a lean one.** Most changes
> need *no* edit at all. Adding a wrong or verbose line makes it worse.

## Input
- `$ARGUMENTS`: optional diff range. Default: uncommitted + staged (`git diff HEAD`); fall back to `main...HEAD`.
- **Scope: the project's rules file(s).** `AGENTS.md` at the root **and** `.opencode/AGENTS.md`, plus
  `CONTEXT.md` for state-shaped entries and `docs/adr/*.md` for decisions this repository cites from
  code. Everything else under `docs/` is documentation of the product, not rules, and is out of
  scope. This skill exists to keep the *rules and decisions* honest, nothing else.

## Process

### 0. Run the deterministic pre-pass
The generic pre-pass (`scripts/ref-check.ps1`) is **not part of this repository** and must not be
invoked here. This project's equivalent already runs on every commit:

```
C:/vllm-env/Scripts/python.exe tools/release/check_doc_links.py
C:/vllm-env/Scripts/python.exe tools/release/check_doc_citations.py
C:/vllm-env/Scripts/python.exe tools/release/check_text_encoding.py
C:/vllm-env/Scripts/python.exe tools/release/check_profile_consistency.py
C:/vllm-env/Scripts/python.exe tools/release/check_rule_count.py
```

A citation the gates already reject is not a finding for the table below — it is a broken
mechanical check, so fix that first and say so. Everything the gates cannot see is what this skill
is for: a rule whose *wording* is still true but whose *effect* is no longer what it was, a
consequence that has since been measured, or a rationale that the code has overtaken.

### 1. See what changed
`git diff <range>` + `git status`. Note: moved/renamed/removed files, new modules, changed conventions,
and any new invariant the change establishes.

### 2. Read the rules file as it is now
Load the project's rules file, `CLAUDE.md` or `AGENTS.md` (and any package-scoped ones). Hold each claim against the change set.

### 3. Flag ONLY these three things
1. **A stated rule or fact is now false**, e.g. "routes live in `src/routes/`" but they moved. → fix it.
2. **The architecture map drifted**. A path or "where things live" pointer no longer matches reality.
   → fix the wrong entry (don't catalog every new file).
3. **A new durable invariant must hold going forward**. The change introduces a rule that must stay true
   (e.g. "never call the DB from handlers, go through `repository/`"). → add it as **one line**.

Everything else, leave alone. Do **not** suggest an edit to *record that a feature was added* (that's a
changelog. The codebase is the source of truth), to restate what the code already makes obvious, or to add
background/rationale/prose that doesn't steer future work.

### 4. Write each suggestion the way CLAUDE.md should read
- **One bullet, not a paragraph.** A rule is a line, not an essay.
- **Keep the map current. Don't grow it.** Fix the wrong path; don't enumerate the new ones.
- **State rules in natural language; reference the codebase, never paste code.** Copied code goes stale; the
  codebase stays true. Good: "follow the error pattern in `src/core/errors/`." Bad: pasting the class.

## Output
```
## Rules-file drift check — range: <range>

### Fix (now false)
| Where | What's wrong | Minimal fix |
|-------|--------------|-------------|
| "Architecture" map | routes moved `src/routes/` → `src/api/routes/` | update the one path |

### Add (new invariant only)
- <one-line rule> — established by <the change that made it durable>

### Checked, still true — no edit
- <areas you verified need no change>
```
If nothing drifted: **"The rules file is still accurate for these changes, no edits needed."**

## Rules
- **Advisory.** Report the drift; only apply/piv-commit edits if the caller explicitly asks.
- **Rules file only** (`CLAUDE.md` / `AGENTS.md`). Not README, not docs.
- **Lean by default.** When in doubt, suggest nothing.
- **Run it before every merge** (or as part of `/piv-review-changes`) so your rules never drift behind the code.
