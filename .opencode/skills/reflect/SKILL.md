---
name: reflect
description: "Spawn three parallel review subagents over the active session, surface learnings, and route each to a concrete edit on an existing skill. Use when the user says reflect."
disable-model-invocation: true
---

> **Port note (this repo):** this host does not expose an `agent-transcripts/` directory. The reviewers work from a session digest the parent writes (task, decisions, dead ends, the working path, the user's corrections), plus the show-me-your-work trail when one exists. The routing targets are the skills installed under `.opencode/skills/`.

# Reflect

Mine the current conversation for durable learnings, then route them into skill edits.

## When to invoke

- The user said "reflect" or "/reflect".
- A complex task (5+ tool calls) just landed cleanly and the recipe is worth keeping.
- The agent hit dead ends, found the working path, and the path generalizes.
- The user corrected the agent's approach mid-task.
- A non-trivial workflow emerged that isn't captured anywhere.

Skip when the conversation is trivial, off-topic, or already covered by an existing skill the parent followed correctly. One-offs are not learnings.

## Process

### 1. Build the session digest

Write a tight digest of the session before fanning out: the task, the approach that worked, the dead ends and what revealed the working path, every user correction verbatim when short, and the durable facts that were created (paths, decisions, test names). If a show-me-your-work trail exists, pass its path; the digest then stays a pointer, not a retelling. The digest is the reviewers' "transcript".

### 2. Spawn three reviewers in parallel

One message, three subagents (`subagent_type: general`, session model), tools allowed (they may verify a finding against the repo or a Glider query; the prompt forbids file writes, the parent applies edits).

| Lens | Prompt template |
|---|---|
| Judgment | `references/judgment-reviewer.md` |
| Tooling | `references/tooling-reviewer.md` |
| Divergent | `references/divergent-reviewer.md` |

Pass each template verbatim, substituting the digest (and trail path) where marked. Reviewers return findings in the subagent response.

### 3. Synthesize

One subagent (`subagent_type: general`, session model), tools allowed. The synthesizer's quality check includes spot-verifying citations against the repo. Use `references/synthesizer.md` verbatim, with each reviewer's full output inlined where marked. The synthesizer returns a structured Accepted / Rejected / Backlog list.

### 4. Structural enforcement check

Sanity-check the synthesizer's Accepted list. For any item that would be enforced more reliably by a lint rule, MSTest pin, script, or runtime check, move it from Accepted to Backlog (this repo's house style is the pin: the `FontCacheEvictionTests`-style boundary test, the lockstep option-array pin). The synthesizer already applies this criterion; this is a final pass before edits land. See the **encode-lessons-in-structure** principle skill.

### 5. Apply

Before applying any Accepted edit, present the synthesizer's full Accepted/Rejected/Backlog output to the user and wait for explicit approval. The user picks which subset to apply and may redirect routings. Skill changes affect every future agent in this repo; do not auto-apply.

Backlog items land in the repo's own record: `CONTEXT.md` (a glossary entry or a design-decision row) when the learning is architectural, `docs/adr/` when it is a decision, or a todo in this session's trail when it is tooling. There is no external tracker.

For each approved Accepted item, follow the Routing field exactly:

- Trivial existing-skill edit (a one-line bullet, a tightened sentence, a stale fact corrected): parent does directly.
- Substantive existing-skill edit (a new section, a new pattern table, more than ~10 lines): route through the `authoring-a-skill` playbook (poteto-mode), draft, validate, prove it changes behavior.
- `tune description: <skill path>` (the skill exists but didn't trigger when it should have): tighten the skill's `description` frontmatter so it fires next time.
- `new skill: <kebab-name>`: create it under `.opencode/skills/<kebab-name>/` via the `authoring-a-skill` playbook. Do not invent the shape ad hoc.

If your environment ships a SKILL.md validator, run it on every touched skill before declaring done. Skip this step if it doesn't.

### 6. Summarize for the user

Short list, no preamble:

- Edits applied: `<skill path>`. What changed, one line each.
- New skills created: `<skill path>`. One line each (rare).
- Backlog filed to: `<CONTEXT.md section / ADR / trail>`. One line each.
- Dropped: one line per rejected finding + reason from the synthesizer.