---
name: how
description: "Use for \"how does X work\", code walkthroughs before changing something, and placement / ownership / layering questions (\"where should this live\", \"which project owns this\", \"is this the right layer\"). Explains subsystem architecture, runtime flow, onboarding mental models. Can critique architecture. For motivation (why was it built this way), use docs/adr/, CONTEXT.md, and git history."
---

> **Tool mapping (this project):** for C# symbols, prefer the indexed tools over raw search: `codegraph_explore` (one call returns symbol source + call paths + blast radius) and Glider (`glider_search_symbols`, `glider_find_references`, `glider_find_callers`, `glider_get_structure`). Raw Glob/Grep are for non-C# assets and after the index is cold. This host has no `why` skill; motivation questions route to `docs/adr/`, `CONTEXT.md`, and `git log`/`git blame`.

# How

Explore the codebase to answer "how does X work?" questions. Produce clear architectural explanations at the level of a senior engineer onboarding onto a subsystem. Enough to build a working mental model, not annotated source code.

Two modes:

1. **Explain** (default). Explore the codebase and produce a clear explanation
2. **Critique.** Explain first, then spawn multiple reviewers to independently identify architectural issues

## Explain Mode

### Step 1. Understand the Question and Assess Complexity

Parse what the user is asking about:

- "How does the rate limiter work?", a subsystem
- "How do we handle billing for on-demand usage?", a feature flow
- "How is the auth service structured?", an architectural overview
- "Walk me through what happens when a user submits a form", a runtime trace

In this repo, typical shapes: "How does the weather cluster resolve a city?", "How does a frame get from the compositor to the display?", "Walk me through what happens when touch lands on a widget".

Identify the scope. If ambiguous, state your best-guess interpretation before exploring. Don't ask. Let the user redirect if you're off.

**Assess complexity to decide the approach:**

- **Simple** (a single module, a small utility, a narrow question like "how does X work"): skip explorer agents; the explainer explores and explains in a single pass. Go to Step 2b.
- **Complex** (a subsystem spanning multiple files/projects, a cross-cutting feature, a full architectural overview): spawn parallel explorer agents first, then hand off to the explainer. Go to Step 2a.

When in doubt, lean simple. You can always spawn explorers if the explainer hits a wall.

### Step 2a. Explore (complex questions only)

Decompose the question into 2-4 parallel exploration angles, each a distinct slice of the subsystem so explorers don't duplicate work. Example split for "how does a frame reach the display?":

- Explorer 1: the composer and the frame buffer lifecycle
- Explorer 2: the delivery pipeline (encode, pool, pacing, drop accounting)
- Explorer 3: the transport (connect, init verdict, send, touch)

The right decomposition depends on the question. Use your judgment. Narrow questions: 2 explorers is fine. Broad subsystems: up to 4.

Spawn all explorers in a single message (`subagent_type: general`, read-only framing, session model):

- Each explorer gets the same base prompt from `references/explorer-prompt.md` plus a specific exploration angle naming its slice. Each explorer should:
- Locate the entry points (indexed symbol search, not raw grep for C#)
- Follow the thread: from an entry point, trace the call chain (callers, callees, data flow, type definitions), riding dynamic-dispatch hops (the index follows these; grep doesn't)
- Read the actual code, don't guess from file names
- Stop when it can describe the full path from input to output (or trigger to effect) without hand-waving any step
- Note things that are surprising, non-obvious, or that a newcomer would get wrong (this repo's ADR-referenced invariants are the house flavor of "surprising")

Each explorer returns structured findings: components found, flow traced, files read, anything non-obvious. Overlap between explorers is fine; the explainer reconciles.

Then proceed to Step 3.

### Step 2b. Direct Explain (simple questions)

Spawn a single subagent that explores and explains in one pass (`subagent_type: general`, read-only framing, session model).

The agent does its own exploration and writes the explanation directly. Read `references/explainer-prompt.md` for the communication style and output format. Same structure, just no explorer findings as input.

Proceed to Step 4.

### Step 3. Synthesize (complex questions only)

Once all explorers return, spawn a single subagent to synthesize their findings into one coherent explanation (`subagent_type: general`, read-only framing, session model).

The explainer gets all explorers' findings and writes the human-facing explanation (output format below). Read `references/explainer-prompt.md` for the full prompt template. The explainer reconciles overlapping findings, resolves contradictions, and weaves the slices into a unified picture.

### Step 4. Present

Present the explainer's output to the user. You may lightly edit for clarity or add context from the conversation, but don't substantially rewrite. The explainer's communication is the product.

### Output Format

Follow this structure, adapted to the question. Not every section is needed for every question.

**Overview.** 1-2 paragraphs. What it is, what it does, why it exists. Enough to decide whether to keep reading.

**Key Concepts.** The important types, abstractions, or seam interfaces. Brief definition of each, and the CONTEXT.md glossary entry when one exists. Not exhaustive, just the ones needed to understand the rest.

**How It Works.** The core of the explanation. Walk through the flow: what triggers it, what happens step by step, where data goes, the decision points. Prose, not pseudocode. Reference specific files and functions so the reader can go look, but don't dump code blocks unless a snippet is genuinely necessary.

**Where Things Live.** A brief map of the relevant files/directories. Not every file, just the ones needed to start working in this area.

**Gotchas.** Non-obvious or surprising things that would trip someone up. Historical context that explains why something looks weird (the ADRs are the source of truth for the deliberate-looking oddities: the synchronous transport, the static stores, the inert engine constructor). Known sharp edges.

## Critique Mode

Triggered when the user asks for architectural issues, problems, or improvements, not just understanding.

### Step 1. Explain First

Run the full explain flow above (Steps 1-4). You must understand the architecture before critiquing it.

### Step 2. Spawn Critics

After the explanation is complete, spawn 2-3 independent architectural critics as fresh subagents (`subagent_type: general`, read-only), all in a single message. One local model here, so the independence comes from fresh contexts, distinct framing, and the rubric, not from model diversity.

For each critic:
- `subagent_type`: `general`
- Scope: the explanation from Step 1 plus the relevant file paths. The critics are at minimum reasoning-level readers; escalate the depth of the framing when the architecture warrants it (the layered Sdk/Core/Widgets/Hardware/App boundary deserves the deeper pass).

Read `references/critic-prompt.md` for the prompt template. Each critic gets:
1. The explanation from Step 1 (so they don't re-explore)
2. The relevant file paths (so they can read the actual code)
3. The architectural critique rubric from `references/critique-rubric.md`

### Step 3. Lead Judgment

You're a pragmatic lead, not an aggregator.

Categorize findings:
- **Act on.** Architectural problems worth fixing now
- **Consider.** Real concerns, but the cost/benefit is unclear
- **Noted.** Valid observations, low priority
- **Dismissed.** Wrong, missing context, style preference, or a deliberate project decision documented in CONTEXT.md or an ADR (the synchronous transport and the reflection-instantiated widgets are the two the repo will keep pointing at you)

Present the explanation first (from Step 1), then the critique verdict below it. The explanation should stand on its own; someone who just wants to understand the system shouldn't wade through critique.