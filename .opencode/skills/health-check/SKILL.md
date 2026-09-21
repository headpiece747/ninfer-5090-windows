---
name: health-check
description: >
  Multi-dimensional health assessment for .NET projects with letter grades (A-F)
  using Roslyn MCP tools. Evaluates 8 dimensions: build health, code quality,
  architecture, test coverage, dead code, API surface, security posture, and
  documentation. Produces a structured report card with actionable recommendations.
  Load this skill when: "health check", "how healthy is this", "project health",
  "code quality report", "grade this project", "assess codebase", "quality audit",
  "technical assessment", "codebase review", "report card".
---

> **Tool mapping (this project):** Roslyn MCP tools map to Glider as follows:
> `get_diagnostics` → glider_get_diagnostics; `detect_antipatterns` →
> glider_semantic_query; `find_dead_code` → glider_find_unused_symbols /
> glider_find_unused_parameters; `get_test_coverage_map` → no Glider
> equivalent, approximate with test-class name matching against
> ModernWigiDash.Tests. `get_project_graph` → glider_get_project_graph.

# /health-check: 8-Dimension Project Assessment

## What

Runs a data-driven health assessment across 8 dimensions, each graded A-F with
the specific data points that produced the grade, and rolls them into a GPA.
Gut feeling is not a grade: every dimension uses MCP tools or CLI commands, and
every grade below A comes with specific, prioritized, effort-estimated fixes. "Add test classes for OrderService, PaymentProcessor, ShippingCalculator" is
actionable; "improve test coverage" is not.

This skill owns the **canonical grading system** for the kit. The full rubrics,
GPA scale, and report template live in `references/grading-rubric.md`, load
that file when running an assessment.

Tone is diagnostic, not punitive: a C grade is an improvement path, not a
failure.

## When

- Onboarding to an unfamiliar or new project, set the baseline
- "How healthy is this?", "grade this project", "codebase review", "report card"
- Pre-release quality gate, or monthly/quarterly maintenance review
- After a cleanup sprint (`/de-sloppify`), re-grade to show progress
- Tech-debt prioritization. Lowest grades get the next sprint's attention

## How

### Step 1: Choose Scope

| Scenario | Dimensions |
|----------|------------|
| Full assessment (onboarding, pre-release, monthly review) | All 8 |
| Quick health (mid-sprint checkpoint, before a demo, after a merge) | 1-4 only |
| After major refactor | 1 (Build), 3 (Architecture), 4 (Tests) |
| Post-dependency update | 1 (Build), 7 (Security) |
| After cleanup sprint | Re-grade only the cleaned dimensions |

### Step 2: Run the Dimensions

Read `references/grading-rubric.md` for the grade thresholds, then collect data
per dimension. For deep code-quality dimensions, delegate to the
`code-reviewer` agent with the `code-review` skill.

| # | Dimension | Data source |
|---|-----------|-------------|
| 1 | Build Health | `dotnet build --no-restore`, errors + warnings |
| 2 | Code Quality | MCP `detect_antipatterns`, read `summary`, grade high-confidence only |
| 3 | Architecture | MCP `get_project_graph` + `detect_circular_dependencies` (projects AND types) |
| 4 | Test Coverage | MCP `get_test_coverage_map`, **check `applicable` first** (structural, not line coverage) |
| 5 | Dead Code | MCP `find_dead_code(scope: "solution")`, grade high-confidence; ignore `conventionFiltered` |
| 6 | API Surface | MCP `get_public_api` + `find_references`, overexposure, return-type consistency |
| 7 | Security Posture | `dotnet list package --vulnerable --include-transitive` + secrets/auth spot check (deep dive: `/security-scan`) |
| 8 | Documentation | XML doc coverage on public APIs + README currency |

### Step 2.5: Triage Gate (before any grade is assigned)

Detector output is evidence, not a grade. Pass every finding through this gate
first. It is what stops a noisy count becoming a wrong letter.

1. **Read `summary`, not the violation list.** `summary.byId` is complete even
   when the list is truncated. Never sample a truncated list and extrapolate.
2. **Drop `suppressed`.** Record the count and `summary.suppressionConfig` in the
   report so suppression stays visible.
3. **Set aside `medium`.** These are review items, not grade inputs. Summarise
   them by category; do not fix or count them.
4. **Grade `high` only.** These are wrong regardless of context.
5. **Check invariants.** If a signal contradicts something the target repo's
   CLAUDE.md documents as deliberate, the invariant wins, verify before grading.
6. **Read before asserting.** Never describe a finding you have not opened. If
   you cannot open all of them, report the ones you did and say so.

Fill in the triage table from `references/grading-rubric.md` as you go. If the
table cannot be filled, the dimension is not ready to grade.

### Step 3: Grade and Aggregate

Apply the rubric thresholds exactly, never grade on a curve ("pretty good for
a project this size" is how standards erode; 15 warnings is a C regardless of
project size). Convert to GPA (A=4.0 … F=0.0), averaging only the dimensions
actually graded. A dimension marked "Not assessed" is excluded from the GPA,
never scored as an F.

### Step 4: Report

Produce the report card from the template in `references/grading-rubric.md`:
grades table with key findings, overall GPA, and priority recommendations,
each with named types/files, priority order, and effort estimates. If a
previous report exists, append the trend comparison table.

## Example

```
User: /health-check

Claude: Running full 8-dimension assessment...

| Dimension | Grade | Key Finding |
|-----------|-------|-------------|
| Build Health | A | 0 errors, 2 warnings |
| Code Quality | B | 3 high-confidence findings in 4.2K lines; 31 medium untriaged |
| Architecture | A | Clean direction, 0 cycles |
| Test Coverage | Not assessed | Integration-driven suite — structural metric invalid |
| Dead Code | B | 5 unused methods (79 convention-discovered, not counted) |
| API Surface | B | 2 overexposed service types |
| Security | A | 0 vulnerable packages |
| Documentation | D | 12/30 public APIs documented |

Overall GPA: 3.1 (B) — averaged over 7 graded dimensions.

Triage: 44 AP005 raw → all log-and-rethrow wrappers (medium); 2 AP004 real.

Priority: (1) `SystemSeeder` → `TimeProvider`, ~15 min; (2) XML docs on the 8
endpoint classes, ~1 day; (3) review the 44 catch blocks or suppress by path.
```

## Related

- `references/grading-rubric.md`: canonical rubrics, GPA scale, report template
- `/de-sloppify`: cleanup pipeline for the issues a health check surfaces
- `/security-scan`: deep 6-layer scan behind Dimension 7
- `/code-review`: per-change review (this skill grades the whole project)
- `/verify`: pass/fail pipeline for a change set, not a graded assessment
