# Health Check: Grading Rubric Reference

Loaded by `/health-check` Step 2. This is the **canonical** grading system for
the kit. No other skill defines its own.

## The Cardinal Rule: Never Grade a Raw Count

A detector count is evidence, not a grade. Every tool in this kit returns a
`confidence` field and a `summary` histogram for exactly this reason.

- **Grade on `high` confidence findings only.** These are wrong regardless of context.
- **`medium` findings are review items, never grade inputs.** The pattern is
  suspicious but has legitimate uses the detector cannot rule out, a catch block
  that logs and rethrows, an EF query in a command handler that may edit downstream.
- **`suppressed` findings count for nothing**, but the report must state the count
  and the config path. Suppression must be visible, or it becomes a way to game
  the grade silently.
- Read `summary.byId`. It is complete even when the violation list is truncated.
  Never sample a truncated list and extrapolate.

If a raw signal contradicts an invariant documented in the target repo's
CLAUDE.md, **the invariant wins pending verification**. Verify, then grade.

## Dimension Rubrics

### Dimension 1: Build Health

Tool: `dotnet build --no-restore`, metric: error count, warning count.

| Grade | Criteria |
|-------|----------|
| A | 0 errors, 0 warnings |
| B | 0 errors, 1-5 warnings |
| C | 0 errors, 6-15 warnings |
| D | 0 errors, 16-30 warnings |
| F | Any errors, or 30+ warnings |

### Dimension 2: Code Quality

Tool: MCP `detect_antipatterns`, metric: **high-confidence** findings per 1K
production lines. The tool defaults to production scope; generated code is never
reported and test/migration code is excluded.

| Grade | Criteria |
|-------|----------|
| A | 0 high-confidence findings |
| B | < 0.1 per 1K lines |
| C | 0.1 - 0.5 per 1K lines |
| D | 0.5 - 1.5 per 1K lines |
| F | > 1.5 per 1K lines |

**Unreviewed-medium cap:** a project cannot be graded A while more than 25
medium-confidence findings remain untriaged. Cap at B and name the categories in
the report. A clean high-confidence count with hundreds of unexamined mediums is
not an A. It is an unexamined codebase.

These thresholds are far tighter than raw-count grading because the detectors are
precise now. A well-maintained 100K-line codebase should produce single-digit
high-confidence findings; dozens means something real is wrong.

Common findings: async void, sync-over-async, `new HttpClient()`, `DateTime.Now`,
broad catch blocks, string interpolation in logging, missing CancellationToken.

### Dimension 3: Architecture

Tools: MCP `get_project_graph` (dependency direction), `detect_circular_dependencies`
at both `scope: "projects"` and `scope: "types"`.

| Grade | Criteria |
|-------|----------|
| A | Correct dependency direction, 0 circular deps (project or type level) |
| B | Correct dependency direction, 1-2 type-level cycles (no project cycles) |
| C | 1-2 minor direction issues, or 3-5 type-level cycles |
| D | Project-level circular dependency, or significant layer violations |
| F | Multiple project-level cycles, no discernible architecture |

### Dimension 4: Test Coverage

Tool: MCP `get_test_coverage_map(projectFilter: each production project)`,
metric: % of production types with corresponding test classes.

**Check `applicable` first.** When the tool returns `applicable: false`, the
structural metric is invalid for this codebase and the percentage is meaningless.
Record the dimension as **Not assessed**, quote `notApplicableReason` and
`testMethodCount`, and **exclude it from the GPA** (divide by the number of
dimensions actually graded). A structurally invalid metric must never become an F.

| Grade | Criteria |
|-------|----------|
| A | 90%+ types have test classes |
| B | 75-89% |
| C | 50-74% |
| D | 25-49% |
| F | < 25% |
| Not assessed | `applicable: false`, excluded from GPA |

This is structural coverage (test class exists), not runtime line coverage. It
only works for suites written one test class per production type. Integration-
and feature-driven suites test by behaviour, and name matching cannot see that,
which is what `applicable: false` reports. To grade those, run a runtime coverage
tool (`dotnet test --collect:"XPlat Code Coverage"`) or cite an existing coverage
audit; do not substitute the structural number.

### Dimension 5: Dead Code

Tool: MCP `find_dead_code(scope: "solution", kind: "all", maxResults: 50)`.

Grade on `high`-confidence symbols. `conventionFiltered` reports symbols the tool
excluded because a reference search cannot see how they are bound (EF entity
configurations, hosted services, migrations, extension-method hosts). These are
**not** debt and are never penalised. `medium` symbols carry a `note` explaining
which convention their name matches; verify before removing.

| Grade | Criteria |
|-------|----------|
| A | 0-2 dead symbols |
| B | 3-8 |
| C | 9-15 |
| D | 16-25 |
| F | 25+ |

### Dimension 6: API Surface

Tools: MCP `get_public_api` (design review), `find_references` (overexposure).

| Grade | Criteria |
|-------|----------|
| A | Minimal public surface, proper return types, consistent naming |
| B | Mostly clean, 1-2 overexposed types |
| C | Several types expose internal details, inconsistent return types |
| D | Public APIs leak implementation, mixed return type patterns |
| F | No API design consideration, everything is public |

Check for: services that should be `internal`, methods returning `Task` instead
of `Task<Result<T>>` for fallible operations, mixed `TypedResults`/`IResult`,
public setters on types that should be immutable.

### Dimension 7: Security Posture

Tools: `dotnet list package --vulnerable --include-transitive`, MCP
`detect_antipatterns` (security patterns), scan for hardcoded secrets and
missing auth attributes.

| Grade | Criteria |
|-------|----------|
| A | 0 vulnerable packages, no hardcoded secrets, auth on all endpoints |
| B | 0 critical/high vulns, 1-2 low/medium vulns, clean auth |
| C | 1-2 medium vulns, or minor auth gaps |
| D | High-severity vuln, or missing auth on sensitive endpoints |
| F | Critical vuln, hardcoded secrets, or systemic auth gaps |

For a deep dive, run `/security-scan`. This dimension is a spot check.

### Dimension 8: Documentation

Scan: XML docs on public APIs; README existence and currency.

| Grade | Criteria |
|-------|----------|
| A | 90%+ public APIs have XML docs, README is comprehensive |
| B | 70-89% coverage, README covers basics |
| C | 50-69% coverage, README exists but sparse |
| D | < 50% coverage, minimal README |
| F | No XML docs, no README or severely outdated |

## GPA Calculation

Letter grades to points: A=4.0, B=3.0, C=2.0, D=1.0, F=0.0.
GPA = average across the dimensions **actually graded**. Dimensions marked
"Not assessed" are excluded from both numerator and denominator.

| GPA Range | Overall Assessment |
|-----------|--------------------|
| 3.5 - 4.0 | Excellent: production-ready, well-maintained |
| 3.0 - 3.4 | Good: solid foundation, minor improvements needed |
| 2.5 - 2.9 | Fair: functional but accumulating tech debt |
| 2.0 - 2.4 | Needs Work: significant improvements required |
| < 2.0 | Critical: major structural issues to address |

## Report Card Template

```markdown
## Project Health Report

**Project:** MyApp | **Date:** 2026-03-04 | **Assessed by:** Claude (MCP-assisted)

### Grades

| Dimension | Grade | Score | Key Finding |
|-----------|-------|-------|-------------|
| Build Health | A | 95 | 0 errors, 2 pre-existing warnings |
| Code Quality | B | 82 | 3 high-confidence findings in 4.2K lines; 31 medium untriaged |
| Architecture | A | 92 | Clean dependency direction, 0 circular deps |
| Test Coverage | C | 68 | 34/50 production types have test classes |
| Dead Code | B | 85 | 5 unused methods identified |
| API Surface | B | 80 | 2 overexposed service types |
| Security | A | 94 | 0 vulnerable packages, auth coverage complete |
| Documentation | D | 55 | 12/30 public APIs have XML docs |

### Overall GPA: 3.0 (B-)

### Detector Triage

Mandatory whenever Dimension 2 or 5 is graded. Shows how the raw signal became a
grade, so a reader can audit the judgement rather than trust it.

| Id | Raw | Suppressed | Medium | High | Verdict |
|----|-----|------------|--------|------|---------|
| AP005 | 44 | 0 | 44 | 0 | All log-and-rethrow resilience wrappers — review, not debt |
| AP010 | 9 | 0 | 8 | 1 | 1 genuine read-only query; 8 in command handlers |
| AP004 | 2 | 0 | 0 | 2 | Real — `SystemSeeder` should take `TimeProvider` |

Suppression config: `.cwm-navigator.json` (none / path). State it either way.

### Priority Recommendations

1. **Test Coverage (C -> B):** Add test classes for these 16 untested types:
   - `OrderService`, `PaymentProcessor`, `ShippingCalculator` (critical path)
   Estimated effort: 2-3 days

2. **Documentation (D -> C):** Add XML docs to the 8 endpoint classes first
   (user-facing), then the 10 public service interfaces.
   Estimated effort: 1 day

3. **Code Quality (B -> A):** Fix 3 anti-patterns:
   - `OrderService.cs:47` — `DateTime.Now` → `TimeProvider.GetUtcNow()`
   - `PaymentClient.cs:23` — `new HttpClient()` → `IHttpClientFactory`
   - `NotificationHandler.cs:12` — `async void` → `async Task`
   Estimated effort: 1 hour
```

Every recommendation must be specific (named types, file:line), prioritized,
and effort-estimated. "Improve test coverage" is not actionable.

## Trend Tracking

If a previous health report exists, compare grades:

```markdown
### Trend

| Dimension | Previous | Current | Change |
|-----------|----------|---------|--------|
| Build Health | B | A | Improved — fixed 4 warnings |
| Code Quality | C | B | Improved — resolved 7 anti-patterns |
| Test Coverage | C | C | No change — still 68% |
```

Improving grades validate cleanup efforts; regressions flag where review
discipline slipped.
