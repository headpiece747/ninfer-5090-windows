# Upstream's unmerged work: the candidate backlog, and what decides when to take any of it

**Why this file exists.** The port's reference corpus is upstream's tracker, and that tracker has two
halves: issues, which the survey recipe named from the start, and **open pull requests**, which it did
not. Six of Wallawalla47's sixteen engine changes were readable in the open PRs for a month while only
issues were searched -- an issue search cannot return a PR title, and no artifact survey surfaces an
engine change. The ranked list lived only in `.audit/models-rebuild.tsv`, which is untracked by design,
so a fresh clone could not see it. This is that list, in the tree.

**How to refresh it**, both halves and both query shapes:

    gh pr   list --repo Neroued/ninfer --state open
    gh issue list --repo Neroued/ninfer --search <term>

## Wait, with the trigger that would change the answer

| PR | what it does | state | why wait | revisit when |
|---|---|---|---|---|
| 273 | text profile of `ops::rmsnorm_rope`, routed to the two full-attention call sites | open, active author | platform-neutral, not a fix; a fork measured the same idea at 14-22%, and this PR's own figures are -30.0%/-21.8% at its 256 threshold | it stalls -- its predecessor #222 sat a month unreviewed |
| 264 | NVFP4 fused SwiGLU partial last M tile, so ragged widths stop falling back | open | finishes work already merged (`5f5fccab`, `05507ab0`); the `rows % 256` gate under `Rows::kPaired` is in this tree | merge |
| 292 | shared Q5 K-split MMA for small-batch linear, +11% MTP3 decode claimed | draft | not ready | leaves draft |
| 355 / 324 | nvfp4 tiled prefill KV split / FP8 TMA pilot for the attn prefill | draft | not ready | leaves draft |
| 167 | fp8 A8 GEMM stages operands through TMA | open | this tree's `fp8_mma_shared_byte` already computes the swizzle from the absolute byte address, so its headline defect is not ours; end-to-end -0.12%/+0.71%/+0.97% | merge |
| 335 | hybrid prefix cache replacing the checkpoint catalog | open, unreviewed | conflicts with this port's divergences; the cheapest route is being current when it merges | merge |
| 234 | n-gram copy drafting with reuse after compaction | proposal | the fork's commits are cherry-pickable if it stalls; the drafter-alignment finding makes this the *same axis* as this port's most promising open lever | it stalls, or the drafter work starts |
| 173 | rk2v4-e8 compressed KV (E8-root, 208 B/head-token) | open | platform-neutral | merge |
| 316 / 318 / 299 / 295 / 382 | agent and tool-call compatibility: Copilot/OpenAI shapes, quoted markers, duplicate parameters, reasoning summaries, custom tools | open, several authors | **the most user-visible items on this list** -- these lanes serve a coding agent | any of them merges |
| 165 | YaRN context extension (`--rope-yarn-factor`), with a linked implementation | open proposal; PR 130 closed | a *product* call: positional scaling to exceed the trained window | the product wants more than 262,144 |
| 233 | Windows native MSVC build with vcpkg-managed dependencies | open | this port already carries equivalent compiler fixes and surveyed it on 2026-09-18; upstream is Linux-only, so its mergeability buys nothing | upstream ever merges it -- then adopt it and delete this port's duplicate |
| 383 / 384 | noncausal single-pass decision inference for `pplx-decider-v1.1-27b` | open | a different model family, not this product's 27B | that family is ever targeted |

## Do, because nothing upstream is coming

| item | why not wait | where it stands |
|---|---|---|
| `SetConsoleOutputCP(CP_UTF8)` in the three apps | the manifest changes the *process* code page and was measured not to change the console's, so console rendering is still the user's console's business | named in `cmake/windows-utf8.manifest`; unshipped |
| the drafter's own precision, target held fixed | no upstream work on this engine's draft encoding; it is a lever directly on acceptance | open, and `lane-coverage-2026-10-08.md` calls it the most promising build work |
| the precision-assignment search | no upstream arm isolates q/k from v/o, or moves MLP activations from A8 to A16 | open |

## Closed by measurement rather than by merge

- The A8 NVFP4 activation route: refused -- `a8policy` loses 25.3% acceptance and 32% decode on the
  measured prompt set, and a kernel cannot change a penalty that lives in the logits.
- The `attn8` recipes: refused on the real-text instrument (-22.9%), since re-measured across four
  prompt treatments, all of them at or below parity.
- The A8 kernel path as a project: closed. The activation-format axis is not the lever the drafter is.

## The constraint every figure here is read under

Acceptance is protocol-dependent: the same artifact pair flips sign between prompt sets and between
greedy and the served sampling, and the depth choice is degenerate at this card's noise level. Nothing
in this area is quotable without its prompt and its sampling, which is why the instruments now print
their source and record `domain`/`sampling_applied` beside every number.
