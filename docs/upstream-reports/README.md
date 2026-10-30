# Upstream reports

Issues found while working on this port, each written up with a reproduction and
evidence so it can be filed as-is. **None has been posted.** They are drafts, and posting them is
an outbound action rather than a code change.

**Pulling an upstream release?** Read [`fork-divergence.md`](fork-divergence.md) first. It records
which of our changes touch files upstream also owns, and which are Windows-permanent and cannot
conflict. `dev` is aligned with `upstream/dev`, so the only risk is
recognising our own patches when upstream edits the same shared logic.

## The removed record

`docs/research/` is not in the tree. It held 68 dated investigation notes, and was deleted from
`dev` by `8475086f` and `f8bbc3fd` on 2026-10-13 because nothing in the tree referenced them. The
deletion was reviewed and approved; the mistake was deleting a file that a 2026-10-06 restoration
pass had put back after an earlier merge dropped it.

The corpus survives under a tag, and reading one note is one command:

    git show docs-archive-2026-10-13:docs/research/<name>.md

The tag is at `f8bbc3fd^` = `4315b6e0`, the last commit that had all 68. Do not treat
`C:\AI\actions-runner\_work\...` as the archive: that is an Actions runner work directory, and
`actions/checkout` wipes it on the next run.

**The ratchet cannot see a file a merge made identical to upstream.** On 2026-10-17 a 45-commit
merge reverted nine reviewed files, and `check_port_delta.py --check` reported them as
"stale -- trim them" -- the gate's advice would have erased the evidence of the loss, because a
file at zero net divergence is invisible to it. The merge base of that merge was `919721877`, not
`438ae67c`; `merge-base HEAD upstream/dev` is only the post-merge tip. Compare against the merge
base of the two parents, or tag the pre-merge state (`pre-merge-<sha>`) so `git diff
pre-merge-<sha> <merge> -- <path>` answers per path what the merge took.

| Report | Target | What it is | Strength |
| --- | --- | --- | --- |
| `ninfer-tma-descriptor-graph-capture.md` | `Neroued/ninfer`, aimed at the open Windows/TMA PRs `#233` and `#82` | the device-buffer TMA descriptor path is not safe under CUDA Graph capture | strongest: a `compute-sanitizer` trace, a named mechanism, and a fix |
| `codegraph-init-replaces-junction.md` | `@colbymchenry/codegraph` 1.5.0 | `init` replaces a junction/symlink with an empty directory, then reports "No files found to index" | reproduction, no fix |
| `ninfer-reasoning-effort-mismatch.md` | `Neroued/ninfer` | the engine advertises six `reasoning_effort` values; the artifact's chat template accepts four, so `minimal` and `high` return HTTP 400 after passing validation | a two-line change either way |
| `ninfer-small-context-reuse-hang.md` | `Neroued/ninfer` | `--max-context 512` with prefix reuse resolves a 1.00 GiB pool that startup accepts, then the first request never completes; an explicit capacity too small for a unit deadlocks instead of being refused | reproduction with controls, no Engine fix |
| `ninfer-thinking-grammar-escape.md` | `Neroued/ninfer` | a GBNF literal with a 2-token thinking budget returns the literal plus one trailing byte, so a constrained response is not exact | deterministic reproduction; cause is a hypothesis |

## The formerly failing test, and the fix

The suite is green against the recorded baseline, which is the authority for how many tests there
are and which failures are expected by construction: `tools/release/test_baseline.json`.
`ninfer_resource_manager_test`'s
`test_candidate_search_prefers_deep_reuse_without_eviction` used to fail and was recorded in
`tools/release/test_baseline.json`; that record is now empty.

The earlier reading here -- a 5 ms **wall-clock** budget, so the outcome depends on the machine --
was wrong. `950c87cb` had already let the planner take a caller-supplied clock, and the outcome is
unchanged with it, so the failure was deterministic policy rather than timing. The optional search's
initial grant was `min(5 ms, incumbent_cost / 20, allowance)`, and the flat 5 ms cap dominated
because a root incumbent's `/20` term is hundreds of milliseconds and never binds. The search
verified too few targets to reach the two-action preserving closure, so one-step eviction won the
incumbent; on a busy engine the same cap let a reuse target go unconfirmed and admission fell back
to the root incumbent, re-prefilling the whole prompt (upstream issue #229, TTFT 142 s -> 1.2 s
once the grant scales).

The fix scaled the grant up to a 250 ms ceiling (`materialization_budget.h`, `kMaximumGrantNs`),
with the economic term still governing below the ceiling and the boundary allowance capping both.
*(Note: With upstream's `bea5a04ec` / PR #366 context-cache unification, `materialization_budget.h`
was retired and issue #229 closed upstream. See `fork-divergence.md`.)*

## Not a bug: the fused DFlash2 binding

Worth recording so nobody re-derives it. This tree at one point required a fused
`dflash2/layers/*/attention/query_key_value` parameter. **Upstream is consistent and correct
here**: `load/dflash.cpp` binds the separate `attention/query`, `attention/key` and
`attention/value`, and `parameters.cpp` assembles the fused parent itself with
`ops::prepare_attn_input_proj_weights`. `tools/convert/qwen3_5.py` groups those three rather
than emitting a fused name. The fused *binding* was our own divergence, and it is why
upstream-shaped artifacts (the official Qwen3.8-27B image, community fuller-NVFP4 images)
were refused at startup. It is fixed in `9898ecb1`; there is nothing to report upstream.
