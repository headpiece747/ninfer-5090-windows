# Upstream reports

Three issues found while working on this port, each written up with a reproduction and
evidence so it can be filed as-is. **None has been posted.** They are drafts, and posting them is
an outbound action rather than a code change.

**Pulling an upstream release?** Read [`fork-divergence.md`](fork-divergence.md) first. It records
which of our changes touch files upstream also owns, and which are Windows-permanent and cannot
conflict. `dev` is a strict superset of `upstream/master` (0 commits behind), so the only risk is
recognising our own patches when upstream edits the same shared logic.

| Report | Target | What it is | Strength |
| --- | --- | --- | --- |
| `ninfer-tma-descriptor-graph-capture.md` | `Neroued/ninfer`, aimed at the open Windows/TMA PRs `#233` and `#82` | the device-buffer TMA descriptor path is not safe under CUDA Graph capture | strongest: a `compute-sanitizer` trace, a named mechanism, and a fix |
| `codegraph-init-replaces-junction.md` | `@colbymchenry/codegraph` 1.5.0 | `init` replaces a junction/symlink with an empty directory, then reports "No files found to index" | reproduction, no fix |
| `ninfer-reasoning-effort-mismatch.md` | `Neroued/ninfer` | the engine advertises six `reasoning_effort` values; the artifact's chat template accepts four, so `minimal` and `high` return HTTP 400 after passing validation | a two-line change either way |

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

The fix scales the grant up to a 250 ms ceiling (`materialization_budget.h`, `kMaximumGrantNs`),
with the economic term still governing below the ceiling and the boundary allowance capping both.
It was committed as `2fcffaa9`, reverted as `7ae7bb40` with no recorded reason, and re-landed here.
See `docs/research/prefix-state-eviction.md`.

## Not a bug: the fused DFlash2 binding

Worth recording so nobody re-derives it. This tree at one point required a fused
`dflash2/layers/*/attention/query_key_value` parameter. **Upstream is consistent and correct
here**: `load/dflash.cpp` binds the separate `attention/query`, `attention/key` and
`attention/value`, and `parameters.cpp` assembles the fused parent itself with
`ops::prepare_attn_input_proj_weights`. `tools/convert/qwen3_5.py` groups those three rather
than emitting a fused name. The fused *binding* was our own divergence, and it is why
upstream-shaped artifacts (the official Qwen3.8-27B image, community fuller-NVFP4 images)
were refused at startup. It is fixed in `9898ecb1`; there is nothing to report upstream.
