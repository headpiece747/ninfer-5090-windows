# Upstream reports

Two issues found while working on this port, each written up with a reproduction and
evidence so it can be filed as-is. **Neither has been posted.** They are drafts.

| Report | Target | What it is |
| --- | --- | --- |
| `codegraph-init-replaces-junction.md` | `@colbymchenry/codegraph` | `init` replaces a junction/symlink with an empty directory and then indexes nothing |
| `ninfer-tma-descriptor-graph-capture.md` | `Neroued/ninfer` (and the open Windows/TMA PRs `#233`, `#82`) | the device-buffer TMA descriptor path is not safe under CUDA Graph capture; reproduced with `compute-sanitizer`, fix identified |

The second report also notes a timing-dependent test
(`test_candidate_search_prefers_deep_reuse_without_eviction`, 5 ms wall-clock search budget)
that fails here and passes elsewhere.

## Not a bug: the fused DFlash2 binding

Worth recording so nobody re-derives it. This tree at one point required a fused
`dflash2/layers/*/attention/query_key_value` parameter. **Upstream is consistent and correct
here**: `load/dflash.cpp` binds the separate `attention/query`, `attention/key` and
`attention/value`, and `parameters.cpp` assembles the fused parent itself with
`ops::prepare_attn_input_proj_weights`. `tools/convert/qwen3_5.py` groups those three rather
than emitting a fused name. The fused *binding* was our own divergence, and it is why
upstream-shaped artifacts (the official Qwen3.8-27B image, community fuller-NVFP4 images)
were refused at startup. It is fixed in `9898ecb1`; there is nothing to report upstream.
