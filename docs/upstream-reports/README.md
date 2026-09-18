# Upstream reports

Three issues found while working on this port, each written up with a reproduction and
evidence so it can be filed as-is. **None has been posted.** They are drafts, and posting them is
an outbound action rather than a code change.

| Report | Target | What it is | Strength |
| --- | --- | --- | --- |
| `ninfer-tma-descriptor-graph-capture.md` | `Neroued/ninfer`, aimed at the open Windows/TMA PRs `#233` and `#82` | the device-buffer TMA descriptor path is not safe under CUDA Graph capture | strongest: a `compute-sanitizer` trace, a named mechanism, and a fix |
| `codegraph-init-replaces-junction.md` | `@colbymchenry/codegraph` 1.5.0 | `init` replaces a junction/symlink with an empty directory, then reports "No files found to index" | reproduction, no fix |
| `ninfer-reasoning-effort-mismatch.md` | `Neroued/ninfer` | the engine advertises six `reasoning_effort` values; the artifact's chat template accepts four, so `minimal` and `high` return HTTP 400 after passing validation | a two-line change either way |

## The one failing test, and why it is not being fixed

The suite reports 121 of 122. The failure is `ninfer_resource_manager_test`, specifically
`test_candidate_search_prefers_deep_reuse_without_eviction`.

`tests/test_resource_manager.cpp:3443` sets `allowance.limit_ns = 5'000'000`, a 5 ms **wall-clock**
search budget. The planner in `src/runtime/engine/context_cache/materialization_planner.h:209`
starts a clock, and `:700` compares elapsed time against that budget. So the planner explores
however many candidates fit in 5 ms on the machine running it, and the test then asserts a
specific eviction ordering. On this machine the search gets less far, a different candidate wins,
and the assertion fires. It passes on the author's machine.

Three ways to make it green, and why none is being taken:

1. **Raise the budget in the test.** One line, but it changes the test to match the machine while
   still asserting an outcome the planner never promised.
2. **Add a deterministic search mode to the planner.** Correct in principle, but it is production
   surface added to satisfy a test, which `AGENTS.md` rules out for hypothetical needs.
3. **Leave it and record it.** This is the choice.

The reasoning for 3: the test is not in the shipped path. It asserts an internal planner ordering,
the engine serves correctly at 262,144 across all four profiles, and the failure is a property of
the test rather than of what ships. A timing-sensitive assertion is the author's to fix, and the
TMA report already names it so the finding travels upstream with the rest.

If a green suite is ever wanted more than an accurate one, option 1 is the change to make, and the
commit should say plainly that the budget was raised to match the machine.

## Not a bug: the fused DFlash2 binding

Worth recording so nobody re-derives it. This tree at one point required a fused
`dflash2/layers/*/attention/query_key_value` parameter. **Upstream is consistent and correct
here**: `load/dflash.cpp` binds the separate `attention/query`, `attention/key` and
`attention/value`, and `parameters.cpp` assembles the fused parent itself with
`ops::prepare_attn_input_proj_weights`. `tools/convert/qwen3_5.py` groups those three rather
than emitting a fused name. The fused *binding* was our own divergence, and it is why
upstream-shaped artifacts (the official Qwen3.8-27B image, community fuller-NVFP4 images)
were refused at startup. It is fixed in `9898ecb1`; there is nothing to report upstream.
