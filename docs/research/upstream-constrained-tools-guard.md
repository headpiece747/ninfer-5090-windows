# Upstream's constrained-tool guard rejects six of upstream's own prefix scenarios

**Status: not filed upstream, and not fixed upstream** as of 2026-10-06. This note is the report, kept
here because the port does not file on the upstream tracker without the owner's say-so. It is ready to
paste as an issue on `Neroued/ninfer`; everything below was checked against `upstream/dev`, not
inferred.

## What is wrong

Upstream `41e50d0d` ("feat: add constrained tool calling") makes a request that declares tools take a
constrained contract. The guard that enforces the constraint's preconditions rejects a merged stop
policy with `include_model_defaults = false`:

    src/models/qwen3_5/frontend/frontend.cpp:919
    "constraints require default EOS, text output, no custom stops, and one output language"

`41e50d0d` did not touch `tests/models/qwen3_5/test_engine_prefix_real.cpp`, and that file sets
`include_model_defaults = false` at eleven sites. Six of its scenarios therefore throw at the guard:

| scenario | in this tree |
|---|---|
| `exercise_explicit_prefix` | `tests/models/qwen3_5/test_engine_prefix_real.cpp:553` |
| `exercise_nested_tool_markers` | `:642` |
| `exercise_anthropic_prefix_regression` | `:701` |
| `exercise_rewrite_checkpoints` | `:1084` |
| `exercise_agent_continuation` | `:1206` |
| `exercise_rewrite_branch` | `:1363` |

`ninfer_qwen3_5_agent_continuation_real_test` is registered by upstream's own
`tests/models/qwen3_5/tests.cmake`, so this is upstream's coverage, not a downstream test.

**The control is inside the data**: `exercise_shared_rewrite_materialization`
(`tests/models/qwen3_5/test_engine_prefix_real.cpp:844`) also declares tools and passes, because it
never disabled the model's default stops. That separates "a declared tool implies a constraint" from
"this particular stop policy cannot carry one", and it is why the guard is the suspect rather than the
scenario construction.

## Why it is silent

The throw is an unhandled `RequestError` escaping `Engine::generate`, so the binary fail-fasts
(`0xC0000409`, `BEX64`) printing nothing. A crash dump names frames
(`EngineCore<ModelInstance>::submit -> Engine::submit -> Engine::generate -> lambda -> terminate`),
which says a throw escaped but not which one. Only after guarding the test's `main` so a scenario
reports its failure did all six print the same message.

## Reproduction

1. Build with `BUILD_TESTING=ON`; set `NINFER_TEST_ARTIFACT` to any v3 artifact.
2. `ctest -R ninfer_qwen3_5_prefix_real_test`, or run the binary and select a scenario with the
   `NINFER_PREFIX_REAL_SCENARIO` environment variable (`tests/models/qwen3_5/test_engine_prefix_real.cpp:1990`).
3. The six above throw at the guard; the other seven pass.

## The two candidate fixes

* The scenarios keep the model's default stops when they declare tools. This is what this port did to
  unblock itself: every assertion unchanged, the six sites at
  `tests/models/qwen3_5/test_engine_prefix_real.cpp` (`:561` and `:656` carry the port's comment).
* A tool-only constraint stops requiring `include_model_defaults`. This changes what production
  accepts, so it needs the contract's intent checked rather than assumed; the guard's other clauses
  (single output language, default EOS, no raw output) are not obviously relaxable.

## What was checked before treating this as unreported

Searched on 2026-10-06, all returning nothing on this defect:

* `gh api search/issues` for `include_model_defaults` (only #197, about `ignore_eos`),
  `"one output language"` (0), `InvalidToolConstraint` (0), `"constrained tool"` (5 hits, none about
  the guard), `test_engine_prefix_real` (only #313 and #335, unrelated);
* `git log 41e50d0d..upstream/dev -- src/models/qwen3_5/frontend/frontend.cpp
  tests/models/qwen3_5/test_engine_prefix_real.cpp` — **no commits**: upstream has not touched the
  guard or the test file since the commit that introduced the disagreement.

The maintainer's own note in #366 says "next is constrained decoding which improves tool calling", so
this guard is new work and a report about it is expected rather than unwelcome.
