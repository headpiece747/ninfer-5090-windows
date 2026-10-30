# Thinking + a literal grammar lets one byte escape the constraint

**Target:** `Neroued/ninfer`
**Severity:** medium — a constrained response is no longer exactly the constrained text. The
constraint holds for every byte except a trailing one, so a downstream consumer that trusts the
grammar (a JSON parser, a tool-call router) can receive text the grammar does not allow.

**Found:** 2026-10-30, integrating `upstream/dev` `458863be2` into this port. Upstream's own
assertion in `tests/models/qwen3_5/test_engine_grammar_real.cpp` (line 197-202) fails; the port did
not add this case.

## Summary

A GBNF literal with a 2-token thinking budget returns the literal **plus one trailing byte**:

```
expected  "  {\"value\":\"你好\",\"literal\":\"<tool_call>x</tool_call>\"}"
observed  "  {\"value\":\"你好\",\"literal\":\"<tool_call>x</tool_call>\"}]"
```

The thinking block is injected as expected (`thinking.applied == 1`, `injected_tokens == 25`), and
the same literal grammar without thinking is exact — so the literal, the model and the artifact are
consistent, and the loss is in the thinking/grammar composition.

## Reproduction

```
ninfer_qwen3_5_grammar_real_test          # NINFER_TEST_ARTIFACT set, no extra argv
```

against a Qwen3.8-27B `nvfp4qat` v3 artifact, prints:

```
choice/regex: literals, fullmatch, thinking, continuation, EOS, truncation and mixed rows passed
literal temp=0 reuse=8 prepare_ms=0.94 decode_ms=289.9
literal temp=0.8 reuse=8 prepare_ms=0.18 decode_ms=268.5
GBNF integration: thinking control broke constrained content
```

The failing case is deterministic (greedy, `temperature = 0`): the literal is constrained, so
sampling cannot explain the extra byte.

## What the port does

Nothing is worked around. The assertion is upstream's and the behaviour is wrong, so the port leaves
the test failing and records the case here rather than baselining it — a baseline entry would make
the gate green while the defect stayed in the shipped engine.

## Candidate cause, not confirmed

The observed shape — every constrained byte correct, then one unconstrained byte before EOS — is
what an accepted-state or concatenation boundary that completes one byte early produces. Read from
the tree, not measured: `src/text/grammar.cpp` composes the thinking barrier with the user grammar
through `xgrammar::Grammar::Concat`, and `1407c62f6` rewrote `reasoning_prefix` from a hand-built DFA
to a `TagDispatch` + `AddScalarStringFSM`. That is a hypothesis about where to look, not a
demonstrated cause: the port did not bisect the composition, because the code is in the vendored
`third_party/xgrammar` and a fix belongs with its owner.

## Controls that passed

- The same literal without thinking, greedy and at 0.8, exact at both temperatures.
- Choice, regex, EOS, truncation, continuation and mixed-row cases in the same binary.
- `thinking.applied` and `injected_tokens > 0` both hold, so the failing half is the content
  boundary, not the thinking injection.

## Configurations tested

Measured on 2026-10-30, same artifact:

| argv | result |
| --- | --- |
| (none) — `none`, CUDA graphs | literal case fails |
| `none eager` — no CUDA graphs | literal case fails |
| `mtp` | fails earlier, at the choice + thinking case ("choice constrained the thinking channel or lost the content boundary") |

So this is not graph capture and not one backend. It is the thinking channel composed with a
constraint.

One difference between this artifact and upstream's is known and unconfirmed as the cause: every
artifact this port builds embeds a template that emits the `reasoning_effort` instruction as a
synthetic system message (this is why the same test's thinking-token golden is 58 here against
upstream's 16; see `ninfer-reasoning-effort-mismatch.md`). Whether that template and the injected
thinking block compose correctly with a grammar is not separated here.
