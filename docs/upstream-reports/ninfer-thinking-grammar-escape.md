# Thinking + a literal grammar lets one byte escape the constraint

**Target:** `Neroued/ninfer`
**Status:** **fixed in this port** by `4840b66f3`, which restored the pre-`1407c62f6`
deterministic byte DFA in `src/text/grammar.cpp`'s `reasoning_prefix`. Upstream `458863be2` still
carries the TagDispatch version and the regression; this report remains a candidate to send them.
`ninfer_qwen3_5_grammar_real_test` passes on the merged tree (measured 2026-10-31, 22.25 s in the
suite), and the divergence is recorded in `tools/release/port_delta_baseline.json`.
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

The port restored the **deterministic exact byte DFA** that `1407c62f6` replaced with
`TagDispatch` + `AddScalarStringFSM`. The DFA consumes through the first complete delimiter with a
single unambiguous path, including a delimiter split across tokens; the generated EBNF is fed to
`xgrammar::Grammar::FromEBNF` as before. With that, the same assertion passes on the same artifact
and the same engine options (measured 2026-10-31). The change is a recorded port delta on an
upstream-owned file, with the reason in `tools/release/port_delta_baseline.json`.

The port does not baseline the test and does not modify the assertion.

## Cause, as the fix establishes it

`1407c62f6` built the reasoning barrier as `TagDispatch` and then added an epsilon edge from **every**
end state to one tail state before appending the delimiter bytes. Where the delimiter repeats or
shares a prefix with its own suffix, those parallel end states make the Earley item set ambiguous:
the parser can accept a delimiter byte while another path is still mid-delimiter, so the user
grammar starts one byte late. The observed shape — every constrained byte correct, then one
unconstrained byte before EOS — is exactly that one-byte-late boundary. Restoring the DFA removes
the ambiguity and the byte with it.

That matches the hypothesis this report carried before the fix (a composition boundary completing
one byte early); it is now established by the repaired behaviour rather than by reading.

## Controls that passed

- The same literal without thinking, greedy and at 0.8, exact at both temperatures.
- Choice, regex, EOS, truncation, continuation and mixed-row cases in the same binary.
- `thinking.applied` and `injected_tokens > 0` both hold, so the failing half is the content
  boundary, not the thinking injection.

## Configurations tested, before the fix

Measured on 2026-10-30, same artifact:

| argv | result |
| --- | --- |
| (none) — `none`, CUDA graphs | literal case fails |
| `none eager` — no CUDA graphs | literal case fails |
| `mtp` | fails earlier, at the choice + thinking case ("choice constrained the thinking channel or lost the content boundary") |

So this was not graph capture and not one backend. It was the thinking channel composed with a
constraint.

The port's reasoning-effort template was ruled out when the same artifact and template passed after
only the grammar fix: every artifact this port builds embeds a template that emits the
`reasoning_effort` instruction as a synthetic system message (the same test's thinking-token golden
is 58 here against upstream's 16; see `ninfer-reasoning-effort-mismatch.md`), and that difference was
a candidate cause until the repaired `reasoning_prefix` made the case pass unchanged.
