# `reasoning_effort` advertises six values; the chat template accepts four

**Target:** `Neroued/ninfer`
**Severity:** low (misleading error surface), user-visible as a 400 on a value the engine
itself said was valid.

## Summary

The engine validates `reasoning_effort` against six values and says so in its rejection
message, but the artifact's `chat_template.jinja` implements only four. Two of the
advertised values therefore fail at request time, from the template rather than the
validator, with a different error code.

## Reproduction

Send one request per value to any v3 artifact:

```bash
curl -s localhost:8086/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"Reply OK."}],"reasoning_effort":"high"}'
```

Observed:

| value | result |
| --- | --- |
| `none` | 200, no reasoning emitted |
| `low` | 200 |
| `medium` | 200 |
| `xhigh` | 200 |
| `minimal` | **400** |
| `high` | **400** |

The 400 body:

```json
{"error":{"code":"invalid_prompt","message":"artifact:chat_template.jinja:
------------
While executing CallExpression at line 55, column 28 in source:
...', 'low') %}?        {{- raise_exception('Unexpected reasoning effort ...
```

Meanwhile the validator in `src/serve/openai_chat_request.cpp:840` rejects anything outside
the six with:

```
reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or ...
```

So `minimal` and `high` pass validation and are then refused by the template, as
`invalid_prompt` rather than `bad_request`. A client that trusts the validator's list
cannot tell which subset is real.

## Suggested fix

Whichever direction is intended:

1. implement `minimal` and `high` in the template, mapping them onto the nearest supported
   effort; or
2. narrow the engine's validation and its message to the four the template supports, so the
   two agree and the error is `bad_request` naming the allowed set.

Option 2 is a two-line change and removes the trap immediately. Option 1 is better if the
efforts are meant to be distinct.

## Why it matters here

We drive these models from opencode, which sets `reasoningEffort` per model via
`options.reasoningEffort` and offers built-in variants named `minimal`, `low`, `medium`,
`high`. Selecting a built-in variant named `minimal` or `high` hits this 400 with an error
that points at the template rather than at the setting.
