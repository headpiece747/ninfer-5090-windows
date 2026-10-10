# The multi-needle long-context task, and what it measures at 8k, 32k and 128k

Written 2026-10-09 to replace the literal NIAH configuration that
[docs/research/planned-projects-checked-2026-10-09.md](planned-projects-checked-2026-10-09.md) item 1 asked to
replace: a multi-needle retrieval task in this port's own corpus text, scored through the existing
`ninfer-perplexity --per-token-logprobs` path, with the answer's NLL as the endpoint and a
mismatched-needle control that shows the instrument can see what it claims to measure.

## What it is

Each text is built from `eval/corpora/perplexity-1m/data/ninfer/00.txt`, truncated to the target depth, with
five single-fact notes planted at even fractions:

```
 Note for the record: <subject> is <value>.
```

All five questions are then asked at the end, followed by one reference answer naming all five values. The
scorer reports the per-token logprob of every token, and the metric is the mean logprob of the answer's last
40 (and 20) tokens.

This is a **RULER-shape retrieval task**: the question names the subject in different words from the note
where it can, and shares terms where the subject's own name is the only natural phrasing — so it is
multi-needle retrieval and **not** NoLiMa's latent-association construction, where question and needle share
no terms at all. Labelling it otherwise would overstate what was built.

**Endpoint limits, stated because the numbers below are small.** This scores the *reference answer's
likelihood*, not whether the model generates it: it is task-adjacent, and the eval harness's generation
grader is the instrument for generation accuracy. A model that retrieves but cannot answer would read well
here. The token boundary is also estimated by row count rather than read from the tokenizer's own byte
offsets, which is why two window sizes are reported.

## The control that makes it an instrument

Scored with the needle values replaced by values that are not in the context — same subjects, same questions,
same reference answer:

| depth | window | matched | mismatched | rise |
|---|---|---|---|---|
| 8,192 | last 40 | -0.1720 | -1.1311 | **+0.96 nats** |
| 8,192 | last 20 | -0.0687 | -0.7449 | +0.68 |
| 131,072 | last 40 | -0.0655 | -1.0078 | **+0.94 nats** |
| 131,072 | last 20 | -0.0741 | -0.8230 | +0.75 |

The instrument sees retrieval: when the context contradicts the answer, the answer's tokens lose roughly a
nat each. Without this table, a flat curve could not be distinguished from a test that cannot see anything.

## Result: no depth-dependent degradation up to 131,072 tokens

`qwen3_8_27b_nvfp4nvidia`, context/stride {8192/4096, 32768/16384, 131072/65536}, `--kv-dtype fp8`:

| depth | mean logprob, last 40 | mean logprob, last 20 | scored tokens |
|---|---|---|---|
| 8,192 | -0.1720 | -0.0687 | 10,090 |
| 32,768 | -0.0685 | -0.0631 | 37,547 |
| 131,072 | -0.0655 | -0.0741 | 150,344 |

The curve is flat and 8,192 is the *worst* of the three: there is no degradation to report. Read against the
field's own effective-length rule — the longest context where a metric stays above roughly 85 % of its
short-context baseline — the effective length of this construction on this artifact is **at least 131,072
tokens, the deepest depth measured**: not a 100 % pass rate, but no measurable loss of answer confidence
across a 16x span of context.

## Limits

- The haystack in the first version is one corpus file **repeated** to depth, not a natural multi-document
  pile; the code-repository variant below uses this repository's own source instead and carries the same
  result, so the repetition limit applies only to the table above.
- One artifact (nvidia), one scorer, one sampling. No other lane was run.
- The task is retrieval; the item's second suggestion, a code-repository task with no verbatim overlap,
  is still unbuilt.

## Records

`profiles/bench/needle-2026-10-09/`: the five per-token CSVs (three matched depths, two mismatched controls)
and the two run logs that carry each scorer's own summary lines. The construction is deterministic from the
corpus file, the depth, and the needle values printed in the logs.

---

## The code-repository variant, built the same day

The item asked for a multi-needle **or** code-repository task without verbatim overlap; this is the second
half, and it also removes the repeated-haystack limit above. Same instrument, different haystack and facts:
the text is this repository's own source (`src/**` and `include/**`, 6.4 M characters available, natural
rather than repeated), with five code-flavoured notes planted at even fractions — a symbol that enforces a
retry budget, a component owning an admission policy, a byte ceiling, a switch name, a function name — and
questions phrased by role. The control is the same construction with different values.

| depth | matched (last 40 / last 20) | control rise |
|---|---|---|
| 8,192 | -0.5636 / -0.1758 | +0.75 / +0.69 nats |
| 32,768 | -0.6333 / -0.1853 | not run |
| 131,072 | -0.5558 / -0.1585 | +0.72 / +0.65 nats |

Flat to 131,072 again, and the control separates retrieval from a floor again, so the effective length of
this construction is at least the deepest depth measured — on natural code text, with symbol and number
answers rather than prose. Residual limits: the questions still share terms with the notes where a subject
has only one natural phrasing, and the endpoint is still the reference answer's likelihood rather than a
generation grade. The CSVs are `code-needle-*` and `code-needle-control-*` beside the first version's, in the
same `profiles/bench/needle-2026-10-09/` directory.
