# ADR-0010: incremental prompt encoding, spliced at a verified seam

Status: accepted (2026-09-23)

## Context

Host prompt preparation is `render + tokenize`, and on a warm prefix cache it is the whole of
time-to-first-token: measured 119 ms for a 229-message, 130,869-token conversation, of which
tokenize is 41 ms. `docs/research/prompt-preparation-cost.md` records that measurement and the
process-level defect that used to dwarf it.

Tokenization is linear in the prompt and the prompt is re-sent in full every turn, so its cost grows
with the conversation: about 0.3 ms per thousand tokens, so roughly 80 ms at the 262,144-token
ceiling. With automatic prefix caching, the engine already reuses that prefix; the frontend is the
only layer that recomputes it.

This is not a new problem elsewhere. vLLM PR #47583, *"Add opt-in incremental prompt-encoding cache
for multi-turn chat"*, describes it exactly: every request re-sends the conversation, the rendered
prompt of turn N is almost always a strict prefix of turn N+1, and `_tokenize_prompt` re-tokenizes all
of it. Their measurement is 54.9x on per-turn encode at 128K and a median multi-turn time to first
token of 0.92 s to 0.47 s. SGLang's SMG L1 prefix tokenization cache is the same idea.

## Decision

Cache the full encode of a prompt in `Tokenizer` and, on a later request whose text is a
prefix-extension of a cached one, answer from the cache plus an encode of the tail alone.

The seam is chosen and then **verified**, in this order:

1. The cached text must be a strict prefix of the new text, at equal `max_tokens` and with
   `parse_added_tokens` set. Anything else falls through to a full encode.
2. The seam is a byte offset that was a **boundary of the cached encode** — so its token frontier is
   known without re-deriving it — and the boundary one before it, `prev`, must exist too.
3. Every boundary of the new request below the seam must also be a boundary of the cached encode, at
   the same offset. They are the ones whose results can only come from the cache.
4. The seam must not lie inside a literal span.
5. **The window `[prev, seam)` is re-encoded as its own input** and its token ids must equal the
   cached ids for the cached frontiers at `prev` and `seam`. This is the whole safety argument, and
   it is a real test rather than a formality.

If all five hold, the result is the cached ids up to the seam frontier, followed by the ids of
`text.substr(seam)` encoded on its own with the boundaries and literal spans above the seam shifted
by the seam, and the boundary results come from the cache below the seam and from the tail encode
above it, shifted by the seam frontier. Any failed step means a full encode, and the full result
replaces the cache entry.

### An exact repeat needs none of that, and was missing

Step 1 requires a *strict* prefix, so this design never served a prompt equal to a cached one: a
retried, duplicated or re-sent request re-encoded the whole conversation while a growing one spliced.
That is the opposite of what the cache is for, and it was found only by reading the live preparation
split — `prepared 31.4 ms, render 3.79 ms, tokenize 27.4 ms` on a warm request, with the counters
showing `splices=0 fallbacks=0` on every request.

An exact repeat is answered from the entry directly, before the seam search: same text, same
`max_tokens`, same boundary offsets in the same order, and equal literal spans. Nothing is spliced and
nothing is re-derived, because the entry *is* the answer to that text, that boundary set and that
option set — there is no seam to place and no window to re-encode, so steps 2 to 5 have nothing to
check. The boundary set and the literal spans are part of the comparison rather than assumed equal,
because both are part of what the entry answers.

Measured on the request that showed the problem (229 messages, 92,373 tokens): `tokenize`
28.5 ms → 192 µs, `prepared` 32.6 ms → 4.21 ms, wall 23.98 s → 0.018 s. `test_repeated_encoding_matches_full`
checks the repeat against a cold tokenizer as the oracle, checks it agrees with the encode it came
from, and checks that a boundary set the cache never answered is not served as a repeat.

## Why step 5 is the right test

`encode_with_boundaries` walks the text as alternating *ordinary stretches* and *added tokens*: it
scans for added-token matches and hands each stretch between them to the BPE. Two consequences
decide this design.

- **A splice is exact only at a stretch boundary.** BPE merges do not cross an added token, so a
  stretch encodes identically whether it is passed whole or as the whole text; but a seam *inside* a
  stretch hands the tail a shorter stretch, and the merge order there can differ.
- **An added token spanning the seam cannot be missed.** The tail's scan starts at the seam, so a
  match beginning before it would be dropped from the tail while the prefix kept its own.

Both are exactly what step 5 detects, and neither needs a separate check. If the seam is mid-stretch,
the window's shortened first stretch tokenizes differently and the ids differ. If a token spans the
seam, the window's text is cut at the seam and its last id is a partial encoding, so the ids differ
there too. Step 4 is a cheap belt on the same braces: `encode_with_boundaries` rejects literal spans
that are not ordered and disjoint, and a span crossing the seam would have to be split.

## Consequences

- **Tokenization stops growing with the conversation.** A turn that appends one message encodes about
  one message, whatever the conversation's length, instead of the whole prompt.
- **It is opt-out by construction, not by flag.** A request that cannot be spliced is encoded exactly
  as before; the cache can only ever replace work, never change an id, because step 5 compares ids. A
  splice that throws falls back to a full encode and increments a counter rather than failing the
  request, so the fallback is visible instead of silent — and it is expected to stay at zero.
- **The render is not covered, and cannot be by this design.** A chat template is a whole-
  conversation loop that may legitimately rewrite its earlier output — the Qwen3.8 template gates its
  ` thinking` block on `last_query_index`, which moves forward every turn — so the rendered prefix is
  not a function of the messages before it and there is nothing to reuse. That conclusion stands: the
  render needed a different answer, and got one in
  [ADR-0012](0012-native-render-for-the-registered-template.md), which renders the registered template
  in C++ instead of interpreting it. On the same 229-message conversation this record measured at
  78 ms, that path renders in **3.79 ms**.
- **Concurrency:** `prepare` runs on `httplib::ThreadPool` workers, so the cache is a small ring
  guarded by a mutex, and a miss on contention is a full encode rather than a correctness question.
- **Evidence: a byte-identity test, and a live measurement.** `test_incremental_encoding_matches_full`
  encodes a prompt, then the same prompt extended, and asserts the spliced ids *and boundary results*
  equal a fresh `Tokenizer`'s full encode of the extension — a tokenizer that never saw the shorter
  prompt, so the oracle is independent of the cache. It also asserts the splice counter moved and the
  fallback counter did not, because an id comparison passes whether or not the splice fired and a
  mechanism that never fires proves nothing.
- **Measured on the shipped server**, same 229-message, 130,869-token conversation the field log
  produced: tokenization fell from **40.8 ms to 462 us** (88x) on the request that extended the cached
  one, `prepared` from 121 ms to 79.5 ms, and a fully cached request's time to first token from 137 ms
  to **93.8 ms**. The engine reported `cache 100.0%`, which is the independent check: spliced ids that
  differed from the full encode would not have matched its cached state.
- **Tokenization no longer grows with the conversation.** It is now a per-extension cost, and an
  exact repeat costs a cache lookup. What remained of preparation after both was measured on the
  shipped server and the split is now `prepared 4.02 ms, render 3.69 ms, tokenize 94 us` on a warm
  229-message request — so neither of the two terms this record was written about is the largest one
  any more, and the term that is (a constant ~10 ms of engine queue wait) is outside preparation.

## Alternatives rejected

- **Keying the cache on the message list and reusing the rendered text.** Rejected: it would skip the
  render as well, but it asserts that rendering is append-stable, which the template's own semantics
  contradict (above). The correctness argument would rest on the template rather than on a check.
- **A hash of the cached text instead of a prefix comparison.** Rejected: the seam needs the cached
  text to compare a window against, and a hash of the whole text cannot localize a seam.
- **Splicing at an arbitrary pre-token boundary, as vLLM's design does for a HuggingFace fast
  tokenizer.** Our tokenizer exposes no pre-token boundary list, and the boundaries the frontend
  already requests sit exactly at the stretch boundaries a splice needs, so they are the natural
  seam set. Requiring one of them is stricter than vLLM's rule, and the window check is what makes
  either sound.
