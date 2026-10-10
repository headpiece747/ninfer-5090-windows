# Context

Domain vocabulary for this repo. **Concepts and names only.** No ceilings, ports, speeds or
artifact sizes: those live in one profile table under `tools/release/` and drift every time
they are restated. If a fact here needs a number, it belongs there or in the launcher that
ships it.

Two names carry an artifact, not a preference: the **QUASAR artifact** is our own
quantisation-aware-trained image, and the **NVFP4-full artifact** is the fuller requantisation of
the unsloth line. Both are built locally by this port's converter from the Hugging Face source
checkpoints; neither is fetched prebuilt. "ninfer" on its own is ambiguous and should not be used.

## Launch surface

**profile** — one shippable combination of artifact, spec route, vision and ceiling. A profile is
data, not a file: the launcher, the verifier, the measurement harnesses and the docs are all
consumers of it.

**launcher** — the `.bat` a user double-clicks to start one profile. It resolves the engine and
artifact beside itself, checks both exist, and starts one engine.

**model id** — the string the engine enforces on every request. The engine rejects a request
naming anything else, so a launcher's model id and the entry a client uses must match exactly.
The display name a client shows is free text and can change without consequence.

**ceiling** — the highest context the engine accepts for a profile. It is refused above that
point with byte accounting, never silently reduced, so a ceiling is a measured value rather
than a configured one.

## Speculation

**spec route** — how draft tokens are proposed: **MTP** or **DFlash2**. MTP is lighter and needs
only the artifact's MTP head; DFlash2 needs a companion module and proposes deeper.

**draft depth** — how many tokens a route proposes per round. Chosen per profile by measurement.

**acceptance** — the fraction of proposed tokens the target commits. It does not predict
throughput: tokens committed per round matters more, so depth is chosen on measured decode rate.

**proposal head** — the optimised head the `--lm-head-draft` flag selects. Its value differs per
profile, and it can cost headroom as well as buy speed.

## Memory

**artifact** — the `.ninfer` file. The **v3 container** has a JSON index, declared bindings and
named components; the **v2 container** is rejected outright by a v3 engine, so upgrading is
mandatory rather than optional.

**device weights** — what the artifact costs in VRAM once materialised. This, not the file size,
is what bounds a ceiling.

**KV pool** — the context state, sized automatically from the VRAM left after weights. Its dtype
trades precision for room.

**prefix cache** — retention of prepared context so a later request reuses it instead of
re-prefilling. Retention is bounded by one shared **Host quota** (`--host-context-mib`), which
upstream b9114396 put in place of the separate shared-prefix, private-continuation and anchor
budgets this glossary used to name. That quota, not the host pool's size, is what limits it.

## Practices

**measured, not assumed** — every number a profile ships is backed by a record. Where a claim
cannot be measured, say so rather than inferring it.

**generated, not transcribed** — a fact restated by hand is a drift site. Two documented drifts
in this repo's docs came from transcription, not from wrong measurement.

**maximin — "the worst domain decides"** — a lane's depth, width or recipe is chosen by the workload
where it does *worst*, not by its average. **Compare each option's worst case, not the same cell across
options**: reading a per-cell delta as the comparison flipped a lane's depth twice in one afternoon before
the rule's own form was applied, and it is the error the code-weighted bound below is written to prevent
recurring. The domains are the matrix's `code`, `prose`, `chinese`,
`dialogue` and `repetition` prompts (`v3_profile_matrix.py`). A candidate that wins three domains and
loses one badly is rejected for it: see the quasar MTP4 row in `profiles.py`, where depth 5 was faster
on prose and dialogue and lost code and chinese by 37 and 15 tok/s. Introduced 2026-09-30 (`4456fac6`)
when two depths were found to have been chosen from stale figures. Two caveats the rule is read under:
it is only as good as the domains it is run on, which is why a prompt can now come from a file
(`v3_profile_matrix.py --domain-from-file`), and a margin smaller than the measurement's own spread is
not a decision -- the deciding margins have been 0.3 to 1.6 %, against spreads that reach 21 % when the
protocol compares lane restarts. Evaluated 2026-10-09: the alternative of constraining the worst case to a
floor and then ranking by the mean was refused by measurement -- on Swift 1.5 it would have shipped depth 9,
which is 28 % slower than depth 7 on long Chinese, acceptance falling from 30.0 % to 16.9 %.

**The scenario set is the workload's (2026-10-09).** The rule above was applied over all five domains, and
because Chinese is consistently the worst domain for speculative decoding it decided every lane -- so a
domain this product is not used for vetoed choices that were best for the one it is. The product is used for
coding (owner's statement, 2026-10-09), so the **deciding scenarios are the code ones**: the 225-character
`code` prompt and 36,000 characters of real code, with the worst of the two deciding, as before. Every other
domain is still measured on every candidate, recorded, and **disclosed in the row** -- a regression there no
longer vetoes, and it is never hidden. The quality axis is weighted the same way: the deciding perplexity
domain is `ninfer_code`, with the others reported. The Swift 1.5 case the paragraph above refused on is
reopened by this and re-measured the same day: at served length depth 7 wins long code 208.2 against 182.0
and long Chinese 171.9 against 165.6, so the incumbent stands and its Chinese cost never had to be weighed.
What the reopening did change is the figure that reading rested on -- the "+19.3 % for depth 9 on long code"
is withdrawn, because no record carries it and the measurement is the other sign.

**The bound that makes disclosure a rule rather than a note:** a change is adopted when the deciding cells'
worst case improves by at least as much as the worst disclosed non-code cell gives up. Without it,
disclosure is decoration -- of three lanes whose code cells were all positive, only one was a trade worth
making (swift dflash2's +24.6 % against -4.2 %, against ninfer dflash2's +4.9 % against -25.4 % and nvidia
dflash2's +3.8 % against -31.5 %).

What the rule is *for*, after that evaluation: at served length the candidates are separated by 10 to 28 %
against spreads of 0.2 to 1.0 %, so the arithmetic decides nothing a measurement does not and any sane
criterion agrees. Its value is the **discipline** -- measure every candidate across domains rather than only
the one you care about, and protect the worst case *of the scenarios that decide* -- plus three practices
that keep it honest:

* **select on served-length prompts.** The short domains give margins of 0.3 to 1.6 %, which invert: Swift
  1.5 loses 3.4 % of its worst short domain and 28 % of its worst served one, the same direction but a
  different decision weight.
* **a near-tie at short length gets a long-length measurement before it is decided.** The nvidia lane's
  "+0.3 %, well inside one invocation's spread" was left on the incumbent, and at served length depth 7
  leads by 9.8 % -- a tie is a reason to measure again, not a reason to keep what is there.
* **record the mean beside the worst case**, so a trade is visible rather than implicit.
* **when the worst-case gap is itself inside the spread, the mean decides.** That is what the nvidia lane's
  depth change of 2026-10-09 turned on: a 0.3 % tie on the short set, and on the served set a worst case
  that favours depth 9 by 29.6 % while the mean over both sets favours it by 4.1 %. A tie is not a licence
  to keep the incumbent; it means the criterion has no signal, and something with signal has to decide.
