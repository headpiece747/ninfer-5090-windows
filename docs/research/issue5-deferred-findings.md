# Issue 5: what is known, what was falsified, and what is deferred

**Status:** deferred. The bug is not reproduced and not refuted. This note replaces the claim in
`resource-underflow-issue-5.md` that our soak was evidence about it — that claim was void.

## Why the earlier note was wrong

Our soak ran `--max-concurrency 1`. The reporter states plainly that the fault is **never** observed
there: *"0 incidents in all tests and in ~9 h of single-lane operation."* A non-reproduction in the
configuration the reporter excludes tests nothing, so the note's negative result was a null experiment
reported as evidence.

## What is established

Verified from source, with the counter or line that proves it:

| finding | evidence |
|---|---|
| `failed_` is set once and never cleared, so the engine latches | `engine_core.h:1916`, the only assignment in `src/runtime` |
| New requests after that answer 503 `service_unavailable` | `engine_core.h:187-190` → `generation_service.cpp:81-86` |
| `/health` answers 503 permanently | `http_server.cpp:452-457`, `is_available() = !stopping_ && !failed_` |
| The process never exits on an engine-wide failure | `main()` returns non-zero only for bind/warmup/listen; nothing stops the server |
| No `Retry-After` is ever emitted | absent from `src/`, `apps/`, `include/` |
| The fail-stop design is **upstream's**, not this port's | `d6af046a` (Neroued), ancestor of `upstream/dev`, byte-identical `fail_all_locked` |
| Cache-hit reuse requires `session_key` **and** a retention hint | `reused=14,394` once set; six HTTP attempts without them read `cache 0.0%` |
| A capture requires a `PromptCacheMarker` | `active_captures_offered: 0 → 1` after adding one |
| Demotion and restore work on this build | `exercise_host_restore` passes, asserting `PrivateTurnClosure` + `state_h2d > 0`; all 15 prefix scenarios pass |
| An owner cannot be demoted while a lane holds it | `private_has_active_edge`, `resource_manager.h:1389-1395`, gates every candidate filter |
| The demotion choice prices rebuild against recovery | `resource_manager.h:2115-2118` |

## What was falsified (ten runs, each a hypothesis read from code)

Concurrency count is not the whole story, and neither is any of these: pressure ratio · session key ·
retention hint · request ordering · holder timing · pool capacity · session size · capture eligibility
after edge release. Every run reported `state_h2d=0` after pressure and never underflowed.

**The methodological error, recorded so it is not repeated:** each run was a *hypothesis test* built by
reading code first. The `diagnosing-bugs` skill says the opposite — build a red-capable loop before
forming any theory — and a loop that has only ever passed has not been shown to catch the bug. Ten runs
of a loop asserting my own proximate conditions (`which reuse path`, `did it demote`) rather than the
reporter's symptom.

## The loop that now exists

`tests/models/qwen3_5/test_engine_issue5_race.cpp`, its own binary, the existing suite untouched. It
drives the reporter's named trigger — a cache-hitting continuation submitted while another lane is
generating — and asserts only their symptom (a request failing with the underflow, then unavailability).
It is deterministic, runs in seconds, and reports `ok`. **It has never gone red, so red-capability is
unproven.**

## Why a faithful reproduction cannot be run here

Their command line needs **20,316,679,168 bytes** of runtime capacity; this machine offers
**15,289,286,656**, as the engine reported when refusing to start. This host has **48 GB** of RAM where
their issue states **64 GB**. Scaling every quantity by the same factor (context 262,144 → 104,857,
KV 294,912 → 117,964, host KV 24,576 → 9,830 MiB) makes the instance start and its context-cache line
match theirs feature for feature — `3 active + 3 cached device states | host 16 states | private 8 |
shared 7 | anchors 4` — but the regime is not theirs.

## Neither project documents concurrency 3

| source | value |
|---|---|
| upstream `README.md`, both worked examples | **2** |
| upstream `docs/serving.md` example | **2** |
| upstream's default / valid range | `1` / `1..8` |
| this port's shipped launchers (v1.1.0 and today) | **1** |
| the reporter ran | **3** |

Their Expected 3 asks that the defaults scale with `--max-concurrency` or be documented. They already
scale — `device_state_slots = C`, `private = 2C`, `shared = max(C, 4)` — and the only value either
project demonstrates is 2, which their table does not test.

## The one question worth asking the reporter — **and it was already answered by the report**

They offered to run variants: *"the reproduction takes two minutes."* This note read their table as
testing 1 (never) and 3 (fails) and asked whether `repro_dsh.py` underflows at `--max-concurrency 2`.

**Read against the tracker on 2026-10-07, that premise is wrong: the report is about 2, not 3.** The
issue's own title is "`serve` wedges with 503 for all requests after `Qwen3.5 resource subtraction
underflow` (checked_resource_difference, context_work.cpp:222) **under `--max-concurrency 2`**"
(`gh api repos/Neroued/ninfer/issues/339 --jq .title`), and its body says "With `--max-concurrency 2`,
when two large-context requests are in flight at the same time (60k-70k input tokens each …)" and "9
total occurrences that day, all with `--max-concurrency 2` under heavy concurrent load", against the
control "`--max-concurrency 1`, continuous all-day usage: 0 occurrences". A tracker-wide search finds
no underflow report at concurrency 3 at all
(`gh api "search/issues?q=%22resource+subtraction+underflow%22+repo:Neroued/ninfer"` → #339 only).

So the question is moot: **the reporter already ran 2 and it underflowed there**, and the only value
below 2 is 1, where they observed nothing. The boundary is *at* 2, and what remains genuinely
untested is not a question for them but the port's own re-test of their workload against the
replacement cache — the re-test `#366` asks every reporter for, whose partial form is recorded in
`server-open-items.md` item 1 (their settings, 360/360 clean, but prompts of ~2,000 tokens against
their 60,000-70,000, no tools and no reasoning effort).

**This paragraph previously asked the reporter for a run they had already done**, which is what
reading the primary source rather than the paraphrase is for: this note's table of "their table"
came from the earlier note's summary, not from the issue.

## Correction 2026-10-06: three claims in this note have been overtaken

The note is otherwise still the record it claims to be, but these three no longer hold, and a reader
who takes them as current would be misled in the same way this note accuses the earlier one of being.

1. **"Why a faithful reproduction cannot be run here" is out of date, and the sibling note already says
   so.** `server-open-items.md` re-measured on 2026-10-04: 30.9 GiB of 32.6 GiB VRAM free and 36.4 GB of
   47.8 GB RAM free, so the 15,289,286,656-byte ceiling quoted here no longer describes this machine.
   The *reason* the reproduction was not run is now the reporter's client shape (their tools, their
   reasoning effort, their 60,000-70,000-token prompts), not capacity.
2. **The context-cache line this note matches "feature for feature" is the pre-`b9114396` one.** `3
   active + 3 cached device states | host 16 states | private 8 | shared 7 | anchors 4` describes a
   cache whose separate budgets and flags (`--host-kv-mib`, `--host-state-slots`,
   `--max-private-continuations`, `--max-shared-prefixes`) no longer exist; they are one
   `--host-context-mib` quota now. The comparison to the reporter's instance still says something about
   *their* build, not about this one.
3. **"It has never gone red, so red-capability is unproven" is now weaker than that: for the symptom it
   names, the test cannot go red at all.** Upstream's replacement removed `checked_resource_difference`
   and its throw with the subsystem they belonged to, so no code path in this tree emits `resource
   subtraction underflow`; the test's match on that string is a reintroduction detector. What the loop
   still asserts — the engine accepts work after the race rather than latching — is the reporter's other
   half and remains live (`docs/research/server-open-items.md` item 1, correction 2026-10-06, which also
   records the concurrency-3 case that has since run against the new cache: 360/360, no failure).
