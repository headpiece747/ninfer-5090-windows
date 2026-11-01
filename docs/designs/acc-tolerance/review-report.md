# Review of the acceptance-tolerance decision

Run 2026-10-31 by four fresh-context reviewers (`egm-zoom-out`, `egm-comprehension`, `egm-design-critic`,
`egm-codebase-auditor`) against the first draft of `design.md`. That draft argued for widening
`ACC_TOLERANCE` from 0.5 to 2.0 and was already applied to the working tree.

Verdict: the draft's premise was false. The change was reverted and replaced.

## Blockers

| Finding | Disposition |
|---|---|
| The table is a fold of `lane_fingerprint.py`, not an interleaved window (`tools/release/profiles.py:64-71`); the rejected "one session" option is the 10-minute fold, not an hour | **Accepted.** This is the finding that overturned the decision. |
| The quoted 45.4/46.2 failure was never produced by this gate: `tools/release/check_lane_figures.py:94-97` skips the acceptance comparison when the engine identity differs, and it always did here | **Accepted.** The failure came from the withheld refresh's own criterion, not from `ACC_TOLERANCE`. The doc now says so. |
| "The widened tolerance cannot hide a stale table because the identity check is separate" is false: identity compares the engine and artifacts, not the published number | **Accepted.** Withdrawn with the revert. |
| Phase 1 described an edit already on disk | **Accepted.** The doc was rewritten to record a decision, not to propose one. |

## Majors

| Finding | Disposition |
|---|---|
| The 14.5% / 21.1% counter-example is a draft-depth comparison on the long-Chinese cell (`tools/release/profiles.py:181`), not a sampling-config error, and not the cell the gate reads | **Accepted.** The comment carrying it was reverted. |
| Two acceptance tolerances exist: the gate's 0.5 and `lane_fingerprint.py --tolerance-acc`, default 3.0, on the same comparison | **Accepted, not fixed.** Out of scope for this change and worth its own decision: the tool can call clean a gap the gate then rejects. |
| The proposed negative control duplicates `tests/release/test_lane_figures.py::test_table_acceptance_mismatch_fails`, which doctors by 5 points and cannot distinguish 0.5 from 4.9 | **Accepted.** Verification now specifies a boundary control (+0.6 fails, +0.4 passes). |
| A fingerprint whose `code` cell has no numeric `accept_rate` skips the acceptance comparison silently | **Accepted, not fixed.** Same class as the missing-fingerprint case; noted for the gate's owner. |
| "There is no measurement of the tok spread across sessions" was false: `tools/bench/lane_fingerprint.py:26-28` records 5.1-7.3%, and the same file records 10.3% over six hours | **Accepted.** The doc now says the tok tolerance is thin rather than unmeasured. |
| `docs/performance/lanes.md:44-59` still said the table was stale and the decision open | **Accepted.** Updated in the same change. |
| "It reads only `lane_figures.json` and `profiles.py`" — it also hashes the binary and stats artifacts | **Accepted.** Wording corrected. |
| "0.79 points over the gate" — 0.79 is the gap; the excess was 0.29 | **Accepted.** Wording corrected. |

## Rejected

| Finding | Reason |
|---|---|
| "Acceptance is deterministic at this sampling, so six runs cannot vary by 0.1-1.2 points; the design's ranges are impossible" | The ranges are in `scratch/spread2.log`, and the interleaved A/B reproduces the endpoints on two binaries. The contradiction between the determinism claim and the measurements is real and is recorded as an open question rather than resolved in either direction. |
| "The 1.2-point gap is an instrument offset, not a session effect, so the rejected option failed on principle" | Plausible and consistent with the determinism note, but not established: the A/B holds the binary fixed and still sees the table's value move. Recorded as open. |

## What the review was worth

The draft was internally consistent, cited real files, and measured the thing it claimed to measure. It was
still wrong, because it described the table's provenance from memory instead of reading
`tools/release/profiles.py:62-76`. Three reviewers found that independently; none of them saw the conversation
that produced the draft.

## Second pass: upstream personas

After the first verdict, the upstream `style` and `structural` personas were run against the corrected design,
each in a fresh context via the "read the persona file first" fallback their own skill documents. They found
defects the first pass missed. The decision did not change.

| Finding | Source | Disposition |
|---|---|---|
| The A/B never named its lane | style | Fixed: `start_swift_v3_dflash2_vision.bat`, `code` cell |
| Verification item 3 was a recipe with no recorded outcome | style | Fixed: both arms and their results recorded |
| Percent, points, and fractions of a rate were mixed | style | Fixed in the same edit |
| The document never named `ACC_TOLERANCE`, the symbol it changes | structural | Fixed |
| The document is not listed in the repository's docs index | structural | Open: a separate index edit |
| "Resolution of the record" was undefined | style | Fixed: now the table's one-decimal quantisation |
| The primary evidence is an uncommitted scratch log | style | Open: recorded in the design as a limitation |
| The design template does not fit a decision record; `docs/adr/` does | structural | Open: a repository convention question, not this change |

The style persona earned a permanent place in the skill as `egm-style`, and stage 3 now dispatches it
alongside the critic and the codebase auditor. The structural persona's output was mostly conformance to EGM's
own design blueprint (Goals/Non-Goals, CUJs, heading case, bullet style), which this repository does not use;
its three repository-agnostic checks are kept and two of them appear above.

Measured cost of the second pass: two subagent runs, no GPU.
