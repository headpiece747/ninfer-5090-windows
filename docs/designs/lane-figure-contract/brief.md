# Which instrument publishes a lane figure

Status: brainstormed — the hypothesis was refuted; see `design.md` in this directory

The swarm established that the interleaved window has no lane-level pen (`mode_profile` measures one lane per
call and nothing reads its record into the table), so "the window publishes" was not a role the tree
implements. The recommendation and its conditions are in `design.md`.

## Problem

Three files state incompatible rules for producing a published figure:

- `tools/bench/lane_fingerprint.py:21-49` — a single run is not comparable with the published table, must not
  be folded, and folding a warm run publishes a figure "~6-8% optimistic". Publication uses the interleaved
  window in `tools/release/v3_profile_matrix.py`.
- `tools/release/profiles.py:64-73` — the table *is* the 2026-10-30 fold of `lane_fingerprint.py`, and the
  three-round window "was attempted the same day and abandoned".
- `tools/release/check_lane_figures.py:145-146` — the failure remedy tells a reader to run the fingerprint and
  commit `lane_figures.json` "with the table figures it refreshes".

The gate now passes with tolerances sized to the instrument's session spread (`ACC_TOLERANCE = 2.0`,
`TOK_TOLERANCE = 0.12`), so the immediate symptom is gone. What remains is that Phase 4 — publishing a figure
for the current engine — has no single agreed procedure, and the three statements above cannot all be true.

## Hypothesis

The two tools have different jobs and the conflict is a category error: **the window publishes, the fingerprint
witnesses.**

- `tools/release/v3_profile_matrix.py` (interleaved `profile` mode) produces published figures.
- `tools/bench/lane_fingerprint.py` records provenance and detects condition changes; it is never a published
  figure.
- `check_lane_figures.py` compares the published table against the fingerprint, with tolerances sized to the
  measured cross-session spread, and its identity check catches engine and artifact changes.

The 2026-10-30 fold then reads as a one-off forced by an encoding change, not as standing policy, and
`profiles.py` should say so.

## Constraints

- The table is the published claim; it must be produced by an instrument whose bias is documented.
- A rebuild must be able to make the gate meaningful again in minutes, not an hour — the fingerprint does that.
- A published figure must be reproducible by a later reader, which means the procedure has to be one a reader
  can follow.
- Card time: a fingerprint is ~10 minutes; an interleaved window is ~1 hour.

## Options

1. **Retire the prohibition and fold.** Delete `tools/bench/lane_fingerprint.py:21-49`, make the fold the
   publishing path. Cheapest, but publishes the bias the docstring predicts (measured +6.6% on 2026-10-31).
2. **Window only.** Delete the fold path and `fold_lane_figures.py`, make the window the only source of table
   figures. Cleanest, but every table update costs an hour, and `fold_lane_figures.py` currently consumes only
   `lane_figures.json`.
3. **Dual roles (the hypothesis).** Keep both, state each one's job in all three files, and change the gate's
   remedy text to send a reader to the window for the table and to the fingerprint for provenance.

## Open questions

- Under option 3, what does `fold_lane_figures.py` become — a tool for the window's output, or retired?
- Does the gate's remedy message need to distinguish "the engine moved" from "the number is wrong"?
- What evidence would make option 2's hour-per-update acceptable, or option 1's bias acceptable?
