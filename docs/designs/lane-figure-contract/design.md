# Which instrument publishes a lane figure

Status: approved (implemented 2026-11-01)
Owner: Tobias

## Problem

`docs/designs/lane-figure-contract/brief.md` asked whether the two lane instruments can be given separate
jobs. The shoot-down swarm found that the question was malformed: one of the two jobs does not exist in code.

This document records what the swarm established, and the recommendation that follows.

## What the swarm established

**1. The interleaved window has no pen at the lane level.** `tools/release/v3_profile_matrix.py:913-924`
(`mode_profile`) measures one lane per call, sequentially (`:1045-1048`), and prints the record; `--rounds` is
consumed only by `mode_widths` (`:1032`), which sweeps draft widths inside a single artifact. Nothing reads
`matrix_v3.jsonl` into `tools/release/profiles.py`. So "use the interleaved window to produce figures", which
`tools/bench/lane_fingerprint.py:48-49` tells the reader, is advice this tree cannot follow.

**2. The fold is the only mechanical writer.** `tools/release/fold_lane_figures.py:22-43` reads
`lane_figures.json` and writes the table; it is what produced the 2026-10-30 and 2026-10-31 figures. The
prohibition in `tools/bench/lane_fingerprint.py:43-46` names an alternative that has no implementation.

**3. A green gate does not certify the table.** `tools/release/check_lane_figures.py:88-101` compares the
*record's* engine hash with the binary, and every lane is marked stale from that one value. The table carries no
engine of its own. The tree is in that state now: the gate passes on the 2026-11-01 fingerprint, and the table
still describes `874fce8a` (`docs/performance/lanes.md:44-47`). "Identity matches" means a fingerprint exists
for this engine, never that the published figure describes it.

**4. Nothing watches between refreshes.** Each `lane_fingerprint.py` run overwrites `lane_figures.json`
(`:316-318`), so the previous entry — the anchor a drift check compares against — does not survive. The tool's
own drift report exits 0 by design (`:333-346`). After a fold, a later "drift" therefore means the instruments
disagreed, not that the engine's conditions changed.

## Recommendation

**Sanction the fold, and make the table carry its own engine.**

1. Replace the flat prohibition in `tools/bench/lane_fingerprint.py:21-49` with the conditional that
   `tools/release/profiles.py:64-73` already states: a fold is published only with its run level recorded and
   the decode column labelled as that window's reading. The 2026-10-30 fold did this at -10.2%; the 2026-10-31
   fold would have done it at +6.6%.
2. Add the engine the table was measured on to `tools/release/profiles.py`, and have
   `tools/release/check_lane_figures.py` compare it with the tree. Without it, the gate cannot fail for the
   reason it exists.
3. Keep the fingerprint as the provenance and drift witness, which is what it already is.

This is option 1 from the brief with the two conditions the swarm demanded, and it is the only option that
requires no new instrument. Option 2 needs a lane-level interleaved loop and a second fold input; option 3 is
option 2 without the plumbing.

## What would falsify this

Paired window and fingerprint on one engine, in the same session: if the two disagree by more than the gate's
tolerance in one direction, the published figure and its witness are not comparable and the fold needs a
different certification. The 2026-10-31 datum (6.3% decode, 0.79 points acceptance) is one observation, and it
points at estimator resolution rather than at the session — so the test is not yet decisive.

## Implemented (2026-11-01)

1. `tools/bench/lane_fingerprint.py` — the flat prohibition is replaced by the two conditions (run level
   recorded, decode labelled as that window's reading), and it points at `fold_lane_figures.py --apply`.
2. `tools/release/profiles.py` — `MEASURED_ENGINE` records the engine the table came from;
   `fold_lane_figures.py --apply` rewrites it on every fold.
3. `tools/release/check_lane_figures.py` — compares `MEASURED_ENGINE` with the tree and fails when they
   differ; `--table-engine` lets a test drive the check. Covered by
   `tests/release/test_lane_figures.py::test_table_engine_change_fails`.
4. The table was folded onto the 2026-11-01 fingerprint (engine `00947874`, run level +7.8%, range +3.4% to
   +10.3%), and `docs/performance/lanes.md` was updated with it.

## Not decided here

Deferred, not open — none blocks a shipped lane:

- The shape of the per-row provenance (one table-level engine, or an instrument and date per lane). The
  table-level value is what shipped; a per-row field would cost a schema change for no failure it prevents.
- Whether `tools/release/fold_lane_figures.py` should also accept a window record once one exists. No such
  record exists: `tools/release/v3_profile_matrix.py --mode profile` prints per-lane records and nothing
  aggregates them.
- Whether the fingerprint should keep history instead of overwriting. It overwrites, so a drift series does
  not exist; the log keeps what the record drops only for the run that wrote it.
