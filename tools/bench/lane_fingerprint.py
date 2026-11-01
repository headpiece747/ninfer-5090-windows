"""Fingerprint the shipped lanes: measure every lane on the cells the product depends on, and record it.

This exists because most of the defects found on 2026-10-09 were not missing card time. They were recorded
figures relied on past their conditions: a throughput column taken before its lane's depth changed, an
acceptance figure from an instrument that could not reach the served regime, a context derived and never
served. Every one is caught by re-measuring a small, fixed set of cells and comparing.

What it does NOT do is enumerate the space. Five artifacts x two backends x ~15 depths x five KV dtypes x
domains x samplings x contexts is order 10^5 configurations, weeks of exclusive card time, invalidated by
the next rebuild. The cells below are the ones a shipped product is judged on.

Comparison rules, which the first run got wrong in two ways worth keeping:

  * Only the **`code`** cell is comparable with the profile table, because that is the cell the published
    figures were measured on. The other cells read far lower -- decode falls with context, 381.8 tok/s on a
    225-character prompt against 185.1 on 36,000 characters is physics, not staleness -- so they are
    compared against this file's previous entry for the same cell, which is what a fingerprint is for.
  * Acceptance is read from **`accept_rate`** explicitly. A substring search for "accept" picks
    `spec_accepted`, which is a token count: the first run reported "acceptance 1328.0 against 71.2".

**A single run of this tool is one session. Folding it is the port's publishing path, under two conditions.**

This tool already records the reason, and does not need it restated: the drift block below
cites one lane reading 216.6 (the committed table), 230.5 and 239.0 within six hours -- a
10.3% span -- against a within-run spread of 0.3 to 5.1%, so no statistic available inside a
run can size it. Two runs of this tool, same binary, an hour apart, differed on **all
twenty-four lane x cell comparisons, every one in the same direction**: 5.1% to 7.3%,
uniformly.

A uniform one-directional shift is the card's clocks settling, not engine behaviour: a code
change moves acceptance, and acceptance on the seeded `code` cell was bit-identical across all
eight lanes in both runs (0.5175, 0.5612, 0.4493, 0.4772, 0.548, 0.597, 0.509, 0.5804). Decode
is a clock measurement and is not evidence about the code.

**The first lane is the worst case, and the tool can see it.** Lanes run sequentially in table
order with no warm-up, and this file records the card state per lane. In the run of
2026-10-13T22:13:51Z the first lane's "before" reads **37 C, 637 MHz, 61.73 W** -- an idle card
that has not ramped -- and it reaches 2865 MHz only inside that lane. Every later lane starts at
57-66 C and 2812-2895 MHz. **So the first lane is measured on a card in a different power state
from the other seven, and its figure is the least trustworthy in the file.** Compare it against
its own previous entry, never against the table, and never fold it.

So: **fold a run only when both conditions hold.** The published figures come from a fold, and the warnings
above are why the conditions exist rather than a prohibition:

* the run level is recorded beside the figures (`run_level`, written by this tool) and stated wherever the
  numbers are quoted, because a whole-run shift is the card's condition, not the engine's;
* the decode column is presented as that window's reading. `tools/release/profiles.py` says so, and records
  -10.2% for the 2026-10-30 fold and +7.8% for the 2026-11-01 fold.

There is no lane-level interleaved window to use instead: `tools/release/v3_profile_matrix.py --mode profile`
measures one lane per call, in table order, and `--rounds` belongs to widths mode. The interleaving that
exists is inside one artifact's widths sweep, not across lanes.

Use this tool for provenance and drift detection. To publish, run `tools/release/fold_lane_figures.py
--apply`: it rewrites the table from this record and stamps `MEASURED_ENGINE`, so the gate can tell whether
the published numbers belong to the tree.

Usage:
    python tools/bench/lane_fingerprint.py                      # all eight lanes, three cells
    python tools/bench/lane_fingerprint.py --file <launcher> --cell code
    python tools/bench/lane_fingerprint.py --no-write --tolerance-tok 12

Exit status: 0 clean or drift-reported, 2 a lane refused to serve -- a scan that did not run is not a
pass. A *drift* is reported rather than failed, and that is measured: it compares against a recorded
number, and this card's session-to-session spread is the size of the tolerance -- one lane read 216.6,
230.5 and 239.0 within six hours (10.3%) while each run's own spread was 0.3 to 5.1%, so nothing inside
a run can size it. A drift means the committed provenance no longer describes the lanes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

from profiles import PROFILES  # noqa: E402
import v3_profile_matrix  # noqa: E402
from v3_profile_matrix import DOMAINS, run_profile  # noqa: E402

FIGURES = REPO / "tools" / "release" / "lane_figures.json"
SERVE = REPO / "build" / "apps" / "ninfer-serve.exe"
MODELS = Path(r"C:\AI\models")
TABLE_CELL = "code"      # the cell the published tok/acc were measured on, and the only comparable one

# (cell name, prompt-chars, prompts). The `code` cell uses the matrix's own domain, which is what the
# published row figures were measured on and the only cell comparable with them.
CELLS = [
    ("code", 225, 1),
    ("long-code", 12_000, 3),
    ("long-chinese", 12_000, 3),
]

# Where a cell's prompt comes from, for the cells that are not a DOMAINS name: generated at run time from
# tracked files, as (path, characters to take). A CI checkout is fresh, and the slices this session used
# lived under profiles/, which .gitignore:62 excludes -- the same trap the gpu workflow's header records for
# the FFmpeg tree. Everything below is tracked, so the fingerprint has no inputs outside the repository.
#
# A cell built from tracked source is only as stable as those sources. `long-code` includes
# `src/ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh`, which this port edits: on 2026-10-10 a ten-line alignment fix
# in that file's first 12,000 characters produced fourteen drifts across six lanes -- acceptance down 5
# to 27 points, decode both ways -- that were a prompt change and not a code effect. What identified it
# was that the other two cells did not move, `code` being a slice of a file the change did not touch.
# Read a `long-code` drift against that file's revision before reading it as a regression, and prefer
# `code` for a claim about a code change.
CELL_SOURCES = {
    "long-code": [("tools/release/v3_profile_matrix.py", 12_000),
                  ("src/ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh", 12_000),
                  ("tools/convert/official_recipes.py", 12_000)],
    "long-chinese": [("eval/corpora/perplexity-1m/data/zhwiki/01.txt", 36_000)],
}


def stage_cells(names: list[str]) -> None:
    """Register each file-backed cell in DOMAINS, from its tracked sources."""
    for name in names:
        if name not in CELL_SOURCES:
            continue
        DOMAINS[f"fingerprint-{name}"] = "".join(
            (REPO / path).read_text(encoding="utf-8", errors="replace")[:take]
            for path, take in CELL_SOURCES[name])


def acceptance_rate(record: dict) -> float | None:
    """`accept_rate` only, taken as a rate: a count read as a rate is how this reports 1328.0 against 71.2."""
    value = record.get("accept_rate")
    if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        return None
    return float(value)


def gpu_state() -> str:
    """One sample of the card's state, recorded with the figures it qualifies.

    Recorded, not required. A figure measured at one temperature is a sample of a range, and a reader's
    own machine, ambient and power delivery will place theirs somewhere else - so pinning a state would
    imply a canonical one that does not exist, and the fingerprint has no temperature gate. What the
    record can do is say what state this figure came from, which is what this is for.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,"
             "utilization.gpu,pstate", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def load_figures() -> dict:
    if not FIGURES.is_file():
        return {"what": "Lane fingerprints, per lane and cell, with the identity of what measured them.",
                "lanes": {}}
    return json.loads(FIGURES.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", dest="files", action="append",
                        help="shipped launcher; repeatable, default all")
    parser.add_argument("--cell", action="append", help="cell name; repeatable, default all")
    parser.add_argument("--tolerance-tok", type=float, default=10.0,
                        help="percent a cell may differ from the table's figure, or from the last run")
    parser.add_argument("--tolerance-acc", type=float, default=3.0, help="acceptance points, same basis")
    parser.add_argument("--serve", type=Path, default=SERVE,
                        help="the binary to measure with and to hash into the provenance; a CI checkout "
                             "has no build/, so the runner passes the machine's own binary")
    parser.add_argument("--models", type=Path, default=MODELS,
                        help="where the artifacts live")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the warm-up lane (it is run and discarded by default)")
    parser.add_argument("--no-write", action="store_true", help="measure and report, record nothing")
    parser.add_argument("--kv-dtype", choices=["bf16", "int8", "fp8", "nvfp4", "k8v4"],
                        help="override every selected lane's KV format, for a format A/B; the record "
                             "carries the value actually passed, because the profile dict is what the "
                             "matrix builds arguments from -- editing the launcher file does nothing")
    args = parser.parse_args()

    selected = [profile for profile in PROFILES if not args.files or profile["file"] in args.files]
    cells = [cell for cell in CELLS if not args.cell or cell[0] in args.cell]
    if not selected or not cells:
        raise SystemExit("no lane or cell selected")
    if args.kv_dtype:
        for profile in selected:
            profile["kv_dtype"] = args.kv_dtype
        print(f"    KV format overridden to {args.kv_dtype} for {len(selected)} lane(s)")
    stage_cells([cell[0] for cell in cells])

    # The matrix launches its own EXE constant, which lives inside its own tree -- and a CI checkout has no
    # build/, which is how the first dispatch failed with FileNotFoundError from Popen. Point it at the
    # binary this run hashes, so the figures come from the engine they are recorded against.
    if Path(v3_profile_matrix.EXE).resolve() != args.serve.resolve():
        print(f"    engine for this run: {args.serve} (overriding the matrix's own)")
        v3_profile_matrix.EXE = args.serve.resolve()
    if not args.serve.is_file():
        raise SystemExit(f"no serving binary at {args.serve}")

    payload = load_figures()
    lanes = payload.setdefault("lanes", {})
    print(f"    {len(selected)} lane(s) x {len(cells)} cell(s); the table is compared on "
          f"{TABLE_CELL!r} only, other cells against the previous fingerprint")

    drift: list[str] = []
    refused: list[str] = []
    # (tok, table figure) per lane on the table's own cell, for the run-level statement below: a drift
    # check compares against a past number and cannot see that a whole run moved together.
    run_level: list[tuple[float, float]] = []
    if selected and not args.no_warmup:
        # Run the first selected lane once and throw the numbers away. The card is idle when
        # this tool starts, and the first lane is otherwise measured while it is still
        # ramping -- 637 MHz on the recorded run of 2026-10-13, against 2812-2895 MHz for
        # every later lane. That is a property of the run, not of the lane.
        warm = selected[0]
        print(f"\n=== warm-up: {warm['file']} (discarded)")
        for name, _chars, _prompts in cells:
            domain = name if name not in CELL_SOURCES else f"fingerprint-{name}"
            try:
                run_profile(profile=warm, domain=domain, sampling="default")
            except Exception as exc:  # noqa: BLE001 - a warm-up must never fail the run
                print(f"    warm-up cell {name} did not run: {exc}")
        print("    card warmed; every measured lane now starts ramped")

    for profile in selected:
        launcher = profile["file"]
        print(f"\n=== {launcher}")
        artifact = args.models / profile["art"]
        recorded: dict[str, Any] = {
            "artifact": {"path": artifact.name, "bytes": artifact.stat().st_size,
                         "mtime": artifact.stat().st_mtime},
            "cells": {},
            "gpu_state": {"before": gpu_state()},
        }
        prior_cells = (lanes.get(launcher) or {}).get("cells", {})
        for name, chars, prompts in cells:
            domain = name if name not in CELL_SOURCES else f"fingerprint-{name}"
            record = run_profile(profile=profile, domain=domain, sampling="default")
            if record.get("ready") is not True:
                refused.append(f"{launcher} / {name}: {str(record.get('refusal'))[:70]}")
                print(f"    {name:<14} REFUSED TO SERVE")
                continue
            tok = float(record.get("decode_avg") or 0.0)
            acc = acceptance_rate(record)
            spread = (max(record.get("decode_tps") or [0]) - min(record.get("decode_tps") or [0]))
            line = f"    {name:<14} {tok:>7.1f} tok/s  spread {spread:>5.1f}"
            if acc is not None:
                line += f"  accept {acc * 100:>5.1f}%"

            # The table is the reference on its own cell; every other cell's reference is the last run.
            #
            # Measured 2026-10-12: an earlier record can describe conditions nothing reproduces today,
            # so a drift against one is not evidence of a regression until the same prompt has been
            # re-measured on a known engine. This lane's long-chinese record read 305.3 tok/s; the
            # pre-merge binary reads 104.3 on that cell now and the merged one reads 187.9, so the
            # record was the outlier -- and a drift against it was reported as a regression that did
            # not exist. Compare engines before concluding.
            if name == TABLE_CELL:
                published = float(profile.get("tok") or 0.0)
                table_acc = float(str(profile.get("acc", "0")).rstrip("%"))
                if published:
                    run_level.append((tok, published))
                    # Per lane, not only per run: the record accumulates one write per lane, so a
                    # payload-level field would describe only the last one. This is the measurement's
                    # own provenance - what the table said when this figure was taken - and it is why
                    # the field exists: a figure and the table can differ because the lane changed
                    # (a format moved) or because the run was long, and this says which by how much.
                    recorded["run_level_pct"] = round((tok / published - 1.0) * 100.0, 1)
                if published and abs(tok / published - 1.0) * 100.0 > args.tolerance_tok:
                    line += f"  DRIFT: table says {published:.1f} tok/s"
                    drift.append(f"{launcher} / {name}: {tok:.1f} against the table's {published:.1f}")
                if acc is not None and abs(acc * 100.0 - table_acc) > args.tolerance_acc:
                    line += f"  DRIFT: table says {table_acc:.1f}%"
                    drift.append(f"{launcher} / {name}: acceptance {acc * 100:.1f} against "
                                 f"{table_acc:.1f}")
            else:
                prior = prior_cells.get(name) or {}
                prior_tok = float(prior.get("decode_avg") or 0.0)
                if prior_tok and abs(tok / prior_tok - 1.0) * 100.0 > args.tolerance_tok:
                    line += f"  DRIFT: last run {prior_tok:.1f} tok/s"
                    drift.append(f"{launcher} / {name}: {tok:.1f} against the last run's {prior_tok:.1f}")
                prior_acc = prior.get("accept_rate")
                if acc is not None and isinstance(prior_acc, (int, float)) \
                        and abs(acc * 100.0 - float(prior_acc) * 100.0) > args.tolerance_acc:
                    drift.append(f"{launcher} / {name}: acceptance {acc * 100:.1f} against the last "
                                 f"run's {float(prior_acc) * 100:.1f}")
            print(line)
            recorded["cells"][name] = {"decode_avg": round(tok, 1), "accept_rate": acc,
                                       "spread": round(spread, 1),
                                       # the runs the average and the spread came from, so a reader can
                                       # tell a real difference from this lane's own noise - the record
                                       # kept only the average until 2026-10-13, which is why a 2% gap
                                       # could not be told from a 3% artefact without re-measuring
                                       "decode_tps": record.get("decode_tps") or [],
                                       "sampling": "default", "source": "measured"}
        recorded["gpu_state"]["after"] = gpu_state()
        lanes[launcher] = recorded

    # The run's own level, before any per-lane drift is read. A grouped run measures lanes in
    # sequence, and the 2026-10-13 refresh showed what that costs: all eight read below the table,
    # with a step between lanes 4 and 5 (-2 to -10%, then -11 to -16%). That is a condition of the
    # run, not a per-lane result, and a drift check against a past number cannot see it -- so it is
    # computed here, recorded with the figures it qualifies, and printed above the drift list.
    if run_level:
        ratios = sorted(tok / published for tok, published in run_level)
        median_ratio = ratios[len(ratios) // 2]
        payload["run_level"] = {
            "lanes": len(ratios),
            "median_vs_table_pct": round((median_ratio - 1.0) * 100.0, 1),
            "min_pct": round((ratios[0] - 1.0) * 100.0, 1),
            "max_pct": round((ratios[-1] - 1.0) * 100.0, 1),
        }
    payload["engine"] = {"file": str(args.serve), "sha256": _sha(args.serve),
                         "bytes": args.serve.stat().st_size,
                         "built_utc": datetime.fromtimestamp(args.serve.stat().st_mtime, timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ")}
    payload["measured_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload.pop("figures", None)  # the per-lane-per-cell schema below supersedes the first flat one
    if not args.no_write:
        FIGURES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\n    recorded {len(lanes)} lane(s) into {FIGURES.relative_to(REPO)}")

    if refused:
        print(f"    {len(refused)} cell(s) refused to serve:")
        for line in refused:
            print(f"        {line}")
        return 2
    if payload.get("run_level"):
        level = payload["run_level"]
        print(f"    run level: median {level['median_vs_table_pct']:+.1f}% against the table over "
              f"{level['lanes']} lane(s), range {level['min_pct']:+.1f}% to {level['max_pct']:+.1f}%")
        if abs(level["median_vs_table_pct"]) > args.tolerance_tok:
            print("    a whole-run shift is a condition of the run - order, warmup, session - not a "
                  "per-lane result: fold only after re-measuring the lane solo, or state the "
                  "condition wherever the figures are quoted")
    if drift:
        # Reported, not fatal, and that is a measured decision rather than a preference. The card's
        # session-to-session spread is the size of the tolerance: one lane read 216.6 (the committed
        # table), 230.5 and 239.0 within six hours -- a 10.3% span -- while each of those runs measured
        # a within-run spread of 0.3 to 5.1%, so no statistic available inside a run can size this. And
        # a drift check compares against a *past* number, which cannot be interleaved with the present,
        # so the honest verdict is "this no longer matches the record": an instruction to re-measure and
        # commit the provenance, not a failed gate. A lane that refuses to serve is still exit 2.
        print(f"    {len(drift)} drift(s) beyond tolerance, reported not fatal:")
        for line in drift[:12]:
            print(f"        {line}")
        print("    re-run the fingerprint where the artifacts live, then commit the new "
              "tools/release/lane_figures.json")
        return 0
    print("    every measured cell is inside tolerance")
    return 0


def _sha(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
