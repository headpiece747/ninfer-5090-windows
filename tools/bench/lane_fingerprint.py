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

Usage:
    python tools/bench/lane_fingerprint.py                      # all eight lanes, three cells
    python tools/bench/lane_fingerprint.py --file <launcher> --cell code
    python tools/bench/lane_fingerprint.py --no-write --tolerance-tok 12

Exit status: 0 clean, 1 drift, 2 a lane refused to serve -- a scan that did not run is not a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
CELL_SOURCES = {
    "long-code": [("tools/release/v3_profile_matrix.py", 12_000),
                  ("src/ops/linear/nvfp4/nvfp4_a4_tma.cuh", 12_000),
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
    parser.add_argument("--no-write", action="store_true", help="measure and report, record nothing")
    args = parser.parse_args()

    selected = [profile for profile in PROFILES if not args.files or profile["file"] in args.files]
    cells = [cell for cell in CELLS if not args.cell or cell[0] in args.cell]
    if not selected or not cells:
        raise SystemExit("no lane or cell selected")
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
    for profile in selected:
        launcher = profile["file"]
        print(f"\n=== {launcher}")
        artifact = args.models / profile["art"]
        recorded = {
            "artifact": {"path": artifact.name, "bytes": artifact.stat().st_size,
                         "mtime": artifact.stat().st_mtime},
            "cells": {},
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
            if name == TABLE_CELL:
                published = float(profile.get("tok") or 0.0)
                table_acc = float(str(profile.get("acc", "0")).rstrip("%"))
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
                                       "sampling": "default", "source": "measured"}
        lanes[launcher] = recorded

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
    if drift:
        print(f"    {len(drift)} drift(s) beyond tolerance:")
        for line in drift[:12]:
            print(f"        {line}")
        return 1
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
