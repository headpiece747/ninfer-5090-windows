#!/usr/bin/env python3
"""Browse a flashinfer-trace (FIB) dataset to pick workloads for profiling.

flashinfer-bench ships its benchmark workloads as a dataset with this layout:

    <dataset_root>/
    ├── definitions/<category>/<definition_name>.json      # axes, shapes, dtypes, reference impl
    ├── workloads/<category>/<definition_name>.jsonl       # one line per workload (uuid, axes, input paths)
    └── blob/workloads/<category>/<definition_name>/
        └── <definition_name>_<uuid>.safetensors           # raw tensors per workload

Scalar inputs (like `scale`) are stored inline in the jsonl; tensor inputs
are `{"type": "safetensors", "path": "./blob/...", "tensor_key": "<name>"}`.

This script helps pick representative workloads when writing a harness:

  # Summarize all workloads for a definition (shape distribution)
  python3 list_flashinfer_workloads.py --definition <your_definition_name>

  # Filter to specific axis values
  python3 list_flashinfer_workloads.py --definition <def> --filter <axis>=<value>

  # Print the absolute safetensors path for a specific uuid
  python3 list_flashinfer_workloads.py --definition <def> --uuid <uuid>

  # Pick one workload per distinct (axis1, axis2) tuple (useful for dispatch coverage)
  python3 list_flashinfer_workloads.py --definition <def> --unique-axes <axis1>,<axis2>

The dataset root is taken from $FIB_DATASET_PATH if set, otherwise --dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable


def locate_dataset(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit
    else:
        env = os.environ.get("FIB_DATASET_PATH")
        if not env:
            sys.exit("Set $FIB_DATASET_PATH or pass --dataset <path>.")
        p = Path(env)
    if not p.is_dir():
        sys.exit(f"Dataset path {p} is not a directory.")
    if not (p / "definitions").is_dir() or not (p / "workloads").is_dir():
        sys.exit(f"{p} does not look like a flashinfer-trace dataset "
                 f"(expected definitions/ and workloads/ subdirs).")
    return p


def find_definition_file(dataset: Path, definition: str) -> Path:
    """Search <dataset>/definitions/<category>/<definition>.json."""
    for cat_dir in sorted((dataset / "definitions").iterdir()):
        cand = cat_dir / f"{definition}.json"
        if cand.is_file():
            return cand
    sys.exit(f"Could not find definition '{definition}' under {dataset}/definitions/*/")


def find_workloads_file(dataset: Path, definition: str) -> Path:
    for cat_dir in sorted((dataset / "workloads").iterdir()):
        cand = cat_dir / f"{definition}.jsonl"
        if cand.is_file():
            return cand
    sys.exit(f"Could not find workloads for '{definition}' under {dataset}/workloads/*/")


def read_workloads(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_definition(def_path: Path) -> None:
    d = json.loads(def_path.read_text())
    print(f"Definition: {d['name']}")
    if d.get("description"):
        print(f"  {d['description']}")
    print(f"  Axes:")
    for axis, info in d.get("axes", {}).items():
        kind = info.get("type", "?")
        desc = info.get("description", "")
        val = info.get("value", "")
        extra = f"= {val}" if kind == "const" else ""
        print(f"    {axis} ({kind}) {extra}   {desc}")
    print(f"  Inputs:")
    for name, info in d.get("inputs", {}).items():
        shape = info.get("shape", "?")
        dtype = info.get("dtype", "?")
        optional = "  [optional]" if info.get("optional") else ""
        print(f"    {name}: shape={shape} dtype={dtype}{optional}")
    print(f"  Outputs:")
    for name, info in d.get("outputs", {}).items():
        print(f"    {name}: shape={info.get('shape')} dtype={info.get('dtype')}")


def parse_filter(filter_args: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in filter_args or []:
        if "=" not in f:
            sys.exit(f"Bad --filter '{f}': expected key=value")
        k, v = f.split("=", 1)
        # try to coerce to int
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def matches(axes: dict, filters: dict) -> bool:
    for k, v in filters.items():
        if axes.get(k) != v:
            return False
    return True


def summarize_workloads(workloads: list[dict], axes_keys: list[str]) -> None:
    """Print distribution: {(axis1, axis2, ...): count}."""
    dist = Counter()
    for w in workloads:
        axes = w["workload"]["axes"]
        key = tuple(axes.get(k) for k in axes_keys)
        dist[key] += 1

    print(f"\nDistribution over {axes_keys} ({len(workloads)} workloads total):")
    print(f"  {' '.join(f'{a:>15}' for a in axes_keys)}  count")
    for key, cnt in sorted(dist.items(), key=lambda x: x[0]):
        print(f"  {' '.join(f'{str(v):>15}' for v in key)}  {cnt}")


def safetensors_path_for(dataset: Path, workload: dict) -> Path | None:
    """Return absolute path to the safetensors file for this workload, if any."""
    for _, inp in workload["workload"].get("inputs", {}).items():
        if isinstance(inp, dict) and inp.get("type") == "safetensors":
            rel = inp.get("path")
            if rel:
                abs_path = dataset / rel.lstrip("./")
                return abs_path.resolve()
    return None


def list_workloads(dataset: Path, workloads: list[dict], filters: dict, show_paths: bool) -> None:
    print(f"\n{'UUID':<38} {'Axes':<40} {'Safetensors' if show_paths else ''}")
    print("-" * (38 + 40 + (60 if show_paths else 0)))
    n_shown = 0
    for w in workloads:
        axes = w["workload"]["axes"]
        if not matches(axes, filters):
            continue
        uuid = w["workload"]["uuid"]
        axes_str = " ".join(f"{k}={axes[k]}" for k in sorted(axes.keys()))
        path_str = ""
        if show_paths:
            p = safetensors_path_for(dataset, w)
            path_str = str(p) if p else "(no safetensors)"
        print(f"{uuid:<38} {axes_str:<40} {path_str}")
        n_shown += 1
    print(f"\n({n_shown} matching workloads)")


def pick_unique_axes(dataset: Path, workloads: list[dict], axis_keys: list[str],
                     show_paths: bool) -> None:
    """Print one representative workload per distinct tuple of axis values."""
    seen: OrderedDict = OrderedDict()
    for w in workloads:
        axes = w["workload"]["axes"]
        key = tuple(axes.get(k) for k in axis_keys)
        if key not in seen:
            seen[key] = w
    print(f"\nOne representative per unique {axis_keys} ({len(seen)} combinations):")
    print(f"  {'Axes':<40} {'UUID':<38}{' Safetensors' if show_paths else ''}")
    for key, w in seen.items():
        axes = w["workload"]["axes"]
        axes_str = " ".join(f"{k}={axes[k]}" for k in axis_keys)
        path_str = ""
        if show_paths:
            p = safetensors_path_for(dataset, w)
            path_str = f" {p}" if p else " (none)"
        print(f"  {axes_str:<40} {w['workload']['uuid']:<38}{path_str}")


def find_by_uuid(dataset: Path, workloads: list[dict], uuid: str) -> None:
    for w in workloads:
        if w["workload"]["uuid"] == uuid:
            axes = w["workload"]["axes"]
            print(f"UUID: {uuid}")
            print(f"Axes: {axes}")
            path = safetensors_path_for(dataset, w)
            if path:
                print(f"Safetensors: {path}")
                if path.is_file():
                    print(f"  (file exists, {path.stat().st_size / 1e6:.1f} MB)")
                else:
                    print("  (file does NOT exist at this path)")
            # inline scalars
            print("Scalars:")
            for name, inp in w["workload"].get("inputs", {}).items():
                if isinstance(inp, dict) and inp.get("type") == "scalar":
                    print(f"  {name} = {inp.get('value')}")
            return
    sys.exit(f"UUID {uuid} not found in workloads.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Path to flashinfer-trace root. Defaults to $FIB_DATASET_PATH.")
    ap.add_argument("--definition", required=True,
                    help="Definition name (without .json), e.g. <your_definition_name>.")
    ap.add_argument("--show-definition", action="store_true",
                    help="Print the full definition (axes, input/output shapes, dtypes) and exit.")
    ap.add_argument("--filter", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Keep only workloads with the given axis = value. Repeatable.")
    ap.add_argument("--axes", default=None,
                    help="Comma-separated axis keys for summary histogram (default: auto-pick the 'var' axes).")
    ap.add_argument("--unique-axes", default=None,
                    help="Comma-separated axis keys. Print one representative workload per unique tuple.")
    ap.add_argument("--uuid", default=None,
                    help="Print details (including safetensors path) for a specific workload UUID.")
    ap.add_argument("--list", action="store_true",
                    help="List every workload (uuid + axes + safetensors path).")
    ap.add_argument("--no-paths", action="store_true",
                    help="Omit safetensors paths from output.")
    args = ap.parse_args()

    dataset = locate_dataset(args.dataset)
    def_path = find_definition_file(dataset, args.definition)

    if args.show_definition:
        summarize_definition(def_path)
        return

    wl_path = find_workloads_file(dataset, args.definition)
    workloads = list(read_workloads(wl_path))
    filters = parse_filter(args.filter)

    if args.uuid:
        find_by_uuid(dataset, workloads, args.uuid)
        return

    if args.list:
        list_workloads(dataset, workloads, filters, show_paths=not args.no_paths)
        return

    if args.unique_axes:
        keys = [k.strip() for k in args.unique_axes.split(",") if k.strip()]
        # Filter first, then pick uniques
        filtered = [w for w in workloads if matches(w["workload"]["axes"], filters)]
        pick_unique_axes(dataset, filtered, keys, show_paths=not args.no_paths)
        return

    # Default: summary.
    summarize_definition(def_path)
    if args.axes:
        axis_keys = [k.strip() for k in args.axes.split(",") if k.strip()]
    else:
        # Auto-pick: use 'var' axes from the definition
        d = json.loads(def_path.read_text())
        axis_keys = [k for k, info in d.get("axes", {}).items() if info.get("type") == "var"]
        if not axis_keys:
            # fall back to all keys from first workload
            axis_keys = list(workloads[0]["workload"]["axes"].keys())
    filtered = [w for w in workloads if matches(w["workload"]["axes"], filters)]
    summarize_workloads(filtered, axis_keys)


if __name__ == "__main__":
    main()
