#!/usr/bin/env python3
"""Validate captured `--request-log-jsonl` files against the schema docs/serving.md documents.

The request log is the port's diagnostic surface: `server_start` records the artifact's identity, the
resolved engine configuration and the argv the server was started with, and every `request_done`
records the finish reason, the token counters and the complete speculative counters. Nothing checked a
captured log against that contract until now, so a lane that silently ran without its draft backend, or
a run whose log was truncated, would only be noticed by reading.

It validates what it can be sure of:

  * every line parses, with the declared `artifact_type` and a schema version
  * exactly one `server_start`, naming a `.ninfer` artifact
  * for a log that carries requests: at least one `request_done` with a finish reason and positive
    completion tokens, and -- when the argv selects a speculative backend -- drafts issued and accepted

A log with no requests is not a failure. `ceiling` probes start the engine and read its accounting
without sending a request, and a ladder step that is refused logs no `server_start` at all; both are
counted and reported separately rather than being called defects.

    python3 tools/release/check_request_logs.py --logs "C:/AI/bench/req_*.jsonl"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

EXPECTED_TYPE = "ninfer_serve_request_log"


def flag(argv: list[str], name: str) -> str:
    """Value of `--name` in a captured argv, or an empty string."""
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return ""


def inspect(path: str) -> dict:
    row: dict = {
        "path": os.path.basename(path),
        "parse_errors": 0,
        "problems": [],
        "drafted": 0,
        "accepted": 0,
        "completions": 0,
        "requests": 0,
    }
    starts: list[dict] = []

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                row["parse_errors"] += 1
                continue
            if record.get("artifact_type") != EXPECTED_TYPE:
                row["problems"].append(f"artifact_type={record.get('artifact_type')!r}")
            event = record.get("event")
            if event == "server_start":
                starts.append(record)
            elif event == "request_done":
                row["requests"] += 1
                spec = record.get("speculative") or {}
                row["drafted"] += int(spec.get("drafted_tokens") or 0)
                row["accepted"] += int(spec.get("accepted_tokens") or 0)
                result = record.get("result") or {}
                row["completions"] += int(result.get("completion_tokens") or 0)
                if not result.get("finish_reason"):
                    row["problems"].append("request_done without a finish_reason")

    if row["parse_errors"]:
        row["problems"].append(f"{row['parse_errors']} unparseable line(s)")
    if not starts:
        row["started"] = False
        return row
    row["started"] = True
    if len(starts) > 1:
        row["problems"].append(f"{len(starts)} server_start events")

    argv = starts[0].get("argv", [])
    row["artifact"] = os.path.basename(argv[1]) if len(argv) > 1 else "?"
    row["model_id"] = flag(argv, "--model-id")
    row["max_context"] = flag(argv, "--max-context")
    row["spec"] = flag(argv, "--spec")
    if not row["artifact"].endswith(".ninfer"):
        row["problems"].append(f"artifact argument is not a .ninfer path: {row['artifact']!r}")
    if not starts[0].get("schema_version"):
        row["problems"].append("server_start without a schema_version")

    if row["requests"] == 0:
        return row
    if row["completions"] == 0:
        row["problems"].append("requests completed with no completion tokens")
    if row["spec"] and row["drafted"] == 0:
        row["problems"].append(f"--spec {row['spec']} selected but no drafts were issued")
    if row["spec"] and row["accepted"] == 0:
        row["problems"].append(f"--spec {row['spec']} selected but nothing was accepted")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, help="glob of request-log files to validate")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.logs))
    if not paths:
        print(f"no logs matched {args.logs}")
        return 1

    rows = [inspect(path) for path in paths]
    unstarted = [r for r in rows if not r.get("started")]
    probes = [r for r in rows if r.get("started") and r["requests"] == 0]
    live = [r for r in rows if r.get("started") and r["requests"] > 0]
    broken = [r for r in rows if r["problems"]]

    print(f"{len(paths)} request logs: {len(live)} with requests, {len(probes)} started without one, "
          f"{len(unstarted)} that never started")
    by_artifact: dict[str, list[dict]] = collections.defaultdict(list)
    for row in live:
        by_artifact[row.get("artifact", "?")].append(row)
    print(f"{'artifact':<40} {'logs':>5} {'requests':>9} {'drafts':>8} {'accepted':>9} "
          f"{'completions':>12}")
    for artifact in sorted(by_artifact):
        group = by_artifact[artifact]
        print(f"{artifact:<40} {len(group):>5} {sum(r['requests'] for r in group):>9} "
              f"{sum(r['drafted'] for r in group):>8} {sum(r['accepted'] for r in group):>9} "
              f"{sum(r['completions'] for r in group):>12}")

    if broken:
        print()
        print(f"{len(broken)} log(s) have problems:")
        for row in broken:
            print(f"  {row['path']}: {'; '.join(row['problems'])}")
        return 1
    print()
    print("every log is well-formed: parseable lines, one server_start naming a .ninfer artifact, and "
          "request_done events carrying completions and a speculative backend that issued and accepted "
          "drafts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
