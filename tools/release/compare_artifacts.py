#!/usr/bin/env python3
"""Compare two .ninfer v3 entry files: header, JSON index, and payload bytes.

Answers one question that recurs whenever an artifact is republished: are these two files the same
model? Written because a locally-upgraded copy and the repository's own v3 produced different tokens
(61.8% against 62.5% acceptance, different digests) while being only 4,283 bytes apart -- 4,096 of
that JSON trailing-whitespace padding -- which no size check can see.

Per docs/maintainer/artifact-container.md 3.1 the entry header is 32 bytes (magic, json_bytes,
artifact_id), the JSON occupies [32, metadata_end), and the payload starts at
align_up(metadata_end, 4096).

  header   magic, json_bytes, artifact_id, payload start
  index    canonical JSON diff (parsing then re-dumping removes the writer's padding)
  payload  sha256 of [payload_start, EOF) -- equal means identical weights

usage: compare_artifacts.py A.ninfer B.ninfer [--metadata-only] [--diff-lines N] [--find PATTERN]
"""
from __future__ import annotations

import difflib
import hashlib
import json
import struct
import sys
from pathlib import Path


def read_entry(path: Path) -> tuple[bytes, int, bytes, dict]:
    with path.open("rb") as handle:
        head = handle.read(32)
        json_bytes = struct.unpack("<Q", head[8:16])[0]
        index = json.loads(handle.read(json_bytes).decode("utf-8"))
    return head[:8], json_bytes, head[16:32], index


def sha_region(path: Path, start: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = sys.argv[1:]
    paths: list[Path] = []
    metadata_only = False
    diff_lines = 40
    find: str | None = None
    at = 0
    while at < len(args):
        arg = args[at]
        if arg == "--metadata-only":
            metadata_only = True
        elif arg in ("--diff-lines", "--find"):
            at += 1
            if at >= len(args):
                print(f"{arg} needs a value")
                return 2
            if arg == "--diff-lines":
                diff_lines = int(args[at])
            else:
                find = args[at]
        elif arg.startswith("--"):
            print(f"unknown option: {arg}")
            return 2
        else:
            paths.append(Path(arg))
        at += 1

    if len(paths) != 2:
        print(__doc__)
        return 2

    info: dict[str, dict] = {}
    for path in paths:
        magic, json_bytes, artifact_id, index = read_entry(path)
        payload_start = -(-(32 + json_bytes) // 4096) * 4096
        info[path.name] = dict(magic=magic, json_bytes=json_bytes, artifact_id=artifact_id,
                               payload_start=payload_start, index=index)
        declared = index.get("files", [{}])[0].get("payload_bytes")
        print(f"{path.name}")
        print(f"  magic={magic.decode('latin1')!r} json_bytes={json_bytes:,} "
              f"artifact_id={artifact_id.hex()} payload_start={payload_start:,} "
              f"file_bytes={path.stat().st_size:,}")
        print(f"  components={list(index.get('components', {}))} files={len(index.get('files', []))} "
              f"payload_bytes={declared:,}")

    first, second = (info[path.name] for path in paths)
    print("\n--- header and metadata ---")
    for key in ("magic", "json_bytes", "artifact_id", "payload_start"):
        left, right = first[key], second[key]
        print(f"  {key:<14} {'same' if left == right else 'DIFFERENT'}   {left!r} vs {right!r}")

    left = json.dumps(first["index"], indent=1, sort_keys=True).splitlines()
    right = json.dumps(second["index"], indent=1, sort_keys=True).splitlines()
    print(f"  artifact_id    {'same' if first['artifact_id'] == second['artifact_id'] else 'differs by construction: it identifies the file set'}")

    if find:
        print(f"\n--- index entries matching {find!r}, with three lines either side ---")
        for path, lines in zip(paths, (left, right)):
            print(f"  [{path.name}]")
            for at, line in enumerate(lines):
                if find in line:
                    for context_line in lines[max(0, at - 3):at + 4]:
                        print(f"    {context_line.strip()[:150]}")

    if left == right:
        print("  index JSON     identical")
    else:
        diff = list(difflib.unified_diff(left, right, "A", "B", n=1, lineterm=""))
        print(f"  index JSON     DIFFERENT ({len(diff)} changed lines; first {diff_lines})")
        for line in diff[:diff_lines]:
            print(f"    {line[:160]}")

    print("\n--- payload region (the weights) ---")
    if metadata_only:
        print("  skipped (--metadata-only)")
    else:
        digests = [sha_region(path, info[path.name]["payload_start"]) for path in paths]
        for path, digest in zip(paths, digests):
            print(f"  {path.name}: {digest}")
        print(f"  payloads {'identical' if digests[0] == digests[1] else 'DIFFERENT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
