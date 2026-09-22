"""Compare two .ninfer artifacts object by object: ids, shape, format, layout, payload size.

The converter's contract is that a converted artifact carries the same layout as the official one, and
the BOM defect showed how quietly a resource-level difference can ship. This is the check for that: it
reads only metadata, so it is fast on a 17 GB artifact and needs no GPU.

    python -m tools.artifact.inspect_diff LEFT.ninfer RIGHT.ninfer [--hash]

Exit code is 0 when the layouts match, 1 when anything differs. --hash also compares payload digests,
which reads every object and is therefore slow; without it only metadata and payload sizes are compared.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .reader import Artifact

# Object records carry either {id, encoding, offset, bytes} for raw files or
# {id, shape, format, layout, offset, bytes} for tensors, so the comparison reads attributes by name
# rather than assuming one shape.
ATTRIBUTES = ("encoding", "shape", "format", "layout", "bytes")


def describe(art: Artifact) -> dict[str, tuple]:
    described = {}
    for record in art.objects:
        described[record.id] = tuple(getattr(record, name, None) for name in ATTRIBUTES)
    return described


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hash_payloads = "--hash" in sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m tools.artifact.inspect_diff LEFT.ninfer RIGHT.ninfer [--hash]")
        return 2
    left_path, right_path = (Path(a) for a in args)

    with Artifact.open(left_path) as left, Artifact.open(right_path) as right:
        left_objects = describe(left)
        right_objects = describe(right)
        digest_mismatches = []
        if hash_payloads:
            for object_id in sorted(set(left_objects) & set(right_objects)):
                pair = []
                for art in (left, right):
                    digest = hashlib.sha256()
                    for block in art.iter_object(object_id):
                        digest.update(block)
                    pair.append(digest.hexdigest())
                if pair[0] != pair[1]:
                    digest_mismatches.append(object_id)

    only_left = sorted(set(left_objects) - set(right_objects))
    only_right = sorted(set(right_objects) - set(left_objects))
    mismatched = [
        object_id
        for object_id in sorted(set(left_objects) & set(right_objects))
        if left_objects[object_id] != right_objects[object_id]
    ]

    print(f"left : {left_path.name}  objects {len(left_objects)}")
    print(f"right: {right_path.name}  objects {len(right_objects)}")
    for label, ids in (("only in left", only_left), ("only in right", only_right)):
        if ids:
            print(f"  {label}: {len(ids)}  e.g. {ids[:3]}")
    if mismatched:
        print(f"  attribute mismatches: {len(mismatched)}")
        for object_id in mismatched[:8]:
            print(f"    {object_id}: {dict(zip(ATTRIBUTES, left_objects[object_id]))} != "
                  f"{dict(zip(ATTRIBUTES, right_objects[object_id]))}")
    if hash_payloads and digest_mismatches:
        print(f"  payload digest mismatches: {len(digest_mismatches)}  e.g. {digest_mismatches[:3]}")

    identical = not (only_left or only_right or mismatched)
    print(f"layout: {'IDENTICAL' if identical else 'DIFFERS'}"
          + (f"   payloads: {'identical' if not digest_mismatches else 'differ'}"
             if hash_payloads else ""))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
