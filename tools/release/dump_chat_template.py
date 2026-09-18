#!/usr/bin/env python3
"""Dump the artifact's chat template so the reasoning_effort mapping is visible.

The template (not the server) is what rejected minimal/high, so it also decides what
`low`, `medium` and `xhigh` actually mean -- how many thinking tokens each allows and what
it puts in the system prompt. Reading it is the difference between tuning the server and
guessing.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

MODEL = Path(r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer")
OUT = Path(r"C:\AI\bench\chat_template.jinja")


def main() -> int:
    with MODEL.open("rb") as handle:
        header = handle.read(32)
        json_bytes = struct.unpack("<Q", header[8:16])[0]
        index = json.loads(handle.read(json_bytes).decode("utf-8"))
        payload_start = 32 + json_bytes

        objects = index["objects"]
        by_id = {o["id"]: o for o in objects}
        wanted = [o for o in objects if "jinja" in o["id"] or "template" in o.get("id", "")]
        print(f"   template-ish objects: {[o['id'] for o in wanted]}")

        target = next((o for o in objects if o["id"].endswith("chat_template.jinja")), None)
        if target is None:
            print("   no chat_template.jinja object")
            return 1
        print(f"   using object {target['id']}")

        handle.seek(payload_start + target["offset"])
        data = handle.read(target["bytes"])
        OUT.write_bytes(data)
        print(f"   wrote {OUT} ({len(data)} bytes)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
