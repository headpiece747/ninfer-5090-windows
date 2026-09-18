#!/usr/bin/env python3
"""Set the compaction config for our models, and show what it changes.

Compaction's V2 trigger is:

    estimated >= min(input_limit - buffer, context_limit - max(output_reserve, buffer))

with the output reserve capped at 32,000 tokens. Our per-model limit.input already reserves
exactly 32,768 (context - 32768), so the default buffer of 20,000 stacks on top of that and
costs roughly 12k of usable context for nothing. 8,000 keeps a real margin for the
compaction call -- its own summary prompt plus output allowance must fit -- without
double-counting the output reserve.

keep.tokens is raised from the 15,000 default to 20,000 because these are coding sessions,
where the recent file contents and diffs are exactly what a summary should not lose.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(r"C:\Users\tobia\.config\opencode\opencode.json")
BUFFER = 8000
KEEP = 20000
OUTPUT_RESERVE = 32768  # what compaction caps the output reserve at, and what our input limit already holds back


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["compaction"] = {"auto": True, "keep": {"tokens": KEEP}, "buffer": BUFFER}
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"   compaction: auto=True, keep.tokens={KEEP}, buffer={BUFFER}")

    check = json.loads(CONFIG.read_text(encoding="utf-8"))
    print(f"   {'model':<46} {'triggers':>9}  {'was':>9}  context")
    for provider in check["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" not in key:
                continue
            limit = model["limit"]
            ceiling = min(limit["input"] - BUFFER, limit["context"] - OUTPUT_RESERVE)
            before = min(limit["input"] - 20000, limit["context"] - OUTPUT_RESERVE)
            print(f"   {key:<46} {ceiling:>9,}  {before:>9,}  {limit['context']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
