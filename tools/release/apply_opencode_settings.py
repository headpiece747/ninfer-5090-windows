#!/usr/bin/env python3
"""Apply the measured opencode settings to the v3 model entries.

From docs/opencode-settings.md: `none` matched every other effort's pass rate at a
fraction of the time, and the failures at higher effort were truncation (reasoning tokens
billed against a 2,048-token output cap) rather than bad answers. So default to no
thinking, and give output enough room that turning thinking on later cannot silently
truncate the answer.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(r"C:\Users\tobia\.config\opencode\opencode.json")
OUTPUT_TOKENS = 32768


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    touched = 0
    for provider in config["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" not in key:
                continue
            model.setdefault("options", {})["reasoningEffort"] = "none"
            model["limit"]["output"] = OUTPUT_TOKENS
            touched += 1
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"   applied reasoningEffort=none and output={OUTPUT_TOKENS} to {touched} v3 models")

    check = json.loads(CONFIG.read_text(encoding="utf-8"))
    for provider in check["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" in key:
                effort = model.get("options", {}).get("reasoningEffort", "-")
                print(f"   {key:<46} effort={effort:<5} "
                      f"out={model['limit']['output']:<6} ctx={model['limit']['context']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
