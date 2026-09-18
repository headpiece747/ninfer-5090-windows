#!/usr/bin/env python3
"""Give every v3 model an `xhigh` variant alongside the `low` one.

Measured guidance (docs/opencode-settings.md): `none` stays the default because it matches
xhigh's score on these tasks at a fraction of the time, and `xhigh` is exposed because it
solves tasks `none` fails -- interval scheduling -- even though it fails others that `none`
passes.

The server keeps --default-thinking-budget 4096: raising it produced 5.5x the reasoning and
5x the time at an identical score. limit.output stays 32768 so reasoning can never truncate
the answer.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(r"C:\Users\tobia\.config\opencode\opencode.json")


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    touched = 0
    for provider in config["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" not in key:
                continue
            model.setdefault("options", {})["reasoningEffort"] = "none"
            model["variants"] = {
                "think": {"reasoningEffort": "low"},
                "xhigh": {"reasoningEffort": "xhigh"},
            }
            touched += 1
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"   {touched} v3 models: default=none, variants think=low and xhigh=xhigh")

    check = json.loads(CONFIG.read_text(encoding="utf-8"))
    for provider in check["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" not in key:
                continue
            variants = ", ".join(f"{name}={spec['reasoningEffort']}"
                                 for name, spec in model.get("variants", {}).items())
            print(f"   {key:<46} default={model['options']['reasoningEffort']:<5} "
                  f"out={model['limit']['output']:<6} variants: {variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
