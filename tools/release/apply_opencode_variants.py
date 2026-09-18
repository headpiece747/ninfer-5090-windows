#!/usr/bin/env python3
"""Add the measured thinking variant to each v3 model entry.

Hard-task testing showed `none` fails tasks that need a real algorithm (interval
scheduling: 0/2) while `low` and above pass them all, and that `low` matches `medium` and
`xhigh` at a third of the time. So the default stays `none` for speed and a `low` variant
carries the cases that need deliberation.

Deliberately no variants named `minimal` or `high`: the artifact's chat template rejects
both with HTTP 400.
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
            model["variants"] = {"think": {"reasoningEffort": "low"}}
            touched += 1
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"   {touched} v3 models: default effort=none, plus a 'think' variant at low")

    check = json.loads(CONFIG.read_text(encoding="utf-8"))
    for provider in check["provider"].values():
        for key, model in provider.get("models", {}).items():
            if "v3" in key:
                effort = model.get("options", {}).get("reasoningEffort")
                variants = ", ".join(f"{k}={v['reasoningEffort']}"
                                     for k, v in model.get("variants", {}).items())
                print(f"   {key:<46} default={effort:<5} variants: {variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
