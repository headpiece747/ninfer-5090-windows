#!/usr/bin/env python3
"""Generate QUASAR_ARGS from the CLI flag set, not the server's.

The first attempt generated the server's argument list into launcher_env.bat, and running
test_prompt.bat showed the mistake: that helper invokes ninfer.exe, the offline CLI, which
rejects --host, --port, --model-id, --max-concurrency, the state slots, the host KV pool, the
cache bounds and the timeouts. The original hand-written value was closer to right than the
generated one; what it actually lacked was --lm-head-draft and --prefill-chunk.

profiles.cli_args now holds that subset, so the two interfaces stay separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES, cli_args  # noqa: E402

ENV = Path(__file__).resolve().parents[2] / "launcher_env.bat"


def main() -> int:
    profile = next(p for p in PROFILES if "mtp4" in p["file"])
    args = cli_args(profile)
    line = 'set "QUASAR_ARGS=' + " ".join(args) + '"'

    # newline="" on both the read and the write, and both are load-bearing. A default-mode read silently
    # converts this file's CRLF to LF in memory, and write_text then writes that back verbatim, so the
    # shipped file loses its line endings. git diff does not show it -- core.autocrlf=true normalises
    # both sides on commit -- but this file is compared as text between machines, where a checkout on
    # Windows restores CRLF and a regeneration would not. Path.open rather than Path.read_text because
    # read_text only gained a newline parameter in Python 3.13 and this runs on 3.12.
    with ENV.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    current = next((l for l in text.splitlines() if l.startswith('set "QUASAR_ARGS=')), None)
    if current is None:
        print("  ANCHOR NOT FOUND: QUASAR_ARGS")
        return 1
    if current == line:
        print(f"  already current ({len(args)} tokens)")
        return 0
    ENV.write_text(text.replace(current, line, 1), encoding="utf-8", newline="")
    print(f"  QUASAR_ARGS written from cli_args: {len(args)} tokens")
    for token in args:
        print(f"     {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
