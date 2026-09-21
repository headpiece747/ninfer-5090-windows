"""Checks for the launcher generator's per-profile branches.

The four shipped profiles all set vision, a speculative backend and the proposal head, so the
other side of every branch in varying_flags is live code that nothing exercises: a profile with
no vision, no speculation, or a head without a backend is a case the shipped set cannot reach.

Synthetic profiles cover those, and the flag order the generator renders is asserted directly,
because the launchers are compared byte-for-byte against it and order is therefore part of the
contract.

Runs as a ctest test, following tests/artifact/writer_interop.py: exit 0 for pass, 1 for fail.
This tree has no Python test runner and does not need one for this.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

from profiles import INVARIANT_FLAGS, cli_args, launcher_args, ordered_flags, varying_flags  # noqa: E402

failures: list[str] = []


def expect(condition: bool, label: str) -> None:
    if not condition:
        failures.append(label)


def profile(**overrides):
    """A profile that takes no optional flag, so each branch can be turned on one at a time."""
    base = dict(file="start_test.bat", port=8090, art="a.v3.ninfer", device_state_slots=1,
                label="test", model_id="test-model", spec="none", draft=0, vision=False,
                lm_head=False, ctx=4096, tok=0, acc="0%", runtime="", free="", note="")
    base.update(overrides)
    return base


expect(varying_flags(profile()) == [], "no vision and no speculation adds no flag")
expect(varying_flags(profile(vision=True)) == ["--vision"], "vision alone adds --vision")
expect(varying_flags(profile(spec="mtp", draft=3)) == ["--spec mtp", "--draft-tokens 3"],
       "speculation alone adds the backend and its depth")
expect(varying_flags(profile(spec="dflash2", draft=7, lm_head=True))
       == ["--spec dflash2", "--draft-tokens 7", "--lm-head-draft"],
       "the proposal head rides with a speculative backend")
expect(varying_flags(profile(lm_head=True)) == [],
       "a head without a backend adds nothing, because there is no head to select")

flags = ordered_flags(profile(vision=True, spec="mtp", draft=4, lm_head=True, port=8080,
                              model_id="probe-model"))
names = [flag for flag, _ in flags]
expect(names[0] == "--vision", "the per-profile flags come first")
expect(names[1:3] == ["--spec mtp", "--draft-tokens 4"],
       "the backend and its depth follow, each carrying its value in one token")
expect(names.index("--port") < names.index("--kv-capacity"),
       "the bound flags precede the invariant ones")
expect(names[-len(INVARIANT_FLAGS):] == [flag for flag, _ in INVARIANT_FLAGS],
       "every invariant flag is rendered last, in its declared order")
expect([flag for flag, value in INVARIANT_FLAGS if value is None] == ["--preserve-thinking"],
       "only --preserve-thinking is declared without a value")

args = launcher_args(profile(vision=True, spec="mtp", draft=4, lm_head=True, port=8080,
                             model_id="probe-model"))
expect(args[args.index("--port") + 1] == "8080", "the port reaches the argument list")
expect(args[args.index("--model-id") + 1] == "probe-model", "the model id reaches it")
expect(args[args.index("--max-context") + 1] == "4096", "the ceiling reaches it")
expect(len(args) == sum(1 if value is None and " " not in flag
                        else len(flag.split()) + (1 if value is not None else 0)
                        for flag, value in flags),
       "each ordered flag flattens to its tokens, or to one token with its value")

server_only = {"--host", "--port", "--model-id", "--max-concurrency", "--host-state-slots",
               "--host-kv-mib", "--pending-timeout-ms"}
cli = cli_args(profile(vision=True, spec="mtp", draft=4, lm_head=True))
expect(not (set(cli) & server_only), "the CLI flag set names no server-only flag")
expect("--lm-head-draft" in cli, "the CLI flag set includes the proposal head it accepts")
expect("--vision" in cli and "--spec" in cli, "the CLI flag set includes what both interfaces share")

if failures:
    for failure in failures:
        print(f"  FAIL  {failure}")
    raise SystemExit(1)
print("  launcher generation checks passed")
raise SystemExit(0)
