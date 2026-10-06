#!/usr/bin/env python3
"""Fail when a Python test or tool spawns a subprocess in text mode without stating its encoding.

Why this exists
---------------
Twice in two days a Python oracle in this tree failed on Windows for the same reason, and neither
time was it the code under test:

  * tests/text/test_json_schema.py -- "UnicodeEncodeError: 'charmap' codec can't encode character
    '\\u4f60'" on its first case, before the probe ran;
  * tests/models/qwen3_5/test_tool_schema.py -- the same, "characters in position 549-550".

Both wrote `subprocess.run(..., text=True, ...)` with no `encoding`, and both build their payload with
`ensure_ascii=False`, so the bytes are deliberately non-ASCII. With `text=True` and no encoding,
Python uses the LOCALE encoding: UTF-8 on the maintainer's Linux host, cp1252 on Windows. So the test
passes where it was written and fails on the port, and upstream has no CI that would have said so.

Two instances is the threshold this repo uses for a gate rather than another paragraph in AGENTS.md.

What it checks, and what it deliberately does not
------------------------------------------------
AST, not a text pattern. A regex over a call's arguments cannot tell an argument of this call from a
keyword in a neighbouring one, and this repo has been bitten by checks that pass for a structural
reason. `ast` sees the call node, so the check is exact:

  FAIL  subprocess.run(..., text=True, ...)                with no encoding= or errors=
  PASS  subprocess.run(..., text=True, encoding="utf-8")   explicit
  PASS  subprocess.run(..., capture_output=True)           binary mode -- no decoding happens
  PASS  subprocess.run(..., encoding="utf-8")              encoding implies text

Scope is `tests/` and `tools/`: the places that drive a built binary. It is not a general style rule
about subprocess use, and it says nothing about a call that never decodes.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("tests", "tools")
SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}
TEXT_KWARGS = ("text", "universal_newlines")


def spawned_text_calls(tree: ast.AST) -> list[tuple[int, bool, bool]]:
    """(line, text_mode, states_encoding) for every subprocess spawn in this module."""
    found: list[tuple[int, bool, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name not in SPAWNERS:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        # A loop rather than a comprehension: mypy narrows `isinstance(value, ast.Constant)` on a
        # bound local, but not across two separate subscripts of the same dict -- the original
        # `isinstance(keywords.get(kw), ast.Constant) and keywords[kw].value` left the second lookup
        # typed as `ast.expr`, and the hook's mypy step could not run over this file until it was
        # bound once. Same behaviour: a missing key is simply not a constant.
        text_mode = False
        for kw in TEXT_KWARGS:
            value = keywords.get(kw)
            if isinstance(value, ast.Constant) and value.value is True:
                text_mode = True
                break
        states_encoding = "encoding" in keywords or "errors" in keywords
        found.append((node.lineno, text_mode, states_encoding))
    return found


def main() -> int:
    problems: list[str] = []
    checked = 0
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as error:
                problems.append(f"{rel}: could not be parsed -- {error}")
                continue
            for line, text_mode, states_encoding in spawned_text_calls(tree):
                checked += 1
                if text_mode and not states_encoding:
                    problems.append(
                        f"{rel}:{line}: spawns a subprocess in text mode without stating an "
                        f"encoding -- the locale decides, and cp1252 is not UTF-8"
                    )

    if problems:
        print("  FAIL: a subprocess in text mode inherits the locale encoding:")
        for problem in problems:
            print(f"    {problem}")
        print("  State it: subprocess.run(..., text=True, encoding=\"utf-8\")")
        return 1

    print(f"  {checked} subprocess spawn(s) checked across {'/ and '.join(ROOTS)}; "
          f"every text-mode one states its encoding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
