"""Count the rule bullets in AGENTS.md's rules section and check the stated count against it.

Written because that number has been wrong in this file more than once, and each time the error was
invisible: a reader cannot count 60-odd bullets, so a stale count reads exactly like a correct one.
The count sentence lives just before the bullets, so this measures from there to the next `## `
heading -- the same region a reader is looking at.

Fails on any drift rather than reporting it, and prints the region it measured so the number is
checkable rather than asserted.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RULES_FILE = REPO / "AGENTS.md"

# The sentence is split across two lines in the file ("Sixty-four " / "rules, each earned ..."), so
# the phrase is matched against the two lines joined. A regex over the raw text finds only "Sixty-"
# followed by a newline and matches nothing -- which is how the first version of this reported "no
# word-number count found" on a file that plainly carries one.
COUNT_SENTENCE = re.compile(r"\b([A-Z][a-z]+)-([a-z]+)\s+rules\b")
WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}


def main() -> int:
    text = RULES_FILE.read_bytes().decode("utf-8")
    marker = text.find("rules, each earned by a failure")
    if marker < 0:
        print("  FAIL: the count sentence was not found, so there is nothing to check")
        return 1
    start = text.rfind("\n", 0, text.rfind("\n", 0, marker)) + 1
    end = text.find("\n## ", marker)
    region = text[start : end if end > 0 else len(text)]

    bullets = len(re.findall(r"^- \*\*", region, re.M))
    # `marker` sits INSIDE the sentence, at "rules, each earned...", so text[start:marker] holds only
    # the number and never the word "rules". Take the count's whole line instead, then normalise the
    # wrap. The first two versions both sliced to `marker` and reported "no word-number count found"
    # on a file that plainly carries one.
    line_end = text.find("\n", marker)
    sentence = " ".join(text[start : line_end if line_end > 0 else len(text)].split())
    match = COUNT_SENTENCE.search(sentence)
    if match is None:
        print("  FAIL: no word-number count found in the sentence")
        return 1
    tens, units = WORDS.get(match.group(1).lower()), WORDS.get(match.group(2).lower())
    claimed = None
    if tens is not None and units is not None:
        # "sixty-four" is tens=60, units=4. An earlier version multiplied tens by 10 as well, giving
        # 604 for a claim of sixty-four -- a number no reader would believe, and the check still
        # reported it without noticing, because a FAIL is a FAIL either way. Multiplication is right
        # only when the tens word is a single digit ("twenty-four" = 24, not 204).
        claimed = tens + units if tens >= 20 else tens * 10 + units
    elif units is not None:
        claimed = units

    print(f"  {RULES_FILE.name}: {bullets} rule bullets; the sentence claims {claimed}")
    if claimed != bullets:
        print(f"  FAIL: the count says {claimed} and the section holds {bullets}")
        print("  A stale count reads exactly like a correct one, so it has to be checked mechanically.")
        return 1
    print("  PASS: the stated count matches the bullets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
