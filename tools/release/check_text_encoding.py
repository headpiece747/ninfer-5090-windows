#!/usr/bin/env python3
"""Fail on tracked text files whose bytes were re-encoded instead of preserved.

A Windows PowerShell round-trip through Get-Content followed by WriteAllLines is not a no-op on a
UTF-8 file. Get-Content decodes with the console codepage, so every byte at or above 0x80 becomes a
single-character code point; WriteAllLines then encodes those code points back as UTF-8, which is one
extra generation of encoding. Reading through the shell a second time adds another, and the file ends
up carrying a three-code-point sequence where it was written with one em dash. Nothing fails: the file still parses, still renders,
and every gate that reads it still passes, because the damage is confined to the prose.

Two files in this repository carried it for a fortnight before this gate existed:
docs/maintainer/qwen3.8-27b-artifact.md from 13f5f219 (2026-09-24, one generation, seven lines) and
docs/active-work.md from 6d9ae366 (three generations, 181 lines). Both were introduced by scripted
edits of the kind the project already records as a trap for BOMs; this is the same trap one step
further along, and a check is the only thing that catches it.

The markers are multi-character sequences rather than individual code points on purpose. A single
U+00C2 or U+00C3 is a real character in several languages, so flagging it would be a false positive in
a document that legitimately contains one. The sequences below cannot occur in correctly encoded
prose: U+00E2 U+20AC is what a UTF-8 em dash becomes after one mis-decode, U+00C3 U+0192 is the
second-generation .NET signature, and U+00EF U+00BF U+00BD is a byte-order mark that was itself
decoded as text. The C1 controls U+0080..U+009F are included because .NET's 1252 codepage falls back
to them for its five undefined slots, so their presence means a byte was read that no correct decode
of this text would produce.

They are written as escapes rather than as the characters themselves, which is not a style choice: a
file that spelled out the sequences it forbids would forbid itself.

Repair is the exact inverse and exists because the corruption is a bijection, not a loss. Each
code point that a .NET 1252 decode could have produced maps back to its single byte, everything else
is already correct UTF-8 and passes through, and applying the map until it converges reverses any
number of generations. It is offered here rather than applied silently so that the fix is reviewable
and re-runnable.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from xml.etree import ElementTree

# Multi-character sequences that cannot appear in correctly encoded prose. See the module docstring.
MOJIBAKE_SEQUENCES: tuple[str, ...] = (
    "\u00e2\u20ac",  # U+00E2 U+20AC: one mis-decode of a smart quote, en dash or em dash
    "\u00c3\u0192",  # U+00C3 U+0192: the second-generation .NET signature
    "\u00c3\u02c6",  # "Ã" + MODIFIER LETTER CIRCUMFLEX ACCENT
    "\u00c3\u02dc",  # "Ã" + SMALL TILDE
    "\u00ef\u00bf\u00bd",  # U+00EF U+00BF U+00BD: a byte-order mark decoded as text
)

# .NET's 1252 codepage returns the C1 control of the same value for its undefined slots, so a
# correct decode of UTF-8 text never produces one of these.
C1_RANGE: tuple[int, int] = (0x80, 0x9F)

# Only these are read. A .ninfer, .zip or .png has bytes at or above 0x80 by construction, and
# scanning it would report every file in the repository as corrupt.
#
# `.bat` and `.jinja` were both missing when this gate first landed, and both matter. `.bat` is the
# half of the CRLF rule below that no file could reach, so all 13 tracked launchers went unread while
# the rule sat there looking enforced. `.jinja` is the two chat templates: those are PROMPTS, and
# mojibake in a prompt is a functional defect rather than a cosmetic one -- the exact fault this gate
# exists to catch, in the one place it cannot afford to miss.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".bat",
        ".c",
        ".dot",
        ".cc",
        ".cfg",
        ".cmd",
        ".cpp",
        ".cmake",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".ini",
        ".jinja",
        ".js",
        ".json",
        ".jsonl",
        ".manifest",
        ".md",
        ".mjs",
        ".ps1",
        ".py",
        ".rst",
        ".sh",
        ".svg",
        ".toml",
        ".ts",
        ".tsv",
        ".txt",
        ".yml",
        ".yaml",
    }
)

# Files this gate reads that carry no suffix, or whose suffix is not descriptive. CMakeLists.txt,
# README.md and RELEASE_NOTES.md were listed here when this landed and could never match, because
# `.txt` and `.md` are in TEXT_SUFFIXES above; they are listed by their suffix instead, so editing this
# set has an effect on every entry in it.
NAME_ONLY = frozenset(
    {
        ".clang-format",
        ".clang-tidy",
        ".clangd",
        ".dockerignore",
        ".gitattributes",
        ".gitignore",
        "Dockerfile",
        "LICENSE",
        "NOTICE",
        "pre-commit",
    }
)

# .gitattributes pins every tracked text file to LF except these, which are CRLF by definition and
# which two gates read byte for byte. The rule here restates the attribute rather than deriving it,
# because the gate reads the worktree and the worktree is where the split actually lived: 70 files
# CRLF against 16 LF, decided by whichever tool last wrote each one, invisible in a diff and fatal to
# any byte-level comparison. It is the same tooling that re-encoded 181 lines of documentation.
CRLF_SUFFIXES = frozenset({".bat", ".cmd"})

# No path carries an exemption. An earlier revision skipped the C1 rule under bench/fixtures/ and
# examples/, because the TTFT prompts held a "\x97" written in Python 3 as an em dash and became
# U+0097 -- a generator bug, not a mis-decode, and one this gate must not "repair", since the
# inverse maps U+0097 back to the bare byte 0x97. That was fixed by regenerating the corpus from its
# seed, so the exemption it needed is gone rather than left behind: an exclusion whose cause is gone
# is a blind spot waiting for the next defect to hide behind.

MAX_PASSES = 12


def build_reverse_table() -> dict[str, bytes]:
    """Map each code point a .NET 1252 decode produces back to the byte it came from.

    Python's cp1252 codec raises on the five undefined slots where .NET returns the C1 control of
    the same value, so those are filled in here. Without them the map is not total and a line
    containing one cannot be repaired at all -- which is what left 181 lines stuck on the first
    attempt at this repair.
    """
    table: dict[str, bytes] = {}
    for value in range(256):
        try:
            char = bytes([value]).decode("cp1252")
        except UnicodeDecodeError:
            char = chr(value)
        table[char] = bytes([value])
    return table


def to_bytes(text: str, table: dict[str, bytes]) -> bytes:
    """Re-encode text to the byte stream a mis-decode would have consumed.

    Code points the table knows become their single original byte. Everything else -- Chinese prose,
    arrows, any correctly encoded text -- is already UTF-8 and is passed through unchanged, which is
    what lets a document hold mojibake and correct text in the same file.
    """
    out = bytearray()
    for char in text:
        original = table.get(char)
        out += original if original is not None else char.encode("utf-8")
    return bytes(out)


def to_text(raw: bytes) -> str:
    """Decode a byte stream, preferring valid UTF-8 and falling back one byte at a time.

    A whole-stream decode is wrong for this input in both directions. It fails on the legitimate
    section sign in docs/maintainer/qwen3.8-27b-artifact.md, which U+00A7 maps back to the bare byte
    0xa7 and which is not valid UTF-8 on its own; and it fails on the mojibake runs, which only
    become valid UTF-8 once their bytes are put back together. Preferring the longest valid sequence
    at each position resolves both, because the mojibake recombines into valid UTF-8 while a lone
    section sign does not.
    """
    out: list[str] = []
    index = 0
    size = len(raw)
    while index < size:
        byte = raw[index]
        if byte < 0x80:
            out.append(chr(byte))
            index += 1
            continue
        decoded = False
        for width in (2, 3, 4):
            chunk = raw[index : index + width]
            if len(chunk) < width:
                continue
            try:
                out.append(chunk.decode("utf-8"))
            except UnicodeDecodeError:
                continue
            index += width
            decoded = True
            break
        if not decoded:
            try:
                out.append(bytes([byte]).decode("cp1252"))
            except UnicodeDecodeError:
                out.append(chr(byte))
            index += 1
    return "".join(out)


def count_markers(text: str) -> int:
    """Count every marker occurrence, so a repair can only be accepted as progress."""
    total = sum(text.count(sequence) for sequence in MOJIBAKE_SEQUENCES)
    total += sum(1 for char in text if C1_RANGE[0] <= ord(char) <= C1_RANGE[1])
    return total


def repair(text: str, table: dict[str, bytes]) -> tuple[str, int]:
    """Reverse any number of generations of the corruption.

    Stops as soon as no marker remains rather than continuing until the text is byte-stable: the
    inverse is only defined on mojibake, so applying one more pass to text that is already correct
    destroys it. A first attempt bounded at seven passes reported success because the markers hit
    zero at pass seven, and pass eight then turned a section sign into a decode error.
    """
    current = text
    passes = 0
    while passes < MAX_PASSES:
        if count_markers(current) == 0:
            return current, passes
        following = to_text(to_bytes(current, table))
        if following == current:
            return current, passes
        current = following
        passes += 1
    if count_markers(current) == 0:
        return current, passes
    raise ValueError(f"did not converge within {MAX_PASSES} passes")


def tracked_text_files() -> list[Path]:
    """List tracked files this check reads, from the index rather than the working tree.

    Reading the index is deliberate: the check has to describe what is about to be committed, and
    an untracked scratch file is not part of that.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True, text=False
    )
    files: list[Path] = []
    for entry in result.stdout.decode("utf-8").split("\0"):
        if not entry:
            continue
        path = Path(entry)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in NAME_ONLY:
            files.append(path)
    return files


def describe(path: Path) -> str:
    """Return the findings for one file, or an empty string when it is clean."""
    raw = path.read_bytes()
    findings: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        findings.append("byte-order mark: Set-Content -Encoding utf8 writes one under PowerShell 5.1")
        raw = raw[3:]
    wants_crlf = path.suffix.lower() in CRLF_SUFFIXES
    if b"\r\n" in raw:
        if not wants_crlf:
            findings.append(
                "CRLF line endings: .gitattributes pins every text file except *.bat and *.cmd to LF"
            )
    elif wants_crlf and b"\n" in raw:
        findings.append("LF line endings: *.bat and *.cmd are CRLF by definition")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        # Report what was already found alongside the decode failure. Returning here instead would
        # drop a BOM verdict and a CRLF verdict for a file that has both, and --repair strips the BOM
        # on a different path than the one that reports it, so the two would disagree about the same
        # file.
        findings.append(f"not valid UTF-8 at byte {error.start}")
        return f"{path}: {'; '.join(findings)}"
    if count_markers(text):
        found = sorted({sequence for sequence in MOJIBAKE_SEQUENCES if sequence in text})
        controls = sum(1 for char in text if C1_RANGE[0] <= ord(char) <= C1_RANGE[1])
        parts = [f"mojibake from a mis-decoded round trip ({len(found)} sequences"]
        parts.append(f", {controls} C1 controls)" if controls else ")")
        findings.append("".join(parts))
    if path.suffix.lower() == ".manifest":
        # These carry prose in comments, which is why they are read at all, and the linker parses
        # them rather than ignoring them: a parse failure is a build failure for whoever builds next,
        # and an XML comment may not contain a double hyphen. One was committed with exactly that, and
        # the only reason it surfaced is that a probe tried to link with it -- the product build had
        # already succeeded from the previous revision and would not have noticed until its next run.
        try:
            ElementTree.fromstring(text)
        except ElementTree.ParseError as error:
            findings.append(
                f"not well-formed XML ({error}); the linker parses this file, and an XML comment may "
                "not contain a double hyphen"
            )
    return f"{path}: {'; '.join(findings)}" if findings else ""


def check(files: list[Path]) -> int:
    """Report every offending file, and fail if there is one."""
    print(f"  text files checked: {len(files)}")
    broken = [message for message in (describe(path) for path in files) if message]
    for message in broken:
        print(f"    {message}")
    if broken:
        print("  FAIL: a tracked text file is not byte-preserved UTF-8.")
        print("        Re-encode it with a tool that reads UTF-8 (the Read/Edit tools, or Python's")
        print("        own open()), never with Get-Content followed by WriteAllLines. To repair an")
        print("        existing file: python tools/release/check_text_encoding.py --repair")
        return 1
    print("  PASS: every tracked text file is byte-preserved UTF-8.")
    return 0


def repair_files(files: list[Path]) -> int:
    """Repair the offending files in place, and prove the repair rather than asserting it."""
    table = build_reverse_table()
    changed = 0
    for path in files:
        raw = path.read_bytes()
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        if had_bom:
            raw = raw[3:]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            print(f"  SKIP  {path}: not valid UTF-8 at byte {error.start}")
            continue
        # One FIXED line per file, whatever combination of faults it had. Counting the BOM and the
        # markers separately reported the same file twice and inflated `files repaired`, and writing
        # twice meant the second write could disagree with the first.
        notes: list[str] = []
        payload = raw
        if had_bom:
            notes.append("byte-order mark removed")
        if count_markers(text):
            try:
                fixed, passes = repair(text, table)
            except ValueError as error:
                print(f"  SKIP  {path}: {error}")
                continue
            notes.append(f"{passes} generation(s) reversed, {count_markers(fixed)} markers left")
            payload = fixed.encode("utf-8")
        if not notes:
            continue
        path.write_bytes(payload)
        changed += 1
        print(f"  FIXED {path}: {', '.join(notes)}")
    print(f"  files repaired: {changed}")
    if changed:
        print("  Review the diff before committing: this reverses bytes, it does not know the prose.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="rewrite the offending files in place instead of reporting them",
    )
    arguments = parser.parse_args()
    files = tracked_text_files()
    if not files:
        print("  FAIL: no tracked text files were found, so nothing was checked.")
        return 1
    return repair_files(files) if arguments.repair else check(files)


if __name__ == "__main__":
    raise SystemExit(main())