"""What the text-encoding gate must catch, and what it must not "repair".

The corruption this gate exists for is not lossy: a Windows PowerShell Get-Content followed by
WriteAllLines decodes every byte at or above 0x80 through the console codepage and encodes the result
as UTF-8, which adds one generation of encoding and is exactly reversible. Three things about that
are easy to get wrong, and each is a test rather than a comment.

The repair has to stop when the markers are gone. Applying the inverse until the text is byte-stable
destroys text that is already correct, because the inverse is only defined on mojibake. A first
attempt bounded at seven passes looked successful because the marker count reached zero on the
seventh, and an eighth pass then turned a section sign into a decode error.

The repair has to tolerate mixed content. docs/maintainer/qwen3.8-27b-artifact.md holds both
mis-decoded prose and a legitimate section sign, and U+00A7 maps back to the bare byte 0xa7, which
is not valid UTF-8 on its own. A whole-stream decode fails on it; preferring the longest valid UTF-8
sequence at each byte does not, because the mojibake recombines into valid UTF-8 and a lone section
sign does not.

And the gate has to fail on a file that is genuinely corrupt. A checker that cannot fail is worse
than no checker, which is the same reason the hook names its own test file explicitly: a gate whose
failure path has never run is indistinguishable from one that always passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "release"))

from check_text_encoding import (  # noqa: E402
    MOJIBAKE_SEQUENCES,
    build_reverse_table,
    count_markers,
    describe,
    is_corpus_data,
    repair,
    to_text,
)

EM_DASH = "—"
RIGHT_QUOTE = "’"

TABLE = build_reverse_table()


def _corrupt(text: str, generations: int) -> str:
    """Encode text the way a Windows PowerShell round trip does, ``generations`` times.

    One generation is exactly the inverse of ``to_bytes``: the file's UTF-8 bytes are read as if
    they were console-codepage text, and the resulting code points are what a later UTF-8 write
    stores. It has to be the inverse rather than a ``bytes.decode("cp1252")`` because Python's
    codec raises on the five slots where .NET returns the C1 control, and ``errors="replace"`` would
    make the corruption lossy -- which is the one property the repair relies on.
    """
    forward = {value[0]: char for char, value in TABLE.items()}
    out = text
    for _ in range(generations):
        raw = bytearray()
        for char in out:
            raw += char.encode("utf-8")
        out = "".join(forward[byte] for byte in raw)
    return out


def test_correct_prose_carries_no_marker() -> None:
    assert count_markers(f"a {EM_DASH} b and the {RIGHT_QUOTE} character\n") == 0


def test_one_generation_of_corruption_is_detected() -> None:
    assert count_markers(_corrupt(f"a {EM_DASH} b\n", 1)) > 0


def test_three_generations_of_corruption_are_detected() -> None:
    # This is docs/active-work.md as it stood: seven reversals were needed, and the first repair
    # attempt stalled at 181 lines because a string-level inverse cannot encode a line containing a
    # character outside cp1252.
    assert count_markers(_corrupt(f"a {EM_DASH} b\n", 3)) > 0


def test_ascii_only_text_is_not_mistaken_for_corruption() -> None:
    # The control for the two cases above: a round trip over pure ASCII is a genuine no-op, so a
    # detector that flagged it would be flagging the act of editing rather than the damage.
    assert count_markers(_corrupt("a dash\n", 3)) == 0


def test_repair_restores_the_original_exactly() -> None:
    original = f"a {EM_DASH} b, the {RIGHT_QUOTE} quote, and a section § too\n"
    for generations in (1, 2, 3):
        fixed, passes = repair(_corrupt(original, generations), TABLE)
        assert fixed == original, f"{generations} generation(s)"
        assert passes == generations


def test_repair_preserves_a_legitimate_section_sign() -> None:
    # The case that stops a whole-stream decode: U+00A7 reverses to the bare byte 0xa7.
    original = f"values § 3.5 {EM_DASH} done\n"
    assert repair(_corrupt(original, 1), TABLE)[0] == original


def test_repair_does_not_damage_text_that_is_already_correct() -> None:
    original = f"a {EM_DASH} b\n"
    fixed, passes = repair(original, TABLE)
    assert fixed == original
    assert passes == 0


def test_repair_stops_at_zero_markers_rather_than_running_to_a_stable_text() -> None:
    # A section sign is what a further pass would destroy, so its survival is the proof that the
    # repair stopped on the marker count instead of on byte-stability.
    original = f"§ {EM_DASH} {RIGHT_QUOTE}\n"
    fixed, _ = repair(_corrupt(original, 2), TABLE)
    assert fixed == original


def test_a_byte_order_mark_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "launcher.cmake"
    path.write_bytes(b"\xef\xbb\xbf" + b"target_sources(ninfer PRIVATE a.cpp)\n")
    assert "byte-order mark" in describe(path)


def test_a_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"caf\xe9 au lait\n")
    assert "not valid UTF-8" in describe(path)


def test_a_corrupt_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_bytes(_corrupt(f"a {EM_DASH} b\n", 2).encode("utf-8"))
    assert "mojibake" in describe(path)


def test_a_clean_file_reports_nothing(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_bytes(f"a {EM_DASH} b\n".encode())
    assert describe(path) == ""


def test_corpus_data_is_exempt_from_the_c1_rule_but_not_the_sequences() -> None:
    # bench/fixtures/ttft/text holds generated prompts in which a "\x97" written in Python 3 as an
    # em dash became U+0097. That is a generator bug, and the mojibake inverse would map it back to
    # a bare 0x97, so the C1 rule must skip it -- while a real round trip there is still caught.
    fixture = Path("bench/fixtures/ttft/text/rotation_55k_0.json")
    assert is_corpus_data(fixture)
    assert not is_corpus_data(Path("docs/active-work.md"))

    text = fixture.read_bytes().decode("utf-8")
    controls = [char for char in text if 0x80 <= ord(char) <= 0x9F]
    assert controls, "the fixture stopped carrying C1 controls, so this exemption needs re-deciding"
    assert count_markers(text) > 0, "the C1 controls must be counted when the rule is enabled"
    assert describe(fixture) == "", "a corpus fixture with C1 controls is exempt from the rule"

    # The exemption is narrow: the multi-character sequences are still counted on that same path, so
    # a genuine round trip inside a corpus fixture is not hidden by it.
    assert count_markers(_corrupt(f"a {EM_DASH} b\n", 2), not is_corpus_data(fixture)) > 0


def test_the_sequences_are_the_ones_the_documents_actually_carried() -> None:
    # Spelled out from the two repaired files rather than from the encoder's own table, so a change
    # to the table alone cannot make this test agree with itself.
    assert "\u00e2\u20ac" in MOJIBAKE_SEQUENCES  # docs/active-work.md
    assert "\u00c3\u0192" in MOJIBAKE_SEQUENCES  # docs/active-work.md, three generations


def test_to_text_prefers_valid_utf8_over_a_lone_byte() -> None:
    assert to_text("é §".encode("utf-8")) == "é §"