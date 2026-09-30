#!/usr/bin/env python3
"""Tests for the per-domain KL instrument.

Each expectation below is derived from the definition of KL and written out by hand, not recorded
from a run. A recorded expectation would have recorded the bug: the three cases in
`test_position_divergence_*` are exactly the shapes where a plausible implementation and a correct
one differ, and they are the reason the Op's own commit message insists the key be order-EXACT.

Run: python tools/release/test_per_domain_kl.py
"""
from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from per_domain_kl import (  # noqa: E402
    DEFAULT_LOG_FLOOR, Record, Stream, check_comparable, compare, position_divergence,
    read_record, token_digest, write_record,
)


class PositionDivergence(unittest.TestCase):
    def test_identical_distributions_diverge_by_zero(self):
        # Same ids, same logprobs. Every term is weight * (lp - lp) = 0.
        value, floored, mass = position_divergence([7, 3], [-0.1, -2.0], [7, 3], [-0.1, -2.0],
                                                   DEFAULT_LOG_FLOOR)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(floored, 0)
        self.assertAlmostEqual(mass, math.exp(-0.1) + math.exp(-2.0), places=12)

    def test_rank_is_not_identity(self):
        # The candidate carries the same two tokens in the opposite rank order. Looking the
        # candidate up by rank instead of by token id would report a large divergence here, which
        # is the whole reason the lookup is keyed on the id.
        value, floored, _ = position_divergence([7, 3], [-0.1, -2.0], [3, 7], [-2.0, -0.1],
                                                DEFAULT_LOG_FLOOR)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(floored, 0)

    def test_one_shared_token_diverges_by_the_log_ratio(self):
        # Reference {7: 0.5, 3: 0.5}; candidate {7: 0.25, 3: 0.75}. BOTH tokens moved, so the
        # divergence is the full two-term KL and not the single token's log-ratio:
        #   0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln2 + 0.5*ln(2/3) = 0.5*ln(4/3).
        # An earlier revision of this expectation asserted ln(1.5), which is token 3's ratio alone
        # and ignores that token 7 halved. The code was right and the expectation was not.
        value, floored, _ = position_divergence(
            [7, 3], [math.log(0.5), math.log(0.5)],
            [3, 7], [math.log(0.75), math.log(0.25)], DEFAULT_LOG_FLOOR)
        self.assertAlmostEqual(value, 0.5 * math.log(4.0 / 3.0), places=6)
        self.assertEqual(floored, 0)

    def test_abandoned_token_is_floored_not_dropped(self):
        # The candidate does not carry token 3 at all. Dropping it would report the divergence of
        # token 7 alone -- here zero -- which is the failure the instrument exists to prevent: a
        # candidate that abandoned a reference-important token must be charged for it.
        value, floored, _ = position_divergence(
            [7, 3], [math.log(0.5), math.log(0.5)],
            [7], [math.log(1.0)], DEFAULT_LOG_FLOOR)
        self.assertEqual(floored, 1)
        expected = 0.5 * (math.log(0.5) - 0.0) + 0.5 * (math.log(0.5) - DEFAULT_LOG_FLOOR)
        self.assertAlmostEqual(value, expected / 1.0, places=6)
        self.assertGreater(value, 13.0, "a floor at -27.6 must dominate the term it replaces")

    def test_normalisation_is_by_retained_mass(self):
        # Two positions with the same per-token log-ratio but different retained mass must give
        # the same mean. If the divisor were 1, the second would read smaller purely because the
        # reference spread its mass over more tokens.
        top = [math.log(0.9), math.log(0.05)]
        tail = [math.log(0.02)] * 4
        ref_a, ref_b = [7, 3], [3, 7, 11, 19]
        cand_a = [math.log(0.45), math.log(0.025)]
        cand_b = [math.log(0.01)] * 4
        a, _, _ = position_divergence(ref_a, top, ref_a, cand_a, DEFAULT_LOG_FLOOR)
        b, _, _ = position_divergence(ref_b, tail, ref_b, cand_b, DEFAULT_LOG_FLOOR)
        self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(a, math.log(2.0), places=6)


class RecordRoundTrip(unittest.TestCase):
    def _record(self) -> Record:
        return Record(
            k=4, corpus_id="ninfer-ppl-1m-v1", context=4096, stride=2048, kv_dtype="fp8",
            prefill_signature="ed709b7d",
            streams=[Stream("code_a", "ninfer_code", token_digest([1, 2, 3]),
                            [[5, 6, 7, 8], [9, 10, 11, 12]],
                            [[-0.1, -1.2, -2.3, -3.4], [-0.5, -1.5, -2.5, -3.5]])])

    def test_round_trip_is_byte_exact_in_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.bin"
            original = self._record()
            write_record(path, original)
            back = read_record(path)
        self.assertEqual(back.k, original.k)
        self.assertEqual(back.corpus_id, original.corpus_id)
        self.assertEqual(back.kv_dtype, original.kv_dtype)
        self.assertEqual(back.prefill_signature, original.prefill_signature)
        self.assertEqual(back.positions, original.positions)
        self.assertEqual([s.id for s in back.streams], ["code_a"])
        self.assertEqual([s.domain for s in back.streams], ["ninfer_code"])
        self.assertEqual(back.streams[0].token_digest, original.streams[0].token_digest)
        self.assertEqual(back.streams[0].indices, original.streams[0].indices)
        # logprobs go through FP32 on disk, so compare at the Op's own resolution rather than
        # exactly: the record stores what the device produced, not an FP64 intermediate.
        for got, want in zip(back.streams[0].logprobs, original.streams[0].logprobs):
            for g, w in zip(got, want):
                self.assertEqual(g, struct.unpack("<f", struct.pack("<f", w))[0])

    def test_truncated_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.bin"
            write_record(path, self._record())
            data = path.read_bytes()
            path.write_bytes(data[:len(data) // 2])
            with self.assertRaises((ValueError, struct.error)):
                read_record(path)

    def test_foreign_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.bin"
            path.write_bytes(b"this is not a record" * 8)
            with self.assertRaises(ValueError):
                read_record(path)


class Comparability(unittest.TestCase):
    def _record(self, **kwargs: Any) -> Record:
        # Built field by field rather than through a dict[str, object] splat: the splat types every
        # value as `object`, so mypy rejects the whole construction even though each override is the
        # right type at the call site. The overrides below are the four comparability axes.
        streams = kwargs.get("streams")
        return Record(
            k=kwargs.get("k", 2),
            corpus_id=kwargs.get("corpus_id", "c"),
            context=kwargs.get("context", 4096),
            stride=kwargs.get("stride", 2048),
            kv_dtype=kwargs.get("kv_dtype", "fp8"),
            prefill_signature=kwargs.get("prefill_signature", "sig"),
            streams=streams if streams is not None
            else [Stream("s", "d", token_digest([1, 2]), [[1, 2]], [[-1.0, -2.0]])],
        )

    def test_identical_records_are_comparable(self):
        self.assertEqual(check_comparable(self._record(), self._record()), [])

    def test_token_digest_mismatch_is_reported(self):
        # This is the guard that stops a re-tokenized stream from reading as a result.
        other = self._record()
        other.streams[0].token_digest = token_digest([1, 2, 3])
        problems = check_comparable(self._record(), other)
        self.assertTrue(any("token digest differs" in p for p in problems), problems)

    def test_window_plan_mismatch_is_reported(self):
        problems = check_comparable(self._record(), self._record(stride=1024))
        self.assertTrue(any("window plan differs" in p for p in problems), problems)

    def test_k_mismatch_is_reported(self):
        problems = check_comparable(self._record(), self._record(k=8))
        self.assertTrue(any("k differs" in p for p in problems), problems)

    def test_kv_dtype_mismatch_is_reported(self):
        problems = check_comparable(self._record(), self._record(kv_dtype="bf16"))
        self.assertTrue(any("KV dtype differs" in p for p in problems), problems)

    def test_prefill_signature_difference_is_allowed(self):
        # The reference and the candidate are different images, so their arithmetic signatures
        # differ by construction. Requiring them to match would make the comparison impossible.
        self.assertEqual(check_comparable(self._record(), self._record(prefill_signature="other")),
                         [])


class DomainReduction(unittest.TestCase):
    def test_domains_reduce_independently(self):
        reference = Record(2, "c", 4096, 2048, "fp8", "sig", [
            Stream("a", "code", token_digest([1]), [[1, 2]], [[math.log(0.5), math.log(0.5)]]),
            Stream("b", "prose", token_digest([1]), [[1, 2]], [[math.log(0.5), math.log(0.5)]]),
        ])
        # Candidate halves token 1's probability in the code stream only.
        candidate = Record(2, "c", 4096, 2048, "fp8", "sig", [
            Stream("a", "code", token_digest([1]), [[1, 2]],
                   [[math.log(0.25), math.log(0.5)]]),
            Stream("b", "prose", token_digest([1]), [[1, 2]],
                   [[math.log(0.5), math.log(0.5)]]),
        ])
        domains = compare(reference, candidate)
        self.assertEqual(sorted(domains), ["code", "prose"])
        # code: token 1 moved by ln2, token 3 not at all, equal weights -> mean ln2 / 2.
        self.assertAlmostEqual(domains["code"].mean, math.log(2.0) / 2.0, places=6)
        self.assertAlmostEqual(domains["prose"].mean, 0.0, places=12)
        self.assertEqual(domains["prose"].floored, 0)

    def test_overall_is_token_weighted_not_domain_averaged(self):
        # Two domains with 1 and 3 positions must not average to the midpoint of the two means.
        # Each position holds two tokens at 0.5, and the candidate halves token 1's probability in
        # the one-position domain only, so that position diverges by 0.5*ln2 / 1.0 and the overall
        # is that spread over all four positions.
        def stream(name, domain, n, token_one):
            return Stream(name, domain, token_digest([1]), [[1, 2]] * n,
                          [[math.log(0.5), token_one]] * n)

        reference = Record(2, "c", 4096, 2048, "fp8", "sig",
                           [stream("a", "small", 1, math.log(0.5)),
                            stream("b", "large", 3, math.log(0.5))])
        candidate = Record(2, "c", 4096, 2048, "fp8", "sig",
                           [stream("a", "small", 1, math.log(0.25)),
                            stream("b", "large", 3, math.log(0.5))])
        domains = compare(reference, candidate)
        self.assertEqual(domains["small"].positions, 1)
        self.assertEqual(domains["large"].positions, 3)
        self.assertAlmostEqual(domains["small"].mean, 0.5 * math.log(2.0), places=6)
        self.assertAlmostEqual(domains["large"].mean, 0.0, places=12)
        overall = (sum(r.total for r in domains.values())
                   / sum(r.positions for r in domains.values()))
        self.assertAlmostEqual(overall, 0.5 * math.log(2.0) / 4.0, places=6)
        # The domain-unweighted mean would be half the small domain's value, which is the error
        # this test exists to catch.
        unweighted = (domains["small"].mean + domains["large"].mean) / 2.0
        self.assertNotAlmostEqual(overall, unweighted, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
