#!/usr/bin/env python3
"""Per-domain KL against a reference distribution, over top-k scoring records.

Perplexity reduces the [vocab, columns] logits to one number per position, so it cannot see a
distribution that moved. `docs/research/per-domain-kl-instrument.md` specifies the replacement: the
`topk_logprobs` Op reduces a tile to k per column, a scoring mode writes those k indices and
log-probabilities per position to disk, and this module compares a candidate's record against a
reference's, per domain. The two design decisions worth re-reading before changing anything here
are in that file; the short version is that the record is top-k rather than full-vocabulary (480
bytes per position against 483 GiB over this corpus) and that a reference token outside the
candidate's own top-k has to be floored rather than dropped, because dropping it silently reports
a divergence that is really an artefact of the truncation.

**The reference is not BF16, and cannot be.** `docs/research/per-domain-kl-instrument.md` was
designed around per-domain KL against a BF16 reference. Qwen3.8-27B is 27.781 B parameters, so its
BF16 weights are 51.75 GiB, measured from the safetensors headers rather than estimated; the RTX
5090 has 31.85 GiB. A BF16 reference cannot be resident, and CPU-offloaded weight streaming is not
something this engine has. So the reference is the highest-fidelity image of the *same weights* that
does fit -- the groupwise-int recipe over the same BF16 source checkpoint -- and every number this
module prints is KL between two quantized images. That is a real limitation and it is stated in the
output rather than left for the reader to discover: the headline is the candidate's divergence from
a near neighbour, not from unquantized arithmetic.

What the comparison *does* separate cleanly, because the reference is built from the same source
checkpoint, is quantization error from finetune difference. Two artifacts built from different
finetunes differ in both; building the reference from the candidate's own BF16 source removes the
second, which is why `tools/release` records the reference's own recipe and sources beside the
number.

The floor is the one genuinely arbitrary constant here, so it is a parameter, it is reported, and
the count of positions that reached it is printed next to every mean: a mean that is really a floor
in disguise is visible rather than plausible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"NINFKL01"
# The header is fixed-width and explicitly little-endian, because the two sides of a comparison are
# written by different toolchains -- this one by MSVC, the writer by CPython in the tests -- and a
# byte-order assumption would not fail loudly, it would just misalign every field after it.
# Layout, matching apps/perplexity/topk_record.h:
#   magic 8B | k u32 | stream_count u32 | positions u64 | corpus_id 32B | context u32 | stride u32
#   | kv_dtype 32B | prefill_signature 64B
# kv_dtype is 32 bytes rather than 8 because the engine's own names are longer: "fp8-e4m3-r256" is 14.
_HEADER = struct.Struct("<8sIIQ32sII32s64s")
_STREAM_HEAD = struct.Struct("<H")
_U32 = struct.Struct("<I")
DEFAULT_K = 60
# arXiv 2606.19558 Eq. 5's construction: a log-probability floor keeps the divergence finite when a
# reference-important token falls outside the candidate's own top-k. 1e-12 in probability is
# log = -27.6, which is below any real top-60 logit gap on this vocabulary and far above a
# denormal, so a position that reaches it is genuinely reporting "the candidate abandoned this".
DEFAULT_LOG_FLOOR = -27.631021115928547


def _pad(text: str, width: int) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) >= width:
        raise ValueError(f"{text!r} does not fit in {width} bytes")
    return raw + b"\0" * (width - len(raw))


def _unpad(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


@dataclass
class Stream:
    """One corpus stream's positions, in scoring order."""

    id: str
    domain: str
    token_digest: str
    indices: list = field(default_factory=list)  # list[list[int]], one per position
    logprobs: list = field(default_factory=list)  # list[list[float]]


@dataclass
class Record:
    """A top-k scoring record: the whole distributional substrate for one artifact."""

    k: int
    corpus_id: str
    context: int
    stride: int
    kv_dtype: str
    prefill_signature: str
    streams: list[Stream]

    @property
    def positions(self) -> int:
        return sum(len(s.indices) for s in self.streams)


def token_digest(ids) -> str:
    """Digest of a stream's token ids.

    This is the guard that makes the comparison mean anything. The two sides are scored by
    different artifacts, each tokenizing with its own embedded tokenizer, and the KL positions are
    matched by ordinal. If the tokenizations differed, every position after the first divergence
    would be a comparison of two unrelated contexts and the mean would be meaningless while still
    looking like a number. Asserting the digest is what stops that from reading as a result.
    """
    hasher = hashlib.sha256()
    for token in ids:
        hasher.update(struct.pack("<i", int(token)))
    return hasher.hexdigest()


def write_record(path: Path, record: Record) -> None:
    with open(path, "wb") as handle:
        handle.write(_HEADER.pack(
            MAGIC, record.k, len(record.streams), record.positions,
            _pad(record.corpus_id, 32), record.context, record.stride,
            _pad(record.kv_dtype, 8), _pad(record.prefill_signature, 64)))
        for stream in record.streams:
            handle.write(_STREAM_HEAD.pack(len(stream.id.encode("utf-8"))))
            handle.write(stream.id.encode("utf-8"))
            handle.write(_STREAM_HEAD.pack(len(stream.domain.encode("utf-8"))))
            handle.write(stream.domain.encode("utf-8"))
            handle.write(_STREAM_HEAD.pack(32))
            handle.write(bytes.fromhex(stream.token_digest))
            handle.write(_U32.pack(len(stream.indices)))
            for position in range(len(stream.indices)):
                handle.write(struct.pack(f"<{record.k}i", *stream.indices[position]))
                handle.write(struct.pack(f"<{record.k}f", *stream.logprobs[position]))


def read_record(path: Path) -> Record:
    with open(path, "rb") as handle:
        raw = handle.read(_HEADER.size)
        if len(raw) < _HEADER.size:
            raise ValueError(f"{path}: truncated header")
        (magic, k, stream_count, positions, corpus, context, stride, kv, signature) = \
            _HEADER.unpack(raw)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a top-k scoring record (magic {magic!r})")
        streams: list[Stream] = []
        for _ in range(stream_count):
            sid = _read_string(handle)
            domain = _read_string(handle)
            digest_len = _STREAM_HEAD.unpack(handle.read(_STREAM_HEAD.size))[0]
            if digest_len != 32:
                raise ValueError(f"{path}: stream {sid}: token digest is {digest_len} bytes, "
                                 f"expected 32")
            digest = handle.read(digest_len).hex()
            count = _U32.unpack(handle.read(_U32.size))[0]
            indices: list[list[int]] = []
            logprobs: list[list[float]] = []
            for _ in range(count):
                indices.append(list(struct.unpack(f"<{k}i", handle.read(4 * k))))
                logprobs.append(list(struct.unpack(f"<{k}f", handle.read(4 * k))))
            streams.append(Stream(sid, domain, digest, indices, logprobs))
        actual = sum(len(s.indices) for s in streams)
        if actual != positions:
            raise ValueError(f"{path}: header says {positions} positions, streams carry {actual}")
    return Record(k, _unpad(corpus), context, stride, _unpad(kv), _unpad(signature), streams)


def _read_string(handle) -> str:
    length = _STREAM_HEAD.unpack(handle.read(_STREAM_HEAD.size))[0]
    return handle.read(length).decode("utf-8")


def check_comparable(reference: Record, candidate: Record) -> list[str]:
    """Everything that must agree before a per-position comparison means anything.

    Returns the list of mismatches, empty when the two are comparable. This is deliberately a list
    of strings rather than an exception on the first problem: a reader who ran the two passes with
    different flags should be told all of what differs, not the first thing.
    """
    problems: list[str] = []
    if reference.k != candidate.k:
        problems.append(f"k differs: reference {reference.k}, candidate {candidate.k}")
    if reference.corpus_id != candidate.corpus_id:
        problems.append(f"corpus differs: {reference.corpus_id!r} against {candidate.corpus_id!r}")
    if (reference.context, reference.stride) != (candidate.context, candidate.stride):
        problems.append(f"window plan differs: {reference.context}/{reference.stride} against "
                        f"{candidate.context}/{candidate.stride}")
    if reference.kv_dtype != candidate.kv_dtype:
        problems.append(f"KV dtype differs: {reference.kv_dtype!r} against {candidate.kv_dtype!r}")
    if len(reference.streams) != len(candidate.streams):
        problems.append(f"stream count differs: {len(reference.streams)} against "
                        f"{len(candidate.streams)}")
        return problems
    for ref, cand in zip(reference.streams, candidate.streams):
        if ref.id != cand.id:
            problems.append(f"stream id differs: {ref.id!r} against {cand.id!r}")
        if ref.domain != cand.domain:
            problems.append(f"stream {ref.id}: domain differs: {ref.domain!r} against "
                            f"{cand.domain!r}")
        if ref.token_digest != cand.token_digest:
            problems.append(f"stream {ref.id}: token digest differs, so the two sides scored "
                            f"different token sequences and no position is comparable")
        if len(ref.indices) != len(cand.indices):
            problems.append(f"stream {ref.id}: position count differs: {len(ref.indices)} against "
                            f"{len(cand.indices)}")
    return problems


@dataclass
class DomainResult:
    name: str
    positions: int = 0
    total: float = 0.0
    floored: int = 0
    reference_mass: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.positions if self.positions else float("nan")


def position_divergence(ref_indices, ref_logprobs, cand_indices, cand_logprobs,
                        log_floor: float) -> tuple[float, int, float]:
    """KL(p_reference || p_candidate) at one position, over the reference's top-k support.

    Returns (divergence, entries that reached the floor, retained reference mass).

    The candidate's own top-k is looked up by token id, not by rank, so a candidate that orders the
    same tokens differently is not charged for it. A reference token the candidate does not carry has
    its log-probability floored: it was abandoned, and the divergence is real however small the
    candidate's residual mass for it is, so substituting the candidate's renormalised top-k mass
    would understate exactly the case the instrument exists to catch.

    The result is normalised by the reference's retained mass rather than by 1. A top-60 record does
    not hold all the probability, and the mass it does hold varies by position, so dividing by 1
    would fold "the reference spread its mass widely" into the divergence and make the mean
    incomparable across domains.
    """
    lookup = {int(t): float(lp) for t, lp in zip(cand_indices, cand_logprobs)}
    total = 0.0
    mass = 0.0
    floored = 0
    for token, ref_lp in zip(ref_indices, ref_logprobs):
        ref_lp = float(ref_lp)
        weight = math.exp(ref_lp)
        cand_lp = lookup.get(int(token))
        if cand_lp is None:
            cand_lp = log_floor
            floored += 1
        total += weight * (ref_lp - cand_lp)
        mass += weight
    if mass <= 0.0:
        return 0.0, floored, 0.0
    return total / mass, floored, mass


def compare(reference: Record, candidate: Record,
            log_floor: float = DEFAULT_LOG_FLOOR) -> dict[str, DomainResult]:
    """Per-domain mean divergence, plus the overall token-weighted mean."""
    domains: dict[str, DomainResult] = {}
    for ref, cand in zip(reference.streams, candidate.streams):
        for position in range(min(len(ref.indices), len(cand.indices))):
            value, floored, mass = position_divergence(
                ref.indices[position], ref.logprobs[position],
                cand.indices[position], cand.logprobs[position], log_floor)
            result = domains.setdefault(ref.domain, DomainResult(ref.domain))
            result.positions += 1
            result.total += value
            result.floored += floored
            result.reference_mass += mass
    return domains


def render(domains: dict[str, DomainResult], log_floor: float, meta: dict) -> str:
    lines = [
        "per-domain KL(p_reference || p_candidate), top-k support, expectation under the reference",
        f"  k                 {meta['k']}",
        f"  corpus            {meta['corpus_id']} ({meta['context']}/{meta['stride']}, "
        f"kv {meta['kv_dtype']})",
        f"  reference         {meta['reference']}",
        f"                    prefill signature {meta['reference_signature'] or '(none)'}",
        f"  candidate         {meta['candidate']}",
        f"                    prefill signature {meta['candidate_signature'] or '(none)'}",
        f"  log-prob floor    {log_floor:.3f}  (probability {math.exp(log_floor):.3e})",
        f"  positions         {meta['positions']:,}",
        "",
        f"  {'domain':<22} {'mean KL (nats)':>15} {'positions':>12} {'floored':>12} "
        f"{'ref mass/pos':>13}",
    ]
    for name in sorted(domains):
        result = domains[name]
        share = result.reference_mass / result.positions if result.positions else float("nan")
        lines.append(f"  {name:<22} {result.mean:>15.6f} {result.positions:>12,} "
                     f"{result.floored:>12,} {share:>13.6f}")
    overall_positions = sum(r.positions for r in domains.values())
    if overall_positions:
        overall = sum(r.total for r in domains.values()) / overall_positions
        lines.append(f"  {'OVERALL':<22} {overall:>15.6f} {overall_positions:>12,} "
                     f"{sum(r.floored for r in domains.values()):>12,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="the reference top-k scoring record")
    parser.add_argument("candidate", type=Path, help="the candidate top-k scoring record")
    parser.add_argument("--log-floor", type=float, default=DEFAULT_LOG_FLOOR,
                        help="log-probability substituted for a reference token the candidate's "
                             f"top-k does not carry (default {DEFAULT_LOG_FLOOR:.3f})")
    parser.add_argument("--json", type=Path, help="also write the table as JSON")
    args = parser.parse_args(argv)

    reference = read_record(args.reference)
    candidate = read_record(args.candidate)
    problems = check_comparable(reference, candidate)
    if problems:
        print("the two records are not comparable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    domains = compare(reference, candidate, args.log_floor)
    meta = {
        "k": reference.k,
        "corpus_id": reference.corpus_id,
        "context": reference.context,
        "stride": reference.stride,
        "kv_dtype": reference.kv_dtype,
        "positions": reference.positions,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_signature": reference.prefill_signature,
        "candidate_signature": candidate.prefill_signature,
    }
    print(render(domains, args.log_floor, meta))
    if args.json:
        payload = {
            "metric": "top-k KL(p_reference || p_candidate), expectation under the reference",
            "log_floor": args.log_floor,
            **meta,
            "domains": {
                name: {"mean_kl_nats": r.mean, "positions": r.positions,
                       "floored_entries": r.floored,
                       "reference_mass_per_position":
                           r.reference_mass / r.positions if r.positions else None}
                for name, r in sorted(domains.items())
            },
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
