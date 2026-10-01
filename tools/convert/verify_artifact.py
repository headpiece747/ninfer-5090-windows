#!/usr/bin/env python3
"""Re-validate a built v3 artifact against the sources it claims to be built from.

Why this exists. docs/maintainer/artifact-conventions.md:113-115 records that the fork published
``verify_nvfp4qat`` and ``verify_nvfp4full``, which "revalidate the complete ordered directory, both
W8 endpoints against base rows, and the input divisors". Neither entry point is in this tree. The
substitute recorded in the same paragraph -- hashing every binding against the predecessor it
replaces -- only works for an artifact that has a predecessor, and ``qwen3_8_27b_nvfp4nvidia`` has
none, so for that lane no payload was ever checked against anything. The 159 attention and
linear-attention matrices it encodes locally from the BF16 base were unverified.

What this checks, and why each is a check that can fail:

**The complete ordered directory.** Every binding's parts must lie inside its target object, in
non-decreasing order, non-overlapping, and cover exactly the declared element count. A binding that
points at the wrong rows of a tensor produces a model that loads and generates, so nothing else in the
pipeline would notice.

**Every NVFP4 and grouped-integer payload.** The stored bytes are unswizzled, the E2M1 codes and
E4M3FN scales recovered, and the dequantized parent rebuilt, then compared against the source
checkpoint's own matrix. ``tools/artifact/codecs/nvfp4.py`` is the *storage* codec; it is not reused as
the oracle, because a codec that both writes and verifies its own output would agree with a wrong
encoder. The reference here is built from the format definitions instead: E2M1 magnitudes and
round-to-nearest-even from the published table, E4M3FN from its own encoding.

Grouped-integer sites (Q4/Q5/Q6/Q8) were added to this pass on 2026-10-01 and **were not covered by it
before**, which left every quantized vision-tower payload and every NVFP4-draft payload unverified
against its source. They are the format this tree assigns to the vision tower
(``official_recipes.py`` ``_optional``) and to the draft projections (``_nvfp4_draft``). Their bound is
derived from the encoder rather than tabulated -- see ``grouped_bound`` -- and is ``0.5 / qmax`` of the
tensor maximum, against NVFP4's ``1/6 + 1/16 = 0.2292``.

**The weight divisor and every input divisor.** Each must be a positive finite FP32 word. The divisor
words are compared against the source's stored ``input_scale`` where the site is an imported NVFP4 one,
which is where a factor-of-six error would live.

**The W8 endpoints are NOT value-checked, and an earlier revision of this docstring said they were.**
It claimed "Both W8 endpoints against base rows ... compared against the BF16 base's rows rather than
merely checked for a valid encoding". No such check existed in this file; the sentence described an
entry point that ``docs/maintainer/artifact-conventions.md`` records as never having been ported. Now
that grouped-integer sites are claimed, the two endpoints are *attempted* and fail for a reason that
has nothing to do with their encoding: the source factory short-reads at
``248320 x 5120 = 1,270,998,400`` elements ("short source read"), and ``proposal/head`` needs an
explicit logical source that no recipe supplies. Both are reported as **unchecked**, not as failures,
because a reference that cannot be read is a limit of the lookup rather than evidence about the
payload. So the endpoints -- 2.52 GiB, and the subject of ``docs/active-work.md`` item 3 -- remain
unproven by this tool.

Exit code 0 if every check passes, 1 otherwise. Run with ``--only`` to verify a single class, which
is how the time was divided during development.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words  # noqa: E402
from tools.artifact.codecs.row_split import dequantize_row_split  # noqa: E402
from tools.artifact.formats import (  # noqa: E402
    NVFP4_FORMATS,
    QUANT_FORMATS,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import TensorObject  # noqa: E402

# The E2M1 grid, from the format definition rather than from any code in this tree. Magnitudes are
# 0, 0.5, 1, 1.5, 2, 3, 4, 6; a code is sign(1) | exponent(2) | mantissa(1), low nibble first.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# A row block bounds the peak allocation of the value check. The largest parent in the nvidia lane is
# 34816 x 5120 = 178,257,920 elements; at float32 that is 0.66 GiB for the whole parent, and at the
# 8 bytes of float64 it would be 1.33 GiB per temporary. The first version of this function built the
# parent from nested Python lists, which measured 28 bytes per element -- 4.6 GiB for that one parent
# and 688 GiB if every parent were held at once, on a 47.8 GiB machine. It exhausted memory and took
# the machine down. The tables below are built once and the arithmetic is vectorized, so a block costs
# 4 bytes per element, and blocks are the unit of work rather than whole parents.
BLOCK_ELEMENTS = 8 << 20  # 8 Mi elements = 32 MiB per float32 block


def grouped_bound(spec: QuantFormat) -> float:
    """Worst-case max-relative error of grouped absmax, against the tensor's own maximum.

    Derived from the encoder rather than tabulated. ``groupwise.quantize_matrix`` chooses
    ``codes = clamp(round(x / scale), qmin, qmax)`` with ``scale = binary16(group_absmax / qmax)``,
    so a reconstructed value is off by at most half a step, ``0.5 * scale``. Since
    ``group_absmax <= tensor_absmax`` that is at most ``0.5 / qmax`` of the tensor maximum, and the
    binary16 scale carries its own ``2**-11`` relative rounding on top.

    Divided by the tensor maximum rather than the group maximum because that is what the comparison
    measures, which makes this bound conservative: a tensor whose largest group is much smaller than
    its maximum will measure well under it.
    """
    return 0.5 / spec.qmax * (1.0 + 2.0**-11)



def e2m1_value(code: int) -> float:
    """The exact value of one packed E2M1 nibble, sign-magnitude from the format definition."""
    magnitude = E2M1_MAGNITUDES[code & 0x7]
    return -magnitude if code & 0x8 else magnitude


def e4m3fn_value(word: int) -> float:
    """The exact value of one E4M3FN byte, decoded from its own definition.

    1 sign, 4 exponent, 3 mantissa, bias 7, no infinities (S.1111.111 is NaN) and one subnormal step.
    Written here rather than imported so the check does not share a decoder with the encoder.
    """
    if word & 0x80:
        sign = -1.0
    else:
        sign = 1.0
    exponent = (word >> 3) & 0x0F
    mantissa = word & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        raise ValueError("E4M3FN NaN encoding is not a valid NVFP4 scale")
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0 ** (1 - 7)
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _e2m1_table() -> torch.Tensor:
    return torch.tensor([e2m1_value(code) for code in range(16)], dtype=torch.float32)


def _e4m3fn_table() -> torch.Tensor:
    table = torch.zeros(256, dtype=torch.float32)
    for word in range(256):
        try:
            table[word] = e4m3fn_value(word)
        except ValueError:
            table[word] = float("nan")
    return table


E2M1_TABLE = _e2m1_table()
E4M3FN_TABLE = _e4m3fn_table()


@dataclass
class Report:
    """Accumulated verdicts. Counts are reported even on success so a run says what it covered."""

    passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    observed: dict[str, float] = field(default_factory=dict)

    def check(self, label: str, ok: bool, reason: str = "") -> bool:
        if ok:
            self.passed.append(label)
        else:
            self.failures.append(f"{label}: {reason}" if reason else label)
        return ok

    def note(self, label: str, reason: str = "") -> None:
        """Record a site this run could not check, which is not the same as a site that failed.

        A reference that cannot be resolved says nothing about the artifact: it is a limit of the
        lookup, and counting it as a failure makes a verifier that could not see a payload look
        like a verifier that disproved it. These are listed separately so a reader can tell which
        payloads remain unproven.
        """
        self.notes.append(f"{label}: {reason}" if reason else label)
        self.count("unchecked")

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def observe(self, key: str, error_ratio: float) -> None:
        """Record the worst max-relative error seen for a format.

        A pass/fail verdict says whether a payload stayed inside its bound. It does not say how far
        inside, and the margin is the interesting number when the question is whether one format is
        a tighter bet than another: "Q4 measured 0.0718 against a bound of 0.0715" is a fact about
        this artifact, while "PASS" is not.
        """
        if error_ratio > self.observed.get(key, -1.0):
            self.observed[key] = error_ratio


def check_directory(artifact: Artifact, report: Report) -> None:
    """Every binding must address real, ordered, non-overlapping ranges of real objects.

    The Python directory is a different shape from the C++ one this mirrors: a binding is either
    ``{"object": id}`` for a whole tensor or ``{"parts": [{"object": id, "range": [begin, end]}]}``
    for a slice, and ranges are element offsets into that object's shape, not byte offsets. That is
    read from tools/artifact/schema.py rather than assumed from the C++ structs, which carry a
    different and richer form.
    """
    directory = artifact.directory
    by_id = {obj.id: obj for obj in directory.objects}

    for name, binding in sorted(directory.bindings.items()):
        if "object" in binding:
            obj = _object(by_id, binding["object"])
            if obj is None:
                report.check(f"binding {name}", False, f"names missing object {binding['object']!r}")
            else:
                report.check(f"binding {name} names a real object", True)
            report.count("bindings")
            continue

        parts = binding.get("parts") or []
        if not parts:
            report.check(f"binding {name}", False, "has neither an object nor any parts")
            report.count("bindings")
            continue

        ok = True
        reason = ""
        previous_end = -1
        for part in parts:
            obj = _object(by_id, part.get("object", ""))
            if obj is None:
                ok, reason = False, f"part names missing object {part.get('object')!r}"
                break
            span = part.get("range") or []
            if len(span) != 2:
                ok, reason = False, f"part range {span} is not a [begin, end] pair"
                break
            begin, end = int(span[0]), int(span[1])
            if end < begin:
                ok, reason = False, f"range [{begin},{end}) ends before it begins"
                break
            if begin < previous_end:
                ok, reason = False, f"range begins at {begin}, overlapping the previous part"
                break
            total = _object_elements(obj)
            if total and end > total:
                ok, reason = False, f"range ends at {end}, object holds {total} elements"
                break
            previous_end = end
        report.check(f"binding {name} is ordered and in range", ok, reason)
        report.count("bindings")

    for obj in directory.objects:
        end = obj.offset + obj.bytes
        if end > directory.payload_bytes:
            report.check(
                f"object {obj.id} lies within the payload",
                False,
                f"ends at {end}, payload is {directory.payload_bytes}",
            )
        else:
            report.check(f"object {obj.id} lies within the payload", True)
        report.count("objects")

    for entry in directory.uses:
        parameter = entry.get("parameter", "?")
        for aux_name, aux in sorted((entry.get("auxiliaries") or {}).items()):
            obj = _object(by_id, aux.get("object", ""))
            if obj is None:
                report.check(
                    f"auxiliary {aux_name} of {parameter}", False, "names a missing object"
                )
                continue
            report.check(f"auxiliary {aux_name} of {parameter} names a real object", True)
            report.count("auxiliaries")


def _object(by_id: dict[str, Any], object_id: str) -> Any | None:
    return by_id.get(object_id)


def is_tensor(obj: Any) -> bool:
    """True for a TensorObject. Kept as a predicate for readability at the call sites; the mypy
    narrowing comes from the isinstance below where a union attribute is actually read."""
    return isinstance(obj, TensorObject)


def _object_elements(obj: Any) -> int:
    shape = getattr(obj, "shape", None)
    if not shape:
        return 0
    total = 1
    for dim in shape:
        total *= dim
    return total


def check_divisors(artifact: Artifact, report: Report) -> None:
    """Every NVFP4 weight divisor and every stored input divisor must be a positive finite FP32.

    The weight divisor is the last four bytes of the NVFP4 payload; the input divisor is a 4-byte FP32
    auxiliary object on the use. Both are read through the artifact, not recomputed, so this is a check
    on what was stored.
    """
    directory = artifact.directory
    by_id = {obj.id: obj for obj in directory.objects}

    seen: set[str] = set()
    for name, binding in sorted(directory.bindings.items()):
        for object_id in _binding_objects(binding):
            if object_id in seen:
                continue
            seen.add(object_id)
            obj = _object(by_id, object_id)
            if obj is None or getattr(obj, "kind", "") != "tensor":
                continue
            if obj.format not in NVFP4_FORMATS or obj.bytes < 4:
                continue
            (word,) = struct.unpack("<I", artifact.read_range(obj.offset + obj.bytes - 4, 4))
            report.check(
                f"weight divisor of {name}",
                valid_positive_fp32_word(word),
                f"0x{word:08x} is not a positive finite FP32",
            )
            report.count("weight_divisors")

    for entry in directory.uses:
        parameter = entry.get("parameter", "?")
        for aux_name, aux in sorted((entry.get("auxiliaries") or {}).items()):
            obj = _object(by_id, aux.get("object", ""))
            if obj is None or getattr(obj, "format", "") != "fp32" or obj.bytes != 4:
                continue
            (word,) = struct.unpack("<I", artifact.read_range(obj.offset, 4))
            label = f"input divisor {aux_name} of {parameter}"
            report.check(
                label,
                valid_positive_fp32_word(word),
                f"0x{word:08x} is not a positive finite FP32",
            )
            report.count("input_divisors")


def _binding_objects(binding: Any) -> list[str]:
    if "object" in binding:
        return [str(binding["object"])]
    return [str(part.get("object", "")) for part in (binding.get("parts") or [])]


def decode_nvfp4_parent(
    codes: torch.Tensor, scales: torch.Tensor, divisor: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Rebuild the dequantized NVFP4 parent from decoded words, through the format definitions.

    Independent of tools/convert/quantization/nvfp4.py on purpose: the encoder wrote these bytes, and
    an oracle that shared the encoder's arithmetic would agree with a wrong encoder.

    Vectorized, in float32, and built block by block. The first version assembled nested Python lists
    and measured 28 bytes per element, which is 4.6 GiB for the lane's largest parent and 688 GiB
    for the lane as a whole; running it exhausted a 47.8 GiB machine. Lookup tables built once from
    the format definitions cost 4 bytes per element, and a parent is produced one block at a time so
    the peak is set by BLOCK_ELEMENTS rather than by the model.
    """
    n, k = shape[0], shape[1]
    rows, cols = codes.shape
    if (rows, cols) != (n, k // 2):
        raise ValueError(f"code plane is {tuple(codes.shape)}, expected {(n, k // 2)}")
    if tuple(scales.shape) != (n, k // 16):
        raise ValueError(f"scale plane is {tuple(scales.shape)}, expected {(n, k // 16)}")
    global_scale = float(divisor)
    if not global_scale > 0.0 or not global_scale < float("inf"):
        raise ValueError("weight divisor is not positive and finite")

    out = torch.empty((n, k), dtype=torch.float32)
    rows_per_block = max(1, BLOCK_ELEMENTS // k)
    for start in range(0, n, rows_per_block):
        stop = min(n, start + rows_per_block)
        block = codes[start:stop].to(torch.int64)
        low = E2M1_TABLE[block & 0x0F]
        high = E2M1_TABLE[(block >> 4) & 0x0F]
        values = torch.stack((low, high), dim=-1).reshape(stop - start, k)
        scale_values = E4M3FN_TABLE[scales[start:stop].to(torch.int64)]
        if bool(torch.isnan(scale_values).any()):
            raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
        expanded = scale_values.repeat_interleave(16, dim=1).reshape(stop - start, k)
        out[start:stop] = values * expanded / global_scale
    return out


def check_values(
    artifact: Artifact,
    model: Any,
    stores: dict[str, Any],
    report: Report,
    tolerance: float,
    methods: dict[str, str],
) -> None:
    """Decode each NVFP4 and grouped-integer payload and compare it against the source it was made from.

    Grouped-integer sites joined this pass on 2026-10-01. Before that the ``obj.format.startswith
    ("nvfp4")`` filter below meant every quantized vision-tower payload and every NVFP4-draft payload
    was skipped without comment, so a wrong encoding in either would have passed unnoticed.

    The site -> method map decides the reference and the bound for each site; see _compare_site. The
    two families are counted separately, so a run says how many payloads it proved bit-identical and
    how many it bounded against the format's arithmetic, rather than reporting one total that mixes
    a copy check with a lossy one. The bound is per format: NVFP4 takes ``--tolerance``, and a grouped
    format takes ``grouped_bound`` of its own spec.

    This is the check the fork's ``verify_*`` entry points describe and this tree lacked. It is
    deliberately the expensive one -- decoding all 26.1 billion elements of the nvidia lane takes
    about 11 s single-threaded -- and it is separate from the structural pass so a fast run can still
    cover the directory and the divisors.

    The comparison is against the source's own values, not against another kernel and not against
    this tree's encoder. ``model.source(parameter, store, format)`` is the recipe's own mapping, so
    the verifier cannot disagree with the recipe about *which* matrix a site came from; only about
    whether the stored bytes reproduce it.

    Each site is resolved the way the recipe resolves it: ``model.source(name, store)`` with no format
    hint for a locally encoded site, which is what ``_nvfp4_draft`` and the attention recipes both do.
    A hint of ``"nvfp4"`` is only correct for a site whose source is *already* NVFP4.

    The distinction is not cosmetic. Asking the DFlash2 factory for NVFP4 makes it look for
    ``weight_packed`` tensors, which are absent because that companion is Q8-packed upstream; the
    call raises inside ``values()`` rather than at ``source()``, so a mapping that succeeded is not
    proof that the hint was right. A store is accepted only if it can also produce values.

    A binding is usually a *slice* of a larger fused parent -- ``dflash2/layers/0/attention/query``
    covers 4096 rows of a 6144-row object, because query, key, value and output share one packed
    parent. NVFP4's block scale is defined over the parent's 128-row tiles, so a slice cannot be
    decoded on its own: the parent is decoded once and the slice taken from the result. Asking a
    source for the slice's own row count, which is the obvious first thing to write, asks it for
    20,971,520 elements of a 31,457,280-element parent and raises.
    """
    directory = artifact.directory
    by_id = {obj.id: obj for obj in directory.objects}

    # Grouped by parent, and one parent resident at a time. The first version cached every decoded
    # parent in a dict for the whole run, which is 13.68 GiB of NVFP4 becoming 54.5 GiB of float32 --
    # past this machine's 47.8 GiB -- and was the other half of the memory exhaustion. A parent is
    # decoded once, every binding that covers part of it is compared, and it is released before the
    # next one is read.
    claims: dict[str, list[tuple[str, int, int, int]]] = {}
    for name, binding in sorted(directory.bindings.items()):
        object_ids = _binding_objects(binding)
        if len(object_ids) != 1:
            continue
        obj = _object(by_id, object_ids[0])
        if obj is None or getattr(obj, "kind", "") != "tensor":
            continue
        if len(obj.shape) != 2:
            continue
        if not obj.format.startswith("nvfp4") and obj.format not in QUANT_FORMATS:
            continue
        columns = obj.shape[1]
        parts = binding.get("parts")
        if parts:
            begin, end = (int(v) for v in parts[0]["range"])
            start = begin // columns if columns else 0
            rows = (end - begin) // columns if columns else 0
        else:
            start, rows = 0, obj.shape[0]
        claims.setdefault(object_ids[0], []).append((name, start, rows, columns))

    for object_id, entries in sorted(claims.items()):
        obj = _object(by_id, object_id)
        # The key came from a binding that already passed check_directory, so the object is present
        # and a tensor -- but a verifier that assumed its own earlier check would be a verifier that
        # can read a None, so it is asserted rather than assumed.
        if not isinstance(obj, TensorObject):
            for name, _, _, _ in entries:
                report.check(f"object for {name}", False, "binding names a non-tensor object")
            continue
        # Each parent is decoded once and released before the next is read, whichever format it is.
        # Grouped-integer parents are an order of magnitude smaller than the NVFP4 ones, but the
        # release is unconditional rather than per-branch so a future format cannot reintroduce the
        # retention that exhausted this machine once.
        try:
            if obj.format in QUANT_FORMATS:
                # Grouped absmax, not NVFP4. Reuses the storage codec, which the module docstring
                # otherwise declines to do, and that is deliberate here: the oracle for this site is
                # the BF16 source matrix, compared against a bound derived from the format's own
                # arithmetic. A decoder that mis-read the plane order, the nibble order or the sign
                # would produce values uncorrelated with the source and measure O(1) error, not the
                # 0.07 this bound allows, so the source comparison is what validates the decode --
                # not the codec's agreement with itself.
                spec = get_format(obj.format)
                if not isinstance(spec, QuantFormat):
                    raise ValueError(
                        f"{obj.format} is in QUANT_FORMATS but is not a QuantFormat"
                    )
                bound = grouped_bound(spec)
                # float32, not the codec's bfloat16 default. The default rounds the reconstruction to
                # bfloat16, which adds up to 2**-9 relative error of its own -- larger than Q8's entire
                # half-step budget of 0.0039 -- and that error belongs to the decode, not to the
                # encoding, so it would be charged against the format's bound. Casting to float32
                # afterwards does not remove it; the dtype has to be chosen at the call.
                parent = dequantize_row_split(
                    artifact.read_object(object_id),
                    spec,
                    obj.shape,
                    dtype=torch.float32,
                )
            else:
                bound = tolerance
                codes, scales, divisor = decode_nvfp4_words(
                    artifact.read_object(object_id), obj.shape
                )
                parent = decode_nvfp4_parent(codes, scales, divisor, obj.shape)
                del codes, scales
        except Exception as error:  # noqa: BLE001
            for name, _, _, _ in entries:
                report.check(f"decode {name}", False, f"{type(error).__name__}: {error}")
            continue
        try:
            for name, start, rows, columns in entries:
                got = parent[start : start + rows, :columns]
                _compare_site(
                    name,
                    got,
                    model,
                    stores,
                    report,
                    bound,
                    obj.shape[1],
                    methods.get(name, "unknown"),
                    obj.format,
                )
        finally:
            # Released before the next parent is read, not at the end of the loop.
            del parent


def _load_sources(args: argparse.Namespace) -> tuple[Any, Any, dict[str, str]]:
    """Build the model and open the sources, through the same path the converter uses.

    Reusing tools.convert.qwen3_5.build_model and the recipe's own ``model.source`` is the point: the
    verifier then cannot disagree with the recipe about which source matrix a site came from, so a
    disagreement it does report is about the stored bytes and not about the naming.

    Returns the per-site method name as well, because that decides *how* a site is checked. The
    recipe is run for real against a real Recipe object rather than having its mapping guessed at --
    it is what installs the site-to-source factories on the model's parameters, and passing None here
    would leave every ``model.source`` call raising instead of checking anything.
    """
    if not args.recipe:
        raise SystemExit("verify: --values needs --recipe and --source")
    from contextlib import ExitStack

    # Absolute, not relative: this file is run as a script (``python tools/convert/verify_artifact.py``)
    # so it has no parent package, and the relative form raises ImportError at the first --values run
    # rather than at import time. The top-of-file imports are absolute for the same reason.
    from tools.convert.official_recipes import RECIPES
    from tools.convert.qwen3_5 import build_model
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    if args.recipe not in RECIPES:
        raise SystemExit(f"verify: unknown recipe {args.recipe!r}")
    paths: dict[str, str] = {}
    for item in args.source:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"verify: --source needs NAME=PATH, got {item!r}")
        paths[name] = path
    if "base" in paths:
        raise SystemExit("verify: select the base source with --recipe-base, not --source base=")

    base_path = args.recipe_base
    if not base_path:
        raise SystemExit("verify: --values needs --recipe-base PATH")
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorsSource(base_path))
        sources: dict[str, Any] = {"base": base}
        for name, path in paths.items():
            sources[name] = stack.enter_context(SafetensorsSource(path))
        # dflash and dflash2 are *companions*, not sources: __main__ keeps them out of `sources` and
        # passes them to build_model separately. Putting dflash2 in sources made build_model treat the
        # base checkpoint as the drafter, and every dflash2 site then reported a wrong-architecture
        # error that looked like a verification finding.
        companions: dict[str, SafetensorsSource] = {
            key: sources[key] for key in ("dflash", "dflash2") if key in sources
        }
        model = build_model(
            base, components=("text", "vision", "mtp", "dflash2"), companions=companions
        )
        recipe = Recipe(model)
        RECIPES[args.recipe](model, recipe, sources)
        # prepare() is the only place a site's method is recorded, and that is what decides how the
        # site is checked: an imported word must equal its source exactly, while an encoded one is
        # only required to reproduce the BF16 source within NVFP4's own arithmetic. Read from the
        # recipe rather than inferred from the format, because the two are not distinguishable from
        # the artifact alone. Device cpu: it resolves the job graph, and the decode is host-side.
        methods: dict[str, str] = {}
        for job in recipe.prepare(device="cpu").weights:
            for parameter in job.parameters:
                methods[parameter] = job.method_name
        # The model and its sources must outlive this function, so the stack is kept alive on the
        # returned handle rather than closing the file handles here.
        stack.pop_all()
    return model, sources, methods


def _compare_site(
    name: str,
    got: torch.Tensor,
    model: Any,
    stores: dict[str, Any],
    report: Report,
    tolerance: float,
    columns: int,
    method: str,
    fmt: str = "unknown",
) -> None:
    """Compare one decoded slice against the right reference for how the recipe produced it.

    Two families, and conflating them is what made the first run report 32 false failures:

      ``import_encoded`` -- the words are *copied* from an already-NVFP4 source. The reference is
        that encoded source and the error must be **exactly zero**. Checked against the BF16 base
        instead, the comparison measures the quantization gap the import is supposed to preserve, and
        reported up to 0.20 on payloads that are in fact bit-identical to their source.

      ``nvfp4_maxabs`` -- the words are *encoded* from a floating-point source, so the reference is
        that source and the bound is NVFP4's own arithmetic: 1/6 for rounding a value to the nearest
        E2M1 magnitude, plus 1/16 for the block scale's 3-bit E4M3FN mantissa, which scales all 16
        values in the block. 0.2292 in total.

    This is the split artifact-conventions.md section 2 already states -- "Imported words are copied,
    local words are proven" -- and the verifier was not honouring it.
    """
    imported = method == "import_encoded"
    # The store order encodes the intent: an encoded source first for an import, the floating-point
    # source first for a local encode. Trying the other is a fallback, not the plan.
    attempts = (
        (("quantized", "nvfp4"), ("base", None), ("dflash2", None))
        if imported
        else (("base", None), ("dflash2", None), ("quantized", "nvfp4"))
    )
    expected = None
    reason = ""
    used = ""
    for store_name, hint in attempts:
        store = stores.get(store_name)
        if store is None:
            continue
        try:
            source = model.source(name, store, hint)
            values = source.values(0, got.shape[0] * columns)
        except Exception as error:  # noqa: BLE001
            # The FIRST attempt's reason is kept, not the last. Base is tried first, so the last
            # attempt is the quantized store and its "missing source tensor '<name>_packed'" is a
            # consequence of the hint being wrong for a site that is not NVFP4-encoded. Keeping it
            # hid the real cause, which for both W8 endpoints is a "short source read" at 1.27
            # billion elements -- a limit of the lookup, and invisible while a later attempt's
            # unrelated error was printed instead.
            if not reason:
                reason = f"{store_name}/{hint or '-'}: {type(error).__name__}: {error}"
            continue
        expected = values.reshape(got.shape[0], columns)
        used = f"{store_name}/{hint}"
        break
    if expected is None:
        report.note(
            f"source mapping {name}",
            reason or "no opened source maps this site",
        )
        return
    want = expected.to(torch.float32)
    scale = float(want.abs().max())
    if scale == 0.0:
        report.check(f"values {name}", bool(torch.equal(got, want)), "source is all zero")
        report.count("values")
        return
    error_ratio = float((got - want).abs().max()) / scale
    report.observe(fmt, error_ratio)
    if imported:
        report.check(
            f"imported words {name}",
            error_ratio == 0.0,
            f"an imported payload must equal {used} exactly, max relative error {error_ratio:.6f}",
        )
        report.count("imported_words")
        return
    report.check(
        f"encoded values {name}",
        error_ratio <= tolerance,
        f"max relative error {error_ratio:.4f} against {used} exceeds {tolerance:.4f}",
    )
    report.count("encoded_values")


def projected_peak_bytes(artifact: Artifact) -> int:
    """The largest single parent, in float32, plus its raw payload.

    This is the pre-flight guard. The first version of the value check had no such number anywhere,
    assembled parents from nested Python lists, and held every decoded parent for the whole run: the
    nvidia lane needed 688 GiB by arithmetic on a 47.8 GiB machine, and running it exhausted the
    machine. The arithmetic was available before the run and was not done. It is done now, before
    anything is decoded, and the run refuses if the projection is over budget.
    """
    largest = 0
    for obj in artifact.objects:
        # isinstance rather than a kind string: the object type is a union of TensorObject and
        # ResourceObject, and only the former carries shape and format. A getattr would silence the
        # type checker instead of narrowing the union.
        if not isinstance(obj, TensorObject):
            continue
        if not (obj.format or "").startswith("nvfp4") or len(obj.shape) != 2:
            continue
        elements = obj.shape[0] * obj.shape[1]
        largest = max(largest, elements * 4 + obj.bytes)
    return largest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path, help="the .ninfer entry to verify")
    parser.add_argument(
        "--only",
        choices=("directory", "divisors", "all"),
        default="all",
        help="restrict to one class of structural check (default: all)",
    )
    parser.add_argument(
        "--recipe",
        help="recipe name, required for --values: the site-to-source mapping it encodes",
    )
    parser.add_argument(
        "--recipe-base",
        metavar="PATH",
        help="the base checkpoint the recipe was run against, required for --values",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="an additional source checkpoint for --values; repeatable",
    )
    parser.add_argument(
        "--values",
        action="store_true",
        help="also decode every NVFP4 payload and compare it against the source (slow)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 6.0 + 1.0 / 16.0,
        help=(
            "max relative error for a LOCALLY ENCODED site (default 1/6 + 1/16 = 0.2292). "
            "1/6 is NVFP4's own worst-case value rounding -- a 16-element block scaled so its "
            "largest value reaches the E2M1 maximum, then rounded to the nearest representable "
            "magnitude -- and 1/16 is the block scale's own 3-bit E4M3FN mantissa, which is off by "
            "up to half an ulp and scales all 16 values in the block. Imported sites are not bounded "
            "by this: their words must be identical to their source, and the check is exact."
        ),
    )
    parser.add_argument(
        "--max-peak-gib",
        type=float,
        default=4.0,
        help=(
            "refuse --values if the largest parent would need more than this (default 4.0). "
            "A pre-flight guard, not a runtime check: the projection is made before anything is "
            "decoded, because the failure it prevents exhausts the machine rather than raising."
        ),
    )
    args = parser.parse_args(argv)

    report = Report()
    if not args.artifact.is_file():
        print(f"verify: no artifact at {args.artifact}", file=sys.stderr)
        return 1
    if args.artifact.suffix != ".ninfer":
        print(
            f"verify: {args.artifact.name} is not a .ninfer entry; the engine accepts only that "
            "extension, so a copy made for measurement would need a hardlink rather than a rename",
            file=sys.stderr,
        )
        return 1

    with Artifact.open(args.artifact) as artifact:
        if args.only in ("directory", "all"):
            check_directory(artifact, report)
        if args.only in ("divisors", "all"):
            check_divisors(artifact, report)
        if args.values:
            projected = projected_peak_bytes(artifact)
            budget = args.max_peak_gib * 2**30
            print(
                f"  projected peak    {projected / 2**30:.2f} GiB "
                f"(largest NVFP4 parent, float32, plus its raw payload)"
            )
            if projected > budget:
                print(
                    f"verify: projected peak {projected / 2**30:.2f} GiB is over the "
                    f"{args.max_peak_gib:.1f} GiB budget, so --values is refused before any "
                    "decoding happens. Raise --max-peak-gib deliberately, or narrow the artifact.",
                    file=sys.stderr,
                )
                return 1
            model, stores, methods = _load_sources(args)
            check_values(artifact, model, stores, report, args.tolerance, methods)

    for key in sorted(report.counts):
        print(f"  {key:<18} {report.counts[key]}")
    if report.observed:
        print("\n  worst max-relative error against source, by format:")
        for key in sorted(report.observed):
            print(f"    {key:<18} {report.observed[key]:.6f}")
    if report.notes:
        # Printed before the failure count and on the same stream as the counts, because "not
        # checked" is part of what a run covered. A payload listed here is unproven, not disproven.
        print(f"\nverify: {len(report.notes)} site(s) this run could NOT check:")
        for note in report.notes[:40]:
            print(f"  {note}")
        if len(report.notes) > 40:
            print(f"  ... and {len(report.notes) - 40} more")
    if report.failures:
        print(f"\nverify: {len(report.failures)} failure(s):", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  {failure}", file=sys.stderr)
        if len(report.failures) > 40:
            print(f"  ... and {len(report.failures) - 40} more", file=sys.stderr)
        return 1
    print(f"  PASS: {len(report.passed)} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
