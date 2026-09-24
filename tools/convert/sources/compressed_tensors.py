"""Interpret the current compressed-tensors FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .quant_scheme import (
    COMPRESSED_TENSORS_FP8,
    COMPRESSED_TENSORS_NVFP4,
    MODELOPT_FP8,
    MODELOPT_NVFP4,
    Encoding,
    modelopt_scale_word,
    read_declared_scheme,
)
from .safetensors import SafetensorsSource, tensor_source


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str,
    encoding: Encoding | None = None,
) -> LogicalSource:
    """Interpret one producer's encoded matrix.

    `format` is the port's encoding ("nvfp4" or "fp8_e4m3fn_row_bf16"); `encoding`
    says how the producer stores it, defaulting to the compressed-tensors
    convention this file was written for.
    """
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    if encoding is None:
        encoding = (
            COMPRESSED_TENSORS_NVFP4 if format == "nvfp4" else COMPRESSED_TENSORS_FP8
        )
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def scale_word(suffix: str) -> bytes:
        """The bound divisor, inverted first when the producer stored a multiplier."""
        name = f"{prefix}.{suffix}"
        if not encoding.divisors_stored_as_multipliers:
            return _divisor_word(store, name)
        return modelopt_scale_word(store, name)

    def global_divisor() -> bytes:
        return scale_word(encoding.global_scale_key)

    def input_divisor() -> bytes:
        return scale_word(encoding.input_scale_key)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed = f"{prefix}.{encoding.codes_key}"
            scale = f"{prefix}.{encoding.scales_key}"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, global_divisor())
        weight = f"{prefix}.{encoding.codes_key}"
        scale = f"{prefix}.{encoding.scales_key}"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if encoding.row_scale_from_tensor_scale:
            # One stored multiplier, restated on every row. A uniform multiplier is
            # the same number on each row, so this is a faithful restatement of
            # per-tensor scaling in the only geometry this port has for FP8.
            if info.dtype != "F32" or prod(info.shape) != 1:
                raise ValueError(f"{scale}: expected a single FP32 tensor scale")
            row_scale = store.read_flat(scale).reshape(1).to(torch.bfloat16)
            scales = row_scale.expand(end - begin).contiguous()
        else:
            if info.dtype != "BF16" or prod(info.shape) != n:
                raise ValueError(f"{scale}: expected one BF16 scale per row")
            scales = store.read_flat(scale, begin, end)
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = (
                words.codes.view(torch.float8_e4m3fn).float()
                * words.scales.float()[:, None]
            )
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
                last - first, k
            )
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = (
                words.scales.view(torch.float8_e4m3fn)
                .float()
                .repeat_interleave(16, dim=1)
            )
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (global_divisor if format == "nvfp4" and encoding.global_scale_key else None),
        (input_divisor if format == "nvfp4" and encoding.input_scale_key else None),
    )


def _encoding_for(store: SafetensorsSource, prefix: str, format: str) -> Encoding:
    """Which producer's storage this matrix uses.

    When the checkpoint declares its scheme, the declaration decides and a
    disagreement with the tensor names is an error rather than a silent choice:
    reading ModelOpt codes through the compressed-tensors keys would produce
    weights that load and are wrong.

    `lm_head` is the one matrix ModelOpt declares unprefixed while the weight map
    carries it without a `.weight` suffix, so it is also probed at its bare name.
    """
    declared = read_declared_scheme(store)
    if declared:
        algorithm = declared.get(prefix) or declared.get(prefix + ".weight")
        if algorithm is not None:
            if algorithm == "NVFP4":
                if format == "nvfp4":
                    return MODELOPT_NVFP4
                raise ValueError(
                    f"{prefix}: declared NVFP4 but the recipe asked for {format}"
                )
            if algorithm.startswith("FP8"):
                if format == "fp8_e4m3fn_row_bf16":
                    return MODELOPT_FP8
                raise ValueError(
                    f"{prefix}: declared {algorithm} but the recipe asked for {format}"
                )
            raise ValueError(f"{prefix}: unsupported declared quant_algo {algorithm!r}")
        # Declared checkpoints are all-or-nothing: a matrix absent from the
        # declaration is one the producer left unquantized.
        raise ValueError(f"{prefix}: absent from the checkpoint's declared quant scheme")
    # No declaration: identify the producer from the keys it wrote. ModelOpt's
    # NVFP4 carries a `weight_scale_2` where compressed-tensors carries a
    # `weight_packed`; its FP8 carries one FP32 tensor scale where
    # compressed-tensors carries one BF16 scale per row.
    if format == "nvfp4":
        if store.has(f"{prefix}.weight_scale_2"):
            return MODELOPT_NVFP4
        return COMPRESSED_TENSORS_NVFP4
    if store.describe(f"{prefix}.weight_scale").dtype == "F32":
        return MODELOPT_FP8
    return COMPRESSED_TENSORS_FP8


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and (
                store.has(prefix + ".weight_packed")
                or store.has(prefix + ".weight_scale_2")
            ):
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            resolved = (
                tensor_source(store, name, shape)
                if actual is None
                else compressed_matrix_source(
                    store, prefix, shape, actual, _encoding_for(store, prefix, actual)
                )
            )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
