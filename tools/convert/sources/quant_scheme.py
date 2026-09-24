"""Declared quantization schemes, and the encodings that satisfy them.

A checkpoint can say what it holds instead of leaving the reader to infer it from
tensor names. NVIDIA ModelOpt writes `hf_quant_config.json` with a `quant_algo`
per tensor, and that file is authoritative: key-shaped detection is a guess that
happens to be right on the checkpoints seen so far, while the declaration is the
producer's own statement.

Both conventions describe the same two encodings -- NVFP4 with E4M3 block scales,
and FP8 E4M3 with a uniform scale -- so the reader takes an `Encoding` rather
than branching on the producer. What differs is recorded here and nowhere else:

  * which tensor carries the packed codes and which the block scales;
  * whether the global and input scales are stored as divisors
    (compressed-tensors) or as multipliers (ModelOpt, which stores
    `amax / (6 * 448)`) -- so they are inverted;
  * whether the FP8 scale is one value per row or one per tensor, the latter
    restated per row because a uniform multiplier *is* the same number on every
    row.

The inversion and the restatement are conversions between representations of the
same value, not approximations: the reciprocal is the correctly rounded FP32 of
1/x and is computed in FP32 for that reason, and broadcasting a scalar to rows
changes no row's multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import json
from pathlib import Path

import torch

from .safetensors import SafetensorsSource


@dataclass(frozen=True, slots=True)
class Encoding:
    """How one producer stores a quantized matrix.

    Keys are suffixes appended to the matrix prefix; `""` means the key does not
    exist for this producer when the branch does not apply.
    """

    codes_key: str
    scales_key: str
    global_scale_key: str
    input_scale_key: str
    # ModelOpt stores the global and input scales as multipliers (amax / (6 * 448)),
    # which must be inverted to become the divisors this port binds.
    divisors_stored_as_multipliers: bool
    # FP8 only: the stored scale is one value per tensor rather than one per row.
    row_scale_from_tensor_scale: bool


COMPRESSED_TENSORS_NVFP4 = Encoding(
    codes_key="weight_packed",
    scales_key="weight_scale",
    global_scale_key="weight_global_scale",
    input_scale_key="input_global_scale",
    divisors_stored_as_multipliers=False,
    row_scale_from_tensor_scale=False,
)

MODELOPT_NVFP4 = Encoding(
    codes_key="weight",
    scales_key="weight_scale",
    global_scale_key="weight_scale_2",
    input_scale_key="input_scale",
    divisors_stored_as_multipliers=True,
    row_scale_from_tensor_scale=False,
)

COMPRESSED_TENSORS_FP8 = Encoding(
    codes_key="weight",
    scales_key="weight_scale",
    global_scale_key="",
    input_scale_key="",
    divisors_stored_as_multipliers=False,
    row_scale_from_tensor_scale=False,
)

MODELOPT_FP8 = Encoding(
    codes_key="weight",
    scales_key="weight_scale",
    global_scale_key="",
    input_scale_key="",
    divisors_stored_as_multipliers=False,
    row_scale_from_tensor_scale=True,
)


def read_declared_scheme(store: SafetensorsSource) -> dict[str, str]:
    """`quant_algo` per tensor prefix, from ModelOpt's own declaration.

    Returns an empty mapping when the checkpoint declares nothing, which is the
    compressed-tensors case: those checkpoints describe their layout in the
    weight tensors themselves.
    """
    path = Path(store.root) / "hf_quant_config.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    configuration = document.get("quantization") or {}
    per_tensor = configuration.get("quantized_layers") or {}
    declared: dict[str, str] = {}
    for name, entry in per_tensor.items():
        algorithm = (entry or {}).get("quant_algo")
        if not isinstance(algorithm, str):
            continue
        # `lm_head` is stored unprefixed while every other matrix carries
        # `.weight`; the lookup below is by prefix either way.
        declared[name] = algorithm.upper()
    return declared


def modelopt_scale_word(store: SafetensorsSource, name: str) -> bytes:
    """The reciprocal of a stored multiplier, as the FP32 word this port binds.

    One FP32 division, correctly rounded, which is the word compressed-tensors
    would have stored had it written a divisor.
    """
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a single FP32 scale")
    value = store.read_flat(name).reshape(1)
    if not bool(torch.isfinite(value).all()) or float(value) <= 0:
        raise ValueError(f"{name}: scale must be finite and positive")
    return (torch.ones(1, dtype=torch.float32) / value).numpy().tobytes()
