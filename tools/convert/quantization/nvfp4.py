"""NVFP4_MAXABS_DIVISOR_RNE_V1: the local NVFP4 encoder.

Ported from the fork that produced the all-NVFP4 artifacts, so a checkpoint whose text
projections are FP8 can be re-encoded rather than imported. See
[artifact conventions](../../../docs/maintainer/artifact-conventions.md), section 1.

Two scales, per Transformer Engine's NVFP4 recipe and confirmed against a ModelOpt
checkpoint whose scaled codes saturate the format maximum exactly:

    s_global = global_amax / (448 * 6)      # 448 = E4M3FN max, 6 = E2M1 max
    s_block  = (block_amax / 6) / s_global  # stored in E4M3FN, so its max is 448

The word the artifact stores as its weight divisor is therefore ``1 / s_global``, and
the codes are E2M1 with round-to-nearest-even, packed low nibble first.

The divisor belongs to the complete packed parent, not to each logical slice or
streaming chunk. The *activation* divisor is not measured here: the recipe supplies it
through the A4 auxiliary, because the source checkpoint already carries the activation
amax its own producer calibrated with.
"""

from __future__ import annotations

import struct

import torch

from tools.convert.methods import AuxiliaryValue

FULL_RANGE = 2688.0  # 6 * 448


def e2m1_rne_codes(values: torch.Tensor) -> torch.Tensor:
    """E2M1 codes for magnitudes in [0, 6], ties to even.

    The boundaries are the midpoints between representable magnitudes
    (0, .5, 1, 1.5, 2, 3, 4, 6); ``tie_up`` marks the ones whose upper neighbour is
    even, so an exact tie rounds to the even code.
    """
    magnitude = values.abs()
    codes = torch.zeros_like(values, dtype=torch.uint8)
    for boundary, tie_up in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.),
                                (False, True, False, True, False, True, False)):
        codes += ((magnitude > boundary) | ((magnitude == boundary) & tie_up)).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def encode_block(values: torch.Tensor, divisor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one whole-tile block: E2M1 codes and E4M3FN scales for ``[rows, columns]``."""
    rows, columns = values.shape
    blocks = (values.float() * divisor).reshape(rows, columns // 16, 16)
    scales = (blocks.abs().amax(dim=2) / 6.).clamp(max=448.).to(torch.float8_e4m3fn)
    decoded = scales.float()
    nonzero = decoded > 0
    ratios = torch.where(nonzero[..., None],
                         blocks / torch.where(nonzero, decoded, 1.)[..., None], 0.)
    codes = e2m1_rne_codes(ratios).reshape(rows, columns // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scales.view(torch.uint8)


def nvfp4_maxabs(request):
    """Encode a logical NVFP4 parent from floating-point values.

    The parent must be complete: NVFP4's block-scale layout is a 128-row by 16-column
    tile, so a partial parent has no defined divisor.
    """
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError("nvfp4_maxabs requires an NVFP4 matrix")
    if request.parameters:
        raise ValueError("nvfp4_maxabs accepts no numerical parameters")
    n, k = request.target.shape
    if n % 128 or k % 16 or request.source_offsets[-1] != n * k:
        raise ValueError("NVFP4 parent requires complete 128-row and 16-column tiles")
    if request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    chunk = max(128, request.rows_per_chunk // 128 * 128)
    auxiliaries = {}
    for item in request.inputs:
        for use in item.uses:
            key = (*use, "activation_input_divisor")
            if request.policies[use] == "AllowA4":
                if key not in request.auxiliary_overrides:
                    raise ValueError(f"{use}: calibrated activation divisor required")
                auxiliaries[key] = request.auxiliary_overrides[key]

    def produce(output):
        maximum = 0.
        for begin in range(0, n, chunk):
            values = request.values(begin * k, min(n, begin + chunk) * k).float()
            if not torch.isfinite(values).all():
                raise ValueError("NVFP4 source contains NaN or infinity")
            maximum = max(maximum, values.abs().max().item())
        raw = struct.pack("<f", FULL_RANGE / maximum if maximum else 1.)
        AuxiliaryValue.activation_divisor(raw)  # same positive finite FP32 contract
        divisor = struct.unpack("<f", raw)[0]
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k).to(request.device)
            codes, scales = encode_block(values, divisor)
            output.write_codes(begin, codes.cpu(), scales.cpu(), raw)

    return request.job(produce=produce, auxiliaries=auxiliaries)
