from __future__ import annotations

import struct

import torch
from safetensors.torch import save_file

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.quant_scheme import read_declared_scheme
from tools.convert.sources.logical import select_rows


def test_nvfp4_source_preserves_words_and_decodes_independently(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight_packed": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
            "proj.input_global_scale": torch.tensor([1.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        expected = torch.tensor(
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
        expected = torch.stack((expected / 2, expected))
        assert torch.equal(source.values().reshape(2, 16), expected)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        assert source.values(16, 16).numel() == 0


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))


def test_modelopt_nvfp4_preserves_words_and_inverts_the_stored_multiplier(tmp_path):
    """ModelOpt stores the same codes and block scales, and the inverse scale."""
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            # A multiplier: ModelOpt's amax / (6 * 448). The bound divisor is 1/x.
            "proj.weight_scale_2": torch.tensor([0.5], dtype=torch.float32),
            "proj.input_scale": torch.tensor([0.25], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        assert source.input_divisor() == struct.pack("<f", 4.0)


def test_modelopt_fp8_restates_one_tensor_scale_on_every_row(tmp_path):
    """A per-tensor multiplier broadcast to rows, not a per-row scale invented."""
    codes = torch.tensor([[0x38, 0xB8], [0x30, 0xB0]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": torch.tensor([2.0], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 2))
        words = source.read_encoded(0, 2)
        assert words.scales.shape == (2,)
        assert torch.equal(words.scales, torch.full((2,), 2.0, dtype=torch.bfloat16))
        assert torch.equal(
            source.values().reshape(2, 2), torch.tensor([[2.0, -2.0], [1.0, -1.0]])
        )


def test_a_declared_scheme_overrides_the_tensor_names_and_refuses_disagreement(tmp_path):
    """The declaration decides; a recipe asking for the other encoding is an error."""
    import json

    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": torch.tensor([[0x38]], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            "proj.weight_scale_2": torch.tensor([0.5], dtype=torch.float32),
            # Encoded-looking, but the declaration below does not name it.
            "other.weight": codes.clone(),
            "other.weight_scale": torch.tensor([[0x38]], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            "other.weight_scale_2": torch.tensor([0.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quantized_layers": {"proj": {"quant_algo": "NVFP4"}}}})
    )
    with SafetensorsSource(tmp_path) as store:
        assert read_declared_scheme(store) == {"proj": "NVFP4"}
        # Asked for NVFP4 with no explicit format, the declaration resolves it:
        # E4M3FN block scale 0x38 (1.0) over the inverted multiplier 0.5.
        assert torch.equal(
            matrix_source(store, "proj.weight", (1, 16)).values()[:4],
            torch.tensor([0.0, 0.25, 0.5, 0.75]),
        )
        # Asking for FP8 disagrees with what the checkpoint declared.
        try:
            matrix_source(store, "proj.weight", (1, 16), "fp8_e4m3fn_row_bf16").values()
        except ValueError as error:
            assert "declared NVFP4" in str(error)
        else:
            raise AssertionError("a declared scheme was overridden by the asked format")
        # A matrix the checkpoint did not declare is not silently read.
        try:
            matrix_source(store, "other.weight", (1, 16)).values()
        except ValueError as error:
            assert "declared quant scheme" in str(error)
        else:
            raise AssertionError("an undeclared matrix was read as though it were encoded")
