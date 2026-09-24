"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import cast_direct, fp8_row_maxabs, grouped_absmax, import_encoded
from .proposal import add_proposal
from .quantization.nvfp4 import nvfp4_maxabs

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


# The Swift checkpoint's attention and GDN projections are ModelOpt FP8, and importing them is what
# leaves 9 GiB of 8-bit weights in the artifact. They are re-encoded to NVFP4 from the finetune's own
# BF16 source instead -- the choice NVFP4-full made for its source's 233 FP8 matrices. See
# docs/maintainer/artifact-conventions.md, section 1.
_ATTENTION_MODULES = {
    "query": "self_attn.q_proj", "gate": "self_attn.q_proj", "key": "self_attn.k_proj",
    "value": "self_attn.v_proj", "output": "self_attn.o_proj",
}
_GDN_MODULES = {
    "query": "linear_attn.in_proj_qkv", "key": "linear_attn.in_proj_qkv",
    "value": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z",
    "output": "linear_attn.out_proj",
}


def _activation_divisor(store, prefix: str, module: str) -> float:
    """The NVFP4 activation divisor, recovered from the checkpoint's own multiplier.

    ModelOpt exports an FP8 site's `input_scale` as `amax / 448`, and the NVFP4 global scale is
    `amax / (6 * 448)`, so the divisor this port binds is `2688 / amax = 6 / input_scale`. Reusing
    the producer's own calibration is both cheaper and more faithful than re-measuring it over a
    different corpus.
    """
    name = f"{prefix}{module}.input_scale"
    if not store.has(name):
        raise ValueError(f"{name}: no activation scale to derive the divisor from")
    return 6.0 / float(store.read_flat(name).reshape(1))


def qwen3_8_27b_nvfp4_swift(model, recipe, sources):
    """Swift re-encoded to match the shipped artifacts: no FP8 code word reaches the artifact.

    Attention and GDN are encoded to NVFP4 from the finetune's BF16 source, and both W8 endpoints
    are Q8 from that source, which is what the two shipped artifacts bind. The MLP stays imported
    from ModelOpt's NVFP4 codes, which is lossless. Device weights therefore sit inside the
    envelope that reaches the full native context.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    base = sources["base"]
    bf16 = sources["swift_bf16"]
    prefix = "model.language_model." if "text_config" in base.config else "model."
    # Both shipped artifacts bind q8_g32_fp16 to both W8 endpoints, and the base source owns them;
    # taking them from the BF16 source also avoids re-encoding the head from its NVFP4 codes.
    _assign(recipe, "text/token_embedding", Q8,
            source=model.source("text/token_embedding", bf16))
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or not parameter.projection:
            continue
        if name == "text/token_embedding" or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            continue
        if name == "text/output_head":
            _assign(recipe, name, Q8, source=model.source(name, bf16))
            continue
        if "/mlp/" in name:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=model.source(name, base, "nvfp4"),
                activation_policy="AllowA4",
            )
            continue
        role = name.rsplit("/", 1)[1]
        modules = _ATTENTION_MODULES if "/attention/" in name else _GDN_MODULES
        module = modules.get(role)
        if module is None:
            raise ValueError(f"{name}: no NVFP4 site is registered for this projection")
        divisor = _activation_divisor(base, f"{prefix}layers.{name.split('/')[2]}.", module)
        recipe.assign(
            name,
            format="nvfp4",
            method=nvfp4_maxabs,
            source=model.source(name, bf16),
            activation_policy="AllowA4",
        )
        for input_name in parameter.inputs:
            recipe.use(
                name, input_name, auxiliaries={"activation_input_divisor": divisor}
            )
    add_proposal(recipe, source=model.source("text/output_head", bf16))


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_swift": qwen3_8_27b_nvfp4_swift,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
}
