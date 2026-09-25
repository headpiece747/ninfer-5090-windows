"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

import json
from pathlib import Path

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


def _nvfp4_draft(recipe, model, sources):
    """Encode this port's draft projections as NVFP4 instead of upstream's Q8.

    Measured on this port's own bench, interleaved against the fetched QUASAR artifact in one window:
    Q8 accepts 54.8% and NVFP4 58.0%, while the fetched artifact accepts 45.7% -- its table's 62.5%
    was recorded on 2026-09-17 and does not reproduce today on that same file. The skip list is
    upstream's, which measured better than encoding those sites too (42.2% when the convolution kernel
    projections were included). No activation divisor is needed: these sites permit A16 and A8, and
    A4 -- the only policy that would require one -- is not permitted. The text stack is untouched, so
    a rebuild's MTP digests are identical to the Q8 build's.
    """
    skip = (
        "/moe/router",
        "/moe/shared_score",
        "/attention_conv/kernel_projection",
        "/mlp_conv/kernel_projection",
        "/candidate_selector/hidden_projection",
    )
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        component = name.split("/", 1)[0]
        if component not in ("dflash", "dflash2") or component not in model.components:
            continue
        if name.endswith(skip):
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=nvfp4_maxabs,
            source=model.source(name, sources[component]),
            activation_policy="A16Only",
        )


def qwen3_8_27b_nvfp4_qat(model, recipe, sources):
    """QAT-sourced text: nothing is re-encoded, because the source quantized every linear.

    QUASAR's export carries 496 fused NVFP4 sites covering every text linear, and it quantizes
    `gdn/a_projection` and `gdn/b_projection` too, where the other two sources leave them BF16. Those
    two are BF16 from the base checkpoint regardless: they are (96, 5120) and
    `block_scale_k16_m128x4_v1` requires N divisible by 128, so the layout cannot hold them at all --
    which is also why the shipped `nvfp4qat` lists attention, GDN qkv/z/output and MLP but not a or b.
    Measured against the built model, every other text projection resolves to a real NVFP4 site, and
    the only one that does not is `text/output_head`, which this source's quantizer ignores; that head
    and the embedding are Q8. The embedding has no inputs, so it is not a projection and never enters
    the loop below. `gdn/convolution`, `a_log`, `dt_bias` and every norm stay BF16.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _nvfp4_draft(recipe, model, sources)
    base = sources["base"]
    bf16 = sources["bf16"]
    _assign(recipe, "text/token_embedding", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or not parameter.projection:
            continue
        if name == "text/output_head":
            _assign(recipe, name, Q8)
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, bf16))
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, base, "nvfp4"),
            activation_policy="AllowA4",
        )
    add_proposal(recipe, source=model.source("text/output_head", base))


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
    """The NVFP4 activation divisor, recovered from a checkpoint's own multiplier.

    ModelOpt records the multiplier differently by site format, and the difference is a factor of six:
    a site it already stores as NVFP4 records `input_scale = amax / (6 * 448)`, so the divisor is
    `1 / input_scale`; an FP8 site records `amax / 448`, so it is `6 / input_scale`. Which one applies
    is a property of the *source* site, so it is probed rather than assumed -- applying the FP8 form to
    NVFP4 sites scales the divisor six times too large and clips the A4 activations.
    """
    name = f"{prefix}{module}.input_scale"
    if not store.has(name):
        raise ValueError(f"{name}: no activation scale to derive the divisor from")
    scale = float(store.read_flat(name).reshape(1))
    already_nvfp4 = store.has(f"{prefix}{module}.weight_scale_2")
    return (1.0 if already_nvfp4 else 6.0) / scale


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


# nvfp4full keeps these nine parents BF16 rather than encoding them, and the same pattern *lost* on
# Swift's ModelOpt source (docs/maintainer/artifact-conventions.md, section 1). Measured on this source
# it wins: with them encoded, line A scores 4.85576/4.99604 against nvfp4full's 4.82452/4.98768, and
# hashing every binding against that artifact shows the whole difference sits on these 27 projections.
# Which is the rule's point -- a pattern is measured per checkpoint, not transplanted.
_BF16_EXCEPTION_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
_BF16_EXCEPTION_ATTENTION_OUTPUT_LAYERS = (3, 7)
_BF16_EXCEPTION_GDN_OUTPUT_LAYERS = (4,)


def _is_bf16_exception(layer: int, block: str, role: str) -> bool:
    if block == "attention":
        if role in ("query", "gate", "key", "value"):
            return layer in _BF16_EXCEPTION_ATTENTION_LAYERS
        if role == "output":
            return layer in _BF16_EXCEPTION_ATTENTION_OUTPUT_LAYERS
    elif block == "gdn" and role == "output":
        return layer in _BF16_EXCEPTION_GDN_OUTPUT_LAYERS
    return False


# Artifact site name for each re-encoded parameter role, matching the naming the fork's calibration
# recorded, so this port's divisors are comparable with the published profile's site by site.
_CALIBRATION_SITES = {
    "attention": {
        "query": "attention/input_projection", "gate": "attention/input_projection",
        "key": "attention/input_projection", "value": "attention/input_projection",
        "output": "attention/output_projection",
    },
    "gdn": {
        "query": "gdn/input_projection", "key": "gdn/input_projection",
        "value": "gdn/input_projection", "z": "gdn/input_projection",
        "output": "gdn/output_projection",
    },
    "mlp": {
        "gate": "mlp/gate_up_projection", "up": "mlp/gate_up_projection",
        "down": "mlp/down_projection",
    },
}
_CALIBRATION_PATH = Path(__file__).with_name("qwen3_8_27b_nvfp4_calibration.json")


def load_calibration(path=None) -> dict:
    """Measured activation divisors, keyed by site name.

    Regenerate with `python3 -m tools.convert.calibration --model <BF16 checkpoint> --out <path>`; the
    method and corpus are the fork's, and the module's docstring records what validates them.
    """
    payload = json.loads(Path(_CALIBRATION_PATH if path is None else path).read_text(encoding="utf-8"))
    return {name: site["input_scale_divisor"] for name, site in payload["sites"].items()}


def _calibrated_divisor(calibration: dict, name: str) -> float:
    """The measured divisor for a logical parameter, from its artifact site name."""
    parts = name.split("/")
    if len(parts) < 4:
        raise ValueError(f"{name}: not a layer parameter")
    block, role = parts[3], parts[-1]
    sites = _CALIBRATION_SITES.get(block)
    if sites is None or role not in sites:
        raise ValueError(f"{name}: no calibration site is registered for this projection")
    site = f"text/layers/{parts[2]}/{sites[role]}"
    try:
        return float(calibration[site])
    except KeyError as error:
        raise ValueError(f"{site}: the calibration carries no divisor for this site") from error


def qwen3_8_27b_nvfp4_unsloth(model, recipe, sources):
    """The unsloth-sourced line: its NVFP4 MLP codes imported, everything FP8 re-encoded.

    Measured from the checkpoint: unsloth quantizes the MLP of layers 0-55 as NVFP4 (168 matrices)
    and leaves attention, linear-attention including `in_proj_a`/`in_proj_b` and MLP 56-63 as
    row-scaled FP8 (233 matrices, with no activation scale to derive a divisor from).

    The fork's own profile for this source quantized the FP8 side to NVFP4 and kept nine BF16
    exception parents from the Qwen3.6-27B pattern. On Swift that pattern measured *worse* than
    encoding everything -- 4.7701/4.93254 against 4.68429/4.92432 -- so it is not carried here.
    `gdn/a_projection` and `gdn/b_projection` stay BF16: at (96, 5120) the NVFP4 layout cannot hold
    them, which is why the shipped profiles leave them direct.

    Divisors come from `tools/convert/qwen3_8_27b_nvfp4_calibration.json` -- measured on this exact
    BF16 checkpoint with the fork's method and corpus by `tools.convert.calibration`. A divisor is
    `2688 / max|activation|`, and a maximum does not transfer between weight realizations: borrowing
    another quantization's stored scales measured 2.2% / 0.45% worse on the fixed corpus, and the
    sites whose input is a derived intermediate (MLP down) are the ones that move most.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _nvfp4_draft(recipe, model, sources)
    quantized = sources["quantized"]
    calibration = load_calibration()
    _assign(recipe, "text/token_embedding", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or not parameter.projection:
            continue
        if name == "text/token_embedding":
            continue
        if name == "text/output_head":
            _assign(recipe, name, Q8)
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, quantized))
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        if _is_bf16_exception(layer, name.split("/")[3], name.rsplit("/", 1)[1]):
            # BF16 from the base checkpoint: this source keeps these FP8, so the direct values come
            # from `--model`, and the site takes no activation divisor because it runs at A16.
            _assign(recipe, name, "bf16")
            continue
        if "/mlp/" in name and layer < 56:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=model.source(name, quantized, "nvfp4"),
                activation_policy="AllowA4",
            )
            continue
        divisor = _calibrated_divisor(calibration, name)
        recipe.assign(
            name,
            format="nvfp4",
            method=nvfp4_maxabs,
            activation_policy="AllowA4",
        )
        for input_name in parameter.inputs:
            recipe.use(name, input_name, auxiliaries={"activation_input_divisor": divisor})
    add_proposal(recipe, source=model.source("text/output_head", quantized))


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_qat": qwen3_8_27b_nvfp4_qat,
    "qwen3_8_27b_nvfp4_swift": qwen3_8_27b_nvfp4_swift,
    "qwen3_8_27b_nvfp4_unsloth": qwen3_8_27b_nvfp4_unsloth,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
}
