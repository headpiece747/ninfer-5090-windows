"""Activation-divisor calibration for locally encoded NVFP4 sites.

An NVFP4 site that permits A4 activations needs a positive activation divisor, and it is not derivable
from the weights: it is a separate model-role value (docs/maintainer/tensor-formats.md, section 3.3).
A ModelOpt checkpoint records one per site; a compressed-tensors checkpoint does not, so a site that is
encoded locally has to be measured.

The method, the corpus and the constant are the fork's, from `calibrate_nvfp4full.py`, whose output the
published all-NVFP4 profile carries. That artifact's 247 `input_scale_divisor` values are therefore the
acceptance test for this module: the same corpus and the same arithmetic on the same weights must
reproduce them. The fork streamed one decoder layer at a time onto the GPU to bound device memory; here
`device_map="auto"` places what fits and transfers each layer once for a single batched forward pass,
which is enough because the corpus is about ten thousand characters.

    d_x = binary32(2688 / amax_site)        # 2688 = 6 * 448, the A4 block-scale orientation
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Mapping

FULL_RANGE = 2688.0
CORPUS_PATH = Path(__file__).with_name("calibration_corpus.json")

# Artifact site name -> attribute path within a decoder layer. One divisor serves a fused parent,
# because every projection in it reads the same activation: q and gate share `q_proj`, the three GDN
# input projections share `in_proj_qkv`, and `in_proj_z` reads the same hidden state again.
_SITES = (
    ("attention/input_projection", "self_attn.q_proj"),
    ("attention/output_projection", "self_attn.o_proj"),
    ("gdn/input_projection", "linear_attn.in_proj_qkv"),
    ("gdn/output_projection", "linear_attn.out_proj"),
    ("mlp/gate_up_projection", "mlp.gate_proj"),
    ("mlp/down_projection", "mlp.down_proj"),
)


def corpus_documents(path: str | Path = CORPUS_PATH) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload["documents"]
    if not documents:
        raise ValueError(f"{path}: no calibration documents")
    return list(documents)


def detect_prefix(model) -> str:
    """The attribute path under which decoder layer 0 lives.

    The checkpoint's tensor names and the module tree are different namespaces: the safetensors say
    `model.language_model.layers.N`, while `Qwen3_5ForCausalLM` exposes `model.layers.N`. Detect it
    rather than assume either.
    """
    names = dict(model.named_modules())
    for prefix in ("model.", "", "model.language_model.", "language_model."):
        if any(name.startswith(prefix + "layers.0") for name in names):
            return prefix
    raise ValueError("cannot locate decoder layer 0 in the model")


def site_map(model, prefix: str | None = None) -> dict[str, str]:
    """Artifact site name -> module path, discovered from the instantiated model rather than assumed.

    A full-attention block owns `self_attn`; a linear-attention block owns `linear_attn`. Which layers
    are which is read off the model, so no layer index is hardcoded.
    """
    modules = dict(model.named_modules())
    prefix = detect_prefix(model) if prefix is None else prefix
    config = getattr(getattr(model, "config", None), "text_config", None) or model.config
    layers = int(config.num_hidden_layers)
    sites: dict[str, str] = {}
    for layer in range(layers):
        block = f"{prefix}layers.{layer}."
        for suffix, attribute in _SITES:
            if attribute.startswith("self_attn") and f"{block}self_attn" not in modules:
                continue
            if attribute.startswith("linear_attn") and f"{block}linear_attn" not in modules:
                continue
            path = block + attribute
            if path not in modules:
                raise ValueError(f"{path}: calibration site module is missing from the model")
            sites[f"text/layers/{layer}/{suffix}"] = path
    return sites


def measure(
    base: str | Path,
    *,
    corpus: list[str] | None = None,
    device_map: str = "auto",
    max_memory: Mapping[int | str, str] | None = None,
    offload_folder: str | Path | None = None,
    prefix: str | None = None,
    progress=None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (amax per site, divisor per site) from one forward pass over the corpus.

    The weights do not fit in device memory, and how that is handled decides the numbers. Leaving it to
    `device_map="auto"` computes the layers it parks on the CPU *there*, where BF16 matmuls are emulated
    rather than native; the fork streamed each layer onto the GPU explicitly for exactly that reason.
    Here the excess goes to disk and accelerate moves each module to the execution device for its
    forward, so every matmul runs where the engine will run it.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    documents = corpus_documents() if corpus is None else corpus
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if offload_folder is None:
        kwargs = {"dtype": torch.bfloat16, "device_map": device_map, "low_cpu_mem_usage": True}
        if max_memory is not None:
            kwargs["max_memory"] = dict(max_memory)
        model = AutoModelForCausalLM.from_pretrained(str(base), **kwargs)
    else:
        # The documented disk-offload path. Accelerate places what fits on the GPU, writes the excess to
        # the folder, and moves each module to the execution device for its forward -- which is the
        # GPU-side arithmetic the fork achieved with its own per-layer loader. `force_hooks` is NOT
        # passed: it is for multi-device maps and from_pretrained forwards it to the model, which fails.
        # The CPU budget is deliberately tiny so nothing is placed there and computed in emulated BF16.
        folder = Path(offload_folder)
        folder.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "dtype": torch.bfloat16,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "offload_folder": str(folder),
        }
        if max_memory is not None:
            kwargs["max_memory"] = dict(max_memory)
        model = AutoModelForCausalLM.from_pretrained(str(base), **kwargs)
        if progress is not None:
            placement = collections.Counter(str(device) for device in getattr(model, "hf_device_map", {}).values())
            progress(f"module placement: {dict(placement)}")
    model.eval()

    sites = site_map(model, prefix)
    if progress is not None:
        progress(f"calibrating {len(sites)} sites")
    modules = dict(model.named_modules())
    peaks: dict[str, float] = {name: 0.0 for name in sites}
    handles = []

    def capture(site_name):
        def hook(module, args, kwargs=None):
            hidden = args[0] if args else (kwargs or {}).get("hidden_states")
            if hidden is None:
                raise ValueError(f"{site_name}: hook saw no activation")
            value = hidden.detach().abs().amax().item()
            if value > peaks[site_name]:
                peaks[site_name] = value
        return hook

    for site_name, path in sites.items():
        handles.append(modules[path].register_forward_pre_hook(capture(site_name), with_kwargs=True))

    joined = "\n\n".join(documents)
    encoded = tokenizer(joined, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    try:
        with torch.no_grad():
            model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(name for name, peak in peaks.items() if peak <= 0.0)
    if missing:
        raise ValueError(f"{len(missing)} sites saw no positive activation, e.g. {missing[:3]}")
    divisors = {name: FULL_RANGE / peak for name, peak in peaks.items()}
    token_count = int(encoded["input_ids"].shape[1])
    return peaks, divisors, token_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="the BF16 checkpoint to measure")
    parser.add_argument("--out", required=True, help="calibration JSON to write")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default="22GiB")
    parser.add_argument("--cpu-memory", default="2GiB",
                        help="small by design: the excess goes to --offload-folder, so every layer "
                             "computes on the GPU rather than in emulated CPU BF16")
    parser.add_argument("--offload-folder", default=None,
                        help="disk offload folder; set it to force GPU-side arithmetic")
    parser.add_argument("--prefix", default=None, help="override the detected module prefix")
    args = parser.parse_args(argv)

    peaks, divisors, tokens = measure(
        args.model,
        device_map=args.device_map,
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=args.offload_folder,
        prefix=args.prefix,
        progress=print,
    )
    result = {
        "corpus_documents": len(corpus_documents()),
        "corpus_tokens": tokens,
        "full_range": FULL_RANGE,
        "sites": {
            name: {"amax": peaks[name], "input_scale_divisor": divisors[name]}
            for name in sorted(divisors)
        },
    }
    target = Path(args.out)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}: {len(divisors)} sites over {tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
