# Which quantization format is "best" here (2026-10-10)

Asked as "nvfp4, exl3, autoround or something else". The question carries a category error, and NVIDIA's
own guide states the frame: a quantization *method* is a weight precision format + an activation precision
format + a calibration algorithm. NVFP4 and EXL3 are weight formats; AutoRound is a calibration algorithm
(SignRound) with exporters to several formats. They are not three points on one axis.

Sources, all read 2026-10-10: NVIDIA Model-Optimizer `_choosing_quant_methods.rst` and
`_basic_quantization.rst`; exllamav3 `README.md`, `doc/convert.md`, `doc/optimize.md`; intel/auto-round
`README.md`. **No head-to-head measurement on this model was taken**, so nothing here is a ranking.

## The three, by axis

| | NVFP4 | EXL3 | AutoRound |
|---|---|---|---|
| what it is | a weight format: E2M1 values, FP8 block scales per 16, plus a per-tensor second level | a weight format: QTIP-derived trellis, 2-8 bpw, codebook decode | a calibration algorithm (sign-gradient rounding + scale search) exporting W2/W3/W4/W8 INT, NVFP4, MXFP4, FP8 and GGUF |
| hardware path on this GPU | native FP4 tensor cores (Blackwell) | none -- decode the codebook, then FP16 MMA | none -- INT4 dequant kernels (Marlin/GPTQ-style) |
| bitrate | ~4.5 bpw with its scales | 2-8 bpw; per-tensor bitrate recipes are a first-class feature (`sc_optimize.py` allocates bits by measured KLD sensitivity) | whatever it targets |
| quality claim | near-FP8; NVIDIA's docs put QAT/QAD as the recovery path for NVFP4 | SOTA at low bitrate, inherited from QTIP, measured by their own KLD harness | "strong performance even at 2-3 bits, with leading results at 4 bits" |
| who can run it | TensorRT-LLM, vLLM, SGLang, and this engine | exllamav3 / TabbyAPI only | vLLM, SGLang, Transformers, llama.cpp -- per exported format |

## What actually decides it here

The model is dense (64 layers x 5120 hidden, `mlp.gate_proj`/`up_proj` in the artifact's own metadata; no
experts anywhere in it) and is served at batch 1, so decode is memory-bound: a decode round reads
essentially the whole language weight set -- roughly 14 GiB at 4.5 bpw (arithmetic from 27B parameters; the
artifact is 17.65 GiB in total, the remainder being the DFlash2 draft, the vision tower and unquantized
tables). Speculative decoding is what turns one such read into 2-3 emitted tokens, which is how a 27B model
reads 184-380 tok/s on this card.

That makes **bits per weight a direct decode-speed lever**, and gives each option a real case:

- **At ~4 bits the format is not the differentiator -- the calibration is.** The four NVFP4 recipes this
  port ships differ in exactly that way: QUASAR is a QAT model (train-time), NVIDIA's is ModelOpt PTQ,
  unsloth and Swift are their own lines.
- **NVFP4 is the right default here** because it is the only 4-bit format with a native hardware path on
  this GPU and the only one with kernels in this engine.
- **EXL3's case is below 4 bits**: a larger model, or more context, than 32 GiB allows at 4.5 bpw. Its cost
  is a second runtime -- nothing but exllamav3 executes it.
- **AutoRound's case is a fifth artifact**: same NVFP4 container, better PTQ calibration. That is a
  converter-integration question (its NVFP4 export goes through llm-compressor's format, which this port's
  converter has not been shown to read) plus card time, not a kernel project.
- **Named alternatives**: MXFP4 (OCP microscaling -- E2M1 with power-of-two E8M0 scales per 32, a coarser
  standard shipped by GPT-OSS), GGUF K-quants (this engine already carries q4/q5/q6/q8 kernels), FP8
  (near-lossless at 8 bits; three of the eight shipped lanes use it), and QAT/QAD (the strongest lever at
  any fixed bitrate, per NVIDIA's own guide).

## The experiment that would decide it

The cheap half: build an AutoRound NVFP4 artifact of the same base and run it through the apparatus that
already exists -- corpus perplexity, the lane fingerprint, the coding suites -- against the shipped line.
The expensive half: an EXL3 conversion needs exllamav3 to run, so that is an engine comparison rather than
an artifact comparison. Until one of them exists, "best" is a per-axis statement, not a ranking.
