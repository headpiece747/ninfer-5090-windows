# Qwen3.8-27B artifact reference

This reference defines both registered Qwen3.8-27B `.ninfer` storage contracts: identity, object
inventory, shapes, numeric formats, storage layouts, fused row order, aliases, fixed sources, and
source-to-object transforms. Sections 1 through 12 define the `nvfp4` profile and the DFlash2
suffix shared by both profiles; Section 13 defines the `groupwise-int` base allocation.

The nvfp4 profile is a registered Engine identity implemented by the target converter, exact
binder, and Qwen3.8 execution leaves. The generic artifact registry resolves its v3 identity
without a runtime profile flag. Common framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md), byte packing in
[`storage-layouts.md`](storage-layouts.md), and model mathematics and state behavior in
[`qwen3_5-model.md`](qwen3_5-model.md).

The converter modules, recipes and calibration inputs this reference names belong to the fork that
publishes these artifacts (`cometkim/ninfer`). This tree carries the artifact contract, not that
toolchain.

The registered line also defines a `groupwise-int` peer and a plain `nvfp4` profile, documented
here because the engine implements them. This port ships, pins and measures only the two v3 fork
artifacts of Section 14.

## 1. nvfp4 artifact identity and contents

```text
filename   = qwen3_8_27b_nvfp4.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b_nvfp4-v2
```

The current artifact is one complete image containing Text, the optimized proposal head, MTP,
Vision, DFlash2, and six frontend resources. These components are not separate artifacts or
selectable storage profiles. A runtime may choose not to materialize a supported component, but
that does not change the current artifact inventory or identity.

Earlier published artifacts with the same identity contain only the first 1124 objects and no
`dflash2/` objects. They remain valid for Text, Vision, and MTP. Selecting DFlash2 with such an
artifact reports that the DFlash2 capability is absent; no other route requires the suffix. A
current artifact contains the complete 66-object suffix. A partial suffix is malformed rather than
a third compatible inventory.

At startup, `none` and MTP do not materialize DFlash2 weights; DFlash2 does not materialize MTP
weights. Vision and DFlash2 may be resident together. The target always materializes `text/output_head`. The full proposal-head route reuses it; the
optimized route additionally materializes `text/draft_head` and `text/draft_head_token_ids`.
The Engine accepts startup-fixed `draft_tokens=1..15` (recommended 7), independently of the
checkpointâ€™s source block size. See [DFlash2 mathematics and state](dflash.md).

The identity is read from the v3 artifact directory. The filename, object count, and any
representative tensor descriptor do not select the model or weights profile.

## 2. Fixed target facts

All matrix shapes use logical `[N,K] = [output rows,input columns]` notation. nvfp4 groups and all
groupwise integer formats quantize along `K`; row-scaled FP8 owns one scale per `N` row.

| Fact | Value |
|---|---:|
| vocabulary matrix rows | 248320 |
| tokenizer-addressable IDs | 248077 (`0..248076`) |
| Text hidden width | 5120 |
| Text layers | 64 |
| Text MLP intermediate width | 17408 |
| full-attention layers | 16 |
| GDN layers | 48 |
| query heads / KV heads / head width | 24 / 4 / 256 |
| query / KV widths | 6144 / 1024 |
| GDN key heads x width | 16 x 128 = 2048 |
| GDN value heads x width | 48 x 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| MTP layers | 1 full-attention dense-MLP layer |
| optimized draft-head rows | 131072 |
| Vision depth / hidden / intermediate width | 27 / 1152 / 4304 |
| Vision heads / patch input width | 16 / 1536 |
| Vision position rows | 2304 |
| Vision merger input / output | 4608 / 5120 |
| DFlash2 architecture / source dtype | `DFlash2DraftModel / bfloat16` |
| DFlash2 layers / hidden / intermediate width | 5 / 5120 / 17408 |
| DFlash2 query heads / KV heads / head width | 32 / 8 / 128 |
| DFlash2 activation / attention bias | `silu / false` |
| DFlash2 target layers | 64 |
| DFlash2 target-feature layers | `[5,19,33,47,61]` |
| DFlash2 target-feature input width | `5 x 5120 = 25600` |
| DFlash2 source recommended block size / draft positions | 8 / 7; runtime K=1..15 |
| DFlash2 mask token id | 248070 |
| DFlash2 sliding window | 2048 |
| DFlash2 dynamic-conv taps / group size | 2 / 16 |
| DFlash2 selector rank / top-k | 256 / 16 |
| DFlash2 RoPE theta / type | `10000000 / default` |
| DFlash2 maximum positions / RMS epsilon | `262144 / 1e-6` |

DFlash2 has five non-causal sliding-attention layers. Its effective `sample_from_anchor` is false,
input-embedding scale and candidate-logit multiplier are 1.0, and final-logit softcap is disabled.

Full-attention Text layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every other Text layer is GDN. Layer and Vision-block numbers in object names are unpadded decimal
integers.

## 3. Numeric assignment and physical row order

### 3.1 Complete format assignment

The Text backbone preserves the fixed mixed-precision allocation of the quantized Text source. The
embedding is the only additional row-scaled FP8 quantization performed by NInfer.

| Role | Layer domain | Format | Layout | Value provenance |
|---|---|---|---|---|
| token embedding | global | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | encode official bf16 source |
| full-attention input projection | all full-attention layers | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| full-attention output projection | all full-attention layers | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| GDN input projection | all GDN layers | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| GDN output projection | all GDN layers | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| MLP gate/up and down | `0..55` | `nvfp4` | `block_scale_k16_m128x4_v1` | preserve source nvfp4 words |
| MLP gate/up and down | `56..63` | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| full output head | global | `fp8_e4m3fn_row_bf16` | `row_scale_v1` | preserve source FP8 words |
| Text norms, GDN convolution, and fused GDN A/B projection | applicable layers | `bf16` | `contiguous_le_v1` | preserve quantized-source bf16 words |
| GDN `A_log` and `dt_bias` | all GDN layers | `fp32` | `contiguous_le_v1` | expand quantized-source bf16 values |
| nvfp4 input divisors | `0..55` MLP sites | `fp32` | `contiguous_le_v1` | preserve source fp32 words |
| optimized proposal head | global | `q4_g64_fp16` | `row_split_k128_v1` | encode official bf16 head rows |
| optimized draft-head id map | global | `int32` | `contiguous_le_v1` | derived index tensor |
| MTP matrices | MTP | `q8_g32_fp16` | `row_split_k128_v1` | encode official bf16 source |
| MTP norms | MTP | `bf16` | `contiguous_le_v1` | preserve official bf16 words |
| Vision block input/expansion matrices | Vision | `q4_g64_fp16` | `row_split_k128_v1` | encode official bf16 source |
| Vision block output/contraction matrices | Vision | `q5_g64_fp16` | `row_split_k128_v1` | encode official bf16 source |
| Vision patch projection | Vision | `q6_g64_fp16` | `row_split_k128_v1` | encode official bf16 source |
| Vision merger matrices | Vision | `q8_g32_fp16` | `row_split_k128_v1` | encode official bf16 source |
| all other Vision weights and biases | Vision | `bf16` | `contiguous_le_v1` | preserve official bf16 words |
| DFlash2 feature, attention, and MLP matrices | DFlash2 | `q8_g32_fp16` | `row_split_k128_v1` | encode DFlash2 bf16 source |
| DFlash2 norms, dynamic-conv weights, and selector | DFlash2 | `bf16` | `contiguous_le_v1` | preserve DFlash2 bf16 words |

The selected quantized source contains 168 nvfp4 MLP matrices and 233 row-scaled FP8 matrices.
Fusing matrices at the execution-consumer boundary produces 112 nvfp4 parents and 145 FP8 parents.
The locally encoded embedding brings the artifact FP8-parent count to 146. Fusion never decodes or
requantizes a source code or scale word.

### 3.2 Fused parent row order

The artifact parent boundary follows the execution-consumer boundary. Matrices are concatenated
along output rows when they consume the same represented activation at the same semantic rounding
boundary, use the same numeric format and scale semantics, and have no intervening normalization,
nonlinearity, attention, recurrent-state transition, or residual update. The fused families below
are the exhaustive matrix-row fusions in this profile.

The following concatenations define physical output-row order:

- Text full-attention input: `[query,key,output_gate,value]`;
- Text GDN input: `[query,key,value,z]`;
- Text GDN control projection: `[A,B]`;
- Text and MTP MLP input: `[gate,up]`;
- MTP full-attention input: `[query,key,output_gate,value]`;
- DFlash2 attention input: `[query,key,value]`;
- DFlash2 MLP input: `[gate,up]`.

Within every source full-attention q-projection, each of the 24 heads stores
`[query_256,output_gate_256]`. The converter separates those per-head halves before constructing the
fused parent. For a row-scaled FP8 source, every row move applies identically to its E4M3FN code row
and its bf16 scale word.

The source GDN `in_proj_qkv` row order is `[query 2048,key 2048,value 6144]`; the converter appends
the 6144 `z` rows. The two bf16 control projections are one `a_b_projection [96,5120]` parent with
48 A rows followed by 48 B rows. Every stored GDN convolution is tap-major `[4,10240]`.

The following boundaries are intentionally not physical matrix fusions:

- attention and GDN output projections, MLP down projections, Vision attention outputs, Vision
  contractions, and merger stages consume results produced after an intervening semantic stage;
- the bf16 GDN A/B parent remains separate from the row-scaled FP8 Q/K/V/Z parent because their
  formats differ and the control branch and explicit bf16 projection input are distinct semantic
  rounding domains; the convolution remains a separate non-matrix operand;
- the MTP input projection and every Vision QKV projection are already single source matrices and
  require no row-concatenation transform;
- biases, normalization vectors, GDN parameter vectors, and nvfp4 divisors are not matrix rows;
- token embedding and output head are untied independent matrices and cannot alias or fuse.

### 3.3 Scale ownership

Each `fp8_e4m3fn_row_bf16` object is one composite weight containing a row-major E4M3FN code plane
and one bf16 multiplier per logical row. Source `.weight` and `.weight_scale [N,1]` fields therefore
become one artifact object; the source scale is not an independently named artifact tensor.

Each `nvfp4` parent contains its packed E2M1 code plane, swizzled E4M3FN K16 scale plane, and one
trailing fp32 weight divisor `d_w`. Every nvfp4 parent has one separate rank-zero fp32 input divisor
`d_x`, inserted immediately after the parent. The two objects are bound as one calibrated execution
site; neither divisor may be inferred, defaulted to `1`, or folded into rewritten scale words.

For every layer `0..55`, source gate and up projections must have bit-identical `d_w` words and
bit-identical `d_x` words before they can form one `mlp/gate_up` parent. The down projection supplies
its own pair. This yields exactly 112 input-divisor objects.

## 4. Object namespace, order, and frontend resources

### 4.1 Namespace and writer order

- `text/` contains the embedding, 64 Text layers, final norm, full output head, and optimized
  proposal head.
- `mtp/` contains MTP-private tensors.
- `vision/` contains the Vision tower and merger.
- `dflash2/` contains DFlash2-private tensors.
- `text/` and `vision/` contain the six raw frontend resources.

Objects are written in this order:

1. the six frontend resources in Section 4.2;
2. `text/token_embedding`;
3. Text layers `0..63`, using the applicable object order in Sections 5.2 through 5.4;
4. `text/final_norm` and `text/output_head`;
5. `text/draft_head` and `text/draft_head_token_ids`;
6. the twelve MTP tensors in Section 6;
7. the Vision stem, blocks `0..26`, and merger in Section 7;
8. `dflash2/feature_projection`, `dflash2/context_norm`, DFlash2 layers `0..4`,
   `dflash2/final_norm`, and the three selector objects in Section 7.4.

Readers bind by name. Object names contain no format spelling. Logical row views and aliases are
not artifact objects or directory records.

### 4.2 Frontend resources

The artifact preserves exactly these official-source files as `raw_bytes_v1` resources:

| Order | Object name | Source filename | Meaning |
|---:|---|---|---|
| 0 | `text/tokenizer.json` | `tokenizer.json` | base BPE vocabulary, merges, token bytes, and its added-token subset |
| 1 | `text/tokenizer_config.json` | `tokenizer_config.json` | complete added-token decoder, prefix, and special-token policy |
| 2 | `text/chat_template.jinja` | `chat_template.jinja` | registered Qwen template |
| 3 | `text/generation_config.json` | `generation_config.json` | default stop ids |
| 4 | `vision/preprocessor_config.json` | `preprocessor_config.json` | image preprocessing limits and constants |
| 5 | `vision/video_preprocessor_config.json` | `video_preprocessor_config.json` | video sampling and preprocessing |

## 5. Text and optimized proposal inventory

### 5.1 Text-global objects

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/token_embedding` | `[248320,5120]` | `fp8_e4m3fn_row_bf16` |
| after all layers | `text/final_norm` | `[5120]` | `bf16` |
| next | `text/output_head` | `[248320,5120]` | `fp8_e4m3fn_row_bf16` |

The embedding and output head are independent objects with different encoder provenance.

### 5.2 Full-attention Text layer

For every full-attention layer `l` in Section 2, emit these six objects before the MLP tail:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `bf16` |
| 1 | `text/layers/{l}/attention/query_key_gate_value` | `[14336,5120]` | `fp8_e4m3fn_row_bf16` |
| 2 | `text/layers/{l}/attention/query_norm` | `[256]` | `bf16` |
| 3 | `text/layers/{l}/attention/key_norm` | `[256]` | `bf16` |
| 4 | `text/layers/{l}/attention/output` | `[5120,6144]` | `fp8_e4m3fn_row_bf16` |
| 5 | `text/layers/{l}/post_attention_norm` | `[5120]` | `bf16` |

Append the MLP tail from Section 5.4. A full-attention layer in `0..55` therefore has ten physical
objects; layers 59 and 63 each have eight.

### 5.3 GDN Text layer

For every other layer `l` in `0..63`, emit these nine objects before the MLP tail:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/input_norm` | `[5120]` | `bf16` |
| 1 | `text/layers/{l}/gdn/a_log` | `[48]` | `fp32` |
| 2 | `text/layers/{l}/gdn/dt_bias` | `[48]` | `fp32` |
| 3 | `text/layers/{l}/gdn/convolution` | `[4,10240]` | `bf16` |
| 4 | `text/layers/{l}/gdn/a_b_projection` | `[96,5120]` | `bf16` |
| 5 | `text/layers/{l}/gdn/query_key_value_z` | `[16384,5120]` | `fp8_e4m3fn_row_bf16` |
| 6 | `text/layers/{l}/gdn/norm` | `[128]` | `bf16` |
| 7 | `text/layers/{l}/gdn/output` | `[5120,6144]` | `fp8_e4m3fn_row_bf16` |
| 8 | `text/layers/{l}/post_attention_norm` | `[5120]` | `bf16` |

Append the MLP tail from Section 5.4. A GDN layer in `0..55` therefore has thirteen physical
objects; each GDN layer in `56..63` has eleven.

### 5.4 Per-layer MLP tail

For layers `0..55`, append these four objects in order:

| Order within tail | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `nvfp4` |
| 1 | `text/layers/{l}/mlp/gate_up_projection/input_scale_divisor` | `[]` | `fp32` |
| 2 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `nvfp4` |
| 3 | `text/layers/{l}/mlp/down_projection/input_scale_divisor` | `[]` | `fp32` |

For layers `56..63`, append these two objects in order:

| Order within tail | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `text/layers/{l}/mlp/gate_up` | `[34816,5120]` | `fp8_e4m3fn_row_bf16` |
| 1 | `text/layers/{l}/mlp/down` | `[5120,17408]` | `fp8_e4m3fn_row_bf16` |

### 5.5 Optimized proposal head

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `text/draft_head` | `[131072,5120]` | `q4_g64_fp16` |
| 1 | `text/draft_head_token_ids` | `[131072]` | `int32` |

Row `i` of `text/draft_head` represents full-head row `text/draft_head_token_ids[i]`. The ids are
unique and lie in `0..248076`.

## 6. MTP inventory

The MTP module contains exactly twelve physical objects:

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `mtp/input_projection` | `[5120,10240]` | `q8_g32_fp16` |
| 1 | `mtp/embedding_norm` | `[5120]` | `bf16` |
| 2 | `mtp/hidden_norm` | `[5120]` | `bf16` |
| 3 | `mtp/layer/input_norm` | `[5120]` | `bf16` |
| 4 | `mtp/layer/attention/query_key_gate_value` | `[14336,5120]` | `q8_g32_fp16` |
| 5 | `mtp/layer/attention/query_norm` | `[256]` | `bf16` |
| 6 | `mtp/layer/attention/key_norm` | `[256]` | `bf16` |
| 7 | `mtp/layer/attention/output` | `[5120,6144]` | `q8_g32_fp16` |
| 8 | `mtp/layer/post_attention_norm` | `[5120]` | `bf16` |
| 9 | `mtp/layer/mlp/gate_up` | `[34816,5120]` | `q8_g32_fp16` |
| 10 | `mtp/layer/mlp/down` | `[5120,17408]` | `q8_g32_fp16` |
| 11 | `mtp/final_norm` | `[5120]` | `bf16` |

MTP token embedding, full output head, and optimized proposal head are aliases in Section 8.2.

## 7. Vision and DFlash2 inventories

### 7.1 Vision stem

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/patch_embedding` | `[1152,1536]` | `q6_g64_fp16` |
| 1 | `vision/patch_embedding_bias` | `[1152]` | `bf16` |
| 2 | `vision/position_embedding` | `[2304,1152]` | `bf16` |

### 7.2 Vision transformer block

For every block `b` in `0..26`, emit these twelve objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---|---|
| 0 | `vision/layers/{b}/attention/qkv` | `[3456,1152]` | `q4_g64_fp16` |
| 1 | `vision/layers/{b}/attention/qkv_bias` | `[3456]` | `bf16` |
| 2 | `vision/layers/{b}/attention/output` | `[1152,1152]` | `q5_g64_fp16` |
| 3 | `vision/layers/{b}/attention/output_bias` | `[1152]` | `bf16` |
| 4 | `vision/layers/{b}/mlp/fc1` | `[4304,1152]` | `q4_g64_fp16` |
| 5 | `vision/layers/{b}/mlp/fc1_bias` | `[4304]` | `bf16` |
| 6 | `vision/layers/{b}/mlp/fc2` | `[1152,4304]` | `q5_g64_fp16` |
| 7 | `vision/layers/{b}/mlp/fc2_bias` | `[1152]` | `bf16` |
| 8 | `vision/layers/{b}/norm1/weight` | `[1152]` | `bf16` |
| 9 | `vision/layers/{b}/norm1/bias` | `[1152]` | `bf16` |
| 10 | `vision/layers/{b}/norm2/weight` | `[1152]` | `bf16` |
| 11 | `vision/layers/{b}/norm2/bias` | `[1152]` | `bf16` |

### 7.3 Vision merger

| Order | Object name | Shape | Format |
|---:|---|---|---|
| 0 | `vision/merger/fc1` | `[4608,4608]` | `q8_g32_fp16` |
| 1 | `vision/merger/fc1_bias` | `[4608]` | `bf16` |
| 2 | `vision/merger/fc2` | `[5120,4608]` | `q8_g32_fp16` |
| 3 | `vision/merger/fc2_bias` | `[5120]` | `bf16` |
| 4 | `vision/merger/norm/weight` | `[1152]` | `bf16` |
| 5 | `vision/merger/norm/bias` | `[1152]` | `bf16` |

No Vision deep-stack object exists.

### 7.4 DFlash2 companion

The DFlash2 suffix starts after all 1118 base tensors. Its global objects are:

| Order | Object name | Shape | Format |
|---:|---|---:|---|
| 0 | `dflash2/feature_projection` | `[5120,25600]` | `q8_g32_fp16` |
| 1 | `dflash2/context_norm` | `[5120]` | `bf16` |

For every DFlash2 layer `l` in `0..4`, emit these twelve objects:

| Order | Object-name pattern | Shape | Format |
|---:|---|---:|---|
| 0 | `dflash2/layers/{l}/input_norm` | `[5120]` | `bf16` |
| 1 | `dflash2/layers/{l}/attention_conv/base_kernel` | `[2,2,5120]` | `bf16` |
| 2 | `dflash2/layers/{l}/attention_conv/kernel_projection` | `[1280,5120]` | `bf16` |
| 3 | `dflash2/layers/{l}/attention/query_key_value` | `[6144,5120]` | `q8_g32_fp16` |
| 4 | `dflash2/layers/{l}/attention/query_norm` | `[128]` | `bf16` |
| 5 | `dflash2/layers/{l}/attention/key_norm` | `[128]` | `bf16` |
| 6 | `dflash2/layers/{l}/attention/output` | `[5120,4096]` | `q8_g32_fp16` |
| 7 | `dflash2/layers/{l}/post_attention_norm` | `[5120]` | `bf16` |
| 8 | `dflash2/layers/{l}/mlp_conv/base_kernel` | `[2,2,5120]` | `bf16` |
| 9 | `dflash2/layers/{l}/mlp_conv/kernel_projection` | `[1280,5120]` | `bf16` |
| 10 | `dflash2/layers/{l}/mlp/gate_up` | `[34816,5120]` | `q8_g32_fp16` |
| 11 | `dflash2/layers/{l}/mlp/down` | `[5120,17408]` | `q8_g32_fp16` |

The suffix ends with:

| Order | Object name | Shape | Format |
|---:|---|---:|---|
| 0 | `dflash2/final_norm` | `[5120]` | `bf16` |
| 1 | `dflash2/candidate_selector/hidden_projection` | `[256,5120]` | `bf16` |
| 2 | `dflash2/candidate_selector/predecessor_codebook` | `[248320,256]` | `bf16` |
| 3 | `dflash2/candidate_selector/successor_codebook` | `[248320,256]` | `bf16` |

`attention_conv` and `mlp_conv` are distinct weight sets. Every `base_kernel` preserves source
axis order `[side,tap,channel]`. Every `kernel_projection` row axis is the row-major flattening of
`[side,tap,group]`, with `2 x 2 x (5120/16) = 1280` rows.

## 8. Logical views and aliases

All entries in this section are views or aliases of physical objects above. They are not extra
artifact objects.

### 8.1 Fused row views

| Parent object | Logical role | Stored row selection | Shape |
|---|---|---|---|
| `text/layers/{l}/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `text/layers/{l}/gdn/query_key_value_z` | GDN query | `[0,2048)` | `[2048,5120]` |
| same | GDN key | `[2048,4096)` | `[2048,5120]` |
| same | GDN value | `[4096,10240)` | `[6144,5120]` |
| same | GDN z | `[10240,16384)` | `[6144,5120]` |
| `text/layers/{l}/gdn/a_b_projection` | GDN A projection | `[0,48)` | `[48,5120]` |
| same | GDN B projection | `[48,96)` | `[48,5120]` |
| `text/layers/{l}/mlp/gate_up` | MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MLP up | `[17408,34816)` | `[17408,5120]` |
| `mtp/layer/attention/query_key_gate_value` | query | `[0,6144)` | `[6144,5120]` |
| same | key | `[6144,7168)` | `[1024,5120]` |
| same | output gate | `[7168,13312)` | `[6144,5120]` |
| same | value | `[13312,14336)` | `[1024,5120]` |
| `mtp/layer/mlp/gate_up` | MTP MLP gate | `[0,17408)` | `[17408,5120]` |
| same | MTP MLP up | `[17408,34816)` | `[17408,5120]` |
| `dflash2/layers/{l}/attention/query_key_value` | DFlash2 query | `[0,4096)` | `[4096,5120]` |
| same | DFlash2 key | `[4096,5120)` | `[1024,5120]` |
| same | DFlash2 value | `[5120,6144)` | `[1024,5120]` |
| `dflash2/layers/{l}/mlp/gate_up` | DFlash2 MLP gate | `[0,17408)` | `[17408,5120]` |
| same | DFlash2 MLP up | `[17408,34816)` | `[17408,5120]` |

`mtp/input_projection [5120,10240]` is a single input-column parent rather than a row-fused parent.
Columns `[0,5120)` multiply the normalized token embedding and columns `[5120,10240)` multiply the
normalized hidden state. A consumer may evaluate those two column domains and accumulate into the
same output without materializing their concatenated bf16 input.

`dflash2/feature_projection [5120,25600]` has five consecutive 5120-column domains in
target-layer order `[5,19,33,47,61]`. The runtime concatenates captured target hidden states in
that exact order; the converter preserves the source column order.

### 8.2 Aliases

| Logical consumer role | Stored object or view |
|---|---|
| MTP token embedding | `text/token_embedding` |
| MTP full output head | `text/output_head` |
| MTP optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| DFlash2 token embedding | `text/token_embedding` |
| DFlash2 mask embedding | row 248070 of `text/token_embedding` |
| DFlash2 full proposal head | `text/output_head` |
| DFlash2 optimized proposal head | `text/draft_head` plus `text/draft_head_token_ids` |
| GDN channel-major convolution | transpose view of the stored `[4,10240]` convolution |

Full-head DFlash2 top-k selection covers only tokenizer-addressable rows `0..248076`. The
optimized head first maps shortlist rows through `text/draft_head_token_ids`; the resulting global
token ids index both selector codebooks. Padded rows `248077..248319` are never candidates.

### 8.3 Binding and execution-consumer boundary

Every fused matrix parent in Sections 3.2 and 8.1 is one indivisible artifact binding unit and one
complete immutable runtime `Weight`. In particular, this identity binds Text and MTP
`query_key_gate_value`, Text `query_key_value_z`, Text `a_b_projection`, Text and MTP `gate_up`, and
Vision `qkv`, DFlash2 `query_key_value`, and DFlash2 `gate_up` as single-parent payloads. The binder
and primary projection Ops must consume those complete parents; they must not materialize their
logical row ranges as independent persistent weights or introduce a split-payload alternative for
this identity.

Logical row views exist for indexing, verification, and schedules that intentionally evaluate only
a subset of an already-bound parent. Such a schedule may derive a zero-copy view, but the view does
not change parent ownership or replace the complete-parent Op boundary. A complete-parent
projection may write its independently laid-out logical outputs directly and need not materialize a
packed output tensor.

Separate artifact objects do not by themselves require separate semantic Ops or kernel launches.
Biases, normalization vectors, GDN parameters, convolution, and the differently formatted GDN
parents may remain separate immutable arguments to a larger fused Op whenever that Op preserves the
model's semantic boundaries.

## 9. Inventory summary

### 9.1 Object counts

| Component | Derivation | Tensor objects |
|---|---:|---:|
| Text globals | embedding + final norm + full head | 3 |
| full-attention Text layers | `14 x 10 + 2 x 8` | 156 |
| GDN Text layers | `42 x 13 + 6 x 11` | 612 |
| main Text excluding draft | `3 + 156 + 612` | 771 |
| optimized proposal head | weight + id map | 2 |
| MTP | fixed inventory | 12 |
| Vision stem | fixed inventory | 3 |
| Vision blocks | `27 x 12` | 324 |
| Vision merger | fixed inventory | 6 |
| Vision total | `3 + 324 + 6` | 333 |
| DFlash2 globals | feature projection + context norm | 2 |
| DFlash2 layers | `5 x 12` | 60 |
| DFlash2 final norm and selector | fixed inventory | 4 |
| DFlash2 total | `2 + 60 + 4` | 66 |
| all tensors | `771 + 2 + 12 + 333 + 66` | 1184 |
| frontend resources | fixed inventory | 6 |
| complete artifact | `1184 + 6` | 1190 |

### 9.2 Numeric-format counts

| Format | Text | draft | MTP | Vision | DFlash2 | Total |
|---|---:|---:|---:|---:|---:|---:|
| `bf16` | 305 | 0 | 7 | 222 | 45 | 579 |
| `fp32` | 208 | 0 | 0 | 0 | 0 | 208 |
| `int32` | 0 | 1 | 0 | 0 | 0 | 1 |
| `q4_g64_fp16` | 0 | 1 | 0 | 54 | 0 | 55 |
| `q5_g64_fp16` | 0 | 0 | 0 | 54 | 0 | 54 |
| `q6_g64_fp16` | 0 | 0 | 0 | 1 | 0 | 1 |
| `q8_g32_fp16` | 0 | 0 | 5 | 2 | 21 | 28 |
| `nvfp4` | 112 | 0 | 0 | 0 | 0 | 112 |
| `fp8_e4m3fn_row_bf16` | 146 | 0 | 0 | 0 | 0 | 146 |
| total | 771 | 2 | 12 | 333 | 66 | 1184 |

The artifact contains 788 direct tensors using `contiguous_le_v1`, 138 grouped integer tensors
using `row_split_k128_v1`, 112 nvfp4 tensors using `block_scale_k16_m128x4_v1`, and 146 row-scaled
FP8 tensors using `row_scale_v1`.

## 10. Fixed sources and numeric conversion

### 10.1 Source identities and ownership

The official base source is `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The quantized Text source is
`unsloth/Qwen3.8-27B-nvfp4` revision
`60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`. The DFlash2 source is
`z-lab/Qwen3.8-27B-DFlash2` revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

DFlash2 `config.json`, README, and source revision are converter inputs or provenance, not artifact
resources. The only embedded resources remain the six frontend objects in Section 4.2.

| Artifact content | Materialization source |
|---|---|
| Text FP8 and nvfp4 matrices | fixed quantized Text source |
| Text norms, GDN convolution, fused GDN A/B projection, `A_log`, and `dt_bias` | fixed quantized Text source |
| token embedding | official bf16 source |
| optimized draft head and id map | official bf16 source plus fixed ranking input |
| MTP | official bf16 source |
| Vision | official bf16 source |
| six frontend resources | official source |
| DFlash2 | fixed DFlash2 bf16 source |

The quantized source's Vision tensors, MTP tensors, frontend files, and bf16 embedding are not
materialization inputs. Its full-attention `k_scale` and `v_scale` fields are also excluded: they
are not fields of a persistent weight format and do not become artifact tensors. Runtime activation,
KV-cache, and recurrent-state codecs remain outside this artifact contract.

### 10.2 Direct tensors

- Artifact `bf16` objects preserve the selected source bf16 words after the stated concatenate,
  reshape, or transpose.
- GDN `A_log` and `dt_bias` expand selected-source bf16 values exactly to binary32 and store the
  resulting fp32 words.
- Native nvfp4 input-divisor fp32 words are preserved exactly while changing source shape `[1]` to
  artifact shape `[]`.
- `text/draft_head_token_ids` is stored as `int32`.

DFlash2 dynamic-conv weights, selector weights, and norms preserve source bf16 words directly.

No other source bf16 field is promoted to fp32.

### 10.3 Preserved row-scaled FP8

For every source-derived FP8 matrix, the converter validates an E4M3FN `.weight [N,K]` and a bf16
`.weight_scale [N,1]`. It preserves all code and scale words exactly. Splitting, row permutation,
and concatenation operate on `(code row, scale word)` pairs. The resulting logical words are encoded
with `row_scale_v1`; no floating-point decode, scale recomputation, or requantization is permitted.

### 10.4 Embedding FP8 encoder

Only `text/token_embedding` uses the target-specific encoder profile
`MAXABS_bf16S_RECIP_E4M3FN_RNE_V1`. The input is the official bf16 matrix. Every bf16 word expands
exactly to binary32, and each row is encoded independently. All operations below use
round-to-nearest, ties-to-even, with no flush-to-zero:

```text
amax = max_k(abs(x[k]))

if amax == +0:
    scale_bf16 = bfloat16(+0)
    code[k]    = E4M3FN(+0) for every k
else:
    raw_scale32 = round_f32(amax / binary32(448))
    scale_bf16  = round_bf16(raw_scale32)

    if scale_bf16 == +0:
        scale_bf16 = bfloat16_from_bits(0x0001)

    scale32 = exact_bfloat16_to_binary32(scale_bf16)
    inv32   = round_f32(binary32(1) / scale32)

    for every k:
        normalized32 = round_f32(x[k] * inv32)
        bounded32     = clamp(normalized32, -448, 448)
        code[k]       = round_e4m3fn_rne(bounded32)
```

The producer rejects a non-finite source value or a non-positive/non-finite scale for a nonzero
row. This encoder profile applies only to the embedding; it makes no claim about how the upstream
FP8 source selected its words.

### 10.5 Preserved nvfp4

For each source nvfp4 matrix, the converter validates:

```text
weight_packed       U8        [N,K/2]
weight_scale        E4M3FN    [N,K/16]
weight_global_scale fp32      [1]
input_global_scale  fp32      [1]
```

The source `weight_global_scale` is the positive divisor `d_w` in the registered nvfp4 decode
formula, and `input_global_scale` is the positive site divisor `d_x`. The converter copies packed
E2M1 words without decoding, permutes the natural scale matrix into
`block_scale_k16_m128x4_v1`, and preserves both fp32 words exactly. Gate/up fusion concatenates rows
only after the equality checks in Section 3.3.

### 10.6 Grouped integer conversion

Draft-head, MTP, Vision, and DFlash2 matrices first complete the specified split, concatenate,
reshape, or transpose as one contiguous bf16 logical matrix. They then apply
`MAXABS_F16_RECIP_RNE_V1` independently to every output row and G32 or G64 group along `K`, and use
`row_split_k128_v1`. Groups never cross output rows.

## 11. Text source mapping

Let:

```text
P(l) = model.language_model.layers.{l}.
```

### 11.1 Text globals

| Artifact object | Source | Transform |
|---|---|---|
| `text/token_embedding` | official `model.language_model.embed_tokens.weight [248320,5120]` bf16 | apply Section 10.4 |
| `text/final_norm` | quantized-source `model.language_model.norm.weight [5120]` bf16 | preserve words |
| `text/output_head` | quantized-source `lm_head.weight [248320,5120]` E4M3FN and `weight_scale [248320,1]` bf16 | preserve words |

### 11.2 Full-attention source transform

The quantized source q-projection is `[12288,5120]` with per-head
`[query_256,output_gate_256]` rows:

```text
qg    = q_proj.reshape(24,512,5120)
query = qg[:,0:256,:].reshape(6144,5120)
gate  = qg[:,256:512,:].reshape(6144,5120)
```

The same reshape and selection apply to its `[12288,1]` row-scale tensor.

| Artifact suffix under `text/layers/{l}/` | Quantized source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` bf16 | preserve words |
| `attention/query_key_gate_value` | `self_attn.{q,k,v}_proj.weight` and their row scales | form `[query,key,output_gate,value]`, preserve FP8 words |
| `attention/query_norm` | `self_attn.q_norm.weight [256]` bf16 | preserve words |
| `attention/key_norm` | `self_attn.k_norm.weight [256]` bf16 | preserve words |
| `attention/output` | `self_attn.o_proj.weight [5120,6144]` and row scales | preserve FP8 words |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` bf16 | preserve words |

The MLP tail follows Section 11.4.

### 11.3 GDN source transform

The quantized source `linear_attn.in_proj_qkv.weight [10240,5120]` row order is:

```text
query = [0,2048)
key   = [2048,4096)
value = [4096,10240)
```

| Artifact suffix under `text/layers/{l}/` | Quantized source under `P(l)` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` bf16 | preserve words |
| `gdn/a_log` | `linear_attn.A_log [48]` bf16 | exact-expand to fp32 |
| `gdn/dt_bias` | `linear_attn.dt_bias [48]` bf16 | exact-expand to fp32 |
| `gdn/convolution` | `linear_attn.conv1d.weight [10240,1,4]` bf16 | take `[:,0,:]`, transpose to `[4,10240]` |
| `gdn/a_b_projection` | `linear_attn.in_proj_a.weight`, `in_proj_b.weight`, each `[48,5120]` bf16 | concatenate `[A,B]`, preserve words |
| `gdn/query_key_value_z` | `linear_attn.in_proj_qkv.weight`, `in_proj_z.weight [6144,5120]`, and their row scales | form `[query,key,value,z]`, preserve FP8 words |
| `gdn/norm` | `linear_attn.norm.weight [128]` bf16 | preserve words |
| `gdn/output` | `linear_attn.out_proj.weight [5120,6144]` and row scales | preserve FP8 words |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` bf16 | preserve words |

The MLP tail follows Section 11.4.

### 11.4 MLP source transform

For every layer `0..55`, source `mlp.gate_proj` and `mlp.up_proj` each have logical shape
`[17408,5120]`. Concatenate their packed-code and natural scale rows as `[gate,up]` to form
`mlp/gate_up [34816,5120]`; preserve their shared `d_w` in the parent and shared `d_x` in the
following scalar. Map `mlp.down_proj [5120,17408]` to `mlp/down` and its following scalar without a
row transform.

For every layer `56..63`, perform the same `[gate,up]` row concatenation on the source E4M3FN code
rows and bf16 row scales. Map the row-scaled FP8 down projection without a row transform. These
layers have no persistent input-divisor object.

### 11.5 Optimized MTP draft-head construction

The fixed ranking source is:

```text
tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
```

Interpret the file as a little-endian I64 array with row width 248320 and use row 0. Padded rows
`248077..248319` are not candidates. Select ids from `0..248076` as follows:

1. form the ascending forced-id set from entries whose merged
   `added_tokens_decoder[id].special` value is true;
2. stable-sort all candidate ids by descending row-0 count, with ascending-id count ties;
3. take the first `131072 - len(forced)` non-forced ids;
4. append the ascending forced ids and stable-sort the result by descending count;
5. require exactly 131072 unique ids in `0..248076`;
6. store those ids as `int32` in `text/draft_head_token_ids`;
7. gather the same ordered bf16 rows from the official `lm_head.weight` and quantize them to Q4.

## 12. MTP, Vision, and conformance

### 12.1 MTP source mapping

Let `M = mtp.layers.0.` in the official source. Apply the full-attention q/gate extraction from
Section 11.2.

| Artifact object | Official source | Transform |
|---|---|---|
| `mtp/input_projection` | `mtp.fc.weight [5120,10240]` | quantize W8 |
| `mtp/embedding_norm` | `mtp.pre_fc_norm_embedding.weight [5120]` | preserve bf16 |
| `mtp/hidden_norm` | `mtp.pre_fc_norm_hidden.weight [5120]` | preserve bf16 |
| `mtp/layer/input_norm` | `M + input_layernorm.weight [5120]` | preserve bf16 |
| `mtp/layer/attention/query_key_gate_value` | `M + self_attn.{q,k,v}_proj.weight` | form `[query,key,output_gate,value]`, quantize W8 |
| `mtp/layer/attention/query_norm` | `M + self_attn.q_norm.weight [256]` | preserve bf16 |
| `mtp/layer/attention/key_norm` | `M + self_attn.k_norm.weight [256]` | preserve bf16 |
| `mtp/layer/attention/output` | `M + self_attn.o_proj.weight [5120,6144]` | quantize W8 |
| `mtp/layer/post_attention_norm` | `M + post_attention_layernorm.weight [5120]` | preserve bf16 |
| `mtp/layer/mlp/gate_up` | `M + mlp.gate_proj.weight`, `up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize W8 |
| `mtp/layer/mlp/down` | `M + mlp.down_proj.weight [5120,17408]` | quantize W8 |
| `mtp/final_norm` | `mtp.norm.weight [5120]` | preserve bf16 |

### 12.2 Vision source mapping

All sources in this section begin with official-source `model.visual.`.

| Artifact object | Source suffix | Transform |
|---|---|---|
| `vision/patch_embedding` | `patch_embed.proj.weight [1152,3,2,16,16]` | contiguous reshape to `[1152,1536]`, quantize Q6 |
| `vision/patch_embedding_bias` | `patch_embed.proj.bias [1152]` | preserve bf16 |
| `vision/position_embedding` | `pos_embed.weight [2304,1152]` | preserve bf16 |

For block `b`, use source prefix `model.visual.blocks.{b}.`:

| Artifact suffix under `vision/layers/{b}/` | Source suffix | Transform |
|---|---|---|
| `attention/qkv` | `attn.qkv.weight [3456,1152]` | quantize Q4 |
| `attention/qkv_bias` | `attn.qkv.bias [3456]` | preserve bf16 |
| `attention/output` | `attn.proj.weight [1152,1152]` | quantize Q5 |
| `attention/output_bias` | `attn.proj.bias [1152]` | preserve bf16 |
| `mlp/fc1` | `mlp.linear_fc1.weight [4304,1152]` | quantize Q4 |
| `mlp/fc1_bias` | `mlp.linear_fc1.bias [4304]` | preserve bf16 |
| `mlp/fc2` | `mlp.linear_fc2.weight [1152,4304]` | quantize Q5 |
| `mlp/fc2_bias` | `mlp.linear_fc2.bias [1152]` | preserve bf16 |
| `norm1/weight` | `norm1.weight [1152]` | preserve bf16 |
| `norm1/bias` | `norm1.bias [1152]` | preserve bf16 |
| `norm2/weight` | `norm2.weight [1152]` | preserve bf16 |
| `norm2/bias` | `norm2.bias [1152]` | preserve bf16 |

For source prefix `model.visual.merger.`:

| Artifact suffix under `vision/merger/` | Source suffix | Transform |
|---|---|---|
| `fc1` | `linear_fc1.weight [4608,4608]` | quantize W8 |
| `fc1_bias` | `linear_fc1.bias [4608]` | preserve bf16 |
| `fc2` | `linear_fc2.weight [5120,4608]` | quantize W8 |
| `fc2_bias` | `linear_fc2.bias [5120]` | preserve bf16 |
| `norm/weight` | `norm.weight [1152]` | preserve bf16 |
| `norm/bias` | `norm.bias [1152]` | preserve bf16 |

### 12.3 DFlash2 source mapping

Each converter reads only DFlash2 `config.json` and the single-file `model.safetensors` from the
fixed source in Section 10.1. The safetensors file must contain exactly the 81 declared bf16 source
tensors: no missing, extra, differently shaped, or differently typed tensor is accepted.

| Artifact object | DFlash2 source | Transform |
|---|---|---|
| `dflash2/feature_projection` | `fc.weight [5120,25600]` | quantize W8 |
| `dflash2/context_norm` | `hidden_norm.weight [5120]` | preserve bf16 |
| `dflash2/final_norm` | `norm.weight [5120]` | preserve bf16 |
| `dflash2/candidate_selector/hidden_projection` | `candidate_selector.hidden_projection.weight [256,5120]` | preserve bf16 |
| `dflash2/candidate_selector/predecessor_codebook` | `candidate_selector.predecessor_codebook [248320,256]` | preserve bf16 |
| `dflash2/candidate_selector/successor_codebook` | `candidate_selector.successor_codebook [248320,256]` | preserve bf16 |

For each layer `l` in `0..4`, let `S = layers.{l}.` and
`O = dflash2/layers/{l}/`:

| Artifact suffix under `O` | Source under `S` | Transform |
|---|---|---|
| `input_norm` | `input_layernorm.weight [5120]` | preserve bf16 |
| `attention_conv/base_kernel` | `attention_conv.base_kernel [2,2,5120]` | preserve bf16 |
| `attention_conv/kernel_projection` | `attention_conv.kernel_projection.weight [1280,5120]` | preserve bf16 |
| `attention/query_key_value` | `self_attn.q_proj.weight [4096,5120]`, `k_proj.weight [1024,5120]`, `v_proj.weight [1024,5120]` | concatenate `[q,k,v]`, quantize W8 |
| `attention/query_norm` | `self_attn.q_norm.weight [128]` | preserve bf16 |
| `attention/key_norm` | `self_attn.k_norm.weight [128]` | preserve bf16 |
| `attention/output` | `self_attn.o_proj.weight [5120,4096]` | quantize W8 |
| `post_attention_norm` | `post_attention_layernorm.weight [5120]` | preserve bf16 |
| `mlp_conv/base_kernel` | `mlp_conv.base_kernel [2,2,5120]` | preserve bf16 |
| `mlp_conv/kernel_projection` | `mlp_conv.kernel_projection.weight [1280,5120]` | preserve bf16 |
| `mlp/gate_up` | `mlp.gate_proj.weight`, `mlp.up_proj.weight`, each `[17408,5120]` | concatenate `[gate,up]`, quantize W8 |
| `mlp/down` | `mlp.down_proj.weight [5120,17408]` | quantize W8 |

The feature-projection input columns retain target-layer order `[5,19,33,47,61]`. Q/K/V and
gate/up are concatenated as complete bf16 logical matrices before one W8 encoding; separately
quantized code/scale planes are never concatenated.

### 12.4 Producer requirements

Before opening the output, the converter must validate all fixed checkpoint configurations, every
selected source name, shape, dtype, format assignment, source scale geometry, frontend resource,
ranking input, and the complete ordered object plan. The DFlash2 config must match the fixed facts
in Section 2 and the base hidden width, vocabulary, layer count, maximum positions, and RoPE theta.

During materialization, the converter must:

- preserve every source-derived E4M3FN code and bf16 row-scale word exactly after the defined
  split, permutation, and fusion;
- preserve every nvfp4 packed code, natural scale word, `d_w`, and `d_x` exactly, including
  all gate/up equality requirements;
- preserve direct words exactly and perform the specified bf16-to-fp32 expansions;
- encode the embedding with the bit-level profile in Section 10.4;
- encode draft-head, MTP, and Vision weights with `MAXABS_F16_RECIP_RNE_V1`;
- preserve or encode DFlash2 weights according to Section 3.1;
- write the complete object inventory and six resource payloads in the order required by the
  documented logical views and aliases.

Validation must reject an incomplete or alternate mixed-precision allocation. It must not fill a
missing source matrix from the official bf16 checkpoint, silently requantize a preserved FP8 or
nvfp4 field, or add unused source calibration fields as artifact objects.

The canonical nvfp4 conversion entry point is:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --model /path/to/Qwen3.8-27B/base-hf-bf16 \
  --quantized-model /path/to/Qwen3.8-27B/vllm-nvfp4-fp8 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b_nvfp4.ninfer \
  --device cuda
```

## 13. `groupwise-int` peer artifact

The existing registered peer artifact retains this identity:

```text
filename   = qwen3_8_27b.ninfer
model_id   = qwen3.8-27b
weights_id = groupwise-int
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b-v2
```

The current artifact contains the same 66-object DFlash2 suffix, 1118 base tensors, and six
resources. Its base inventory, logical row views, aliases, and writer order are defined by
`tools/convert/qwen3_8_27b/inventory.py`; DFlash2 follows Sections 2 through 12. Its complete format
counts are:

| Format | Tensors |
|---|---:|
| `bf16` | 627 |
| `fp32` | 96 |
| `int32` | 1 |
| `q4_g64_fp16` | 183 |
| `q5_g64_fp16` | 246 |
| `q6_g64_fp16` | 1 |
| `q8_g32_fp16` | 30 |
| total | 1184 |

The complete groupwise artifact has 724 direct tensors using `contiguous_le_v1` and 460 grouped
integer tensors using `row_split_k128_v1`.

`text/token_embedding [248320,5120]` and `text/output_head [248320,5120]` use `q8_g32_fp16`; Text
layers use the registered Q4/Q5 allocation; the optimized draft head uses Q4; MTP matrices and
the Vision merger use W8; and the Vision patch projection uses Q6. All groupwise integer tensors
use `MAXABS_F16_RECIP_RNE_V1` with `row_split_k128_v1`. Base tensors come solely from the official
source revision in Section 10.1; DFlash2 tensors come from the fixed companion source. The artifact
binds through the `Qwen38GroupwiseInt` profile, and its registered nvfp4 peer binds through
`Qwen38Nvfp4`.

Its canonical conversion entry point remains:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b.ninfer \
  --device cuda
```

The converter validates the official and DFlash2 checkpoints, frontend resources, complete object
plan, and numeric recipes before opening the output, then writes the sibling
`qwen3_8_27b.ninfer.conversion.json` report.


## 14. Fork artifact: `nvfp4full`

This fork additionally builds a fuller-nvfp4 Qwen3.8-27B artifact with the memory profile of the
Qwen3.6 nvfp4 recipe, carrying the DFlash2 companion bundle of Section 7 in the same complete
image â€” byte-identical module objects and the same q8_g32_fp16/bf16 suffix contract as the two
registered profiles. It is produced and verified by the fork-local tools
`tools.convert.qwen3_8_27b.{nvfp4_encode, calibrate_nvfp4full, convert_nvfp4full, verify_nvfp4full}`
and binds through the same registered target as an additional weights contract.

Those tools are not part of this port's tree: `tools/convert/` here carries `quantization/` and
`sources/` only. The commands below describe the producing fork, not something this checkout can
run.

### 14.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4full.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4full
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b_nvfp4full-v3
converter  = tools.convert.qwen3_8_27b.convert_nvfp4full
```

The artifact contains 1319 tensors and the same six frontend resources (1325 objects), including
the 66-object DFlash2 companion bundle of Section 7. Its Text
allocation applies the Qwen3.6-27B nvfp4 exception pattern to this checkpoint and object naming:

- `attention/query_key_gate_value` is `bf16` on layers `3,7,11,15,19,23` and `nvfp4` on the other
  ten full-attention layers;
- `attention/output` is `bf16` on layers `3,7` and `nvfp4` on the other fourteen;
- every GDN layer has one nvfp4 `gdn/query_key_value_z` parent; `gdn/output` is `bf16` only on
  layer `4` and nvfp4 on the other 47;
- every Text `mlp/gate_up` and `mlp/down` is nvfp4;
- `text/token_embedding` and `text/output_head` use `q8_g32_fp16` with `MAXABS_F16_RECIP_RNE_V1`;
- MTP, Vision, and the optimized draft head keep the registered formats of Section 13.

This yields 247 nvfp4 parents, 247 site-level fp32 input divisors, and nine bf16 exception
parents, plus the 21 `q8_g32_fp16` DFlash2 module matrices.

| Format | Tensors |
|---|---:|
| `bf16` | 588 |
| `fp32` | 343 |
| `int32` | 1 |
| `q4_g64_fp16` | 55 |
| `q5_g64_fp16` | 54 |
| `q6_g64_fp16` | 1 |
| `q8_g32_fp16` | 30 |
| `nvfp4` | 247 |

| Layout | Tensors |
|---|---:|
| `contiguous_le_v1` | 932 |
| `row_split_k128_v1` | 140 |
| `block_scale_k16_m128x4_v1` | 247 |

### 14.2 Sources and provenance

The base source is `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and owns
every locally quantized nvfp4 parent, every bf16 exception, all direct tensors, both W8 endpoints,
and the draft head, MTP, Vision, and frontend components. The quantized Text source is
`unsloth/Qwen3.8-27B-nvfp4`: the document-pinned revision `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`
was force-pushed out of the upstream repository, so the converter cites the reachable `main`
revision `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` and structurally validates the exact
mixed-precision allocation (168 nvfp4 MLP matrices with `weight_packed`/`weight_scale`/
`weight_global_scale`/`input_global_scale` fields and 233 row-scaled FP8 matrices) before any word
is copied. Its only artifact inputs are the 112 MLP nvfp4 parents of layers `0..55` and their
divisors, copied bit-exactly with the same gate/up shared-divisor equality checks as the registered
profile. The upstream single 22.5 GiB shard is repacked into bounded shards by
`tools.convert.qwen3_8_27b.split_quantized_shards` without touching a value, name, dtype, or shape.
The DFlash2 companion source is the fixed `z-lab/Qwen3.8-27B-DFlash2` revision of Section 7,
encoded through the same `dflash2_inventory`/`dflash2_recipe` machinery as the registered
profiles.

### 14.3 Local encoder profile and site divisors

Locally quantized parents use the documented encoder profile `nvfp4_MAXABS_DIVISOR_RNE_V1`
(`tools/convert/qwen3_8_27b/nvfp4_encode.py`): with `amax = max|W|` over the complete parent and
`2688 = 6 x 448`,

```text
d_w = binary32(2688 / amax)            # 1 when the parent is all zero
y   = binary32(W * d_w)
per K-group of 16:
    s  = E4M3FN(min(max|y| in group / 6, 448))   # RNE, +0 for an all-zero group
    q  = E2M1(y / decode(s))                     # RNE ties-to-even, saturating at +-6
```

Codes and scales are bit-level verified against exhaustive E2M1/E4M3FN decode tables and the
`block_scale_k16_m128x4_v1` writer is byte-for-byte validated against rdtand-sourced objects in the
official Qwen3.6 nvfp4 artifact before production use. Against the bf16 source, locally quantized
parents measure at most 0.0951 relative Frobenius error; the same metric on the unsloth-copied
parents measures 0.107-0.126, so the local profile is at least as accurate per tensor.

Site input divisors for the 135 locally quantized parents come from the fixed calibration
`models/qwen3_8_27b_nvfp4full_calibration.json`: `d_x = binary32(2688 / max|site input|)` over a
committed ten-document corpus (2,690 tokens) evaluated by streaming the official bf16 checkpoint
layer-by-layer through the GPU. As a derivation check the same statistic measured on the 56
unsloth-owned `mlp/gate_up` sites reproduces the amax implied by their stored `input_global_scale`
words with median ratio 0.96 and maximum 1.00, matching the upstream per-tensor-amax derivation.

### 14.4 Production and verification

```bash
python3 -m tools.convert.qwen3_8_27b.calibrate_nvfp4full \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-nvfp4 \
  --out models/qwen3_8_27b_nvfp4full_calibration.json
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4full \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-nvfp4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --calibration models/qwen3_8_27b_nvfp4full_calibration.json \
  --out models/qwen3_8_27b_nvfp4full.ninfer
python3 -m tools.convert.qwen3_8_27b.verify_nvfp4full models/qwen3_8_27b_nvfp4full.ninfer \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-nvfp4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --calibration models/qwen3_8_27b_nvfp4full_calibration.json
```

`verify_nvfp4full` revalidates the complete ordered directory, both W8 endpoints against base rows,
all 112 source nvfp4 payloads word-for-word, all 135 local payloads against both the encoder profile
and the independent decode oracle, all 247 input-divisor words, the nine bf16 exception parents,
all 66 DFlash2 module objects against a reference re-encode of the companion source, and the six
resources.

### 14.5 Measured results (RTX 5090)

The v3 image re-encodes only the DFlash2 companion bundle (q8_g32_fp16) of the earlier v2 build;
the Text weights are unchanged, so the Text-side measurements below carry over, while DFlash2-lane
cells were measured on the superseded nvfp4-module image and need a refresh on the v3 artifact.

| Measurement | `nvfp4full` | official `nvfp4` |
|---|---:|---:|
| device weights | 16.03 GiB | 18.98 GiB |
| free after startup, INT8 KV @ 262,144 | 4.91 GiB | 2.22 GiB |
| MTP3 decode tok/s (8,192-token context, greedy) | 134.7 | 118.9 |
| prefill tok/s (same run) | 913 | 703 |

Greedy MTP3 acceptance was 156/256 tokens (positions 77/49/30) versus 152/256 (78/44/30) for the
official artifact. On GPQA-Diamond under the registered serving profile (thinking, MTP=3, INT8 KV,
262,144-token context; EvalScope 1.9.0, 0-shot, rule scoring, one sample, temperature 0.6, seed 42)
the artifact scores **89.39% (177 / 198)** against the official nvfp4 artifact's currently
published 90.40% (179 / 198) - a two-question difference on single-sample runs, within the
~2.1% sampling error at n=198; the 198-sample run averaged 129.3 tok/s with 11,927 output
tokens per question.


## 15. Fork artifact: `nvfp4qat`

This fork additionally builds a QAT-sourced Qwen3.8-27B artifact: the text weight stack is copied
word-for-word from the QUASAR quantization-aware-trained nvfp4 checkpoint, replacing both the
locally quantized parents and the nine bf16 exception parents of `nvfp4full` (Section 14). It is
produced and verified by the fork-local tools
`tools.convert.qwen3_8_27b.{convert_nvfp4qat, verify_nvfp4qat}` and binds through the same
registered target as an additional weights contract. It carries the upstream DFlash2 companion
bundle of Section 7 (66 objects, matrices `q8_g32_fp16`) in the same complete image.

### 15.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4qat.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4qat
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b_nvfp4qat-v2
converter  = tools.convert.qwen3_8_27b.convert_nvfp4qat
```

The artifact contains 1328 tensors and the same six frontend resources (1334 objects), including
the DFlash2 companion bundle of Section 7. Its Text allocation has no bf16 exception parents:
every `attention/query_key_gate_value`, `attention/output`, `gdn/query_key_value_z`, `gdn/output`,
`mlp/gate_up`, and `mlp/down` is nvfp4 â€” 256 parents with 256 site-level fp32 input divisors.
The token embedding and full output head keep `q8_g32_fp16`; MTP, Vision, and the optimized
draft head keep the registered formats of Section 13.

| Format | Tensors |
|---|---:|
| `bf16` | 579 |
| `fp32` | 352 |
| `int32` | 1 |
| `q4_g64_fp16` | 55 |
| `q5_g64_fp16` | 54 |
| `q6_g64_fp16` | 1 |
| `q8_g32_fp16` | 30 |
| `nvfp4` | 256 |

| Layout | Tensors |
|---|---:|
| `contiguous_le_v1` | 932 |
| `row_split_k128_v1` | 140 |
| `block_scale_k16_m128x4_v1` | 256 |

### 15.2 Sources and provenance

The quantized source is `QUASAR-QAT/Qwen3.8-27B-QUASAR-nvfp4` revision
`d8e6fbfa3e3a78899b440222b827430045a05b44` (compressed-tensors `nvfp4-pack-quantized`, group 16,
E4M3 scale words): one epoch of loss-aware nvfp4 quantization-aware distillation against the
frozen bf16 teacher (QUASAR, arXiv 2608.13966). Every text linear is quantized there â€” 496
source sites â€” under one `weight_global_scale`/`input_global_scale` pair per quantization site,
shared by every constituent tensor of a fused parent; the converter enforces that sharing through
the same-divisor checks before any word is copied. Its only artifact inputs are the 256 fused
nvfp4 parents' packed-code and scale words (copied bit-exactly through the Section 14 row
transforms, including the attention q/gate per-head interleaving), the 256 site input divisors,
and the GDN control `in_proj_a`/`in_proj_b` words decoded to the bf16 `gdn/a_b_projection`.

Every other source is the official base `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` exactly as in Section 14: all direct tensors, both W8
endpoints, the draft head, MTP, and Vision. The converter proves this routing complete at
preflight by byte-comparing every unquantized QAT tensor against the official source (703 tensors,
bit-identical; the QAT export's `linear_attn.convNd` maps to the official `linear_attn.conv1d`),
so the QAT checkpoint differs from the official source only in its 496 quantized text linears.
The DFlash2 companion source is the fixed `z-lab/Qwen3.8-27B-DFlash2` revision of Section 7,
encoded through the same `dflash2_inventory`/`dflash2_recipe` machinery as the registered
profiles. No local encoder run and no calibration corpus are involved.

### 15.3 Production and verification

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4qat \
  --model /path/to/Qwen3.8-27B \
  --quantized-model /path/to/Qwen3.8-27B-QUASAR-nvfp4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out models/qwen3_8_27b_nvfp4qat.ninfer
python3 -m tools.convert.qwen3_8_27b.verify_nvfp4qat models/qwen3_8_27b_nvfp4qat.ninfer \
  --model /path/to/Qwen3.8-27B \
  --quantized-model /path/to/Qwen3.8-27B-QUASAR-nvfp4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2
```

`verify_nvfp4qat` revalidates the complete ordered directory, both W8 endpoints against base
rows, all 256 QAT payloads word-for-word, all 256 input divisors against the source
`input_global_scale` words, the 48 decoded control parents against the independent decode oracle,
all 66 DFlash2 module objects against a reference re-encode of the companion source, the six
resources, and the weight-divisor derivation cross-check `d_w = binary32(2688/amax)`:
single-tensor site families match exactly and the residual envelope (median exactly 1.0, maximum
1.1667) is explained by site-scale sharing with the decoded control tensors and one saturating
E2M1 step at a site's top element.

## 16. Fork artifact: `nvfp4swift`

This fork builds an artifact from UkisAI's Swift-Qwen3.8-27B, a reasoning-efficient fine-tune of
Qwen3.8-27B distributed in NVIDIA ModelOpt's mixed NVFP4/FP8 quantization. It is the first artifact
here whose source keeps its full-attention and GDN projections in E4M3 instead of NVFP4, so its text
stack is part NVFP4 imported from that source and part NVFP4 encoded locally from the fine-tune's
BF16 export. It ships the two lanes of Section 4 like the others; its public file pin lands with its
publication.

### 16.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4swift.v3.ninfer
name       = qwen3.8-27b
recipe     = qwen3_8_27b_nvfp4_swift
converter  = ninfer-v3 (tools.convert)
components = text, vision, mtp, dflash2
bytes      = 19,782,449,156 (18.42 GiB)
sha256     = 6353a46f54dbf9d5cc46d718d88ded9f54bcbbbf25f9f879989ef1a560f01bcc
payload    = 7e9a3bebc65526c9b2aaf6502dafc82c32477ad309f8c306ff9195b2ee272bee
```

The artifact holds 1513 bindings over 1590 objects â€” 1072 reached from bindings and 844 `uses` â€” plus
the six frontend resources. Its Text allocation is all-NVFP4: 256 parents cover every
`attention/query_key_gate_value`, `attention/output`, `gdn/query_key_value_z`, `gdn/output`,
`mlp/gate_up` and `mlp/down`, with 256 site-level fp32 input divisors. The token embedding and the
full output head are `q8_g32_fp16` encoded from the BF16 source. Vision keeps the source's Q4/Q5/Q6,
MTP its Q8, the indexed proposal head its Q4, and the DFlash2 companion its W8G32.

| Format | Objects |
|---|---:|
| `bf16` | 579 |
| `fp32` | 608 |
| `int32` | 1 |
| `nvfp4` | 256 |
| `q4_g64_fp16` | 55 |
| `q5_g64_fp16` | 54 |
| `q6_g64_fp16` | 1 |
| `q8_g32_fp16` | 30 |
| resource | 6 |

| Layout | Objects |
|---|---:|
| `contiguous_le_v1` | 1188 |
| `row_split_k128_v1` | 140 |
| `block_scale_k16_m128x4_v1` | 256 |
| resource | 6 |

### 16.2 Sources and provenance

| source | revision | supplies |
|---|---|---|
| `ukisai/Swift-Qwen3.8-27B-NVFP4` | `4cf1019102c2fe9841c07109ac84acb40dabd9ec` | the MLP's NVFP4 codes and site input scales, Vision, MTP, the proposal head |
| `ukisai/Swift-Qwen3.8-27b` | `6b5cb4859c92dba0a3f195ce9ad4b1b75a6a2418` | the BF16 weights the attention, GDN and both endpoints are encoded from |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the DFlash2 companion |

The two Swift checkpoints agree on every tensor that is not quantized: all **798** BF16 tensors are
byte-identical between them, which is the premise the re-encode rests on and is checked rather than
assumed. The BF16 export is ungated, so the re-encode needs no credentials. ModelOpt stores each
site's activation amax in `input_scale`, and the divisor bound for an FP8 site being re-encoded is
`d_x = 6 / input_scale`, derived in Section 1 of the artifact conventions.

For reproducibility, one thing about those pins is worth stating: both repositories were re-uploaded
on 2026-09-24, after this port's download, as a squashed commit that replaced the pinned revision in
`main`. Comparing the two revisions through the Hub's tree API shows every weight shard, config,
tokenizer and index file carrying an identical size *and* an identical object id; the only changed
files are `.gitattributes` and `README.md`, and a demo video was added. So this artifact is built from
the weights that are current rather than from a superseded quantization, and the pins still name the
revision to reproduce it from.

The FP8-importing build this artifact replaces is published at
`CaptainArni/Swift-Qwen3.8-27B-NInfer` (22,783,241,220 bytes, sha256 `5412a0e7...`). This port's own
build of that recipe carries the same weights and differs from the published file only in the chat
template, so the two are comparable on every axis this section measures.

### 16.3 Production and verification

```bash
python3 -m tools.convert \
  --model /path/to/Swift-Qwen3.8-27B-NVFP4 \
  --recipe qwen3_8_27b_nvfp4_swift \
  --source swift_bf16=/path/to/Swift-Qwen3.8-27b \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b --device cpu \
  --out models/qwen3_8_27b_nvfp4swift.v3.ninfer
```

The recipe imports the MLP's NVFP4 codes and re-encodes the attention and GDN from the BF16 export, so
a rebuild reproduces the artifact on either device: converting the same sources on CPU, against the
published CUDA build, produced identical index JSON, identical `file_bytes`, and the payload digest
above, with only the random `artifact_id` differing. Compare with
`tools/release/compare_artifacts.py`, never by file hash. Every object is reachable from a binding, a
`use`, a resource or a proposal.

Hashing every binding against the FP8-importing build shows the re-encode's blast radius is exactly
the text stack: Vision's 441 bindings, the MLP's 192, DFlash2's 91 and MTP's 16 are byte-identical,
and only attention (80 of 112) and GDN (240 of 528) differ, plus the two endpoints. Vision, MTP and
DFlash2 therefore cannot behave differently because of this conversion.

### 16.4 Measured results (RTX 5090)

Perplexity on the fixed 1M corpus, `fp8` KV, against the FP8-importing build of the same sources:

| build | `--quick` | full |
|---|---:|---:|
| FP8-importing | 4.84938 | 4.93874 |
| re-encoded, endpoints FP8 | 4.85155 | 4.96342 |
| **re-encoded, endpoints Q8** | **4.68429** | **4.92432** |

Three builds separate the two changes, each adjacent pair differing in exactly one. The Q8 endpoints
are worth -3.45% / -0.79% and re-encoding the text stack costs +0.05% / +0.50%, so re-encoding is
bought for resident bytes and context rather than accuracy; the endpoints are where the accuracy comes
from. Section 1 of the artifact conventions carries that argument.

| | re-encoded | FP8-importing |
|---|---:|---:|
| device weights, MTP lane | 15.3 GiB | 18.90 GiB |
| file | 19.78 GB | 22.78 GB |
| KV capacity, int8, `auto`, 252928 context | 367,296 tokens | 283,712 tokens |
| context reach at `fp8` KV | 262,144 | 240,000; 180,224 with DFlash2 |
| DFlash2 lane acceptance | 60.9% | 45.5% |

The DFlash2 acceptance is the re-encode's largest measured win: the z-lab draft was trained against
the stock model's hidden states, and a text stack whose attention is NVFP4 like the stock artifacts'
moves those states closer to it. Upstream records the general form of that dependence: issue #298
section 3 converts a finetune far from stock and measures DFlash2 accepting 3.3-5.1% of drafts while
the MTP head still accepts 67-85%, concluding that the draft is model-specific. Swift sits between the
two cases, which is why its DFlash2 lane ships at a measured 60.9% rather than being assumed.

## 17. Rebuilt from source: the QAT line (`nvfp4qat`)

Section 15 describes the `nvfp4qat` artifact as the fork built it, from
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` at revision `d8e6fbfa3e3a78899b440222b827430045a05b44`. This port
builds the same line from the same source at a later revision, with the recipe
`qwen3_8_27b_nvfp4_qat`. The filename, model id and component set are unchanged, so the launchers and
the lanes do not move with it.

### 17.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4qat.v3.ninfer
name       = qwen3.8-27b
recipe     = qwen3_8_27b_nvfp4_qat
converter  = ninfer-v3 (tools.convert)
components = text, vision, mtp, dflash2
bytes      = 18,946,877,188
sha256     = 814db0dbc367a82f27be23afbde1897dd2e97cc7f797fefd734932efab934053
```

1513 bindings over 1600 objects, 844 `uses`, and 256 NVFP4 parents carrying 256 site-level fp32 input
divisors. No FP8 tensor reaches it: every text linear is imported from the source's quantized codes
with its own activation scale. The token embedding and full output head are Q8 from the base
checkpoint, the draft's projections take the NVFP4 rule of section 16, and norms, `gdn/convolution`,
`a_log` and `dt_bias` stay BF16.

### 17.2 Sources and provenance

| source | revision | supplies |
|---|---|---|
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `15d2e47bffe5d8ad23928879f8f7d2f74909e259` | every text linear's NVFP4 codes and activation scales |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | the endpoints, `gdn/a_projection` and `gdn/b_projection`, Vision, MTP, resources |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the DFlash2 companion |

Two facts about this source were measured rather than assumed. It quantizes 496 fused sites covering
*every* text linear, including `gdn/a_projection` and `gdn/b_projection`, which the other two sources
leave BF16 â€” so those two are taken from the base checkpoint, because at (96, 5120) the
`block_scale_k16_m128x4_v1` layout cannot represent them at all. And all 496 sites carry an
`input_global_scale`, so no calibration is needed for this line: `d_x = 1 / input_scale`.

### 17.3 Production and verification

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B-NVFP4-QUASAR \
  --recipe qwen3_8_27b_nvfp4_qat \
  --source bf16=/path/to/Qwen3.8-27B \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b --device cuda \
  --out models/qwen3_8_27b_nvfp4qat.v3.ninfer
```

Hashing every binding against the published artifact leaves **1394 of 1513 identical**, including every
imported code word: attention 112/112, MLP 192/192, the GDN codes 432/432, MTP 16/16, `proposal` 2/2,
Vision 441/441 and the text globals 131/131. So the source repository moving from `d8e6fbfa` to
`15d2e47b` did not change the weights. The 119 that differ are two groups: `gdn/a_projection` and
`gdn/b_projection` (96, which the fork decoded from the source's quantized control words where this
build takes them pristine from the base) and the DFlash2 draft (23). The NVFP4 draft rule of section 16
is what re-encoded that second group: the same comparison on the build before it counted 59 differing
draft bindings, which is consistent with the acceptance the rule measured on this line (58.0% against
54.8%).

Weights are not the whole artifact, and on this line the frontends differ too. The two builds carry
different chat templates: this one 10,871 bytes at `resource/text/chat_template.jinja`, the published
one 9,712 at `frontend/chat_template.jinja`, and the port's own official artifact a third at 9,897 under
that same `frontend` id. The difference between the first two is two hunks -- a comment, and a block
this port added that maps client reasoning-effort aliases (`high`, `max`, `ultracode`, `extreme` onto
`xhigh`; `minimal` onto `low`) to the three levels the template renders, because the stock template
raised an exception on Claude Code's default `high`. A request that passes no `reasoning_effort`
renders identically either way, so the added block costs nothing when it is not asked for, while a
request that passes `high` fails on the published file and succeeds here. The consequence to keep in
mind is that any comparison of the two by conversation compares the frontends as well as the weights.

The engine loads it at 16.3 GiB of device weights on the MTP lane and 17.2 GiB on the DFlash2 lane,
reaches the full 262,144 context at `fp8` KV with 2.99 and 2.36 GiB free respectively, and answers a
request correctly.

### 17.4 Measured results (RTX 5090)

| protocol | published fork build | this build |
|---|---:|---:|
| `--quick`, `fp8` KV | 4.94879 | **4.88817** |
| full corpus, `fp8` KV | 5.00234 | **4.99097** |

The `--quick` reading of 4.94879 is the artifact as it measures today; section 15's table records
4.89741 for it, which does not reproduce. The `--quick` section of `docs/perplexity-baseline.md`
explains why a `--quick` comparison of this shape is decided by four singleton streams, and the full
corpus settles it: this build is ahead there too, by 0.23% against the subset's 1.22%. The direction is
the same and the magnitude is smaller, which is what a singleton-stream effect looks like when four
streams per domain average it. Per domain it is lower on `chinese_reference` by 1.04% and on
`english_long_form` by 0.05%, and marginally higher on `english_reference` (0.11%) and `ninfer_code`
(0.08%): two domains of four, with the Chinese one carrying the aggregate. That is the pattern all
three rebuilt lines show, tabulated in `docs/perplexity-baseline.md`.

## 18. Rebuilt from source: the unsloth line (`nvfp4full`)

Section 14 describes the `nvfp4full` artifact as the fork built it from
`unsloth/Qwen3.8-27B-NVFP4`. This port builds the same line from the same source with the recipe
`qwen3_8_27b_nvfp4_unsloth`, keeping the filename and component set.

### 18.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4full.v3.ninfer
name       = qwen3.8-27b
recipe     = qwen3_8_27b_nvfp4_unsloth
converter  = ninfer-v3 (tools.convert)
components = text, vision, mtp, dflash2
bytes      = 19,715,597,060
sha256     = f8dc64701daca3eb7d28c9ee74b0d6a9af93cb5d4e4ec6b43d513849e8224201
```

1513 bindings over 1573 objects, 844 `uses`, 278 NVFP4 parents and 485 bound activation divisors. Its
MLP 0-55 is imported from the source's NVFP4 codes; attention, linear-attention and MLP 56-63 are
encoded to NVFP4 from the BF16 base; both W8 endpoints are Q8; the draft takes the NVFP4 rule of
section 16. Nine parents stay BF16, the Qwen3.6-27B pattern §14 records.

### 18.2 Sources and provenance

| source | revision | supplies |
|---|---|---|
| `unsloth/Qwen3.8-27B-NVFP4` | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` | the 112 NVFP4 MLP parents of layers 0-55 and their divisors |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | everything encoded locally, the endpoints, and all unquantized tensors |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the DFlash2 companion |

This source carries no activation scale for its 233 FP8 matrices — verified exhaustively: every
`input_global_scale` in it belongs to one of the 56 NVFP4 MLP layers. A divisor is therefore measured
by `tools/convert/calibration.py` and committed as `qwen3_8_27b_nvfp4_calibration.json`. Borrowing
another quantization's stored scales instead measured 2.2% / 0.45% worse, because an activation maximum
does not transfer between weight realizations.

The calibration uses the fork's method and its committed corpus, and its output was checked against the
247 divisor values the published profile carries: on the sites the fork calibrated, **median ratio
1.0000 and 99% within ±25%**, with layers 56, 58 and 59 exactly equal. The spread reaches 0.5943 to
1.4912, so this is the fork's method applied to this checkpoint rather than a reproduction of its
numbers, and the artifact is measured as what it is. Three candidate causes of that spread were tested
and eliminated: the container's object naming (a build naming every object after its parameters scores
identically), the compute device (a GPU-side calibration returns identical values to a CPU-side one),
and the tokenization (3,625 tokens either way).

### 18.3 Production and verification

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4_unsloth \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b --device cuda \
  --out models/qwen3_8_27b_nvfp4full.v3.ninfer
```

Hashing every binding against the published profile leaves **1490 of 1513 identical**, and every one of
the 23 that differ is a DFlash2 draft binding. `proposal/head` is byte-identical, which is the binding
the defect fixed in `d5165a9f` was about: the proposal was derived from the quantized source while the
head came from the base one. The nine
BF16 exception parents are byte-identical to it, so keeping them was not a deviation from that
artifact — but it *is* a measured choice: encoding them instead scored 4.85576/4.99604 against
4.75750/4.97532, and the same pattern loses on Swift's ModelOpt source (section 1 of the artifact
conventions), which is the rule's point about measuring per checkpoint.

The engine loads it at 17.1 GiB of device weights on the MTP lane and 17.9 GiB on the DFlash2 lane,
reaching the full 262,144 context at `fp8` KV with 2.39 and 1.66 GiB free respectively.

### 18.4 Measured results (RTX 5090)

| protocol | published profile | this build |
|---|---:|---:|
| `--quick`, `fp8` KV | 4.82452 | **4.75750** |
| full corpus, `fp8` KV | 4.98768 | **4.97532** |

Per domain on the full corpus it is lower in three of four and exactly tied in the fourth, with
`chinese_reference` the largest at −0.65% and no domain moving against it by more than three
hundredths of a percent. `docs/perplexity-baseline.md` tabulates it beside the other two lines.

Interleaved against the published artifact in one window, both lanes the port ships are ahead:

| lane | published | this build |
|---|---:|---:|
| DFlash2 d7 acceptance | 49.4% | **68.8%** |
| DFlash2 d7 decode | 286–290 tok/s | **340 tok/s** |
| MTP d5 acceptance | 57.6% | **61.7%** |

Acceptance reproduces to a tenth of a point across windows, which is why it carries the comparison;
throughput does not, so those rows are from one interleaved window.

## 19. New artifact: the NVIDIA line (`nvfp4nvidia`)

This artifact has no published predecessor here. It is built from `nvidia/Qwen3.8-27B-NVFP4`, NVIDIA's
own ModelOpt quantization of the base model, which is a third realization of the same text stack
alongside unsloth's and QUASAR's. It gets its own filename and its own pair of lanes.

### 19.1 Identity and contents

```text
filename   = qwen3_8_27b_nvfp4nvidia.v3.ninfer
name       = qwen3.8-27b
recipe     = qwen3_8_27b_nvfp4_nvidia
converter  = ninfer-v3 (tools.convert)
components = text, vision, mtp, dflash2
bytes      = 18,946,877,188
sha256     = 76131f792241ff0a232abe1fb1234f6b403638940976ecebaf45b94b94d7e58e
```

1513 bindings over 1600 objects, 1082 weight jobs of which 128 are imports (64 layers × 2 fused MLP
parents) and 159 are local encodings, with zero FP8 tensors. Its NVFP4 MLP covers all 64 layers, where
the official stock stops at 55; attention, linear-attention and both W8 endpoints are encoded locally
from the BF16 base.

### 19.2 Sources and provenance

| source | revision | supplies |
|---|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` | `482ca0f3832238542f8f5295dde86b5f22711d80` | the MLP NVFP4 codes and the per-site activation scales |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | everything encoded locally, the endpoints, Vision, MTP, resources |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the DFlash2 companion |

The checkpoint is ModelOpt at producer `0.47.0.dev80+g913f5e224`; the model card states v0.48.0, and the
checkpoint's own field is the one an artifact should be reproduced from. The card agrees with what this
recipe measured and does rather than only what it says: it records mixed NVFP4/FP8 with NVFP4 on the MLP
layers *and* the language model head, FP8 on self-attention and linear-attention, and a 2,048-sample
Local-Hessian calibration -- which is the source of the 401 `input_scale` values that make a corpus
unnecessary here. The divisor probes which form applies, `1 / input_scale` where the site is already
NVFP4 and `6 / input_scale` where it is FP8.

### 19.3 Production and verification

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4_nvidia \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b --device cuda \
  --out models/qwen3_8_27b_nvfp4nvidia.v3.ninfer
```

The sites this line re-encodes are the hidden-state-fed class that transferred exactly when checked
against the published profiles' divisors — 53.4925 at layer 0's GDN input is identical in three
artifacts — which is why a producer's stored scales are usable for them and were not for the
derived-intermediate sites on the unsloth line. The engine loads it at 16.3 GiB of device weights on the
MTP lane and 17.2 GiB on the DFlash2 lane.

### 19.4 Measured results (RTX 5090)

Against the port's local copy of the official artifact (`9f35ba74...`, 23,719,760,043 bytes — note this
is not byte-identical to the published file, whose pin is 23,719,715,844 with sha256 `74d2c571...`, and
whose lane figures are therefore quoted from the copy that was measured rather than from the
repository):

| protocol | official stock | this build |
|---|---:|---:|
| `--quick`, `fp8` KV | 4.80557 | **4.71979** |
| full corpus, `fp8` KV | 4.90169 | **4.90168** |
| file | 23.72 GB | **18.95 GB** |
| FP8 tensors | 146 | **0** |

Both protocols were measured twice and reproduce exactly. The full-corpus figures being
indistinguishable was checked by domain rather than accepted, and it is a cancellation:

| domain | official stock | this build | change |
|---|---:|---:|---:|
| `chinese_reference` | 6.40455 | 6.28941 | −1.80% |
| `english_reference` | 6.55706 | 6.49852 | −0.89% |
| `english_long_form` | 8.17251 | 8.29735 | +1.53% |
| `ninfer_code` | 1.67007 | 1.69030 | +1.21% |

So this build is the same on average and 20% smaller, without FP8, and it reaches the full 262,144 at
the `fp8` KV the launchers use, where the copy of the official artifact measured here stops at 240,000
on MTP and 131,072-163,840 on DFlash2. The published official does reach 262,144 at `int8` KV, which
upstream reports in issue #298, so the reach is a property of the build and the KV dtype together
rather than a wall. This build is not 1.78% better: that figure belongs to the `--quick` protocol,
whose four singleton streams let one of them decide the number.

Its lanes, measured on the artifact this release ships:

| lane | decode | acceptance | runtime / free |
|---|---:|---:|---|
| DFlash2 d7 | 322.4 tok/s | 59.2% | 10.7 GiB / 2.45 GiB |
| MTP d5 | 228.3 tok/s | 56.4% | 10.4 GiB / 2.99 GiB |

`ceiling` mode measured all four spec/vision combinations at the full 262,144, so this line reaches a
context the official stock cannot.

Upstream reached the same conclusion about this checkpoint independently. Issue #214, closed
2026-09-15, switched NInfer's own Qwen3.8-27B NVFP4 base to `nvidia/Qwen3.8-27B-NVFP4` on exactly this
ground -- it is smaller and better calibrated for NVFP4 -- and the interleaved benchmark that closed it
compares an `unsloth` build against an `nvidia hybrid` one, naming the hybrid the shape this recipe
builds: NVFP4 imported where the source has it, the rest re-encoded. That benchmark also puts prefill
and decode within 0.8% of each other, which is what this line's identical full-corpus perplexity
independently shows from the other direction.
