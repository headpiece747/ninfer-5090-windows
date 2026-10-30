# 8-lane matrix: model, artifact, lane, and measured configuration

This document records the eight shipped launcher profiles: four model lines
(QUASAR QAT, NVIDIA ModelOpt, NVFP4-full/unsloth, Swift 1.5), each shipped as
a DFlash2 lane and an MTP lane, all at the native 262,144 context with Vision.
`tools/release/profiles.py` is the single authority for what ships — this
document explains the mapping and the measurement rationale behind it.

## The four model lines

| Model line | Artifact file | Source | Full-corpus PPL (fp8 KV, 2026-10-07) | Role |
|---|---|---|---|---|
| QUASAR QAT | `qwen3_8_27b_nvfp4qat.v3.ninfer` | QUASAR-QAT quantisation-aware training | **4.684860 — best** | Accuracy reference; recommended line |
| NVIDIA ModelOpt | `qwen3_8_27b_nvfp4nvidia.v3.ninfer` | NVIDIA ModelOpt, built by this port | 4.686759 | Within noise of QUASAR; 20% smaller than official stock |
| NVFP4-full | `qwen3_8_27b_nvfp4full.v3.ninfer` | unsloth requantisation, BF16 exceptions kept | 4.724219 | MTP lane image; BF16 exceptions worth +21.4 acceptance points |
| NVFP4-full no-ex | `qwen3_8_27b_nvfp4full_noex.v3.ninfer` | unsloth, nine BF16 exception parents re-encoded to NVFP4 | 4.727636 | DFlash2 lane image; exceptions re-encoded to fit 262k+Vision |
| Swift 1.5 | `qwen3_8_27b_nvfp4swift15.v3.ninfer` | UkisAI Swift 1.5 finetune, re-encoded by this port | 4.755739 | Both Swift lanes |

PPL is fixed-corpus causal perplexity (`ninfer-perplexity --corpus`, full
corpus, fp8 KV). Lower is better. The `nvfp4qat` line is documented as "the
recommended line" in `docs/perplexity-baseline.md`; `nvfp4nvidia` is within
measurement noise of it (4.686759 vs 4.684860). No artifact is best on every
domain — see the per-domain breakdown in `docs/perplexity-baseline.md`.

## The eight lanes

| Launcher | Model line | Lane | Artifact | Draft | KV dtype | Context | Decode tok/s | Draft acceptance |
|---|---|---|---|---|---|---|---|---|
| `start_quasar_v3_dflash2_vision.bat` | QUASAR QAT | DFlash2 | nvfp4qat | 7 | k8v4 | 262,144 | 262.2 | 45.5% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR QAT | MTP4 | nvfp4qat | 4 | fp8 | 262,144 | 188.5 | 57.1% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full no-ex | DFlash2 | nvfp4full_noex | 9 | fp8 | 262,144 | 314.1 | 47.3% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP4 | nvfp4full | 4 | fp8 | 262,144 | 162.8 | 45.6% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 | nvfp4swift15 | 7 | fp8 | 262,144 | 310.1 | 56.7% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP4 | nvfp4swift15 | 5 | k8v4 | 262,144 | 179.8 | 51.7% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA ModelOpt | DFlash2 | nvfp4nvidia | 9 | fp8 | 262,144 | 291.7 | 45.6% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA ModelOpt | MTP4 | nvfp4nvidia | 4 | k8v4 | 262,144 | 165.3 | 46.7% |

Prefill tok/s and acceptance are from `tools/release/profiles.py` (measured
2026-10-08, interleaved window, three rounds, default `code` cell). These are
the published figures; `tools/release/lane_figures.json` records the engine
sha256 and per-cell measurements against which `check_lane_figures.py`
validates.

## Which artifact is "best"

**Accuracy:** QUASAR QAT (`nvfp4qat`) is the recommended line at 4.684860 PPL.
NVIDIA ModelOpt (`nvfp4nvidia`) is within noise at 4.686759. The difference
is not statistically meaningful on this corpus.

**Prefill throughput:** NVFP4-full no-ex DFlash2 (`ninfer` lane) is fastest at
365.6 tok/s, followed by NVIDIA ModelOpt DFlash2 at 314.1 and Swift 1.5
DFlash2 at 319.3. QUASAR QAT DFlash2 is 305.3.

**There is no single "best" artifact.** QUASAR wins on accuracy, NVFP4-full
no-ex wins on prefill throughput. The choice depends on the workload: accuracy-sensitive
workloads should use QUASAR or NVIDIA; throughput-sensitive workloads should use
the NVFP4-full no-ex DFlash2 lane.

## KV dtype: fp8 vs k8v4, decided per lane

KV dtype is a **per-lane** field, not a global choice. A global k8v4 was
refused because the effect is artifact-dependent:

- **k8v4 wins on:** QUASAR DFlash2 (+13.8% decode, +9.5 acceptance), NVIDIA
  MTP (+6.5% decode), Swift MTP (+6.5% decode)
- **fp8 wins on:** QUASAR MTP (+7.0% decode), Swift DFlash2 (+7.3% decode),
  ninfer DFlash2/MTP, nvidia DFlash2

Each lane's `kv_dtype` is recorded in `profiles.py` and rendered into its
launcher. Do not change one lane's KV dtype without re-measuring that lane on
the interleaved window — the effect is artifact-dependent and can reverse
between lanes.

## Draft numbers

Draft numbers were re-swept per lane on the interleaved window:
- QUASAR DFlash2: 7 → 9 (depth 9 wins on the served code cell)
- QUASAR MTP: 5 → 4 (depth 4 wins on maximin over served cells)
- ninfer DFlash2: 7 → 9 (depth 9 leads on all three served treatments)
- ninfer MTP: stays 4 (depth 5 loses long code by 26%)
- Swift DFlash2: stays 7 (maximin over served domains)
- Swift MTP: 4 → 5 → 4 (depth 5 won at fp8, reverted at k8v4)
- NVIDIA DFlash2: 7 → 9 (depth 9 leads on served code)
- NVIDIA MTP: stays 4 (depth 4 vs 5: 223.4 vs 167.8 tok/s)

MTP is hard-capped at 5 by `kMaximumMtpDraftTokens`.

## Maintaining these figures

`lane_figures.json` records the engine sha256 and per-cell measurements. After
any engine rebuild, run `python tools/bench/lane_fingerprint.py` to detect
drift. A single fingerprint run is **not** comparable with the published table
— use `tools/release/v3_profile_matrix.py` (interleaved window) to produce
figures for `profiles.py`. The `check_lane_figures.py` gate verifies that
`lane_figures.json` matches the current engine and the published table; drift
on DFlash2 lanes after a rebuild is expected until re-measured with the
interleaved window.

## Related documents

- `tools/release/profiles.py` — the single source of truth for what ships
- `docs/perplexity-baseline.md` — accuracy protocol and full-corpus PPL figures
- `docs/performance/methodology.md` — measurement and publication rules
- `docs/maintainer/build-system.md` — CMake targets and build configuration
- `docs/upstream-reports/fork-divergence.md` — the port-delta ratchet and merge procedure
