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
| `start_quasar_v3_dflash2_vision.bat` | QUASAR QAT | DFlash2 | nvfp4qat | 7 | k8v4 | 262,144 | 305.1 | 52.4% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR QAT | MTP4 | nvfp4qat | 4 | fp8 | 262,144 | 215.1 | 60.4% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full no-ex | DFlash2 | nvfp4full_noex | 9 | fp8 | 262,144 | 299.6 | 41.7% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP4 | nvfp4full | 4 | fp8 | 262,144 | 193.7 | 55.5% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 | nvfp4swift15 | 7 | fp8 | 262,144 | 309.0 | 53.5% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP4 | nvfp4swift15 | 5 | k8v4 | 262,144 | 206.4 | 54.4% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA ModelOpt | DFlash2 | nvfp4nvidia | 9 | fp8 | 262,144 | 355.3 | 51.1% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA ModelOpt | MTP4 | nvfp4nvidia | 4 | k8v4 | 262,144 | 200.6 | 55.3% |

Decode tok/s and acceptance are from `tools/release/profiles.py`, folded on 2026-11-11 from
`tools/release/lane_figures.json`; that module's docstring carries the protocol note and the run level. These
are the published figures, and `MEASURED_ENGINE` records the engine they came from, so
`check_lane_figures.py` fails if the table and the tree disagree on it.

**Measurement status (2026-11-11).** These figures describe the current engine (`b4ea73c5e91e`, running with `--memory-headroom 1024MiB` in `INVARIANT_FLAGS` to clear Windows WDDM Shared System Memory eviction across all eight lanes). The fold's run
level was -1.5% median against the previous table (range -3.7% to +1.7%), so the decode column is this window's
reading rather than a re-run of the interleaved window; re-measure before quoting it as current steady speed. The earlier 2026-11-03 interleaved comparison notes remain in `tools/release/profiles.py` and the per-lane sections below.

## Which artifact is "best"

**Accuracy:** QUASAR QAT (`nvfp4qat`) is the recommended line at 4.684860 PPL.
NVIDIA ModelOpt (`nvfp4nvidia`) is within noise at 4.686759. The difference
is not statistically meaningful on this corpus.

**Decode throughput:** In the 2026-11-11 fingerprint (`b4ea73c5e91e`), NVIDIA ModelOpt DFlash2 (`nvidia` lane) read
355.3 tok/s on `code`, against Swift 1.5 DFlash2 at 309.0, QUASAR QAT DFlash2 at 305.1 and
NVFP4-full no-ex DFlash2 at 299.6 (under stochastic `temperature=1.0`, `seed=1234`). These are the table's own cells. This page publishes
decode and accuracy; it does not rank lanes on prefill, which is measured per
configuration and is not recorded per lane here.

**There is no single "best" artifact.** QUASAR QAT (`nvfp4qat`) is the accuracy reference (with NVIDIA ModelOpt within noise), while the fastest DFlash2 decode lane has alternated between NVFP4-full no-ex (365.3 tok/s on 2026-11-03) and NVIDIA ModelOpt (358.7 tok/s in the 2026-11-11 fingerprint). The choice depends on the workload: accuracy-sensitive workloads should use QUASAR or NVIDIA; throughput-sensitive workloads should pick whichever DFlash2 lane leads on their domain.

## KV dtype: fp8 vs k8v4, decided per lane

KV dtype is a **per-lane** field, not a global choice. A global k8v4 was
refused because the effect is artifact-dependent:

- **k8v4 wins on:** QUASAR DFlash2 (+13.8% decode, +9.5 acceptance), NVIDIA
  MTP (+6.5% decode), Swift MTP (+2.4% decode on the code cell)
- **fp8 wins on:** QUASAR MTP (+2.6% decode), Swift DFlash2 (+7.3% decode),
  ninfer DFlash2/MTP (+7.0% on MTP), nvidia DFlash2

Each lane's `kv_dtype` is recorded in `profiles.py` and rendered into its
launcher. Do not change one lane's KV dtype without re-measuring that lane on
the interleaved window — the effect is artifact-dependent and can reverse
between lanes.

## Draft numbers

Draft numbers were re-swept per lane on the interleaved window:
- QUASAR DFlash2: 7 → 9 → 7 (depth 9 won at fp8; the merged-engine re-derivation reversed it on the deciding cells)
- QUASAR MTP: 5 → 4 (depth 4 wins on maximin over served cells)
- ninfer DFlash2: 7 → 9 (depth 9 leads on all three served treatments)
- ninfer MTP: stays 4 (depth 5 loses long code by 26%)
- Swift DFlash2: stays 7 (maximin over served domains)
- Swift MTP: 4 → 5 (reverted at k8v4, then re-applied once maximin compared each option's worst deciding cell)
- NVIDIA DFlash2: 7 → 9 (depth 9 leads on served code)
- NVIDIA MTP: stays 4 (unchanged since the contaminated 2026-09-17 sweep; neither the 223.4/61.4% pair nor a later depth-5 arm reproduces, so the value is retained rather than re-decided)

MTP is hard-capped at 5 by `kMtpDecodeMaximumDrafts`.

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
