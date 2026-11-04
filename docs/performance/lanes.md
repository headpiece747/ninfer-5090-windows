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
| `start_quasar_v3_dflash2_vision.bat` | QUASAR QAT | DFlash2 | nvfp4qat | 7 | k8v4 | 262,144 | 302.3 | 46.6% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR QAT | MTP4 | nvfp4qat | 4 | fp8 | 262,144 | 224.2 | 57.3% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full no-ex | DFlash2 | nvfp4full_noex | 9 | fp8 | 262,144 | 365.3 | 47.3% |
| `start_ninfer_v3_mtp4_vision.bat` | NVFP4-full | MTP4 | nvfp4full | 4 | fp8 | 262,144 | 167.1 | 47.3% |
| `start_swift_v3_dflash2_vision.bat` | Swift 1.5 | DFlash2 | nvfp4swift15 | 7 | fp8 | 262,144 | 355.2 | 57.8% |
| `start_swift_v3_mtp4_vision.bat` | Swift 1.5 | MTP4 | nvfp4swift15 | 5 | k8v4 | 262,144 | 213.6 | 51.7% |
| `start_nvidia_v3_dflash2_vision.bat` | NVIDIA ModelOpt | DFlash2 | nvfp4nvidia | 9 | fp8 | 262,144 | 336.1 | 45.4% |
| `start_nvidia_v3_mtp4_vision.bat` | NVIDIA ModelOpt | MTP4 | nvfp4nvidia | 4 | k8v4 | 262,144 | 196.7 | 46.7% |

Decode tok/s and acceptance are from `tools/release/profiles.py`, folded on 2026-11-03 from
`tools/release/lane_figures.json`; that module's docstring carries the protocol note and the run level. These
are the published figures, and `MEASURED_ENGINE` records the engine they came from, so
`check_lane_figures.py` fails if the table and the tree disagree on it.

**Measurement status (2026-11-03).** These figures describe the current engine (`f3bc4e01ca39`). The fold's run
level was +7.8% against the previous table (range +3.4% to +10.3%), so the decode column is this window's
reading rather than the engine's steady speed; re-measure before quoting it as current. A pre-registered
refresh was attempted and **withheld**. The three-round `profile` window was stable (largest
per-lane spread 10.2% against a 15% criterion), agreed with an independent
`lane_fingerprint.py` pass on decode within 6.3% (criterion 10%) and ranked all
eight lanes identically (28/28 pairs), and the greedy digests were identical on
every lane. It failed the acceptance criterion on one lane: NVIDIA ModelOpt
DFlash2 read 45.4% across three profile runs against the fingerprint's 46.2%, a
0.79-point gap over the 0.5-point allowance. The cause is estimator resolution,
not an engine difference: each side draws its acceptance from a different set of
6-7 stochastic requests (`spec_rounds` 542 against 563), so a 0.5-point
comparison is finer than the estimator's own sampling error on this cell. The
table was left unchanged rather than transcribed from numbers the gate could not
certify. Re-measuring needs either a fixed request set for the acceptance cell
or a decision, with evidence, to widen that allowance.

## Which artifact is "best"

**Accuracy:** QUASAR QAT (`nvfp4qat`) is the recommended line at 4.684860 PPL.
NVIDIA ModelOpt (`nvfp4nvidia`) is within noise at 4.686759. The difference
is not statistically meaningful on this corpus.

**Decode throughput:** NVFP4-full no-ex DFlash2 (`ninfer` lane) is fastest at
365.3 tok/s, against Swift 1.5 DFlash2 at 355.2, NVIDIA ModelOpt DFlash2 at 336.1 and
QUASAR QAT DFlash2 at 302.3. These are the table's own cells. This page publishes
decode and accuracy; it does not rank lanes on prefill, which is measured per
configuration and is not recorded per lane here.

**There is no single "best" artifact.** QUASAR wins on accuracy, NVFP4-full
no-ex wins on decode. The choice depends on the workload: accuracy-sensitive
workloads should use QUASAR or NVIDIA; throughput-sensitive workloads should use
the NVFP4-full no-ex DFlash2 lane.

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
