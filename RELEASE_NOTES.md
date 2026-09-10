# NInfer Windows Release Notes

## Version 1.0.5 (2026-09-07)

### Major Features & Engine Upgrades
* **DFlash2 Speculative Decoding**: Integrated z-lab's 5-layer auxiliary draft network (`--spec dflash2 --draft-tokens 1..15`, recommended: 7), achieving burst generation speeds up to **~350–356 tok/s** on NVIDIA GeForce RTX 5090 (`sm_120a`).
* **Upstream Synchronization (`487f8977`)**:
  * **Sparse MoE Warp Merge**: Optimized small-T stage-2 execution using one CTA per token and a warp merge in place of eight dependent rounds.
  * **Exact Agent Prefix Preservation**: Fixed cache state preservation during multi-turn speculative settlement, ensuring zero-loss prompt caching across autonomous agent turns.
  * **Thinking Preservation & Intent**: Hardened `--preserve-thinking` flag and reasoning token boundaries (`<thought>`) to prevent leakage into function calling arguments.
  * **MSVC Runtime Hardening**: Thread-safe Win32 `localtime_s` time formatting for operational and request JSONL logs.

### Client Compatibility & Tooling
* **CORS Support**: Added `--cors` flag by default to `start_ninfer_5090.bat`, `start_ninfer_dflash2.bat`, and `start_ninfer_vision.bat`, enabling seamless out-of-the-box connectivity for browser-based frontends (Open WebUI, LibreChat, VS Code web extensions).
* **Client Compatibility Matrix**: Formally documented behavior and workarounds for OpenCode Desktop, VS Code, Cline, Roo Code, Aider, and Continue.dev in `README.md`.
* **Packaging Tooling**: Added `package_windows_release.bat` to produce clean, portable Windows release archives with binaries, runtime dependencies, and batch launchers.

---

## Version 1.0.4 (2026-08-25)

### Features & Fixes
* **KV Cache Quantization**: Added support for NVFP4 and K8V4 KV cache quantization modes, drastically reducing memory footprint for long-context generation up to 262k tokens.
* **Logging Infrastructure**: Replaced bespoke logging with native `spdlog` integration for fast, structured operational logging.
* **TMA & Decode Kernel Optimizations**: Integrated Fused TMA SwiGLU, W8 8-code decodes, and value-aware prefix scheduling from upstream.
* **Initial Native Windows Port**: Delivered full native Windows MSVC + CUDA 13.x compilation without WSL or Docker overhead.
