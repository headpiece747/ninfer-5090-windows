# NInfer 5090 Windows

> Native Windows MSVC port of [NInfer](https://github.com/Neroued/ninfer) for NVIDIA GeForce RTX 5090 (`sm_120a`). **100% Native Windows (No WSL2 Required).**

NInfer 5090 Windows is a bare-metal C++/CUDA inference engine optimized for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090 (32GB VRAM). It provides OpenAI- and Anthropic-compatible local HTTP endpoints with ultra-low latency and massive context support.

---

## Key Highlights & Performance

* **Blazing Generation Speed**: **Up to 220+ tokens/sec** decode throughput using hardware-native **FP8** (`--kv-dtype fp8`) combined with Multi-Token Prediction **MTP5** (`--draft-tokens 5` + `--lm-head-draft`).
* **Full 262k Context on a Single 32GB GPU**: Pack the 19.7GB Qwen3.8-27B NVFP4 model and a full **262,144-token context** into VRAM with over **1.2 GB of free VRAM headroom**.
* **DDR5 System RAM Host Cache**: Offloads inactive context pages into system memory (`--host-kv-mib 16384`, `--host-state-slots 16`) for instant multi-branch switching.
* **Sub-Second TTFT (<1s) & Reasoning Continuity**: Uses `--preserve-thinking` with `private_response_replay` context caching—asking follow-ups across 160k+ token conversations responds in **under 1.5 seconds**.
* **Multimodal / Vision Ready**: Unified executable supports both pure text (262k context) and vision/image attachments (131k context).

---

## Supported Models

* Primary Benchmark Target: [**Qwen3.8-27B NVFP4**](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) (`qwen3_8_27b_nvfp4.ninfer`)

---

## Quick Start (Pre-compiled Releases)

1. Download the latest release from the [Releases Page](../../releases/latest).
2. Extract the ZIP archive into a folder.
3. Run `download_model.bat` (or manually place `qwen3_8_27b_nvfp4.ninfer` in the directory).
4. Launch the server:
   * **Text-Only (Full 262k Context)**: Double-click `start_ninfer_5090.bat`
   * **Vision / Multimodal (131k Context)**: Double-click `start_ninfer_vision.bat`

---

## Startup Script Configurations

### 1. Text Server (262k Context + FP8 + MTP5) — `start_ninfer_5090.bat`
```bat
@echo off
echo Starting NInfer on RTX 5090 (Full 262k Context + 48GB DDR5 Profile)
build\apps\ninfer-serve.exe qwen3_8_27b_nvfp4.ninfer ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --max-context 262144 ^
  --kv-capacity 262144 ^
  --max-concurrency 1 ^
  --kv-dtype fp8 ^
  --prefill-chunk 1024 ^
  --device-state-slots 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 16384 ^
  --spec mtp --draft-tokens 5 ^
  --lm-head-draft ^
  --preserve-thinking ^
  --pending-timeout-ms 600000
pause
```

### 2. Vision Server (131k Context + FP8 + MTP5) — `start_ninfer_vision.bat`
```bat
@echo off
echo Starting NInfer on RTX 5090 (Vision + 131k Context + FP8 + MTP5)
build\apps\ninfer-serve.exe qwen3_8_27b_nvfp4.ninfer ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --max-context 131072 ^
  --kv-capacity 131072 ^
  --max-concurrency 1 ^
  --kv-dtype fp8 ^
  --prefill-chunk 1024 ^
  --device-state-slots 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 16384 ^
  --spec mtp --draft-tokens 5 ^
  --lm-head-draft ^
  --preserve-thinking ^
  --pending-timeout-ms 600000
pause
```
*(Note: Vision is set to 131k context to reserve ~2–3 GB of VRAM for the ViT visual encoder weights and patch buffers).*

---

## OpenCode Desktop Integration

NInfer exposes an OpenAI-compatible API on `http://127.0.0.1:8080/v1`. 

Add this optimized configuration to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "local-ninfer/qwen3.8-27b",
  "provider": {
    "local-ninfer": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NInfer RTX 5090",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "none"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen 3.8 27B NVFP4 (NInfer)",
          "attachment": true,
          "reasoning": true,
          "limit": {
            "context": 262144,
            "output": 16384
          }
        }
      }
    }
  },
  "compaction": {
    "auto": true,
    "prune": true,
    "maxContext": 262144,
    "buffer": 20480
  },
  "permission": {
    "external_directory": {
      "*": "ask",
      "C:\\Users\\tobia\\.config\\opencode\\**": "allow",
      "C:\\Users\\tobia\\.local\\share\\opencode\\**": "allow",
      "C:\\Users\\tobia\\AppData\\Roaming\\opencode\\**": "allow"
    }
  },
  "agent": {
    "build": {
      "temperature": 0.6,
      "prompt": "You are an elite autonomous software engineer. Follow these operational rules strictly:\n1. INVESTIGATE FIRST: Always inspect existing code, directory structures, and type definitions using tools before making changes. Never assume API signatures, import paths, or external dependencies.\n2. SURGICAL EDITS: Minimize diff footprint. Make precise, isolated edits that match the existing project style, naming conventions, and patterns. Never rewrite, reformat, or restyle unrelated files.\n3. VERIFICATION & SELF-CORRECTION: Validate all modifications by running relevant project build, test, or linter tools where available. Treat error output as diagnostic telemetry to resolve immediately.\n4. ACTION OVER CHAT: Keep conversational output concise and execution-oriented; conduct deep technical analysis inside thinking blocks."
    },
    "plan": {
      "temperature": 1.0,
      "prompt": "You are a principal software architect operating strictly in read-only planning mode.\n1. RECONNAISSANCE: Thoroughly inspect repository architecture, entry points, interfaces, and dependencies using exploration tools.\n2. STRUCTURED ROADMAPS: Deliver phased, step-by-step implementation specs specifying exact file paths, function modifications, and edge-case test criteria.\n3. RISK MITIGATION: Explicitly identify breaking changes, performance bottlenecks, and regressions before implementation begins.\n4. NO FILE MUTATIONS: Do not create, modify, or delete project files, or run state-altering commands."
    }
  }
}
```

* **Recommended Temperatures**:
  * `temperature: 0.6` for the `build` coding agent (optimal for strict compiler languages like Rust, C++, and TypeScript).
  * `temperature: 1.0` for the `plan` agent (encourages architectural exploration).

---

## Building from Source

### Requirements
* 64-bit Windows 11
* NVIDIA GeForce RTX 5090 (`sm_120a`)
* Visual Studio 2022 or 2026 (MSVC x64 Native Tools)
* CUDA 13.1+ (CUDA 13.3 recommended)
* CMake 3.28+ & Ninja

### Compilation
Run the automated build script:
```cmd
build_windows.bat
```
The script will automatically configure your environment, download pre-compiled FFmpeg shared binaries for multimodal support, and build Release binaries to `build\apps\ninfer-serve.exe`.

---

## Acknowledgements & License

* [Neroued/ninfer](https://github.com/Neroued/ninfer) — Original C++/CUDA inference engine for Linux.
* [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) — Windows/MSVC port concepts.
* [QwenLM](https://github.com/QwenLM/Qwen3) — Qwen3.8 foundation model series.

Licensed under the [Apache License 2.0](LICENSE).
