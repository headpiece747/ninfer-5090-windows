# NInfer 5090 Windows

> Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance. **100% Native Windows MSVC (No WSL2 Required!).**

NInfer 5090 Windows is a from-scratch C++/CUDA inference engine optimized for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090, now ported natively to Windows. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs.

This project is a Windows adaptation of the original [Neroued/ninfer](https://github.com/Neroued/ninfer) Linux engine, supporting compilation via Microsoft Visual Studio (MSVC) on Windows 11.

## Supported Models

Like the upstream project, this engine supports a closed set of model artifacts to maximize performance on the RTX 5090 (`sm_120a`).

Our primary target is:
* [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)

## Performance & Benchmarks

The NVFP4 models utilize W4A4 Tensor Core MMA for prefill and A16 NVFP4 kernels for decode. When combined with Multi-Token Prediction (MTP5) and FP8 KV caching:

* **Generation Throughput**: **207 – 220.8 tok/s** (with ~80% MTP5 acceptance rate).
* **Deep Context Throughput**: **160 – 194.2 tok/s** on massive prompts (tested up to 182k+ tokens).
* **Context Capacity**: With FP8 KV compression, full **262,144-token context** fits in 32GB VRAM alongside the 19.4GB model weights with ~1.2 GiB headroom.
* **Host RAM Offloading**: 16GB DDR5 host KV cache offload (`--host-kv-mib 16384`) allows instant context switching across multi-turn sessions.

## Requirements

- 64-bit Windows 11 (Native, **NO WSL2 Required**)
- NVIDIA GeForce RTX 5090 (`sm_120a`)
- Visual Studio 2026 or 2022 (Developer Command Prompt / MSVC)
- CUDA 13.3 Toolkit (or 13.1+)
- CMake 3.28 or newer
- Ninja build system

## Installation (Pre-compiled)

**Download the [Latest Release ZIP](../../releases/latest) from the Releases page.**

The ZIP contains the fully compiled Windows binaries (`ninfer-serve.exe`, `ninfer.exe`) and optimized startup scripts.

1. Extract the ZIP to a folder.
2. Run `download_model.bat` to download the ~20GB `qwen3_8_27b_nvfp4.ninfer` model file into your folder (or download manually from [HuggingFace](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/main/qwen3_8_27b_nvfp4.ninfer)).
3. Double-click `start_ninfer_5090.bat` to launch the server!

---

## Building from Source (For Developers)

### 1. Build Automatically
Simply run:
```cmd
build_windows.bat
```
*(The script automatically downloads the required FFmpeg dev package, locates your MSVC environment, and builds with Ninja).*

### 2. Manual CMake Build
Open the **x64 Native Tools Command Prompt** and run:
```cmd
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="120a" -DNINFER_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Running the Server

### Text Model (Full 262k Context Profile)
Run `start_ninfer_5090.bat`:
```cmd
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
```

### Vision Model (131k Context Profile)
Run `start_ninfer_vision.bat`:
```cmd
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
```

---

## OpenCode Desktop Integration

NInfer exposes OpenAI-compatible `/v1/chat/completions` and `/v1/responses` endpoints.

Edit `~/.config/opencode/opencode.json`:
```json
{
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
          "name": "Qwen 3.8 27B NInfer",
          "attachment": true,
          "reasoning": true,
          "limit": { "context": 262144, "output": 16384 }
        }
      }
    }
  },
  "model": "local-ninfer/qwen3.8-27b",
  "compaction": {
    "auto": true,
    "prune": true,
    "maxContext": 262144,
    "buffer": 20480
  },
  "agent": {
    "build": {
      "temperature": 0.6
    },
    "plan": {
      "temperature": 1.0
    }
  }
}
```

### Key Recommendations:
* **Temperature**: Use `0.6` for precise code construction (e.g. Rust/Tauri) and `1.0` for high-level design planning.
* **Context Buffer**: A `20480` token compaction buffer ensures room for large outputs without early truncation.
* **Ignore Bloat**: Add `.opencodeignore` (`node_modules/`, `target/`, `dist/`, `.git/`) to prevent irrelevant files from consuming VRAM.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
