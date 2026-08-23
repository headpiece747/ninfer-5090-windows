# NInfer 5090 Windows

> Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance. **100% Native Windows MSVC (No WSL2 Required!).**

NInfer 5090 Windows is a from-scratch C++/CUDA inference engine optimized for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090, now ported natively to Windows. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs.

This project is a Windows adaptation of the original [Neroued/ninfer](https://github.com/Neroued/ninfer) Linux engine, incorporating architectural concepts from [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) to support compilation via Microsoft Visual Studio (MSVC) on Windows 11.

## Supported Models

Like the upstream project, this engine supports a closed set of model artifacts to maximize performance on the RTX 5090 (`sm_120a`). 

Our primary benchmark target is:
* [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)

## Performance

The NVFP4 models utilize W4A4 Tensor Core MMA for prefill and A16 NVFP4 kernels for decode. When combined with Multi-Token Prediction (MTP3) and INT8 Group-64 KV caching, performance on a single RTX 5090 is extraordinary:

* **Decode Speed**: ~182 tokens/s (with ~84% MTP acceptance)
* **Context Capacity**: With `int8-group64` KV compression, a single token footprint shrinks to ~33 KB. This allows an RTX 5090 (32GB VRAM) to easily hold a **262,144-token prompt** alongside the 19.4GB model weights in a single GPU!

## Requirements

- 64-bit Windows 11 (Native, **NO WSL2 Required**)
- NVIDIA GeForce RTX 5090 (`sm_120a`)
- Visual Studio 2026 or 2022 (Developer Command Prompt / MSVC)
- CUDA 13.3 Toolkit (or 13.1+)
- CMake 3.28 or newer
- Ninja build system

*(Note: FFmpeg and libcurl dependencies from the original Linux repo are stubbed out in this port for ease of compilation).*

## Installation (Pre-compiled)

**If you just want to run the engine and don't want to compile it yourself, download the [Latest Release ZIP](../../releases/latest) from the Releases page.**

The ZIP contains the fully compiled Windows binaries (`ninfer-serve.exe`) and the optimized startup script. 

1. Extract the ZIP to a new folder.
2. Run `download_model.bat` to automatically download the required ~20GB `qwen3_8_27b_nvfp4.ninfer` model file into your folder (or download it manually from [HuggingFace](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/main/qwen3_8_27b_nvfp4.ninfer)).
3. Double-click `start_ninfer_5090.bat` to launch the server!

---

## Building from Source (For Developers)

### 1. Clone Upstream
Clone the original Linux repository as a base:
```cmd
git clone https://github.com/Neroued/ninfer.git ninfer_upstream
cd ninfer_upstream
```

### 2. Apply Windows/MSVC Patches
To compile natively on Windows, you must patch several files in the `src/` directory to remove POSIX dependencies and fix MSVC-specific linker issues:

1. **`CMakeLists.txt`**: Add MSVC-specific compile definitions:
   ```cmake
   add_compile_options(/Zc:preprocessor)
   add_compile_definitions(NOMINMAX UTF8PROC_STATIC)
   ```
   Disable FFmpeg and curl dependencies if not needed:
   ```cmake
   set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
   ```

2. **POSIX Header Swaps**:
   * Replace `#include <unistd.h>` with `<process.h>` or `<io.h>` in `src/serve/request_log.cpp` and `src/product/load_progress/load_progress.cpp`.
   * Replace `isatty` with `_isatty`.
   * In `src/serve/console_log.cpp`, replace `localtime_r` with `localtime_s` for Windows compatibility.

3. **TMA Alignment Patch**: 
   * In `src/targets/qwen3_6/impl/compute/nvfp4_w4a4_tma.cuh`, ensure TMA allocations are patched from `128` to `64` for Windows compatibility.

4. **Template Instantiation (`api_impl.h`)**:
   * MSVC requires explicit inline template instantiation for `RequestPlan` and `SequencePlan`. In `src/targets/qwen3_6/impl/runtime/api_impl.h`, modify the defaulted move constructors to manually expand the move logic (e.g., `impl_(std::move(other.impl_))`) to resolve `LNK2019` unresolved external symbol errors.

5. **NVFP4 Format Support (`reader.cpp`)**:
   * In `src/artifact/reader.cpp`, add mappings for `FP8_E4M3FN_ROW_BF16S` to `parse_format()` and `row-scale-v1` to `parse_layout()` to correctly recognize Qwen3.8 NVFP4 artifacts.

### 3. Compile
You can compile the project automatically using the provided batch script:
```cmd
build_windows.bat
```
*(The script will automatically attempt to find your MSVC environment and run the required CMake commands).*

Alternatively, if you wish to compile manually, open the **x64 Native Tools Command Prompt**, navigate to the folder, and run:
```cmd
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="120a" -DNINFER_ENABLE_AVX2=ON -DNINFER_BUILD_MEDIA_ACQUIRE=OFF
cmake --build build --config Release
```

**Note:** This repository also includes a GitHub Actions workflow (`build-windows.yml`) that automatically compiles the `.exe` binaries on every push!

## Running the Server

To achieve the maximum benchmark speeds (~182+ tok/s) and support massive context windows, you can simply run the included batch script:
```cmd
start_ninfer_5090.bat
```

*(You must edit the script to point to the location of your downloaded `.ninfer` model file).*

Alternatively, run the generated executable manually with the exact optimized flags:

```cmd
build\apps\ninfer-serve.exe C:\path\to\qwen3_8_27b_nvfp4.ninfer ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --kv-dtype int8 ^
  --max-context 262144 ^
  --lm-head-draft ^
  --prefill-chunk 4096 ^
  --pending-timeout-ms 600000
```

### Flag Explanations:
* `--spec mtp --draft-tokens 3`: Enables Multi-Token Prediction (MTP3), drafting 3 tokens concurrently, leading to massive speedups.
* `--kv-dtype int8`: Compresses the KV cache footprint (to ~33KB per token). Essential for squeezing large contexts onto a single GPU.
* `--max-context 262144`: Overrides the default 8192 software limit, unlocking the full 256k context potential since the 5090 has enough VRAM.
* `--lm-head-draft`: Evaluates the language model head on drafted tokens for slightly better speculative acceptance.
* `--prefill-chunk 4096`: Increases the batch size for prompt ingestion, speeding up the initial processing of massive OpenCode workspaces.
* `--pending-timeout-ms 600000`: Increases the queue timeout to 10 minutes, preventing OpenCode requests from dropping while the server is busy crunching a large prompt.

## OpenCode Desktop & IDE Integration

Once the NInfer server is running, it operates exactly like an OpenAI API endpoint. You can plug it directly into OpenCode Desktop (or Cursor/Continue).

For **OpenCode Desktop**, edit your `~/.config/opencode/opencode.json` configuration file to add NInfer as a local provider and force the global context limit:

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
    "buffer": 8192
  }
}
```

*Note: The `buffer: 8192` ensures OpenCode reserves enough tokens for the AI's reply instead of filling the entire 256k context with your codebase.*

*Note: Make sure the `model` name precisely matches the ID served by NInfer (e.g., `qwen3.8-27b`).*

### Advanced OpenCode Optimizations

**1. Use `.opencodeignore`**
Even with 262k context, injecting build artifacts or binaries into your prompt is a massive waste of VRAM.
Create a `.opencodeignore` file in the root of your project:
```text
node_modules/
dist/
build/
.git/
*.bin
*.lock
package-lock.json
```
This eliminates client-side bloat, ensuring your RTX 5090 uses its VRAM exclusively for actual code comprehension.

## Acknowledgements

- [Neroued/ninfer](https://github.com/Neroued/ninfer) - Original author of the ultra-optimized C++/CUDA NInfer engine for Linux.
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - Reference implementation for compiling the engine under Windows/MSVC.
- Original models derived from Qwen/Qwen3.8-27B.

## License

This project is licensed under the [Apache License 2.0](LICENSE). 

It is derived from the upstream [NInfer](https://github.com/Neroued/ninfer) project, which is also distributed under the Apache-2.0 License. All vendored dependencies and original model weights retain their respective licenses.

## OpenCode Desktop & Advanced Tweaks

### 1. Limiting "Overthinking" (Reasoning Effort)
If you find Qwen3.8 is generating too many thinking tokens and eating into your context window, you can limit its reasoning budget directly in OpenCode.
Edit your ~/.config/opencode/opencode.json and add a thinking budget to your model configuration:
\\\json
"thinking": {
  "budget": 4096
}
\\\

### 2. Running as a Background Windows Service
To avoid keeping a command prompt window open, you can run NInfer as a silent background service using NSSM (Non-Sucking Service Manager):
1. Download [NSSM](http://nssm.cc/).
2. Open an Administrator command prompt and run: 
ssm install NInferServe
3. Set the **Path** to C:\path\to\start_ninfer_5090.bat.
4. Click "Install service" and start it via the Windows Services app. NInfer will now start silently every time you boot your PC!

### 3. WDDM VRAM Warning (Windows Overhead)
Unlike Linux, Windows (via the WDDM display driver) reserves roughly 10-15% of your GPU VRAM for desktop rendering and background apps. 
> **Warning:** Do not run heavy 3D rendering software or modern video games simultaneously while doing 200k+ token code reviews, as you may hit an Out-Of-Memory (OOM) crash. Monitor your VRAM usage via Task Manager.

### 4. Multimodal / Vision Support
**Vision support is now available!** We have introduced a separate, optional vision build to keep the core text-only engine as simple as possible. 
- **For standard use**, run `start_ninfer_5090.bat` (Text-only, maximum 262k context).
- **For vision**, run `start_ninfer_vision.bat` (Multimodal, ~180k context to leave room for the 3GB visual encoders).

If you are building from source, you can use `build_vision_windows.bat`, which automatically downloads the necessary pre-compiled GPL-shared FFmpeg binaries and statically links them against `ninfer-serve-vision.exe` without requiring `vcpkg`.

