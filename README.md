# NInfer 5090 Windows

> Windows port of NInfer. Selected checkpoints. Maximum single-GPU inference performance. **100% Native Windows MSVC (No WSL2 Required!).**

NInfer 5090 Windows is a from-scratch C++/CUDA inference engine optimized for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090 (`sm_120a`), now ported natively to Windows. It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs.

This project is a Windows adaptation of the original [Neroued/ninfer](https://github.com/Neroued/ninfer) Linux engine, supporting compilation via Microsoft Visual Studio (MSVC) on Windows 11.

---

## Supported Models

Like the upstream project, this engine supports a closed set of model artifacts to maximize performance on the RTX 5090 (`sm_120a`).

Our primary target is:
* [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)

---

## Performance & Features (v1.0.5)

The NVFP4 models utilize W4A4 Tensor Core MMA for prefill and A16 NVFP4 kernels for decode. When combined with speculative decoding (MTP5 / DFlash2) and advanced KV caching:

* **Generation Throughput**:
  * **MTP5**: **207 – 220.8 tok/s** (~80% acceptance rate).
  * **DFlash2 (`--spec dflash2 --draft-tokens 7`)**: **265.5 tok/s** on Code, **321.1 tok/s** on Math/Reasoning, and **356.8 tok/s** on Structured JSON.
* **Deep Context Throughput**: **160 – 194.2 tok/s** sustained throughput on massive deep-context prompts (tested up to 200k+ active tokens).
* **Quantized KV Cache Options**:
  * **FP8 (`--kv-dtype fp8`)**: Full **262,144-token context** fits in 32GB VRAM alongside the 19.4GB model weights with ~1.2 GiB headroom.
  * **NVFP4 (`--kv-dtype nvfp4`)**: Sub-4-bit FP4 KV cache compression for ultra-low memory footprint.
  * **K8V4 (`--kv-dtype k8v4`)**: Asymmetric 8-bit Key + 4-bit Value hybrid quantization for high precision retention with half value memory.
* **Host RAM Offloading**: 16GB DDR5 host KV cache offload (`--host-kv-mib 16384`) allows instant context switching across multi-turn sessions.
* **Kernel & Engine Optimizations**:
  * **Sparse MoE**: One CTA per token in small-T S2 with warp merge instead of 8 dependent rounds.
  * **RMSNorm Occupancy**: Warp unrolling tuned for RTX 5090 (170 SMs).
  * **Exact Agent Prefix Reuse**: Prevents token divergence across multi-turn coding agent tool execution loops (OpenCode Desktop, Cline, Cursor).
  * Value-aware prefix cache reuse and instant streaming cancellation.
* **Unified Logging**: Fast, asynchronous operational logging powered by `spdlog` with dynamic weight-loading progress bars and llama.cpp-style request timing.

---

## Requirements

- 64-bit Windows 11 (Native, **NO WSL2 Required**)
- NVIDIA GeForce RTX 5090 (`sm_120a`)
- Visual Studio 2026 or 2022 (Developer Command Prompt / MSVC)
- CUDA 13.3 Toolkit (or 13.1+)
- CMake 3.28 or newer
- Ninja build system

---

## Installation (Pre-compiled)

**Download the [Latest Release ZIP](../../releases/latest) from the Releases page.**

The ZIP contains the fully compiled Windows binaries (`ninfer-serve.exe`, `ninfer.exe`, `ninfer-perplexity.exe`) and optimized startup scripts.

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
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000
```

### Text Model with DFlash2 Speculative Decoding (Up to 356 tok/s Profile)
If your model artifact includes DFlash2 companion weights, run `start_ninfer_dflash2.bat`:
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
  --spec dflash2 --draft-tokens 7 ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
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
  --default-thinking-budget 4096 ^
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

---

## Client Compatibility & Known API Limitations

NInfer is strictly tuned for raw single-GPU autoregressive throughput on the RTX 5090 (`sm_120a`). To maximize performance and keep kernel execution paths minimal, its HTTP server enforces strict OpenAI and Anthropic API validation and intentionally omits non-generation endpoints.

### Summary Compatibility Matrix

| Category / Feature | NInfer Support | Status with Common Clients |
| :--- | :--- | :--- |
| **OpenAI Chat Completions (`/v1/chat/completions`)** |  Full Support | OpenCode Desktop, Aider, Cline, Chat UIs (OpenAI mode). |
| **Anthropic Messages (`/v1/messages`)** |  Supported (Text & Images) | OpenCode Desktop, standard Anthropic chat clients. |
| **Inline Autocomplete (`/v1/completions`)** | ❌ **Not Implemented (404)** | Continue.dev tab completion, Cursor ghost-text, Tabby. |
| **Embeddings (`/v1/embeddings`)** | ❌ **Not Implemented (404)** | RAG tools (AnythingLLM, Dify, Obsidian Smart Connections). |
| **JSON Mode (`response_format: json_object`)** | ❌ **Not Supported (400)** | Instructor, Pydantic structured output, LangChain structured output. |
| **Constrained Decoding (`grammar`, `guided_*`)** | ❌ **Not Supported (400)** | Outlines, Guidance, lm-format-enforcer. |
| **Forced Tools (`tool_choice: "required"`)** | ❌ **Not Supported (400)** | Agent frameworks forcing tool execution (CrewAI, AutoGen). |
| **Sequential Tools (`parallel_tool_calls: false`)** | ❌ **Not Supported (400)** | Agents disabling parallel tool execution. |
| **Logprobs & Logit Bias (`logprobs`, `logit_bias`)** | ❌ **Not Supported (400)** | lm-eval, perplexity runners, SillyTavern token bans. |
| **Multiple Samples (`n > 1`)** | ❌ **Not Supported (400)** | Self-consistency / tree-of-thought multi-candidate sampling. |
| **Browser Web UIs (CORS)** | ⚠️ Requires `--cors` | Open WebUI, LibreChat, LobeChat (fail without `--cors`). |
| **Anthropic Server Tools / Native Documents** | ❌ **Not Supported (400)** | Claude Code CLI (`claude`), Anthropic PDF document blocks. |

---

### Known Breakages & How to Work Around Them

#### 1. Tab Autocomplete & Inline Code Suggestions (Fails: 404 Not Found)
* **Affected Tools**: **Continue.dev (Tab Autocomplete)**, **Cursor (Inline Autocomplete)**, **Tabby**, **Supermaven/Copilot proxies**.
* **Reason**: Inline code completion relies on raw Fill-In-the-Middle (FIM) via the legacy `/v1/completions` endpoint (`prompt` with `<|fim_prefix|>...<|fim_suffix|>`). NInfer only implements `/v1/chat/completions`.
* **Workaround**: Use Continue or Cursor for chat/agent tasks (via `/v1/chat/completions`), but route inline tab completions to a server that supports `/v1/completions` (such as `llama.cpp` or Ollama).

#### 2. RAG, Vector Search, & Knowledge Bases (Fails: 404 Not Found)
* **Affected Tools**: **AnythingLLM**, **Dify**, **Flowise**, **Obsidian Smart Connections**, **PrivateGPT**, **LangChain/LlamaIndex vector stores**.
* **Reason**: Document ingestion, chunking, and semantic search require `/v1/embeddings`. NInfer is an autoregressive generative model engine and does not provide an embedding endpoint.
* **Workaround**: Configure the application's embedding engine to use a local embedding server (e.g. Ollama running `nomic-embed-text` or Hugging Face Text Embeddings Inference), while using NInfer as the chat/generation LLM.

#### 3. Structured Output & Schema Enforcement (Fails: 400 Bad Request)
* **Affected Tools**: **Instructor** (`instructor.from_openai`), **Outlines**, **Guidance**, **LangChain/LlamaIndex** (`with_structured_output`), **Aider** (with `--openai-json-object` / JSON mode), **OpenAI SDK** (`beta.chat.completions.parse`).
* **Reason**: NInfer strictly validates request parameters in `src/serve/openai_chat_request.cpp`:
  * `response_format` other than `{"type": "text"}` returns `400 response_format_not_supported`.
  * Passing `grammar`, `guided_json`, `guided_regex`, `guided_choice`, or `structured_outputs` returns `400 constrained_decoding_not_supported`.
* **Workaround**: Direct the model to output JSON via plain text prompts and system instructions rather than using strict API-level schema constraints.

#### 4. Advanced Agent Frameworks & Tool Orchestrators (Fails: 400 Bad Request)
* **Affected Tools**: **CrewAI**, **AutoGen**, **LangGraph**, **OpenAI Agents SDK / LiteLLM**.
* **Reason**:
  * **Forced Tools**: Forcing tool execution via `tool_choice: "required"` or `tool_choice: {"type": "function", ...}` returns `400 tool_choice_not_supported` (only `"auto"` or `"none"` is supported).
  * **Sequential Tool Execution**: Setting `parallel_tool_calls: false` returns `400 parallel_tool_calls_not_supported`.
  * **Legacy Functions**: Passing legacy `functions` / `function_call` returns `400 legacy_tools_not_supported`.
* **Workaround**: Configure agent frameworks to use `tool_choice: "auto"` and enable parallel tool calls (`parallel_tool_calls: true`).

#### 5. Web-Based Chat Frontends & Web UIs (CORS & Ollama Protocol)
* **Affected Tools**: **Open WebUI**, **LibreChat**, **LobeChat**, **Chatbot UI**, **NextChat**.
* **Reason**:
  * If NInfer is launched without the `--cors` flag, web browser security policies block cross-origin requests (`http://localhost:3000`), failing preflight `OPTIONS` with `404`.
  * If Open WebUI is connected using its native "Ollama" provider instead of "OpenAI", it fails because NInfer does not implement Ollama's `/api/tags` or `/api/generate`.
* **Workaround**: Always include the `--cors` flag in your startup command (included by default in `start_ninfer_5090.bat`) and connect frontends using their **OpenAI-compatible** provider settings.

#### 6. LLM Evaluation & Benchmarking Suites (Fails: 400 Bad Request)
* **Affected Tools**: **EleutherAI LM-Evaluation-Harness (`lm-eval`)**, **Perplexity evaluators**, **SillyTavern** (with token penalty rules).
* **Reason**:
  * Benchmarking suites requesting `logprobs: true` or `top_logprobs > 0` return `400 logprobs_not_supported`.
  * Passing non-zero `logit_bias` (e.g. token ban lists) returns `400 logit_bias_not_supported`.
  * Requesting multiple candidate completions per prompt (`n > 1`) returns `400 n_not_supported`.
  * Specifying non-neutral `repetition_penalty != 1.0` returns `400 repetition_penalty_not_supported`.

#### 7. Anthropic CLI Agents & Native PDF Blocks (Fails: 400 Bad Request)
* **Affected Tools**: **Claude Code CLI (`claude`)**, PDF document upload clients.
* **Reason**:
  * While NInfer exposes `/v1/messages`, it rejects server execution tools (such as Claude Code's `server_tool_use` bash executor) with `400 server tool content blocks require an executor that NInfer does not provide`.
  * Passing Anthropic `document` blocks (PDF uploads) returns `400 document blocks require document and citation semantics that NInfer does not provide`.

---

## License & Attribution

This project is licensed under the [Apache License 2.0](LICENSE).

### Derivative Work & Upstream Attribution
This repository is a Windows MSVC adaptation of the upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) project, originally authored by **Neroued** and licensed under the [Apache License 2.0](LICENSE).

In accordance with Apache License 2.0 Section 4:
- Modifications have been made to support native Windows MSVC compilation, C-runtime portability (random generation, thread-safe time handling, build database locks), FP8/NVFP4/K8V4 KV cache and MTP5/DFlash2 execution profiles on RTX 5090 (`sm_120a`), and Windows dependency tooling.
- All original attribution and copyright notices are retained. See the [NOTICE](NOTICE) file for third-party software details (`cpp-httplib`, `nlohmann/json`, `utf8proc`, `spdlog`, and FFmpeg).
