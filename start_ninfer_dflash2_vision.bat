@echo off
echo Starting NInfer on RTX 5090 (Vision + 131k Context + FP8 + DFlash2)

set BIN=%~dp0build\apps\ninfer-serve.exe
set MODEL=%~dp0..\models\qwen3_8_27b_nvfp4.ninfer
if not exist "%MODEL%" set MODEL=%~dp0qwen3_8_27b_nvfp4.ninfer

"%BIN%" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --model-id qwen3.8-27b-dflash2-vision ^
  --max-context 131072 ^
  --kv-capacity 131072 ^
  --max-concurrency 1 ^
  --kv-dtype fp8 ^
  --prefill-chunk 1024 ^
  --device-state-slots 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 16384 ^
  --spec dflash2 --draft-tokens 7 ^
  --lm-head-draft ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000
pause
