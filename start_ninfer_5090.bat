@echo off
echo Starting NInfer on RTX 5090 (Full 262k Context + FP8 + MTP5)

set BIN=%~dp0build\apps\ninfer-serve.exe
set MODEL=%~dp0..\models\qwen3_8_27b_nvfp4.ninfer
if not exist "%MODEL%" set MODEL=%~dp0qwen3_8_27b_nvfp4.ninfer

"%BIN%" "%MODEL%" ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --model-id qwen3.8-27b ^
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
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000
pause
