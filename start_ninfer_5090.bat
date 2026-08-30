@echo off
echo Starting NInfer on RTX 5090 (Full 262k Context + FP8 + MTP5)
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
