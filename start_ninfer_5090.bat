@echo off
echo Starting NInfer on RTX 5090 with NVFP4 + MTP3 Settings
build\apps\ninfer-serve.exe qwen3_8_27b_nvfp4.ninfer --host 127.0.0.1 --port 8080 --spec mtp --draft-tokens 3 --kv-dtype int8 --max-context 262144 --lm-head-draft --prefill-chunk 4096 --pending-timeout-ms 600000
pause
