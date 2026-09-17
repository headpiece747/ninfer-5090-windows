@echo off
REM ============================================================================
REM  Qwen3.8-27B NVFP4 (official profile) on the v3 engine line - DFlash2 K=7
REM
REM  Measured (32 GB RTX 5090): this artifact carries 19.0 GiB of weights versus
REM  QUASAR's 15.3 GiB, so the context ceiling is materially lower. The engine
REM  requires (runtime + 1 GiB headroom) <= available-after-weights (9.21 GiB):
REM
REM      max-context 262144 -> runtime 10.58 GiB   REFUSED
REM      max-context 240000 -> runtime  9.89 GiB   REFUSED
REM      max-context 212992 -> runtime  9.06 GiB   REFUSED
REM      max-context 180224 -> runtime  8.06 GiB   SERVES (free 1.16 GiB)
REM
REM  (Your v2 launcher used 240000 for this artifact; the v3 engine's accounting
REM  does not fit it, which is why this line is pinned lower.)
REM ============================================================================
setlocal

set "V3=C:\AI\ninfer-v3-windows"
set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer"

"%V3%\build\apps\ninfer-serve.exe" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8087 ^
  --model-id qwen3.8-27b-nvfp4-v3 ^
  --max-context 180224 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --spec dflash2 --draft-tokens 7 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 8 ^
  --host-kv-mib 8192 ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000
