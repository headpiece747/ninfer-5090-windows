@echo off
REM ============================================================================
REM  Qwen3.8-27B QUASAR QAT on the v3 engine line (Windows) - DFlash2 K=7
REM
REM  The v3 line: upstream v3 artifact architecture merged onto the Windows port,
REM  with the fork's fused-NVFP4 DFlash2 draft support re-expressed on upstream's
REM  shape table.
REM
REM  Measured (this machine, 32 GB RTX 5090):
REM      context 262,144 (fp8 KV, auto)   free VRAM 3.04 GiB
REM      decode  ~250-278 tok/s (DFlash2 K=7)     engine ready 5.6 s
REM      prefill 3,682 tok/s @ 200,061-token prompt
REM
REM  Requires the FFmpeg runtime DLLs beside the executable (staged by
REM  build_v3.cmd); without them the process exits 0xC0000135.
REM ============================================================================
setlocal

set "V3=C:\AI\ninfer-v3-windows"
set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer"

"%V3%\build\apps\ninfer-serve.exe" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8086 ^
  --model-id qwen3.8-27b-quasar-v3 ^
  --max-context 262144 ^
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
