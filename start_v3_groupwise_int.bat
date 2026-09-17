@echo off
REM ============================================================================
REM  Qwen3.8-27B groupwise-int on the v3 engine line - no speculative decoding
REM
REM  This artifact is upstream's own published v3 image
REM  (neroued/Qwen3.8-27B-NInfer, 19.03 GB). It serves at the full 262,144 context
REM  without speculation (measured: runtime 9.63 GiB, free 3.8 GiB).
REM
REM  --spec dflash2 is NOT available on it: upstream's published v3 image lacks the
REM  fused dflash2/layers/N/attention/query_key_value binding, so the draft attention
REM  cannot be bound whole. That gap is upstream's, not this build's -- the artifacts
REM  this tree migrates carry the binding (1518 bindings) and do run DFlash2.
REM  Serve this profile without speculation; it exists as a quality baseline.
REM ============================================================================
setlocal

set "V3=C:\AI\ninfer-v3-windows"
set "MODEL=C:\AI\models\qwen3_8_27b.ninfer"

"%V3%\build\apps\ninfer-serve.exe" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8088 ^
  --model-id qwen3.8-27b-groupwise-int-v3 ^
  --max-context 262144 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 8 ^
  --host-kv-mib 8192 ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000
