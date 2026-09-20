# GPQA-Diamond campaign for the shipped Qwen3.8-27B lane, Windows port of
# run_qwen3_8_27b_nvfp4_gpqa_budgets.sh.
#
# Same three thinking-budget tiers (8k/16k/32k) and the same EvalScope suites, but it starts the
# server with the flags a *shipped* launcher passes (fp8 KV, DFlash2 depth 7, prefill 8192, one
# request at a time) instead of the bash script's Linux campaign flags (int8 KV, MTP depth 3,
# prefill 1024, concurrency 8). The published campaign measured the retired neroued NVFP4 image;
# this one measures what users download today, so its numbers describe the shipped profile.
#
# Prerequisites, none of which this script installs:
#   * eval/.venv with eval/requirements.txt (evalscope 1.10.0 + bfcl/ifbench/needle extras)
#   * a .ninfer artifact, and the datasets EvalScope fetches on first run
#   * an RTX 5090 with nothing else holding the model
#
# usage: .\run_qwen3_8_27b_quasar_gpqa_budgets.ps1 [-Tiers 8k,16k,32k] [-Artifact PATH]

[CmdletBinding()]
param(
    [string[]] $Tiers = @('8k', '16k', '32k'),
    [string]   $Artifact = 'C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$server = Join-Path $repo 'build\apps\ninfer-serve.exe'
$config = Join-Path $repo 'eval\configs\qwen3_8_27b_nvfp4_gpqa_budgets.yaml'
$python = Join-Path $repo 'eval\.venv\Scripts\python.exe'
$port = 18080
$modelId = 'qwen3.8-27b'   # must match eval/configs/qwen3_8_27b_nvfp4_gpqa_budgets.yaml
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$outputRoot = Join-Path $repo "profiles\eval\quasar_gpqa_budgets_$stamp"

foreach ($required in @($server, $Artifact, $config, $python)) {
    if (-not (Test-Path $required)) {
        throw "missing required file: $required"
    }
}

# The shipped flag set, per tools/release/profiles.py, plus the tier's own budget and ceiling.
$tierPlan = @{
    '8k'  = @{ suite = 'gpqa_8k';  budget = 8192;  context = 12352 }
    '16k' = @{ suite = 'gpqa_16k'; budget = 16384; context = 20544 }
    '32k' = @{ suite = 'gpqa_32k'; budget = 32768; context = 36928 }
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Write-Host "campaign output: $outputRoot"

foreach ($tier in $Tiers) {
    if (-not $tierPlan.ContainsKey($tier)) {
        throw "unknown tier: $tier; expected 8k, 16k or 32k"
    }
    $plan = $tierPlan[$tier]
    $tierDir = Join-Path $outputRoot $tier
    New-Item -ItemType Directory -Force -Path $tierDir | Out-Null
    $serverLog = Join-Path $tierDir 'server.log'
    $requestLog = Join-Path $tierDir 'server.requests.jsonl'

    Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "starting tier=$tier budget=$($plan.budget) context=$($plan.context)"

    $arguments = @(
        "`"$Artifact`"",
        '--host', '127.0.0.1',
        '--port', "$port",
        '--model-id', $modelId,
        '--max-context', "$($plan.context)",
        '--kv-capacity', 'auto',
        '--kv-dtype', 'fp8',
        '--max-concurrency', '1',
        '--device-state-slots', '1',
        '--host-state-slots', '8',
        '--host-kv-mib', '8192',
        '--max-shared-prefixes', '7',
        '--max-private-continuations', '8',
        '--max-long-anchors-per-continuation', '4',
        '--prefill-chunk', '8192',
        '--vision',
        '--spec', 'dflash2',
        '--draft-tokens', '7',
        '--lm-head-draft',
        '--preserve-thinking',
        '--default-thinking-budget', "$($plan.budget)",
        '--pending-timeout-ms', '86400000',
        '--request-log-jsonl', "`"$requestLog`""
    )
    $proc = Start-Process -FilePath $server -ArgumentList $arguments -PassThru `
        -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.err" -NoNewWindow

    $ready = $false
    for ($attempt = 1; $attempt -le 180; $attempt++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing | Out-Null
            $ready = $true
            break
        } catch {
            if ($proc.HasExited) {
                throw "ninfer-serve exited before becoming ready; see $serverLog"
            }
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) { throw "ninfer-serve did not become ready within 180 seconds; see $serverLog" }

    Write-Host "running suite=$($plan.suite); server_log=$serverLog"
    $env:PYTHONPATH = Join-Path $repo 'eval'
    & $python -m ninfer_eval run --config $config --suite $plan.suite
    if ($LASTEXITCODE -ne 0) { throw "ninfer_eval failed for tier $tier (exit $LASTEXITCODE)" }

    if (-not $proc.HasExited) { $proc | Stop-Process -Force }
    Write-Host "completed tier=$tier"
}

Write-Host "campaign completed: $outputRoot"
