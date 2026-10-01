# Flip the nine PORT-DISPATCH Windows gates so the upstream FP8 A8 TMA route is selected.
#
# Used only by tools/scripts/verify_fp8_tma_route.cmd, which flips, runs and then reverts with
# `git checkout -- src/`. This is a diagnostic mutation of the working tree, NOT a build option:
# a build flag would be better, because a flag cannot leave the tree dirty if the runner dies.
#
# Written as a file rather than inlined through cmd: an earlier inline attempt had `$_` mangled
# into a literal path and created a file named `$_` in the repository root.
#
# Expected site count is asserted, per the repository's rule that a scripted edit needs an expected
# occurrence count and must be idempotent. Nine is derived from `git grep -c PORT-DISPATCH -- src/`.

$ErrorActionPreference = 'Stop'
# Two levels up: this file is in tools/scripts, so the repo root is its grandparent. Taking only the
# parent silently yields tools/, and `git grep -- src/` from there matches nothing -- which reads as
# "expected 9, found 0" and looks like a missing-gate problem rather than a wrong-path one.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $root '.git'))) { throw "repo root not resolved from $PSScriptRoot (got '$root')" }
$files = git -C $root grep -l 'PORT-DISPATCH' -- src/ | ForEach-Object { Join-Path $root $_ }
if ($files.Count -ne 9) { throw "expected 9 PORT-DISPATCH files, found $($files.Count)" }

$flipped = 0
foreach ($f in $files) {
    $lines = [System.IO.File]::ReadAllLines($f)
    $out = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $l = $lines[$i]
        if ($l.Trim() -eq '#ifdef _WIN32') {
            # The shapes/*.cu files put the PORT-DISPATCH comment AFTER the #ifdef, so look both ways.
            $s = [Math]::Max(0, $i - 15)
            $e = [Math]::Min($lines.Count - 1, $i + 15)
            $c = ($lines[$s..($i - 1)] -join '')
            if ($i -lt $lines.Count - 1) { $c += ($lines[($i + 1)..$e] -join '') }
            if ($c -match 'PORT-DISPATCH') {
                $out.Add($l.Replace('#ifdef _WIN32', '#if 0  // VERIFY-FP8-TMA-ROUTE'))
                $flipped++
                continue
            }
        }
        $out.Add($l)
    }
    [System.IO.File]::WriteAllLines($f, $out, (New-Object System.Text.UTF8Encoding($false)))
}
Write-Host "flipped $flipped gate sites"
if ($flipped -ne 9) { throw "expected 9, got $flipped" }
