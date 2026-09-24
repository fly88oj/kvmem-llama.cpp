# KVMem AMD/HIP - quick inference smoke test (baseline vs KVMem slot-pool).
#
# Runs llama-kvmem-cli twice: once WITHOUT KVMem (baseline HIP inference) and
# once WITH the KVMem retrieval path + host spill, then reports each exit code.
# Both runs are logged to files under the build dir so failures are inspectable.
# Requires a prior scripts\windows\build-hip.ps1 build.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\smoke.ps1 -Model D:\models\my.gguf
param(
    [string]$Model    = $env:KVMEM_MODEL,   # no hardcoded developer path: pass -Model or set $env:KVMEM_MODEL
    [string]$BuildDir = 'build-hip'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bin  = Join-Path $root "$BuildDir\bin\Release"
$cli  = Join-Path $bin 'llama-kvmem-cli.exe'
if (-not $Model)             { throw 'No model specified. Pass -Model <path-to.gguf> or set $env:KVMEM_MODEL.' }
if (-not (Test-Path $cli))   { throw "llama-kvmem-cli.exe not found in $bin - run scripts\windows\build-hip.ps1 first" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# Put the HIP runtime DLLs on PATH. build-hip.ps1 already stages them next to
# the exe; also add the ROCm SDK bin when the project venv can report it.
$sdk = Join-Path $root '.venv-rocm\Scripts\rocm-sdk.exe'
if (Test-Path $sdk) {
    $rocmBin = (& $sdk path --bin).Trim()
    if ($rocmBin) { $env:PATH = "$bin;$rocmBin;$env:PATH" } else { $env:PATH = "$bin;$env:PATH" }
} else {
    $env:PATH = "$bin;$env:PATH"
}
$env:GGML_LOG_VERBOSITY = '2'

function Invoke-Cli {
    param([string[]]$CliArgs, [string]$Label, [string]$Log)
    Write-Host "=== $Label ==="
    # Redirect ALL streams to a file and relax ErrorActionPreference for the
    # native call: a child process writing to stderr must not trip 'Stop'
    # (the classic PowerShell '2>&1 + ErrorActionPreference=Stop' trap).
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try     { & $cli @CliArgs *> $Log }
    finally { $ErrorActionPreference = $prev }
    $code = $LASTEXITCODE
    Get-Content $Log -Tail 40
    if ($code -ne 0) { throw "$Label failed (exit=$code); full log: $Log" }
    Write-Host "exit=$code`n"
}

Invoke-Cli -Label '[1/2] baseline HIP inference (no kvmem)' `
    -Log (Join-Path $root "$BuildDir\smoke-baseline.log") `
    -CliArgs @('-m', $Model, '-c', '4096', '-n', '48', '--temp', '0', '--no-prompt',
               'What is the capital of France? Answer in one short sentence.')

Invoke-Cli -Label '[2/2] KVMem slot-pool + host spill (2k ctx, cpu arena 6 GiB)' `
    -Log (Join-Path $root "$BuildDir\smoke-kvmem.log") `
    -CliArgs @('-m', $Model, '--kvmem', '-c', '2048', '-n', '48',
               '--kvmem-method', 'retrieval', '--kvmem-budget', '512', '--kvmem-gen-reserve', '256',
               '--kvmem-block-tokens', '32', '--kvmem-cpu-gb', '6', '--kv-dtype', 'q8_0',
               '--temp', '0', '--no-prompt',
               'List three prime numbers greater than 100, one per line.')

Write-Host 'smoke test PASSED (both runs exit=0)'
