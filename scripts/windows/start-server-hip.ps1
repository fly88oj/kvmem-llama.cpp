# KVMem AMD/HIP — one-command server launcher (no PATH setup needed).
# All ROCm DLLs are staged next to the exe; the rocBLAS Tensile library
# resolves via the project venv, so keep .venv-rocm in place.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\start-server-hip.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\start-server-hip.ps1 -Model D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -Ctx 262144 -Budget 36864 -GenReserve 16384 -Mtp 3
param(
    [string]$Model = $env:KVMEM_MODEL,   # optional: auto-discovers an LM Studio GGUF when omitted
    [int]$Port = 18200,
    [int]$Ctx = 16384,                   # final-recipe defaults (see AMD-HIP-PORT.md)
    [int]$Budget = 4096,
    [int]$GenReserve = 1024,
    [int]$BlockTokens = 128,
    [double]$CpuGb = 6,
    [string]$KvDtype = 'q8_0',
    [int]$Mtp = 0,                       # 0 = off; 3 = draft-mtp n_max 3 (27B recipe)
    [string]$Mmproj = '',
    [switch]$NoUi
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bin  = Join-Path $root 'build-hip\bin\Release'
if (-not (Test-Path (Join-Path $bin 'llama-kvmem-server.exe'))) {
    throw 'build-hip binaries not found - run scripts\windows\build-hip.ps1 first'
}
if (-not $Model) {
    # Auto-discovery: first non-mmproj GGUF under the LM Studio model store.
    # Personal benchmark models are intentionally NOT hard-coded here; prefer
    # -Model <path> or $env:KVMEM_MODEL for a deterministic choice.
    $first = Get-ChildItem (Join-Path $env:USERPROFILE '.lmstudio\models') -Recurse -Filter *.gguf -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notmatch 'mmproj' } | Select-Object -First 1
    if ($first) { $Model = $first.FullName }
    if ($Model) { Write-Host "auto-detected model: $Model" }
}
if (-not $Model) { throw 'No model specified. Pass -Model <path-to.gguf> or set $env:KVMEM_MODEL.' }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
$srvArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', $Port, '-c', $Ctx,
             '--kvmem-budget', $Budget, '--kvmem-gen-reserve', $GenReserve,
             '--kvmem-block-tokens', $BlockTokens,
             '--kvmem-cpu-gb', $CpuGb, '-ctk', $KvDtype, '-ctv', $KvDtype)
if ($Mtp -gt 0) { $srvArgs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', $Mtp) } else { $srvArgs += @('--spec-type', 'none') }
if ($Mmproj)    { $srvArgs += @('--mmproj', $Mmproj) }
if ($NoUi)      { $srvArgs += '--no-ui' }
Write-Host "llama-kvmem-server $($srvArgs -join ' ')"
& (Join-Path $bin 'llama-kvmem-server.exe') @srvArgs
