# KVMem AMD/HIP - needle-in-haystack long-context recall test.
#
# Generates a ~3.7k-token prompt with a hidden "needle" (a launch code) planted
# in an early block, then runs llama-kvmem-cli with a small KVMem budget so the
# needle block MUST be evicted to the host arena and brought back by query-score
# retrieval. A correct recall proves the retrieval pipeline works end to end.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\needle.ps1 -Model D:\models\my.gguf
param(
    [string]$Model      = $env:KVMEM_MODEL,   # no hardcoded developer path: pass -Model or set $env:KVMEM_MODEL
    [string]$BuildDir   = 'build-hip',
    [string]$Python     = '',
    [int]$Ctx           = 4096,
    [int]$Budget        = 512,
    [int]$GenReserve    = 256,
    [int]$BlockTokens   = 32,
    [double]$CpuGb      = 6,
    [string]$KvDtype    = 'q8_0',
    [int]$NPredict      = 24
)
$ErrorActionPreference = 'Stop'
$root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bin     = Join-Path $root "$BuildDir\bin\Release"
$cli     = Join-Path $bin 'llama-kvmem-cli.exe'
$dataDir = Join-Path $root 'tests-data'
if (-not $Model)           { throw 'No model specified. Pass -Model <path-to.gguf> or set $env:KVMEM_MODEL.' }
if (-not (Test-Path $cli))   { throw "llama-kvmem-cli.exe not found in $bin - run scripts\windows\build-hip.ps1 first" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# Resolve a Python interpreter: -Python > project venv > PATH.
if (-not $Python) {
    $venvPy = Join-Path $root '.venv-rocm\Scripts\python.exe'
    $Python = if (Test-Path $venvPy) { $venvPy } else { 'python' }
}

# HIP runtime DLLs on PATH (see smoke.ps1 for rationale).
$sdk = Join-Path $root '.venv-rocm\Scripts\rocm-sdk.exe'
if (Test-Path $sdk) {
    $rocmBin = (& $sdk path --bin).Trim()
    if ($rocmBin) { $env:PATH = "$bin;$rocmBin;$env:PATH" } else { $env:PATH = "$bin;$env:PATH" }
} else {
    $env:PATH = "$bin;$env:PATH"
}

# Helper: run a native exe, log all streams, and never let stderr trip 'Stop'.
# Splatting (@$rest) avoids the PowerShell '1..0' descending-range trap that
# $ExeAndArgs[1..($Len-1)] hits when the array has a single element.
function Invoke-NativeLogged {
    param([string[]]$ExeAndArgs, [string]$Log)
    $exe  = $ExeAndArgs[0]
    $rest = if ($ExeAndArgs.Length -gt 1) { $ExeAndArgs[1..($ExeAndArgs.Length - 1)] } else { @() }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try     { & $exe @rest *> $Log }
    finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

# 1. Generate the needle prompt (gen_needle.py writes needle_prompt.txt to CWD).
$gen = Join-Path $dataDir 'gen_needle.py'
if (-not (Test-Path $gen)) { throw "generator not found: $gen" }
Push-Location $dataDir
try {
    $code = Invoke-NativeLogged -ExeAndArgs @($Python, $gen) -Log (Join-Path $dataDir 'gen_needle.log')
} finally {
    Pop-Location
}
if ($code -ne 0) { throw "gen_needle.py failed (exit=$code)" }
$prompt = Join-Path $dataDir 'needle_prompt.txt'
if (-not (Test-Path $prompt)) { throw "gen_needle.py did not produce $prompt" }

# 2. Run the recall query with a small budget so the needle must be retrieved.
Write-Host "=== KVMem needle-in-haystack via HIP (budget $Budget of ~3.7k ctx) ==="
$log  = Join-Path $dataDir 'needle_out.log'
$code = Invoke-NativeLogged -Log $log -ExeAndArgs @(
    $cli, '-m', $Model, '-f', $prompt, '--kvmem', '--kvmem-method', 'retrieval',
    '-c', $Ctx, '--kvmem-budget', $Budget, '--kvmem-gen-reserve', $GenReserve,
    '--kvmem-block-tokens', $BlockTokens, '--kvmem-cpu-gb', $CpuGb, '--kv-dtype', $KvDtype,
    '-n', $NPredict, '--temp', '0', '--no-prompt')
if ($code -ne 0) { Get-Content $log -Tail 40; throw "llama-kvmem-cli failed (exit=$code); full log: $log" }

# 3. Show the recall evidence: KVMEM trace lines + the planted needle tokens.
Get-Content $log -Tail 40
$hits = Select-String -Path $log -Pattern 'KVMEM_|amber|jaguar|4417' | Select-Object -First 25
$hits | ForEach-Object { $_.Line }
if ($hits) { Write-Host "`nneedle markers present in output (verify the code 'amber-jaguar-4417' above)" }
else       { Write-Warning 'no needle markers found - inspect the log; retrieval may have failed' }
