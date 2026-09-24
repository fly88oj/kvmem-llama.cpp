# KVMem AMD/HIP port — install the ROCm toolchain for Windows (gfx1201).
#
# Method copied from llama.cpp's own windows-setup-rocm GitHub action
# (.github/actions/windows-setup-rocm/action.yml): ROCm is distributed as
# Python wheels from repo.amd.com; `rocm-sdk init` expands the devel tree
# (clang++, hip headers, rocBLAS/hipBLASLt, cmake configs).
#
# Usage (from an ordinary PowerShell):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\rocm-install.ps1
#
# WARNING: downloads several GB into the venv below. Delete .venv-rocm to
# uninstall; nothing touches system state besides the venv.

$ErrorActionPreference = 'Stop'
$Root     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$VenvDir  = Join-Path $Root '.venv-rocm'
$RocmVer  = if ($env:ROCM_VERSION) { $env:ROCM_VERSION } else { '7.14.0' }

# The ROCm wheels currently target CPython <= 3.12. Pick a compatible
# interpreter if the default python is newer.
$py = $null
foreach ($cand in @('python', 'py -3.12', 'py -3.11', 'py -3.10')) {
    $v = & cmd /c "$cand -c `"import sys;print('%d.%d'%sys.version_info[:2])`" 2>nul"
    if ($LASTEXITCODE -eq 0 -and $v -match '3\.(1[0-2])') { $py = $cand; break }
}
if (-not $py) {
    Write-Host 'No Python 3.10-3.12 found. Install one with:'
    Write-Host '  winget install Python.Python.3.12'
    exit 1
}
Write-Host "Using interpreter: $py"

if (-not (Test-Path (Join-Path $VenvDir 'Scripts\python.exe'))) {
    & cmd /c "$py -m venv `"$VenvDir`""
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$venvPy = Join-Path $VenvDir 'Scripts\python.exe'

Write-Host "Installing ROCm $RocmVer wheels (libraries + devel) ..."
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel]==$RocmVer"
if ($LASTEXITCODE -ne 0) { throw 'pip install rocm failed' }

Write-Host 'Expanding the ROCm devel tree (rocm-sdk init) ...'
& (Join-Path $VenvDir 'Scripts\rocm-sdk.exe') init
if ($LASTEXITCODE -ne 0) { throw 'rocm-sdk init failed' }

$root = (& (Join-Path $VenvDir 'Scripts\rocm-sdk.exe') path --root).Trim()
if (-not $root) { throw 'rocm-sdk path --root returned empty' }
Write-Host ''
Write-Host "ROCm installed at: $root"
Write-Host 'Next: powershell -File .\scripts\windows\build-hip.ps1'
