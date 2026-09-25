# make-lms-extension.ps1 - build an LM Studio runtime extension pack from a
# KVMem HIP shared build (build-hip.ps1 -LmsShared).
#
# Method (same overlay approach community forks use for LM Studio 0.4.x):
# clone the official ROCm backend directory (keeps the LM Studio engine/binding
# layer: llm_engine*.dll, *.node, lmstudiocore.dll, MSVC runtime) and overwrite
# exactly the files listed in engine-protocol-server-artifacts.json with our
# shared build, plus the staged ROCm runtime DLLs. The manifest version is
# bumped to an unused patch so the pack installs side-by-side and can be
# selected via Ctrl+Shift+R / `lms runtime`.
#
# KVMem knobs reach the stock llama-server entry point via KVMEM_* environment
# variables (setx / shell env; the adapter applies them when the host never
# calls llama_kvmem_set_params). See docs/lmstudio-runtime-packaging.md.
param(
    [string]$BuildDir = 'build-hip-lms',
    [string]$TemplateBackend = '',      # default: newest official rocm backend in the LM Studio dir
    [string]$NewVersion  = '',          # default: template version +0.0.1
    [string]$OutDir = 'dist-lms',
    [switch]$Install                          # also copy into %USERPROFILE%\.lmstudio\extensions\backends
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

$bin = Join-Path $BuildDir 'bin\Release'
if (-not (Test-Path (Join-Path $bin 'llama-server.exe'))) {
    throw "shared build not found in $bin - run scripts/windows/build-hip.ps1 -LmsShared -BuildDir $BuildDir first"
}

$backendsRoot = Join-Path $env:USERPROFILE '.lmstudio\extensions\backends'
if (-not $TemplateBackend) {
    $TemplateBackend = Get-ChildItem $backendsRoot -Directory -Filter 'llama.cpp-win-x86_64-amd-rocm-avx2-*' |
        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $TemplateBackend -or -not (Test-Path $TemplateBackend)) { throw 'no ROCm template backend found - install the ROCm runtime in LM Studio first' }
$tplManifest = Join-Path $TemplateBackend 'backend-manifest.json'
$tpl = Get-Content $tplManifest -Raw | ConvertFrom-Json
if (-not $NewVersion) {
    $p = $tpl.version.Split('.')
    $NewVersion = "{0}.{1}.{2}" -f $p[0], $p[1], ([int]$p[2] + 1)
}
$name = $tpl.name                                        # llama.cpp-win-x86_64-amd-rocm-avx2
$outBackend = Join-Path $OutDir "$name-$NewVersion"
Write-Host "template: $(Split-Path $TemplateBackend -Leaf)  ->  $name-$NewVersion"

if (Test-Path $outBackend) { Remove-Item $outBackend -Recurse -Force }
New-Item -ItemType Directory -Path $outBackend -Force | Out-Null

# 1. Clone the official pack (engine layer, bindings, MSVC runtime included).
Copy-Item (Join-Path $TemplateBackend '*') $outBackend -Recurse -Force

# 2. Overlay exactly the artifact set LM Studio's engine layer talks to.
#    The unified-ggml build (GGML_BACKEND_DL=OFF) produces ggml.dll instead of
#    ggml-{base,cpu,hip}.dll; the official per-backend dlls then remain from
#    the clone (harmless: our llama.dll embeds its own ggml and never imports
#    them, while the engine layer can still resolve the official ones).
$artifacts = Get-Content (Join-Path $TemplateBackend 'engine-protocol-server-artifacts.json') -Raw | ConvertFrom-Json
foreach ($f in $artifacts.files) {
    $src = Join-Path $bin $f.relative_path
    if (Test-Path $src) {
        Copy-Item $src $outBackend -Force
        Write-Host "  overlaid $($f.relative_path)"
    } elseif ($f.relative_path -match '^(ggml-(base|cpu|hip)|ggml_llamacpp)\.dll$') {
        Write-Host "  kept official $($f.relative_path) (unified ggml build)"
    } else {
        throw "artifact missing from build: $($f.relative_path)"
    }
}
if (Test-Path (Join-Path $bin 'ggml.dll')) {
    Copy-Item (Join-Path $bin 'ggml.dll') $outBackend -Force
    Write-Host '  overlaid ggml.dll (unified backend)'
}

# 3. ROCm runtime DLLs staged by build-hip.ps1 ride along (self-contained).
foreach ($pat in @('amdhip64*.dll', 'rocm_kpack.dll', 'amd_comgr*.dll', 'hipblas*.dll', 'rocblas*.dll', 'rocsolver*.dll', 'libhipblaslt*.dll')) {
    Get-ChildItem (Join-Path $bin $pat) -EA SilentlyContinue | Copy-Item -Destination $outBackend -Force
}

# 4. Manifest: bump version; keep protocol/vendor/minimum fields from template.
#    Write BOM-less UTF-8 (PS5.1 Set-Content -Encoding UTF8 would add a BOM the
#    official packs never ship).
$manifest = Join-Path $outBackend 'backend-manifest.json'
$mm = Get-Content $manifest -Raw | ConvertFrom-Json
$mm.version = "$NewVersion"
[IO.File]::WriteAllText($manifest, ($mm | ConvertTo-Json -Depth 8))

# 5. Display name marker (both languages if present).
$ddPath = Join-Path $outBackend 'display-data.json'
$dd = Get-Content $ddPath -Raw | ConvertFrom-Json
foreach ($entry in $dd) { if ($entry[1].displayName -notmatch 'KVMem') { $entry[1].displayName += ' (KVMem)' } }
[IO.File]::WriteAllText($ddPath, ($dd | ConvertTo-Json -Depth 12 -Compress))

Write-Host "pack: $outBackend"
if ($Install) {
    $dest = Join-Path $backendsRoot "$name-$NewVersion"
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Copy-Item $outBackend $dest -Recurse
    Write-Host "installed to $dest"
    Write-Host 'select it in LM Studio: Ctrl+Shift+R -> llama.cpp (Windows AMD ROCm) -> version'
}
