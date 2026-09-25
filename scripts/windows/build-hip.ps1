# KVMem AMD/HIP port — configure + build llama-kvmem-server/cli for
# Radeon RX 9070 XT (gfx1201) on native Windows, mirroring llama.cpp's
# windows-rocm release job (release.yml): ROCm clang/clang++ as the C/C++
# compilers, hip::device for the GPU objects, GPU targets passed via
# AMDGPU_TARGETS.
#
# Prerequisites:
#   1. scripts\windows\rocm-install.ps1 (or an equivalent ROCm SDK install)
#   2. VS Build Tools C++ workload (for Windows SDK headers/libs)
#   3. cmake + ninja on PATH (winget install Kitware.CMake Ninja.Ninja)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build-hip.ps1
# Options:
#   -GpuTarget gfx1201   -BuildDir build-hip   -RocmRoot <path>   -HostOnly

param(
    [string]$GpuTarget = 'gfx1201',
    # LM Studio runtime pack build: shared llama/ggml libraries with dynamic
    # backend dlls (ggml-base/cpu/hip split), matching the file set in
    # engine-protocol-server-artifacts.json. Used by make-lms-extension.ps1.
    [switch]$LmsShared,
    [string]$BuildDir  = 'build-hip',
    [string]$RocmRoot  = '',
    [int]$Jobs         = [math]::Min(16, [Environment]::ProcessorCount),  # 16C/32T sweet spot: clang HIP TUs peak ~2 GB each, 16x2<47 GB RAM
    [switch]$HostOnly
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

# --- locate ROCm (explicit > venv from rocm-install.ps1 > HIP_PATH) ---
if (-not $RocmRoot) {
    $sdk = Join-Path $Root '.venv-rocm\Scripts\rocm-sdk.exe'
    if (Test-Path $sdk) {
        $RocmRoot = (& $sdk path --root).Trim()
        $RocmBin  = (& $sdk path --bin).Trim()
    } elseif ($env:HIP_PATH) {
        $RocmRoot = $env:HIP_PATH
        $RocmBin  = Join-Path $env:HIP_PATH 'bin'
    }
}
if (-not $HostOnly -and -not $RocmRoot) {
    throw 'ROCm not found. Run scripts\windows\rocm-install.ps1 first or pass -RocmRoot.'
}
# Explicit -RocmRoot skips the block above; derive RocmBin so DLL staging
# never silently no-ops (issue: missing amdhip64/rocsolver next to the exe).
if (-not $RocmBin -and $RocmRoot) { $RocmBin = Join-Path $RocmRoot 'bin' }
Write-Host "ROCm root: $RocmRoot"

# --- locate cmake / ninja (PATH first, then VS Build Tools bundle) ---
$cmakeBin = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmakeBin) {
    $cmakeBin = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cmakeBin) { throw 'cmake.exe not found (install VS C++ workload or winget install Kitware.CMake)' }
$ctestBin = Join-Path (Split-Path $cmakeBin) 'ctest.exe'
$vsCMakeBinDir = Split-Path $cmakeBin
$ninjaBin = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninjaBin) {
    $ninjaBin = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe' -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ninjaBin) { throw 'ninja.exe not found (install VS C++ workload or winget install Ninja-build.Ninja)' }

# --- MSVC environment (Windows SDK headers/libs for clang++) ---
$vcvars = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName
if ($vcvars) {
    Write-Host "Importing MSVC environment from: $vcvars"
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
    }
} else {
    Write-Warning 'vcvars64.bat not found — run this script from an x64 Native Tools prompt instead.'
}

# --- apply the KVMem patches to the pinned submodule if pending ---
Push-Location (Join-Path $Root 'llama.cpp')
$patch = Join-Path $Root 'patches\llama-kvmem-current.patch'
# Presence of the factory header means the cumulative patch is (or was) applied;
# local additions (the GGML_HIP branch in src/CMakeLists.txt) then must not trip
# the strict apply --check.
if (Test-Path 'src\llama-kvmem-factory.h') {
    Write-Host 'KVMem base patch already applied.'
} elseif (git apply --check $patch 2>$null) {
    git apply $patch
    Write-Host 'Applied KVMem base patch.'
} else {
    throw 'llama.cpp tree neither clean-patchable nor patched — inspect manually.'
}
# HIP cmake delta (GGML_HIP branch in src/CMakeLists.txt). Without it a fresh
# tree configures with GGML_HIP=ON but never compiles llama-kvmem-stagein.cu.
$hipPatch = Join-Path $Root 'patches\kvmem-hip-port-cmake.patch'
if (-not $HostOnly) {
    if (Select-String -Path 'src\CMakeLists.txt' -Pattern 'GGML_HIP' -Quiet) {
        Write-Host 'HIP cmake delta already applied.'
    } elseif (git apply --check $hipPatch 2>$null) {
        git apply $hipPatch
        Write-Host 'Applied HIP cmake delta.'
    } else {
        throw 'HIP cmake delta neither clean-patchable nor applied — inspect src/CMakeLists.txt.'
    }
}
Pop-Location

# --- configure ---
# Work around llama.cpp issue #22570: MSVC 14.5x <cmath> overloads collide
# with clang HIP math forward-declares. Patch the project-local ROCm devel
# tree (idempotent) and enable the guard macro.
$pyForPatch = Join-Path $Root '.venv-rocm\Scripts\python.exe'
if (-not (Test-Path $pyForPatch)) { $pyForPatch = 'python' }
& $pyForPatch (Join-Path $PSScriptRoot 'patch-rocm-headers.py') $RocmRoot
if ($LASTEXITCODE -ne 0) { throw 'ROCm header patch failed' }

$cmakeArgs = @(
    '-S', '.', '-B', $BuildDir,
    '-G', 'Ninja Multi-Config',
    '-DLLAMA_KVMEM=ON',
    '-DKVMEM_ENABLE_NVME=ON',
    "-DCMAKE_MAKE_PROGRAM=$($ninjaBin -replace '\\','/')",
    "-DLLAMA_KVMEM_ROOT=$($Root -replace '\\','/')"
)
if ($HostOnly) {
    # CPU-only libkvmem + host tests: no GPU toolchain needed.
    $cmakeArgs += @('-DGGML_HIP=OFF', '-DGGML_CUDA=OFF')
} else {
    $clang    = (Join-Path $RocmRoot 'lib/llvm/bin/clang.exe')
    $clangxx  = (Join-Path $RocmRoot 'lib/llvm/bin/clang++.exe')
    $cmakeArgs += @(
        '-DGGML_HIP=ON', '-DGGML_CUDA=OFF',
        "-DCMAKE_PREFIX_PATH=$($RocmRoot -replace '\\','/')",
        "-DCMAKE_C_COMPILER=$($clang -replace '\\','/')",
        "-DCMAKE_CXX_COMPILER=$($clangxx -replace '\\','/')",
        "-DCMAKE_HIP_COMPILER=$($clang -replace '\\','/')",
        "-DHIP_PATH=$($RocmRoot -replace '\\','/')",
        "-DAMDGPU_TARGETS=$GpuTarget",
        "-DGPU_TARGETS=$GpuTarget",
        '-DGGML_NATIVE=OFF',
        '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types -DKVMEM_HIP_SKIP_MATH_FWD',
        '-DCMAKE_CXX_FLAGS=-DKVMEM_HIP_SKIP_MATH_FWD'
    )
    if ($LmsShared) {
        # Shared tree for the LM Studio extension pack (see artifacts list:
        # llama.dll llama-common.dll llama-server-impl.dll ggml-*.dll mtmd.dll).
        # GGML_BACKEND_DL stays OFF: one unified ggml.dll — the adapter's direct
        # calls into CUDA/HIP backend entry points (gdn_fold) are not resolvable
        # across plugin-dll boundaries. The official per-backend dlls remain in
        # the pack untouched (LM Studio engine layer reads them if it must).
        $cmakeArgs += @('-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=OFF',
                        '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                        '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON',
                        '-DLLAMA_BUILD_APP=ON')
    }
    $env:HIP_PLATFORM = 'amd'
    $env:HSA_OVERRIDE_GFX_VERSION = $null   # unsupported on Windows; gfx1201 is real
}
& $cmakeBin @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }

# --- build ---
if ($LmsShared) {
    # Whitelist targets: KVMem's own test exes use llama-internal C++ classes
    # that a shared llama.dll does not export (fine for the static tree, not
    # for this one) — the LM Studio pack only needs the artifact targets below.
    & $cmakeBin --build $BuildDir --config Release --parallel $Jobs --target llama llama-common llama-server mtmd
} else {
    & $cmakeBin --build $BuildDir --config Release --parallel $Jobs
}
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

# --- model-free tests ---
if (-not $LmsShared) {
    & $ctestBin --test-dir $BuildDir -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { Write-Warning 'ctest reported failures — inspect before GPU validation.' }
}

# --- stage HIP runtime DLLs next to the binaries ---
# Adrenalin ships an amdhip64_7.dll in System32 that the loader finds before
# PATH (llama.cpp issue #26929); copying the matching ROCm DLLs next to the
# exe forces the correct version.
if (-not $HostOnly) {
    if (-not $RocmBin) { throw 'ROCm bin path unresolved — cannot stage HIP runtime DLLs.' }
    $bin = Join-Path $BuildDir 'bin\Release'
    # Dependency chain: ggml-hip -> hipblas -> rocblas -> rocsolver. Without
    # rocsolver.dll next to the exe, double-click launch fails with a
    # "cannot find rocsolver.dll" system error (the message is localized, e.g.
    # zh-CN Windows). rocBLAS resolves its Tensile library via the baked-in
    # devel path, so only the DLL itself must be staged.
    foreach ($pat in @('amdhip64*.dll', 'rocm_kpack.dll', 'amd_comgr*.dll', 'hipblas*.dll', 'rocblas*.dll', 'rocsolver*.dll', 'libhipblaslt*.dll')) {
        Get-ChildItem (Join-Path $RocmBin $pat) -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName $bin -Force
            Write-Host "  staged $($_.Name)"
        }
    }
    Write-Host ''
    Write-Host "Done. Server: $bin\llama-kvmem-server.exe"
    Write-Host "Recipe:   .\llama-kvmem-server.exe -m model.gguf -c 262144 --kvmem-budget 36864 --kvmem-gen-reserve 16384 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3"
}
