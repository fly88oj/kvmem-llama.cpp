@echo off
rem ============================================================================
rem build-server.bat - llama-kvmem-server on the pinned 0.18.0 llamacpp backend
rem (base commit + official patch + fork overlay, all applied by
rem scripts/prepare-backends.py; the submodule HEAD stays at the base commit).
rem
rem Usage: build-server.bat [jobs] [cuda_archs]
rem ============================================================================
setlocal
set JOBS=%1
if "%JOBS%"=="" set JOBS=8
set ARCHS=%2
if "%ARCHS%"=="" set ARCHS=89-real;120a-real

call "%~dp0env.bat"
if errorlevel 1 exit /b 1

cd /d E:\Works\KVMem\kvmem-src

echo [1/3] prepare backend (base + official patch + fork patch)
python scripts\prepare-backends.py --backend llamacpp
if errorlevel 1 exit /b 1

echo [2/3] configure + build llama-kvmem-server (jobs=%JOBS%, archs=%ARCHS%)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build-backends.ps1 ^
  -Backend llamacpp -CudaPath "%CUDA_PATH%" -CudaArchitectures "%ARCHS%" -Jobs %JOBS%
if errorlevel 1 exit /b 1

echo [3/3] artifact check
if not exist build-backends\llamacpp\bin\llama-kvmem-server.exe (
  echo [WARN] build-backends\llamacpp\bin\llama-kvmem-server.exe not found - check the bin layout
)
echo [OK] server build done
endlocal
exit /b 0
