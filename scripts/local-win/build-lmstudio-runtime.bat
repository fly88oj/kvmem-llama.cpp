@echo off
rem ============================================================================
rem build-lmstudio-runtime.bat - one-shot build of the KVMem-enabled LM Studio
rem runtime (stock llama-server with KVMEM_DEFAULT_ON) on the pinned 0.18.0
rem llamacpp backend (base + official patch + fork overlay).
rem
rem Build-time switches:
rem   KVMEM_BUILD_LLAMA=ON / GGML_CUDA=ON / GGML_CUDA_FA_QUANTS=all
rem   KVMEM_DEFAULT_ON (compile define: runtime starts with KVMem ENABLED)
rem Runtime defaults baked in, each overridable via env:
rem   budget=32768 gen_reserve=16384 block=32 method=retrieval
rem   force off: setx KVMEM_ENABLE 0 ; tiers: KVMEM_CPU_BYTES / KVMEM_NVME_BYTES /
rem   KVMEM_NVME_DIR ; KV type: KVMEM_CACHE_TYPE (f16 default is lowered to q8_0)
rem Packaged via E:\Works\KVMem\package-lmstudio-runtime.sh (backend-manifest v4).
rem
rem Usage: build-lmstudio-runtime.bat [jobs] [cuda_archs]
rem ============================================================================
setlocal
set JOBS=%1
if "%JOBS%"=="" set JOBS=8
set ARCHS=%2
if "%ARCHS%"=="" set ARCHS=89-real;120a-real

call "%~dp0env.bat"
if errorlevel 1 exit /b 1

cd /d E:\Works\KVMem\kvmem-src

echo [1/5] prepare backend (base + official patch + fork patch)
python scripts\prepare-backends.py --backend llamacpp
if errorlevel 1 exit /b 1

echo [2/5] configure - KVMEM_DEFAULT_ON runtime
cmake -S . -B build-lmstu-v018 -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DKVMEM_BUILD_LLAMA=ON -DKVMEM_BUILD_SERVER_TESTS=OFF ^
  -DGGML_CUDA=ON -DGGML_CUDA_FA_QUANTS=all -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=%ARCHS% ^
  -DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe -DCUDAToolkit_ROOT=%CUDA_PATH% ^
  -DKVMEM_ENABLE_NVME=ON ^
  -DCMAKE_CXX_FLAGS=/DKVMEM_DEFAULT_ON
if errorlevel 1 goto :fail

echo [3/5] build llama-server with jobs=%JOBS%
cmake --build build-lmstu-v018 --parallel %JOBS% --target llama-server
if errorlevel 1 goto :fail

echo [4/5] package runtime folder
set LMSTU_VER=0.18.0-fork
set LMSTU_SRC=E:\Works\KVMem\kvmem-src\build-lmstu-v018\bin
set LMSTU_OUT=E:\Works\KVMem\lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2-v018
bash "E:\Works\KVMem\package-lmstudio-runtime.sh"
if errorlevel 1 goto :fail

echo [5/5] smoke test with GPU hidden
set CUDA_VISIBLE_DEVICES=
set PKG=%LMSTU_OUT%
"%PKG%\llama-server.exe" --version > "%TEMP%\kvmem_rt_ver.txt" 2>&1
if errorlevel 1 type "%TEMP%\kvmem_rt_ver.txt" & del "%TEMP%\kvmem_rt_ver.txt" >nul 2>&1 & goto :fail
for /f %%i in ('git -C backends\llamacpp rev-parse --short HEAD') do set PORTSHA=%%i
findstr /C:"%PORTSHA%" "%TEMP%\kvmem_rt_ver.txt" >nul
if errorlevel 1 type "%TEMP%\kvmem_rt_ver.txt" & del "%TEMP%\kvmem_rt_ver.txt" >nul 2>&1 & goto :fail
del "%TEMP%\kvmem_rt_ver.txt" >nul 2>&1
echo [OK] runtime at %PKG%
endlocal
exit /b 0

:fail
echo [FAIL] build aborted
endlocal
exit /b 1
