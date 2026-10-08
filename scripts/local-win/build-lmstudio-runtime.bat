@echo off
rem ============================================================================
rem build-lmstudio-runtime.bat - one-shot build of the KVMem-enabled LM Studio
rem runtime with ALL KVMem switches ON by default, on the NEWEST llama.cpp line.
rem
rem Pins: submodule branch llama/trunk (llama.cpp b11347 = LM Studio 2.47+ era),
rem build dir build-lmstu-latest, packaged as ...cuda13-avx2-b11347.
rem The older validated b81c99b47 runtime stays untouched in
rem lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2\.
rem
rem Build-time switches, always ON:
rem   LLAMA_KVMEM=ON / GGML_CUDA=ON / GGML_CUDA_FA_ALL_QUANTS=ON
rem   KVMEM_DEFAULT_ON  (runtime starts with KVMem ENABLED by default)
rem Runtime defaults baked in, each overridable via env:
rem   enabled budget=32768 gen_reserve=16384 block=32 method=retrieval
rem   force off: setx KVMEM_ENABLE 0
rem Note: KVMEM_ENABLE_NVME stays OFF (upstream: not implemented on Windows).
rem MTP speculative decode is NOT wired on this base yet (upstream rewrote the
rem driver; kvmem logical-pos extension deferred) - do not pass --spec-type.
rem
rem Usage: build-lmstudio-runtime.bat [jobs] [cuda_archs]
rem ============================================================================
setlocal
set JOBS=%1
if "%JOBS%"=="" set JOBS=6
set ARCHS=%2
if "%ARCHS%"=="" set ARCHS=89-real;120a-real

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 echo [ERR] MSVC cl.exe not found & goto :fail
where cmake >nul 2>&1
if errorlevel 1 echo [ERR] cmake not found & goto :fail
where ninja >nul 2>&1
if errorlevel 1 echo [ERR] ninja not found & goto :fail

set CUDA_PATH=E:\Works\KVMem\cuda-toolkit
if not exist "%CUDA_PATH%\bin\nvcc.exe" echo [ERR] portable CUDA toolkit missing at %CUDA_PATH% & goto :fail
set PATH=%CUDA_PATH%\bin;%PATH%
"%CUDA_PATH%\bin\nvcc.exe" --version 2>nul | findstr /C:"release 13.2" >nul
if errorlevel 1 echo [ERR] nvcc is not CUDA 13.2.x - project requires 13.2.86 exactly & goto :fail

cd /d E:\Works\KVMem\kvmem-src
echo [0/5] pin submodule to the latest runtime line
git -C llama.cpp checkout -q llama/trunk
if errorlevel 1 goto :fail

echo [1/5] configure - all KVMem switches ON
cmake -S llama.cpp -B build-lmstu-latest -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%ARCHS% ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON ^
  -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT=E:\Works\KVMem\kvmem-src ^
  -DKVMEM_ENABLE_NVME=OFF ^
  -DCMAKE_CXX_FLAGS=/DKVMEM_DEFAULT_ON ^
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF
if errorlevel 1 goto :fail

echo [2/5] build llama-server with jobs=%JOBS%
cmake --build build-lmstu-latest --parallel %JOBS% --target llama-server
if errorlevel 1 goto :fail

echo [3/5] package runtime folder
set LMSTU_VER=0.16.0-rc3-b11347
set LMSTU_SRC=E:\Works\KVMem\kvmem-src\build-lmstu-latest\bin
set LMSTU_OUT=E:\Works\KVMem\lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2-b11347
bash "E:\Works\KVMem\package-lmstudio-runtime.sh"
if errorlevel 1 goto :fail

echo [4/5] smoke test with GPU hidden
set CUDA_VISIBLE_DEVICES=
set PKG=%LMSTU_OUT%
"%PKG%\llama-server.exe" --version > "%TEMP%\kvmem_rt_ver.txt" 2>&1
for /f %%i in ('git -C llama.cpp rev-parse --short HEAD') do set PORTSHA=%%i
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
