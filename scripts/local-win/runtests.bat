@echo off
rem Host-side CTest for the llamacpp backend build (kvmem store/tier/runtime
rem suites; the NVMe suites run on Windows through the fork platform layer).
rem The CUDA runtime DLLs (cublas etc.) live in the packaged runtime folder,
rem not in the portable nvcc toolkit - put both on PATH for the test binaries.
call "%~dp0env.bat"
set PATH=E:\Works\KVMem\lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2;%PATH%
cd /d E:\Works\KVMem\kvmem-src
if not exist build-backends\llamacpp (
  echo [ERR] build-backends\llamacpp missing - run build.bat server first
  exit /b 1
)
ctest --test-dir build-backends\llamacpp --output-on-failure -E "server|responses|session"
