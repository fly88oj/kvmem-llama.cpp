@echo off
rem Host-side CTest for the llamacpp backend build (kvmem store/tier/runtime
rem suites; the NVMe suites run on Windows through the fork platform layer).
call "%~dp0env.bat"
cd /d E:\Works\KVMem\kvmem-src
if not exist build-backends\llamacpp (
  echo [ERR] build-backends\llamacpp missing - run build.bat server first
  exit /b 1
)
ctest --test-dir build-backends\llamacpp --output-on-failure -E "server|responses|session"
