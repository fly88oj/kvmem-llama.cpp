@echo off
rem Host-side CTest for the llamacpp backend build - the full suite
rem (kvmem store/tier/runtime + custom-server unit tests; mtp-kv-test skips
rem itself without an MTP GGUF, by upstream design). CUDA DLLs come from
rem env.bat.
call "%~dp0env.bat"
cd /d E:\Works\KVMem\kvmem-src
if not exist build-backends\llamacpp (
  echo [ERR] build-backends\llamacpp missing - run build.bat server first
  exit /b 1
)
ctest --test-dir build-backends\llamacpp --output-on-failure
