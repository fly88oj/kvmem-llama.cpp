@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
set PATH=E:\Works\KVMem\cuda-toolkit\bin;%PATH%
cd /d E:\Works\KVMem\kvmem-src
ctest --test-dir build-win --output-on-failure -R "^(kvmem_store_test|pinned_kv_tier_test|nvme_disabled_test|kvmem_runtime_test|raw_kv_store_test)$"
