@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
set PATH=E:\Works\KVMem\cuda-toolkit\bin;%PATH%
cmake --build E:\Works\KVMem\kvmem-src\build-win --parallel 8 --target llama-kvmem-server
