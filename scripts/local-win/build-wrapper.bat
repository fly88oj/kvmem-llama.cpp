@echo off
rem Build wrapper: VS dev env + portable CUDA + cmake/ninja, then official build.ps1
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
where cl >nul 2>&1 || (echo NO CL & exit /b 1)
set PATH=E:\Works\KVMem\cuda-toolkit\bin;C:\Users\FLYING\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;C:\Users\FLYING\AppData\Local\Programs\Python\Python314\Scripts;%PATH%
rem pin the custom-server line (official pin + official patch, committed on branch)
git -C "E:\Works\KVMem\kvmem-src\llama.cpp" checkout -q kvmem-custom-b81c99b47
if errorlevel 1 exit /b 1
cd /d E:\Works\KVMem\kvmem-src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -CudaPath E:\Works\KVMem\cuda-toolkit -BuildOnly -Jobs 8
