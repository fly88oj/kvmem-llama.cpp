@echo off
rem ============================================================================
rem env.bat - shared machine-local environment for the fork's build scripts
rem (VS dev shell + portable CUDA 13.2 + ninja/python on PATH). Call with `call`.
rem ============================================================================
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul 2>&1
where cl >nul 2>&1 || (echo [ERR] MSVC cl.exe not found & exit /b 1)
set CUDA_PATH=E:\Works\KVMem\cuda-toolkit
if not exist "%CUDA_PATH%\bin\nvcc.exe" echo [ERR] portable CUDA toolkit missing at %CUDA_PATH% & exit /b 1
rem VS tools first (rc.exe must win over mingw windres); mingw64 supplies tools
rem pip lacks and stays LAST. The packaged runtime folder carries the CUDA
rem 13 runtime DLLs (cublas etc.) - the nvcc toolkit ships none, and test/bench
rem exes link against them (a missing DLL hangs the loader in non-interactive
rem sessions instead of failing loudly).
set PYSCRIPTS=C:\Users\FLYING\AppData\Local\Programs\Python\Python314\Scripts
set PATH=%CUDA_PATH%\bin;%PYSCRIPTS%;E:\Works\KVMem\lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2;%PATH%;C:\Users\FLYING\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin
exit /b 0
