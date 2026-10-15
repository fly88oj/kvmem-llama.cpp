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
set PATH=%CUDA_PATH%\bin;C:\Users\FLYING\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;C:\Users\FLYING\AppData\Local\Programs\Python\Python314\Scripts;%PATH%
exit /b 0
