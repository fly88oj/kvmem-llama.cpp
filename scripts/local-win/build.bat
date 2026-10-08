@echo off
rem ============================================================================
rem build.bat - unified build entry for both deliverables of this fork
rem
rem Usage:
rem   build.bat              same as "build.bat all"
rem   build.bat server       build llama-kvmem-server (custom server, official pin)
rem   build.bat lmstudio     build the LM Studio runtime pack (latest base)
rem   build.bat all          both, sequential
rem   build.bat test         host-side CTest for the custom server line
rem
rem Each target pins its own llama.cpp submodule branch automatically:
rem   server    -> llama/custom-b81c99b47   (official pin + official patch)
rem   lmstudio  -> llama/b11347             (latest upstream + runtime port)
rem Artifacts:
rem   build-win\bin\llama-kvmem-server.exe
rem   lmstudio-runtime\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2-b11347\
rem ============================================================================
setlocal
set TARGET=%1
if "%TARGET%"=="" set TARGET=all

set ROOT=%~dp0

if /i "%TARGET%"=="server"   goto :server
if /i "%TARGET%"=="lmstudio" goto :lmstudio
if /i "%TARGET%"=="all"      goto :server
if /i "%TARGET%"=="test"     goto :test
echo [ERR] unknown target "%TARGET%" ^(server ^| lmstudio ^| all ^| test^)
exit /b 1

:server
echo === target: llama-kvmem-server (custom line) ===
call "%ROOT%build-wrapper.bat" %2 %3
if errorlevel 1 exit /b 1
if /i "%TARGET%"=="all" goto :lmstudio
exit /b 0

:lmstudio
echo === target: LM Studio runtime ===
call "%ROOT%build-lmstudio-runtime.bat" %2 %3
if errorlevel 1 exit /b 1
exit /b 0

:test
call "%ROOT%runtests.bat"
exit /b %errorlevel%
