@echo off
rem ============================================================================
rem build.bat - unified build entry for this fork (upstream 0.18.0 layout)
rem
rem Usage:
rem   build.bat              same as "build.bat server"
rem   build.bat server       llama-kvmem-server via scripts/windows/build-backends.ps1
rem   build.bat lmstudio     KVMEM_DEFAULT_ON llama-server for the LM Studio runtime
rem   build.bat all          server, then lmstudio
rem   build.bat test         host-side CTest suite
rem
rem The llamacpp submodule is pinned to the upstream base commit; prepare-backends
rem applies the official patch and then the fork overlay patch
rem (backends/patches/llamacpp-kvmem-fork.patch). Artifacts:
rem   build-backends\llamacpp\bin\llama-kvmem-server.exe
rem   build-lmstu-v018\bin\llama-server.exe
rem ============================================================================
setlocal
set TARGET=%1
if "%TARGET%"=="" set TARGET=server

if /i "%TARGET%"=="server"   goto :server
if /i "%TARGET%"=="lmstudio" goto :lmstudio
if /i "%TARGET%"=="all"      goto :server
if /i "%TARGET%"=="test"     goto :test
echo [ERR] unknown target "%TARGET%" ^(server ^| lmstudio ^| all ^| test^)
exit /b 1

:server
echo === target: llama-kvmem-server (llamacpp backend) ===
call "%~dp0build-server.bat" %2 %3
if errorlevel 1 exit /b 1
if /i "%TARGET%"=="all" goto :lmstudio
exit /b 0

:lmstudio
echo === target: LM Studio runtime (KVMEM_DEFAULT_ON llama-server) ===
call "%~dp0build-lmstudio-runtime.bat" %2 %3
if errorlevel 1 exit /b 1
exit /b 0

:test
call "%~dp0runtests.bat"
exit /b %errorlevel%
