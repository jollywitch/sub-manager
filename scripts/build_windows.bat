@echo off
setlocal

set ONEFILE=0
set CLEAN=1

:parse_args
if "%~1"=="" goto run_build
if /I "%~1"=="--onefile" (
  set ONEFILE=1
  shift
  goto parse_args
)
if /I "%~1"=="--no-clean" (
  set CLEAN=0
  shift
  goto parse_args
)
echo Unknown argument: %~1
echo Usage: scripts\build_windows.bat [--onefile] [--no-clean]
exit /b 1

:run_build
if "%CLEAN%"=="1" (
  if exist build rmdir /s /q build
)

if "%ONEFILE%"=="1" (
  uv run pyinstaller main.py --name sub-manager --onefile --windowed --noconfirm --clean --add-data "qml;qml"
) else (
  uv run pyinstaller sub-manager.spec --noconfirm --clean
)

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo Build finished. See dist\ directory.
exit /b 0
