@echo off
setlocal

set CLEAN=1

:parse_args
if "%~1"=="" goto run_build
if /I "%~1"=="--no-clean" (
  set CLEAN=0
  shift
  goto parse_args
)
echo Unknown argument: %~1
echo Usage: scripts\build_windows.bat [--no-clean]
exit /b 1

:run_build
if "%CLEAN%"=="1" (
  if exist build rmdir /s /q build
)

uv run pyinstaller sub-manager.spec --noconfirm --clean
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

where wix >nul 2>nul
if errorlevel 1 (
  dotnet tool install --global wix --version 4.*
  if errorlevel 1 (
    echo Failed to install WiX.
    exit /b 1
  )
  set "PATH=%PATH%;%USERPROFILE%\.dotnet\tools"
)

for /f "usebackq delims=" %%v in (`python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"`) do set "RAW_VERSION=%%v"
for /f "tokens=1-3 delims=.-" %%a in ("%RAW_VERSION%") do set "MSI_VERSION=%%a.%%b.%%c"
if "%MSI_VERSION%"=="" set "MSI_VERSION=0.1.0"

wix build installer/windows/sub-manager.wxs -arch x64 -d BuildOutputDir=dist/sub-manager -d ProductVersion=%MSI_VERSION% -o dist/sub-manager.msi

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo Build finished. See dist\ directory.
exit /b 0
