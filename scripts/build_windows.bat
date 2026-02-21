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

set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ISCC%" goto iscc_ready

where choco >nul 2>nul
if errorlevel 1 (
  echo Inno Setup not found and Chocolatey is unavailable.
  echo Install Inno Setup 6 or Chocolatey first.
  exit /b 1
)
choco install innosetup --no-progress -y
if errorlevel 1 (
  echo Failed to install Inno Setup.
  exit /b 1
)

:iscc_ready
if not exist "%ISCC%" (
  echo Inno Setup compiler not found at "%ISCC%".
  exit /b 1
)

for /f "usebackq delims=" %%v in (`python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"`) do set "RAW_VERSION=%%v"
for /f "tokens=1-3 delims=.-" %%a in ("%RAW_VERSION%") do set "INSTALLER_VERSION=%%a.%%b.%%c"
if "%INSTALLER_VERSION%"=="" set "INSTALLER_VERSION=0.1.0"
set "ROOT_DIR=%CD%"
set "BUILD_OUTPUT_DIR=%ROOT_DIR%\dist\sub-manager"
set "OUTPUT_DIR=%ROOT_DIR%\dist"

"%ISCC%" "/DAppVersion=%INSTALLER_VERSION%" "/DBuildOutputDir=%BUILD_OUTPUT_DIR%" "/DOutputDir=%OUTPUT_DIR%" installer\windows\sub-manager.iss

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo Build finished. See dist\ directory.
exit /b 0
