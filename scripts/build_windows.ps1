Param(
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
}

uv run python scripts/prepare_embedded_python.py
uv run pyinstaller sub-manager.spec --noconfirm --clean

$isccPath = Join-Path "${env:ProgramFiles(x86)}" "Inno Setup 6\ISCC.exe"
if (-not (Test-Path $isccPath)) {
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($null -eq $choco) {
        throw "Inno Setup not found and Chocolatey is unavailable. Install Inno Setup 6 or Chocolatey first."
    }
    choco install innosetup --no-progress -y
}
if (-not (Test-Path $isccPath)) {
    throw "Inno Setup compiler not found at '$isccPath'."
}

$rawVersion = (python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])").Trim()
if ($rawVersion -match '^(\d+)\.(\d+)\.(\d+)') {
    $installerVersion = "$($matches[1]).$($matches[2]).$($matches[3])"
}
else {
    $installerVersion = "0.1.0"
}

$buildOutputDir = (Join-Path $PWD "dist/sub-manager")
$outputDir = (Join-Path $PWD "dist")

& $isccPath `
    "/DAppVersion=$installerVersion" `
    "/DBuildOutputDir=$buildOutputDir" `
    "/DOutputDir=$outputDir" `
    "installer\\windows\\sub-manager.iss"

Write-Host "Build finished. See dist/ directory."
