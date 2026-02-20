Param(
    [switch]$OneFile = $false,
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
}

if ($OneFile) {
    uv run pyinstaller main.py --name sub-manager --onefile --windowed --noconfirm --clean --add-data "qml;qml"
}
else {
    uv run pyinstaller sub-manager.spec --noconfirm --clean
}

Write-Host "Build finished. See dist/ directory."
