Param(
    [switch]$OneFile = $false,
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
}

if ($OneFile) {
    uv run pyinstaller sub-manager-onefile.spec --noconfirm --clean
}
else {
    uv run pyinstaller sub-manager.spec --noconfirm --clean
}

Write-Host "Build finished. See dist/ directory."
