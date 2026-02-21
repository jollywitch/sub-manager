Param(
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
}

uv run pyinstaller sub-manager.spec --noconfirm --clean

$wixCommand = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wixCommand) {
    dotnet tool install --global wix --version 4.*
    $toolPath = Join-Path $env:USERPROFILE ".dotnet\tools"
    if ($env:PATH -notlike "*$toolPath*") {
        $env:PATH = "$env:PATH;$toolPath"
    }
}
wix extension add WixToolset.Heat | Out-Null

$rawVersion = (python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])").Trim()
if ($rawVersion -match '^(\d+)\.(\d+)\.(\d+)') {
    $msiVersion = "$($matches[1]).$($matches[2]).$($matches[3])"
}
else {
    $msiVersion = "0.1.0"
}

$wixDir = "build/wix"
if (-not (Test-Path $wixDir)) {
    New-Item -ItemType Directory -Path $wixDir | Out-Null
}
wix harvest directory dist/sub-manager `
    -dr INSTALLFOLDER `
    -cg AppFiles `
    -o build/wix/sub-manager-files.wxs

wix build installer/windows/sub-manager.wxs `
    build/wix/sub-manager-files.wxs `
    -arch x64 `
    -d ProductVersion=$msiVersion `
    -o dist/sub-manager.msi

Write-Host "Build finished. See dist/ directory."
