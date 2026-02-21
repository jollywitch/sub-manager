## Build on Windows

Build from a Windows shell (PowerShell) in the project root.

1. Install dependencies:
```powershell
uv sync
```

2. Build Windows installer:
```powershell
.\scripts\build_windows.ps1
```

Or without PowerShell execution policy changes:
```bat
scripts\build_windows.bat
```

Output:
- Installer: `dist\sub-manager-setup.exe`

Notes:
- QML assets are bundled into the executable build.
- GLM-OCR Python runtime dependencies are installed on demand at first use.
- Runtime writable data (logs, downloaded ffmpeg tools) is stored in:
  - `%LOCALAPPDATA%\sub-manager`
  - override with `SUB_MANAGER_HOME` environment variable.

## Automatic GitHub Release Builds

If you push a git tag that starts with `v` (for example, `v0.1.0`), GitHub Actions will:
- build a Windows installer (`Inno Setup`)
- create/update a GitHub Release for that tag
- attach assets:
  - `sub-manager-setup.exe`

Example:
```bash
git tag v0.1.0
git push origin v0.1.0
```
