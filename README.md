## Build on Windows

Build from a Windows shell (PowerShell) in the project root.

1. Install dependencies:
```powershell
uv sync
```

2. Build folder-based app (recommended):
```powershell
.\scripts\build_windows.ps1
```

Or without PowerShell execution policy changes:
```bat
scripts\build_windows.bat
```

3. Build single-file app (optional):
```powershell
.\scripts\build_windows.ps1 -OneFile
```
```bat
scripts\build_windows.bat --onefile
```

Output:
- Folder build: `dist\sub-manager\sub-manager.exe`
- One-file build: `dist\sub-manager.exe`

Notes:
- QML assets are bundled into the executable build.
- Runtime writable data (logs, downloaded ffmpeg tools) is stored in:
  - `%LOCALAPPDATA%\sub-manager`
  - override with `SUB_MANAGER_HOME` environment variable.

## Automatic GitHub Release Builds

If you push a git tag that starts with `v` (for example, `v0.1.0`), GitHub Actions will:
- build Windows folder and one-file executables
- create/update a GitHub Release for that tag
- attach assets:
  - `sub-manager-windows-folder.zip`
  - `sub-manager.exe`

Example:
```bash
git tag v0.1.0
git push origin v0.1.0
```
