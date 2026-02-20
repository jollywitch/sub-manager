# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# In some CI executions, PyInstaller does not populate `__file__` in the
# spec namespace. Fall back to the current working directory.
_spec_file = globals().get("__file__")
project_root = Path(_spec_file).resolve().parent if _spec_file else Path.cwd().resolve()
qml_dir = project_root / "qml"

datas = []
if qml_dir.exists():
    datas.append((str(qml_dir), "qml"))

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sub-manager",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="sub-manager",
)
