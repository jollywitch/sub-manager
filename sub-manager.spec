# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# In some CI executions, PyInstaller does not populate `__file__` in the
# spec namespace. Fall back to the current working directory.
_spec_file = globals().get("__file__")
project_root = Path(_spec_file).resolve().parent if _spec_file else Path.cwd().resolve()
qml_dir = project_root / "qml"
runtime_python_bundle_dir = project_root / "build" / "runtime-python"

UNUSED_QT_MODULE_EXCLUDES = [
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DQuick",
    "PySide6.Qt3DQuickAnimation",
    "PySide6.Qt3DQuickExtras",
    "PySide6.Qt3DQuickInput",
    "PySide6.Qt3DQuickScene2D",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtPdf",
    "PySide6.QtPositioning",
    "PySide6.QtQuick3D",
    "PySide6.QtQuick3DAssetImport",
    "PySide6.QtQuick3DAssetUtils",
    "PySide6.QtQuick3DEffects",
    "PySide6.QtQuick3DHelpers",
    "PySide6.QtQuick3DParticles",
    "PySide6.QtQuick3DRuntimeRender",
    "PySide6.QtQuick3DSpatialAudio",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSpatialAudio",
    "PySide6.QtStateMachine",
    "PySide6.QtTextToSpeech",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtWebView",
]

LIGHTWEIGHT_PYTHON_EXCLUDES = [
    "PIL",
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "sympy",
    "networkx",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "hf_xet",
    "numpy",
    "numpy.libs",
]

UNUSED_QT_ARTIFACT_PREFIXES = (
    "PySide6/Qt63D",
    "PySide6/Qt6Charts",
    "PySide6/Qt6DataVisualization",
    "PySide6/Qt6Graphs",
    "PySide6/Qt6Location",
    "PySide6/Qt6Multimedia",
    "PySide6/Qt6Pdf",
    "PySide6/Qt6Positioning",
    "PySide6/Qt6Quick3D",
    "PySide6/Qt6RemoteObjects",
    "PySide6/Qt6Scxml",
    "PySide6/Qt6Sensors",
    "PySide6/Qt6SpatialAudio",
    "PySide6/Qt6StateMachine",
    "PySide6/Qt6TextToSpeech",
    "PySide6/Qt6VirtualKeyboard",
    "PySide6/Qt6WebChannel",
    "PySide6/Qt6WebEngine",
    "PySide6/Qt6WebSockets",
    "PySide6/Qt6WebView",
    "PySide6/qml/Qt3D",
    "PySide6/qml/QtCharts",
    "PySide6/qml/QtDataVisualization",
    "PySide6/qml/QtGraphs",
    "PySide6/qml/QtLocation",
    "PySide6/qml/QtMultimedia",
    "PySide6/qml/QtQuick3D",
    "PySide6/qml/QtSensors",
    "PySide6/qml/QtVirtualKeyboard",
    "PySide6/qml/QtWebChannel",
    "PySide6/qml/QtWebEngine",
    "PySide6/plugins/qmltooling",
)

HEAVY_RUNTIME_ARTIFACT_PREFIXES = (
    "PIL/",
    "torch/",
    "torchvision/",
    "torchaudio/",
    "triton/",
    "sympy/",
    "networkx/",
    "transformers/",
    "tokenizers/",
    "safetensors/",
    "huggingface_hub/",
    "hf_xet/",
    "numpy/",
    "numpy.libs/",
)

HEAVY_RUNTIME_ARTIFACT_PATTERNS = (
    "torch_cpu.dll",
    "torch_python.dll",
)

def _filter_unused_qt_artifacts(toc):
    filtered = []
    for entry in toc:
        dest_name = (entry[0] if isinstance(entry, tuple) else str(entry)).replace("\\", "/")
        if any(dest_name.startswith(prefix) for prefix in UNUSED_QT_ARTIFACT_PREFIXES):
            continue
        filtered.append(entry)
    return filtered


def _filter_heavy_runtime_artifacts(toc):
    filtered = []
    for entry in toc:
        dest_name = (entry[0] if isinstance(entry, tuple) else str(entry)).replace("\\", "/")
        if any(dest_name.startswith(prefix) for prefix in HEAVY_RUNTIME_ARTIFACT_PREFIXES):
            continue
        if any(pattern in dest_name for pattern in HEAVY_RUNTIME_ARTIFACT_PATTERNS):
            continue
        filtered.append(entry)
    return filtered


datas = []
if qml_dir.exists():
    datas.append((str(qml_dir), "qml"))
if runtime_python_bundle_dir.exists():
    datas.append((str(runtime_python_bundle_dir), "runtime-python"))

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=UNUSED_QT_MODULE_EXCLUDES + LIGHTWEIGHT_PYTHON_EXCLUDES,
    noarchive=False,
    optimize=0,
)
a.binaries = _filter_unused_qt_artifacts(a.binaries)
a.datas = _filter_unused_qt_artifacts(a.datas)
a.binaries = _filter_heavy_runtime_artifacts(a.binaries)
a.datas = _filter_heavy_runtime_artifacts(a.datas)
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
