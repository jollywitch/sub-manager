from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(str(meipass))
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def app_data_root() -> Path:
    override = str(os.environ.get("SUB_MANAGER_HOME", "") or "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
        if local_app_data:
            return Path(local_app_data) / "sub-manager"
        return Path.home() / "AppData" / "Local" / "sub-manager"
    return Path.home() / ".local" / "share" / "sub-manager"
