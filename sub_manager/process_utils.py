from __future__ import annotations

import os
import subprocess
from typing import Any


def windows_hidden_subprocess_kwargs() -> dict[str, Any]:
    """Return subprocess kwargs that suppress console windows on Windows."""
    if os.name != "nt":
        return {}

    kwargs: dict[str, Any] = {}
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    if creationflags:
        kwargs["creationflags"] = creationflags

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0) or 0)
    kwargs["startupinfo"] = startupinfo
    return kwargs
