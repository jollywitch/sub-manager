from __future__ import annotations

import os
import shutil

from PySide6.QtCore import QSettings

import sub_manager.backend as _backend
from sub_manager.app import main as _app_main
from sub_manager.workers.download_workers import GlmOcrModelDownloadWorker


class AppBackend(_backend.AppBackend):
    def __init__(self) -> None:
        # Keep test monkeypatching compatibility via main.QSettings/main.shutil/main.os.
        super().__init__(settings_factory=QSettings, os_module=os, shutil_module=shutil)


def main() -> int:
    return _app_main()


__all__ = [
    "AppBackend",
    "GlmOcrModelDownloadWorker",
    "QSettings",
    "os",
    "shutil",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
