from __future__ import annotations

import logging
import os
import sys
import threading

from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from sub_manager.backend import AppBackend
from sub_manager.constants import LOG_DIR, SESSION_LOG_PATH
from sub_manager.paths import resource_root

def _log_uncaught_exception(exc_type: object, exc_value: object, exc_traceback: object) -> None:
    if exc_type is KeyboardInterrupt:
        return
    logging.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    logging.critical(
        "Uncaught thread exception in %s",
        args.thread.name if args.thread is not None else "unknown-thread",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _configure_session_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(SESSION_LOG_PATH),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_thread_exception
    logging.info("Session log initialized: %s", SESSION_LOG_PATH)


def main() -> int:
    _configure_session_logging()
    if os.name != "nt" and "WSL_DISTRO_NAME" in os.environ:
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    app = QApplication(sys.argv)
    # Prevent secondary top-level QWidget dialogs from ending the whole app.
    # Main window close explicitly requests quit from QML.
    app.setQuitOnLastWindowClosed(False)
    app.setOrganizationName("sub-manager")
    app.setApplicationName("sub-manager")
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    backend = AppBackend()
    engine = QQmlApplicationEngine()
    engine.setInitialProperties({"backend": backend})
    qml_path = resource_root() / "qml" / "Main.qml"
    logging.info("Loading QML: %s", qml_path)
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        logging.error("QML root object load failed.")
        return 1
    exit_code = app.exec()
    backend.clear_session_ocr_state()
    logging.info("Application exited with code %s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
