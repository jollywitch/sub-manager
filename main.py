from __future__ import annotations

import logging
import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Property, QSettings, QThread, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


VIDEO_FILTER = "Video Files (*.mp4 *.mkv *.avi *.mov *.m4v *.webm)"
TEXT_BASED_SUBTITLE_CODECS = {
    "subrip",
    "ass",
    "ssa",
    "webvtt",
    "mov_text",
    "text",
    "ttml",
}
IMAGE_BASED_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle",
    "dvd_subtitle",
}
GLM_OCR_MODEL_ID = "zai-org/GLM-OCR"
LOG_DIR = Path(__file__).resolve().parent / "logs"
SESSION_LOG_PATH = LOG_DIR / "session.log"


@dataclass
class PgsObjectData:
    width: int
    height: int
    expected_rle_length: int
    rle_data: bytearray


@dataclass
class PgsCompositionObject:
    object_id: int
    window_id: int
    x: int
    y: int


@dataclass
class PgsCue:
    start: float
    end: float
    image_path: Path


class SubtitleEditorWindow(QWidget):
    def __init__(
        self,
        file_path: str,
        stream_index: int,
        language: str,
        codec_name: str,
        subtitle_text: str,
        save_embedded_callback: Callable[[str, int, str], tuple[bool, str]],
    ) -> None:
        super().__init__(None, Qt.Window)
        self._source_file_path = file_path
        self._stream_index = stream_index
        self._save_embedded_callback = save_embedded_callback
        self._confirm_save_box: QMessageBox | None = None
        self._export_dialog: QFileDialog | None = None
        self._baseline_text = subtitle_text

        self.setWindowTitle(f"Subtitle Editor - {os.path.basename(file_path)} [s:{stream_index}]")
        self.setWindowModality(Qt.NonModal)
        self.resize(860, 620)

        layout = QVBoxLayout(self)
        details = QLabel(
            f"File: {os.path.basename(file_path)} | Stream: #{stream_index} | Language: {language or 'und'} | Codec: {codec_name}"
        )
        layout.addWidget(details)

        self.editor = QPlainTextEdit()
        self.editor.setPlainText(subtitle_text)
        self.editor.textChanged.connect(self._update_save_button_state)
        layout.addWidget(self.editor)

        button_row = QHBoxLayout()
        self.save_embedded_btn = QPushButton("Save Embedded")
        self.save_embedded_btn.clicked.connect(self._save_embedded)
        self.save_embedded_btn.setEnabled(False)
        button_row.addWidget(self.save_embedded_btn)

        export_srt_btn = QPushButton("Export SRT")
        export_srt_btn.clicked.connect(self._export_srt)
        button_row.addWidget(export_srt_btn)

        button_row.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        self._update_save_button_state()

    def _update_save_button_state(self) -> None:
        self.save_embedded_btn.setEnabled(self.editor.toPlainText() != self._baseline_text)

    def _save_embedded(self) -> None:
        if self._confirm_save_box is not None and self._confirm_save_box.isVisible():
            self._confirm_save_box.raise_()
            self._confirm_save_box.activateWindow()
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Confirm Overwrite")
        box.setText(
            "This will overwrite the original video file with updated embedded subtitles.\n\nContinue?"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.setModal(False)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.buttonClicked.connect(self._on_confirm_save_clicked)
        box.finished.connect(self._cleanup_confirm_save_box)
        self._confirm_save_box = box
        box.open()

    def _on_confirm_save_clicked(self, button: object) -> None:
        if self._confirm_save_box is None:
            return
        standard_button = self._confirm_save_box.standardButton(button)
        if standard_button != QMessageBox.StandardButton.Yes:
            return

        self._run_save_embedded()

    @Slot(int)
    def _cleanup_confirm_save_box(self, _result: int) -> None:
        self._confirm_save_box = None

    def _run_save_embedded(self) -> None:
        self.save_embedded_btn.setEnabled(False)
        ok, message = self._save_embedded_callback(
            self._source_file_path,
            self._stream_index,
            self.editor.toPlainText(),
        )
        self.save_embedded_btn.setEnabled(True)
        if ok:
            self._baseline_text = self.editor.toPlainText()
            self._update_save_button_state()
            self.status_label.setText(f"Saved embedded subtitle: {message}")
            return
        self.status_label.setText(f"Save failed: {message}")

    def _export_srt(self) -> None:
        if self._export_dialog is not None and self._export_dialog.isVisible():
            self._export_dialog.raise_()
            self._export_dialog.activateWindow()
            return

        dialog = QFileDialog(self, "Export Subtitle as SRT")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("SubRip Subtitle (*.srt);;All Files (*)")
        dialog.setDefaultSuffix("srt")
        dialog.selectFile(self._default_export_path())
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.fileSelected.connect(self._on_export_file_selected)
        dialog.finished.connect(self._cleanup_export_dialog)
        self._export_dialog = dialog
        dialog.open()

    def _default_export_path(self) -> str:
        source = Path(self._source_file_path)
        return str(source.with_name(f"{source.stem}.srt"))

    @Slot(str)
    def _on_export_file_selected(self, selected_path: str) -> None:
        if not selected_path:
            return
        export_path = Path(selected_path)
        if export_path.suffix.lower() != ".srt":
            export_path = export_path.with_suffix(".srt")
        try:
            export_path.write_text(self.editor.toPlainText(), encoding="utf-8")
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        self.status_label.setText(f"Exported SRT: {export_path}")

    @Slot(int)
    def _cleanup_export_dialog(self, _result: int) -> None:
        self._export_dialog = None


class GlmDownloadSetupWindow(QWidget):
    def __init__(
        self, on_download: Callable[[str], None], on_cancel: Callable[[], None]
    ) -> None:
        super().__init__(None, Qt.Window)
        self._on_download = on_download
        self._on_cancel = on_cancel
        self._download_started = False
        self.setWindowTitle("GLM-OCR Setup")
        self.resize(520, 220)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        helper_label = QLabel(
            "Image subtitle editing (PGS/DVD subtitles) needs GLM-OCR "
            "to convert subtitle images into editable text.\n\n"
            "You can optionally set HF_TOKEN to improve authenticated "
            "downloads and rate limits."
        )
        helper_label.setWordWrap(True)
        root_layout.addWidget(helper_label)

        token_label = QLabel("HF Token (optional):")
        root_layout.addWidget(token_label)
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("hf_...")
        self.token_edit.setEchoMode(QLineEdit.Password)
        root_layout.addWidget(self.token_edit)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.download_button = QPushButton("Download Now")
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.download_button)
        root_layout.addLayout(button_row)

        self.status_label = QLabel("")
        root_layout.addWidget(self.status_label)

        self.cancel_button.clicked.connect(self.close)
        self.download_button.clicked.connect(self._download)

    @Slot()
    def _download(self) -> None:
        token = self.token_edit.text().strip()
        self._download_started = True
        self._on_download(token)
        self.close()

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        if not self._download_started:
            self._on_cancel()
        super().closeEvent(event)  # type: ignore[arg-type]


class GlmDownloadProgressWindow(QWidget):
    def __init__(self, on_cancel: Callable[[], None]) -> None:
        super().__init__(None, Qt.Window)
        self._on_cancel = on_cancel
        self._details_visible = False
        self.setWindowTitle("GLM-OCR Download")
        self.resize(560, 190)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        self.status_label = QLabel("Preparing GLM-OCR download...")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root_layout.addWidget(self.progress_bar)

        self.details_box = QPlainTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(140)
        self.details_box.setMaximumHeight(140)
        self.details_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root_layout.addWidget(self.details_box)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.details_button = QPushButton("Details")
        self.cancel_button = QPushButton("Cancel Download")
        button_row.addWidget(self.details_button)
        button_row.addWidget(self.cancel_button)
        root_layout.addLayout(button_row)

        self.details_button.clicked.connect(self._toggle_details)
        self.cancel_button.clicked.connect(self._on_cancel)
        self._set_details_visible(False)

    def _set_details_visible(self, visible: bool) -> None:
        if visible:
            self.details_box.setEnabled(True)
            self.details_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.details_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.details_box.setStyleSheet("")
            return
        self.details_box.setEnabled(False)
        self.details_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_box.setStyleSheet(
            "QPlainTextEdit { border: 0px; background: transparent; color: transparent; }"
        )

    @Slot()
    def _toggle_details(self) -> None:
        self._details_visible = not self._details_visible
        self._set_details_visible(self._details_visible)
        self.details_button.setText("Hide Details" if self._details_visible else "Details")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def append_log(self, text: str) -> None:
        self.details_box.appendPlainText(text)

    def show_details(self) -> None:
        if not self._details_visible:
            self._toggle_details()

    def set_progress(self, percent: int) -> None:
        if int(percent) < 0:
            if self.progress_bar.minimum() != 0 or self.progress_bar.maximum() != 0:
                self.progress_bar.setRange(0, 0)
            return
        if self.progress_bar.minimum() != 0 or self.progress_bar.maximum() != 100:
            self.progress_bar.setRange(0, 100)
        clamped = max(0, min(100, int(percent)))
        self.progress_bar.setValue(clamped)


class FFmpegDownloadWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, download_url: str, archive_ext: str, tools_dir: Path) -> None:
        super().__init__()
        self.download_url = download_url
        self.archive_ext = archive_ext
        self.tools_dir = tools_dir

    @Slot()
    def run(self) -> None:
        temp_dir = self.tools_dir / "_tmp"
        archive_name = Path(self.download_url).name
        archive_path = temp_dir / archive_name
        install_bin_dir = self.tools_dir / "bin"
        try:
            self.progress.emit("Preparing download...")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.tools_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit("Downloading FFmpeg archive...")
            with urllib.request.urlopen(self.download_url) as response:
                archive_path.write_bytes(response.read())

            self.progress.emit("Extracting archive...")
            extract_dir = temp_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            if self.archive_ext == "zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                with tarfile.open(archive_path, "r:xz") as tar_ref:
                    try:
                        tar_ref.extractall(extract_dir, filter="data")
                    except TypeError:
                        tar_ref.extractall(extract_dir)

            ffmpeg_path, ffprobe_path = self._find_extracted_binaries(extract_dir)
            if ffmpeg_path is None or ffprobe_path is None:
                raise RuntimeError("Downloaded archive does not contain ffmpeg and ffprobe.")

            self.progress.emit("Installing ffmpeg/ffprobe...")
            install_bin_dir.mkdir(parents=True, exist_ok=True)
            target_ffmpeg = install_bin_dir / ffmpeg_path.name
            target_ffprobe = install_bin_dir / ffprobe_path.name
            shutil.copy2(ffmpeg_path, target_ffmpeg)
            shutil.copy2(ffprobe_path, target_ffprobe)
            if os.name != "nt":
                os.chmod(target_ffmpeg, 0o755)
                os.chmod(target_ffprobe, 0o755)

            self.finished.emit(str(install_bin_dir))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _find_extracted_binaries(self, root: Path) -> tuple[Path | None, Path | None]:
        ffmpeg_names = {"ffmpeg.exe", "ffmpeg"} if os.name == "nt" else {"ffmpeg"}
        ffprobe_names = {"ffprobe.exe", "ffprobe"} if os.name == "nt" else {"ffprobe"}
        found_ffmpeg: Path | None = None
        found_ffprobe: Path | None = None

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name in ffmpeg_names:
                found_ffmpeg = path
            elif path.name in ffprobe_names:
                found_ffprobe = path
            if found_ffmpeg and found_ffprobe:
                break
        return found_ffmpeg, found_ffprobe


class GlmOcrModelDownloadWorker(QObject):
    progress = Signal(str)
    progressPercent = Signal(int)
    diagnostic = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_id: str, hf_token: str | None = None) -> None:
        super().__init__()
        self.model_id = model_id
        self.hf_token = (hf_token or "").strip() or None
        self._cancel_requested = False
        self._process: subprocess.Popen[str] | None = None

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True
        self._terminate_process()

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
            else:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
        except Exception:
            pass

    def _format_bytes(self, num_bytes: int) -> str:
        value = float(max(0, int(num_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        unit = units[0]
        for candidate in units:
            unit = candidate
            if value < 1024.0 or candidate == units[-1]:
                break
            value /= 1024.0
        if unit == "B":
            return f"{int(value)} {unit}"
        return f"{value:.1f} {unit}"

    def _build_download_env(self) -> tuple[dict[str, str], bool]:
        env = os.environ.copy()
        env["GLM_OCR_MODEL_ID"] = self.model_id
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        xet_raw = str(env.get("HF_XET_HIGH_PERFORMANCE", "") or "").strip().lower()
        if xet_raw in {"0", "false", "no", "off"}:
            return env, False
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        return env, True

    def _resolve_hub_storage_dir(self, env: dict[str, str]) -> Path:
        hub_cache = str(env.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
        if not hub_cache:
            hf_home = str(env.get("HF_HOME", "") or "").strip()
            if hf_home:
                hub_cache = str(Path(hf_home) / "hub")
            else:
                hub_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
        return Path(hub_cache) / f"models--{self.model_id.replace('/', '--')}"

    def _directory_allocated_bytes(self, root: Path) -> int:
        if not root.exists():
            return 0
        total = 0
        for path in root.rglob("*"):
            if not path.is_file() and not path.is_symlink():
                continue
            try:
                stat_result = path.lstat()
            except OSError:
                continue
            total += int(getattr(stat_result, "st_blocks", 0) or 0) * 512
        return max(0, total)

    @Slot()
    def run(self) -> None:
        started_at = time.monotonic()
        if self._cancel_requested:
            self.failed.emit("GLM-OCR download cancelled.")
            return
        self.diagnostic.emit("GLM download worker started.")
        try:
            if self._cancel_requested:
                self.failed.emit("GLM-OCR download cancelled.")
                return
            hub_cache_env = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
            hf_home_env = str(os.environ.get("HF_HOME", "") or "").strip()
            self.diagnostic.emit(
                "Starting isolated snapshot_download process "
                f"(repo_id={self.model_id}, HUGGINGFACE_HUB_CACHE={hub_cache_env or '<default>'}, HF_HOME={hf_home_env or '<default>'}, token={'yes' if self.hf_token else 'no'})."
            )
            self.progress.emit("Downloading or reusing GLM-OCR model files...")
            env, xet_enabled = self._build_download_env()
            self.diagnostic.emit(
                f"transfer_mode=xet_high_performance:{'on' if xet_enabled else 'off'}"
            )
            storage_dir = self._resolve_hub_storage_dir(env)
            self.diagnostic.emit(f"Model storage directory: {storage_dir}")
            code = """
import json
import os
import sys
import traceback
from huggingface_hub import HfApi, snapshot_download

repo = os.environ["GLM_OCR_MODEL_ID"]
token = os.environ.get("HF_TOKEN") or None

def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

try:
    try:
        info = HfApi().model_info(repo_id=repo, token=token, files_metadata=True)
        total = 0
        for sibling in (info.siblings or []):
            size = int(getattr(sibling, "size", 0) or 0)
            if size > 0:
                total += size
        if total > 0:
            emit({"type": "meta", "total_bytes": int(total)})
    except Exception as exc:
        emit({"type": "log", "message": f"Could not resolve metadata size upfront: {type(exc).__name__}: {exc}"})
    path = snapshot_download(repo_id=repo, token=token)
    emit({"type": "done", "path": str(path)})
except Exception as exc:
    emit({"type": "error", "message": f"{type(exc).__name__}: {exc}\\n{traceback.format_exc()}"})
    raise
"""
            self._process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
            event_queue: queue.Queue[dict[str, object]] = queue.Queue()
            stderr_queue: queue.Queue[str] = queue.Queue()

            def consume_stdout() -> None:
                process = self._process
                if process is None or process.stdout is None:
                    return
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        stderr_queue.put(f"[stdout] {line}")
                        continue
                    if isinstance(payload, dict):
                        event_queue.put(payload)

            def consume_stderr() -> None:
                process = self._process
                if process is None or process.stderr is None:
                    return
                for raw_line in process.stderr:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        sys.stderr.write(raw_line)
                        sys.stderr.flush()
                    except Exception:
                        pass
                    stderr_queue.put(line)

            stdout_thread = threading.Thread(target=consume_stdout, daemon=True)
            stderr_thread = threading.Thread(target=consume_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            local_model_path = ""
            last_error = ""
            last_percent = -1
            unknown_progress_reported = False
            total_bytes = 0
            last_status = ""
            while self._process.poll() is None:
                if self._cancel_requested:
                    self._terminate_process()
                    self.failed.emit("GLM-OCR download cancelled.")
                    return
                while True:
                    try:
                        payload = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    event_type = str(payload.get("type", "")).strip().lower()
                    if event_type == "done":
                        local_model_path = str(payload.get("path", "") or "").strip()
                    elif event_type == "error":
                        last_error = str(payload.get("message", "") or "").strip()
                    elif event_type == "meta":
                        total_value = int(payload.get("total_bytes", 0) or 0)
                        if total_value > 0 and total_bytes <= 0:
                            total_bytes = total_value
                            self.diagnostic.emit(
                                f"Resolved expected total model size: {self._format_bytes(total_bytes)}"
                            )
                    elif event_type == "log":
                        message = str(payload.get("message", "") or "").strip()
                        if message:
                            self.diagnostic.emit(message)
                current_allocated = self._directory_allocated_bytes(storage_dir)
                if total_bytes > 0:
                    percent = int((current_allocated * 100) / max(total_bytes, 1))
                    percent = max(0, min(100, percent))
                    status = (
                        "Downloading or reusing GLM-OCR model files: "
                        f"{self._format_bytes(current_allocated)}/{self._format_bytes(total_bytes)} ({percent}%)"
                    )
                    if status != last_status:
                        self.progress.emit(status)
                        last_status = status
                    if percent != last_percent:
                        last_percent = percent
                        self.progressPercent.emit(percent)
                else:
                    if not unknown_progress_reported:
                        self.progressPercent.emit(-1)
                        unknown_progress_reported = True
                    status = "Downloading or reusing GLM-OCR model files... (estimating total size)"
                    if status != last_status:
                        self.progress.emit(status)
                        last_status = status
                while True:
                    try:
                        stderr_line = stderr_queue.get_nowait()
                    except queue.Empty:
                        break
                    self.diagnostic.emit(stderr_line)
                time.sleep(0.1)
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            while True:
                try:
                    payload = event_queue.get_nowait()
                except queue.Empty:
                    break
                event_type = str(payload.get("type", "")).strip().lower()
                if event_type == "done":
                    local_model_path = str(payload.get("path", "") or "").strip()
                elif event_type == "error":
                    last_error = str(payload.get("message", "") or "").strip()
                elif event_type == "meta":
                    total_value = int(payload.get("total_bytes", 0) or 0)
                    if total_value > 0 and total_bytes <= 0:
                        total_bytes = total_value
            while True:
                try:
                    stderr_line = stderr_queue.get_nowait()
                except queue.Empty:
                    break
                self.diagnostic.emit(stderr_line)
            return_code = int(self._process.returncode or 0)
            self._process = None
            if return_code != 0:
                error_text = (
                    last_error
                    or "Hugging Face download process failed. Check terminal output for details."
                ).strip()
                raise RuntimeError(error_text)
            if not local_model_path:
                raise RuntimeError("GLM-OCR download completed without reporting model path.")
            if self._cancel_requested:
                self.failed.emit("GLM-OCR download cancelled.")
                return
            final_allocated = self._directory_allocated_bytes(storage_dir)
            if total_bytes > 0:
                self.progress.emit(
                    "Downloading or reusing GLM-OCR model files: "
                    f"{self._format_bytes(final_allocated)}/{self._format_bytes(total_bytes)} (100%)"
                )
            if last_percent < 100:
                self.progressPercent.emit(100)
            elapsed_s = time.monotonic() - started_at
            self.diagnostic.emit(f"snapshot_download finished in {elapsed_s:.1f}s.")
            self.finished.emit(str(local_model_path))
        except Exception as exc:
            elapsed_s = time.monotonic() - started_at
            self.diagnostic.emit(
                f"snapshot_download failed after {elapsed_s:.1f}s: {type(exc).__name__}: {exc}"
            )
            self.failed.emit(str(exc))
        finally:
            self._terminate_process()
            self._process = None


class ImageSubtitleOcrWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)
    OCR_BATCH_SIZE_CUDA = 4
    OCR_BATCH_SIZE_MPS = 2
    OCR_BATCH_SIZE_CPU = 1
    OCR_HASH_HAMMING_THRESHOLD = 2
    OCR_MAX_NEW_TOKENS = 256
    _model_cache_lock = threading.Lock()
    _cached_model_bundle: tuple[object, object, object] | None = None
    _cached_model_id: str | None = None

    def __init__(
        self,
        file_path: str,
        stream_index: int,
        codec_name: str,
        ffmpeg_exe: str | None,
        ffprobe_exe: str | None,
        hf_token: str | None,
        model_id: str = GLM_OCR_MODEL_ID,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.stream_index = stream_index
        self.codec_name = codec_name
        self.ffmpeg_exe = ffmpeg_exe or shutil.which("ffmpeg")
        self.ffprobe_exe = ffprobe_exe or shutil.which("ffprobe")
        self.hf_token = (hf_token or "").strip() or None
        self.model_id = model_id
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")
            if not self.ffmpeg_exe:
                raise RuntimeError("ffmpeg binary not found.")
            if not self.ffprobe_exe:
                raise RuntimeError("ffprobe binary not found.")

            self.progress.emit("Checking GLM-OCR runtime...")
            torch, processor, model = self._load_model()
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")

            self.progress.emit("Extracting subtitle bitmaps...")
            with tempfile.TemporaryDirectory(prefix="sub_manager_ocr_") as temp_dir:
                cues = self._extract_image_subtitle_cues(Path(temp_dir))
                cue_count = len(cues)
                if cue_count == 0:
                    raise RuntimeError("Could not find image subtitle cues.")

                recognized_texts: list[str] = [""] * cue_count
                pending_indices: list[int] = []
                pending_frame_paths: list[Path] = []
                cue_hashes: list[int] = [0] * cue_count
                hash_text_cache: dict[int, str] = {}
                hash_index: list[tuple[int, str]] = []
                reused_count = 0
                self.progress.emit(f"Running OCR on {cue_count} subtitle images...")
                for idx, cue in enumerate(cues):
                    if self._cancel_requested:
                        raise RuntimeError("OCR cancelled.")
                    hash_value = self._compute_frame_hash(cue.image_path)
                    cue_hashes[idx] = hash_value
                    reused = self._lookup_cached_text_for_hash(hash_value, hash_text_cache, hash_index)
                    if reused is not None:
                        recognized_texts[idx] = reused
                        reused_count += 1
                        continue
                    pending_indices.append(idx)
                    pending_frame_paths.append(cue.image_path)

                batch_size = self._suggest_batch_size(model)
                total_batches = (
                    (len(pending_frame_paths) + batch_size - 1) // batch_size if pending_frame_paths else 0
                )
                for batch_idx, start in enumerate(range(0, len(pending_frame_paths), batch_size), start=1):
                    if self._cancel_requested:
                        raise RuntimeError("OCR cancelled.")
                    batch_paths = pending_frame_paths[start : start + batch_size]
                    self.progress.emit(
                        f"Running OCR batch {batch_idx}/{total_batches} ({len(batch_paths)} frames)..."
                    )
                    batch_texts = self._ocr_image_frames_batched(torch, processor, model, batch_paths)
                    for offset, text in enumerate(batch_texts):
                        original_idx = pending_indices[start + offset]
                        clean_text = self._normalize_ocr_text(text)
                        recognized_texts[original_idx] = clean_text
                        self._store_hash_text(cue_hashes[original_idx], clean_text, hash_text_cache, hash_index)

                if reused_count > 0:
                    self.progress.emit(f"Reused OCR text for {reused_count} duplicate subtitle images.")

                recognized_cues: list[tuple[float, float, str]] = []
                for idx, text in enumerate(recognized_texts):
                    if not text:
                        continue
                    recognized_cues.append((cues[idx].start, cues[idx].end, text))

                merged_cues = self._merge_adjacent_cues(recognized_cues)
                srt_text = self._build_srt(merged_cues)
                if not srt_text.strip():
                    raise RuntimeError("OCR completed but no subtitle text was recognized.")
                self.finished.emit(srt_text)
        except Exception as exc:
            logging.error("Image subtitle OCR worker failed", exc_info=True)
            self.failed.emit(str(exc))

    def _load_model(self) -> tuple[object, object, object]:
        with self._model_cache_lock:
            if self._cached_model_bundle is not None and self._cached_model_id == self.model_id:
                self.progress.emit("Reusing loaded GLM-OCR model...")
                return self._cached_model_bundle

        try:
            import torch  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "GLM-OCR dependencies are missing. Install: transformers, torch, pillow, huggingface_hub."
            ) from exc

        xet_raw = str(os.environ.get("HF_XET_HIGH_PERFORMANCE", "") or "").strip().lower()
        if xet_raw not in {"0", "false", "no", "off"}:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

        has_cuda = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        has_mps = bool(mps_backend) and bool(getattr(mps_backend, "is_available", lambda: False)())
        if not (has_cuda or has_mps):
            raise RuntimeError("GLM-OCR requires a GPU runtime (CUDA or MPS).")
        target_device = "cuda" if has_cuda else "mps"

        self.progress.emit("Downloading or reusing GLM-OCR model files...")
        local_model_path = snapshot_download(repo_id=self.model_id, token=self.hf_token)

        self.progress.emit("Loading GLM-OCR model...")
        try:
            loaded_processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load GLM-OCR processor. Install the latest transformers from GitHub main and run `uv sync`."
            ) from exc
        try:
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception:
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            if hasattr(loaded_model, "to"):
                loaded_model = loaded_model.to(target_device)
        try:
            first_param = next(loaded_model.parameters())
            device_type = str(getattr(first_param, "device", "cpu"))
            if device_type.startswith("cpu"):
                raise RuntimeError("GLM-OCR loaded on CPU. GPU runtime is required.")
        except StopIteration:
            pass
        bundle = (torch, loaded_processor, loaded_model)
        with self._model_cache_lock:
            self._cached_model_bundle = bundle
            self._cached_model_id = self.model_id
        return bundle

    @classmethod
    def clear_model_cache(cls) -> None:
        with cls._model_cache_lock:
            cls._cached_model_bundle = None
            cls._cached_model_id = None

    def _extract_image_subtitle_cues(self, temp_root: Path) -> list[PgsCue]:
        if self.codec_name == "hdmv_pgs_subtitle":
            return self._extract_pgs_cues(temp_root)
        if self.codec_name == "dvd_subtitle":
            raise RuntimeError("DVD subtitle OCR is not implemented yet. Only PGS is supported.")
        raise RuntimeError(f"Unsupported image subtitle codec: {self.codec_name}")

    def _extract_pgs_cues(self, temp_root: Path) -> list[PgsCue]:
        if self._cancel_requested:
            raise RuntimeError("OCR cancelled.")
        sup_path = temp_root / "stream.sup"
        command = [
            self.ffmpeg_exe,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            self.file_path,
            "-map",
            f"0:{self.stream_index}",
            "-c:s",
            "copy",
            str(sup_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except Exception as exc:
            raise RuntimeError(f"Could not run ffmpeg for PGS extraction: {exc}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "Unknown ffmpeg error").strip()
            raise RuntimeError(f"Could not extract PGS subtitle stream: {stderr}")
        if not sup_path.exists():
            raise RuntimeError("PGS extraction did not produce a .sup file.")

        frames_dir = temp_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        return self._parse_pgs_sup_to_cues(sup_path, frames_dir)

    def _parse_pgs_sup_to_cues(self, sup_path: Path, frames_dir: Path) -> list[PgsCue]:
        raw = sup_path.read_bytes()
        palettes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
        objects: dict[int, PgsObjectData] = {}
        current_palette_id = 0
        active_composition: list[PgsCompositionObject] = []
        active_video_size = (0, 0)
        cues: list[PgsCue] = []
        pending_clear_time: float | None = None
        i = 0
        cue_index = 0

        while i + 13 <= len(raw):
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")
            if raw[i : i + 2] != b"PG":
                i += 1
                continue
            pts_90k = int.from_bytes(raw[i + 2 : i + 6], "big", signed=False)
            segment_type = raw[i + 10]
            payload_size = int.from_bytes(raw[i + 11 : i + 13], "big", signed=False)
            payload_start = i + 13
            payload_end = payload_start + payload_size
            if payload_end > len(raw):
                break
            payload = raw[payload_start:payload_end]
            pts_seconds = pts_90k / 90000.0

            if segment_type == 0x14:
                palette_id, palette = self._parse_pds(payload)
                palettes[palette_id] = palette
            elif segment_type == 0x15:
                object_id, parsed = self._parse_ods(payload, objects.get)
                objects[object_id] = parsed
            elif segment_type == 0x16:
                (
                    video_width,
                    video_height,
                    palette_id,
                    composition_objects,
                ) = self._parse_pcs(payload)
                active_video_size = (video_width, video_height)
                current_palette_id = palette_id
                active_composition = composition_objects
                pending_clear_time = pts_seconds if not composition_objects else None
            elif segment_type == 0x80:
                if pending_clear_time is not None:
                    if cues and cues[-1].end <= cues[-1].start:
                        cues[-1].end = max(cues[-1].start + 0.05, pending_clear_time)
                    pending_clear_time = None
                    i = payload_end
                    continue
                if not active_composition:
                    i = payload_end
                    continue
                cue_path = frames_dir / f"cue_{cue_index:08d}.png"
                rendered = self._render_pgs_composition_image(
                    cue_path=cue_path,
                    composition_objects=active_composition,
                    objects=objects,
                    palette=palettes.get(current_palette_id, {}),
                    video_size=active_video_size,
                )
                if rendered:
                    cue_start = pts_seconds
                    if cues and cues[-1].end <= cues[-1].start:
                        cues[-1].end = max(cues[-1].start + 0.05, cue_start)
                    cues.append(PgsCue(start=cue_start, end=cue_start, image_path=cue_path))
                    cue_index += 1
            i = payload_end

        if cues:
            for idx in range(len(cues) - 1):
                if cues[idx].end <= cues[idx].start:
                    cues[idx].end = max(cues[idx].start + 0.05, cues[idx + 1].start)
            if cues[-1].end <= cues[-1].start:
                cues[-1].end = cues[-1].start + 2.0
        if not cues:
            raise RuntimeError("No PGS cues could be extracted from the subtitle stream.")
        return cues

    def _parse_pds(self, payload: bytes) -> tuple[int, dict[int, tuple[int, int, int, int]]]:
        if len(payload) < 2:
            return (0, {})
        palette_id = payload[0]
        palette: dict[int, tuple[int, int, int, int]] = {}
        pos = 2
        while pos + 5 <= len(payload):
            index = payload[pos]
            y = payload[pos + 1]
            cr = payload[pos + 2]
            cb = payload[pos + 3]
            alpha = payload[pos + 4]
            r, g, b = self._ycbcr_to_rgb(y, cb, cr)
            palette[index] = (r, g, b, alpha)
            pos += 5
        return (palette_id, palette)

    def _parse_ods(
        self,
        payload: bytes,
        object_lookup: Callable[[int], PgsObjectData | None],
    ) -> tuple[int, PgsObjectData]:
        if len(payload) < 4:
            raise RuntimeError("Invalid PGS ODS payload.")
        object_id = int.from_bytes(payload[0:2], "big", signed=False)
        sequence_flag = payload[3]
        first_in_sequence = bool(sequence_flag & 0x80)
        data_start = 4
        if first_in_sequence:
            if len(payload) < 11:
                raise RuntimeError("Invalid first PGS ODS payload.")
            object_data_length = int.from_bytes(payload[4:7], "big", signed=False)
            width = int.from_bytes(payload[7:9], "big", signed=False)
            height = int.from_bytes(payload[9:11], "big", signed=False)
            expected_rle_length = max(object_data_length - 4, 0)
            rle_data = bytearray(payload[11:])
            return (
                object_id,
                PgsObjectData(
                    width=width,
                    height=height,
                    expected_rle_length=expected_rle_length,
                    rle_data=rle_data[:expected_rle_length],
                ),
            )
        existing = object_lookup(object_id)
        if existing is None:
            raise RuntimeError("PGS object continuation encountered without prior object data.")
        existing.rle_data.extend(payload[data_start:])
        if existing.expected_rle_length > 0 and len(existing.rle_data) > existing.expected_rle_length:
            existing.rle_data = existing.rle_data[: existing.expected_rle_length]
        return (object_id, existing)

    def _parse_pcs(
        self, payload: bytes
    ) -> tuple[int, int, int, list[PgsCompositionObject]]:
        if len(payload) < 11:
            raise RuntimeError("Invalid PGS PCS payload.")
        video_width = int.from_bytes(payload[0:2], "big", signed=False)
        video_height = int.from_bytes(payload[2:4], "big", signed=False)
        palette_id = payload[9]
        composition_object_count = payload[10]
        composition_objects: list[PgsCompositionObject] = []
        pos = 11
        for _ in range(composition_object_count):
            if pos + 8 > len(payload):
                break
            object_id = int.from_bytes(payload[pos : pos + 2], "big", signed=False)
            window_id = payload[pos + 2]
            crop_flag = payload[pos + 3]
            x = int.from_bytes(payload[pos + 4 : pos + 6], "big", signed=False)
            y = int.from_bytes(payload[pos + 6 : pos + 8], "big", signed=False)
            pos += 8
            if crop_flag & 0x40:
                if pos + 8 > len(payload):
                    break
                pos += 8
            composition_objects.append(
                PgsCompositionObject(object_id=object_id, window_id=window_id, x=x, y=y)
            )
        return (video_width, video_height, palette_id, composition_objects)

    def _render_pgs_composition_image(
        self,
        cue_path: Path,
        composition_objects: list[PgsCompositionObject],
        objects: dict[int, PgsObjectData],
        palette: dict[int, tuple[int, int, int, int]],
        video_size: tuple[int, int],
    ) -> bool:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required for PGS image rendering.") from exc

        if not composition_objects:
            return False
        available: list[tuple[PgsCompositionObject, PgsObjectData]] = []
        for comp in composition_objects:
            obj = objects.get(comp.object_id)
            if obj is None or obj.width <= 0 or obj.height <= 0 or not obj.rle_data:
                continue
            available.append((comp, obj))
        if not available:
            return False

        min_x = min(comp.x for comp, _ in available)
        min_y = min(comp.y for comp, _ in available)
        max_x = max(comp.x + obj.width for comp, obj in available)
        max_y = max(comp.y + obj.height for comp, obj in available)
        if video_size[0] > 0 and video_size[1] > 0:
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, video_size[0])
            max_y = min(max_y, video_size[1])
        canvas_w = max(1, max_x - min_x)
        canvas_h = max(1, max_y - min_y)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for comp, obj in available:
            indices = self._decode_pgs_rle(obj.rle_data, obj.width, obj.height)
            rgba_data = [palette.get(index, (255, 255, 255, 255)) for index in indices]
            sprite = Image.new("RGBA", (obj.width, obj.height))
            sprite.putdata(rgba_data)
            canvas.alpha_composite(sprite, (max(comp.x - min_x, 0), max(comp.y - min_y, 0)))

        bbox = canvas.getbbox()
        if bbox is None:
            return False
        cropped = canvas.crop(bbox)
        cropped.save(cue_path)
        return True

    def _decode_pgs_rle(self, rle_data: bytes | bytearray, width: int, height: int) -> list[int]:
        pixels: list[int] = []
        row_pixels = 0
        i = 0
        target = max(width * height, 0)
        while i < len(rle_data) and len(pixels) < target:
            value = rle_data[i]
            i += 1
            if value != 0:
                pixels.append(value)
                row_pixels += 1
                continue
            if i >= len(rle_data):
                break
            code = rle_data[i]
            i += 1
            if code == 0:
                if width > 0 and row_pixels < width:
                    pixels.extend([0] * (width - row_pixels))
                row_pixels = 0
                continue
            if code & 0x40:
                if i >= len(rle_data):
                    break
                run_length = ((code & 0x3F) << 8) | rle_data[i]
                i += 1
            else:
                run_length = code & 0x3F
            if code & 0x80:
                if i >= len(rle_data):
                    break
                color = rle_data[i]
                i += 1
            else:
                color = 0
            if run_length <= 0:
                continue
            pixels.extend([color] * run_length)
            row_pixels += run_length

        if len(pixels) < target:
            pixels.extend([0] * (target - len(pixels)))
        return pixels[:target]

    def _compute_frame_hash(self, frame_path: Path) -> int:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required for OCR frame hashing.") from exc
        with Image.open(frame_path) as image:
            grayscale = image.convert("L").resize((9, 8))
            pixels = list(grayscale.get_flattened_data())
        hash_bits = 0
        for row in range(8):
            for col in range(8):
                left = pixels[(row * 9) + col]
                right = pixels[(row * 9) + col + 1]
                hash_bits = (hash_bits << 1) | (1 if left > right else 0)
        return hash_bits

    def _lookup_cached_text_for_hash(
        self,
        hash_value: int,
        hash_text_cache: dict[int, str],
        hash_index: list[tuple[int, str]],
    ) -> str | None:
        exact = hash_text_cache.get(hash_value)
        if exact is not None:
            return exact
        best_text: str | None = None
        best_distance = self.OCR_HASH_HAMMING_THRESHOLD + 1
        for previous_hash, previous_text in hash_index:
            distance = self._hamming_distance(hash_value, previous_hash)
            if distance < best_distance:
                best_distance = distance
                best_text = previous_text
                if distance == 0:
                    break
        if best_text is None or best_distance > self.OCR_HASH_HAMMING_THRESHOLD:
            return None
        return best_text

    def _store_hash_text(
        self,
        hash_value: int,
        text: str,
        hash_text_cache: dict[int, str],
        hash_index: list[tuple[int, str]],
    ) -> None:
        if not text:
            return
        if hash_value not in hash_text_cache:
            hash_index.append((hash_value, text))
        hash_text_cache[hash_value] = text

    def _hamming_distance(self, left: int, right: int) -> int:
        return int((left ^ right).bit_count())

    def _suggest_batch_size(self, model: object) -> int:
        model_device = str(self._model_device(model))
        if model_device.startswith("cuda"):
            return self.OCR_BATCH_SIZE_CUDA
        if model_device.startswith("mps"):
            return self.OCR_BATCH_SIZE_MPS
        return self.OCR_BATCH_SIZE_CPU

    def _ocr_image_frames_batched(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_paths: list[Path],
    ) -> list[str]:
        if not frame_paths:
            return []
        try:
            return self._ocr_image_frames_batched_impl(torch_module, processor, model, frame_paths)
        except Exception:
            results: list[str] = []
            for frame_path in frame_paths:
                results.append(self._ocr_image_frame(torch_module, processor, model, frame_path))
            return results

    def _ocr_image_frames_batched_impl(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_paths: list[Path],
    ) -> list[str]:
        prompt = "Text Recognition:"
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("GLM-OCR processor does not support chat templates.")
        conversations = []
        for frame_path in frame_paths:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": str(frame_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        model_device = self._model_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)

        with torch_module.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self.OCR_MAX_NEW_TOKENS)

        input_lengths: list[int] = []
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None and hasattr(attention_mask, "sum"):
                try:
                    mask_lengths = attention_mask.sum(dim=1).tolist()
                    input_lengths = [int(v) for v in mask_lengths]
                except Exception:
                    input_lengths = []
            if not input_lengths:
                input_ids = inputs.get("input_ids")
                if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
                    input_lengths = [int(input_ids.shape[1])] * int(input_ids.shape[0])
        if not input_lengths:
            input_lengths = [0] * len(frame_paths)

        decoded_texts: list[str] = []
        for row_idx, prompt_len in enumerate(input_lengths):
            sample_ids = output_ids[row_idx]
            if prompt_len > 0 and hasattr(sample_ids, "__getitem__"):
                sample_ids = sample_ids[prompt_len:]
            decoded = processor.decode(sample_ids, skip_special_tokens=True)
            decoded_texts.append(decoded if isinstance(decoded, str) else "")
        return decoded_texts

    def _ocr_image_frame(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_path: Path,
    ) -> str:
        prompt = "Text Recognition:"
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("GLM-OCR processor does not support chat templates.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(frame_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = self._model_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)

        with torch_module.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self.OCR_MAX_NEW_TOKENS)

        prompt_tokens = 0
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
                prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > 0 and hasattr(output_ids, "__getitem__"):
            output_ids = output_ids[:, prompt_tokens:]

        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        return self._normalize_ocr_text(decoded[0])

    def _model_device(self, model: object) -> object:
        try:
            first_param = next(model.parameters())
            return getattr(first_param, "device")
        except StopIteration:
            return "cpu"

    def _merge_adjacent_cues(
        self, cues: list[tuple[float, float, str]], max_gap_seconds: float = 0.15
    ) -> list[tuple[float, float, str]]:
        merged: list[tuple[float, float, str]] = []
        for start, end, text in cues:
            if not merged:
                merged.append((start, end, text))
                continue
            prev_start, prev_end, prev_text = merged[-1]
            if text == prev_text and (start - prev_end) <= max_gap_seconds:
                merged[-1] = (prev_start, max(prev_end, end), prev_text)
                continue
            merged.append((start, end, text))
        return merged

    def _build_srt(self, cues: list[tuple[float, float, str]]) -> str:
        lines: list[str] = []
        cue_index = 1
        for start, end, text in cues:
            clean = self._normalize_ocr_text(text)
            if not clean:
                continue
            if end <= start:
                end = start + 0.05
            lines.append(str(cue_index))
            lines.append(f"{self._srt_timestamp(start)} --> {self._srt_timestamp(end)}")
            lines.append(clean)
            lines.append("")
            cue_index += 1
        return "\n".join(lines).strip() + ("\n" if lines else "")

    def _srt_timestamp(self, seconds: float) -> str:
        safe_seconds = max(seconds, 0.0)
        total_ms = int(round(safe_seconds * 1000))
        hours = total_ms // 3_600_000
        total_ms -= hours * 3_600_000
        minutes = total_ms // 60_000
        total_ms -= minutes * 60_000
        secs = total_ms // 1000
        millis = total_ms - (secs * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _normalize_ocr_text(self, text: str) -> str:
        collapsed = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        # Remove multimodal special tokens leaked by model decode.
        collapsed = re.sub(r"<\|[^|>]+?\|>", "", collapsed)
        # Keep only actual OCR content after the prompt echo, if present.
        pivot = "Text Recognition:"
        if pivot in collapsed:
            collapsed = collapsed.rsplit(pivot, 1)[-1]
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    def _ycbcr_to_rgb(self, y: int, cb: int, cr: int) -> tuple[int, int, int]:
        yy = float(y)
        cbb = float(cb) - 128.0
        crr = float(cr) - 128.0
        r = int(round(yy + (1.402 * crr)))
        g = int(round(yy - (0.344136 * cbb) - (0.714136 * crr)))
        b = int(round(yy + (1.772 * cbb)))
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b)),
        )


class AppBackend(QObject):
    videoFilesChanged = Signal()
    ffmpegStatusChanged = Signal()
    ffmpegStatusLevelChanged = Signal()
    hfTokenStatusChanged = Signal()
    hfTokenStatusLevelChanged = Signal()
    hfTokenPreviewChanged = Signal()
    glmOcrModelStatusChanged = Signal()
    glmOcrModelStatusLevelChanged = Signal()
    downloadingChanged = Signal()
    windowGeometryChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("sub-manager", "sub-manager")
        self._selected_paths: set[str] = set()
        self._video_files: list[dict[str, object]] = []
        self.ffmpeg_path: str | None = None
        self.ffprobe_path: str | None = None
        self.download_thread: QThread | None = None
        self.download_worker: FFmpegDownloadWorker | None = None
        self._glm_download_thread: QThread | None = None
        self._glm_download_worker: GlmOcrModelDownloadWorker | None = None
        self._glm_download_progress_window: GlmDownloadProgressWindow | None = None
        self._glm_download_progress_dismissed = False
        self._glm_download_cancel_requested = False
        self._glm_download_started_at: float | None = None
        self._glm_download_diagnostics: list[str] = []
        self._glm_download_followup_action: Callable[[], None] | None = None
        self._add_videos_dialog: QFileDialog | None = None
        self._active_message_box: QMessageBox | None = None
        self._subtitle_editor: SubtitleEditorWindow | None = None
        self._subtitle_editor_key: str | None = None
        self._ffmpeg_override_box: QMessageBox | None = None
        self._ffmpeg_override_download_button: QPushButton | None = None
        self._ffmpeg_override_use_existing_button: QPushButton | None = None
        self._ffmpeg_override_decided = False
        self._glm_setup_window: GlmDownloadSetupWindow | None = None
        self._ocr_progress_box: QMessageBox | None = None
        self._ocr_thread: QThread | None = None
        self._ocr_worker: ImageSubtitleOcrWorker | None = None
        self._ocr_target_key: str | None = None
        self._ocr_target_file_path: str | None = None
        self._ocr_target_stream_index: int | None = None
        self._ocr_target_stream_lang: str | None = None
        self._ocr_target_codec_name: str | None = None
        self._ocr_target_cache_key: tuple[str, int, int, int] | None = None
        self._ocr_srt_cache: dict[tuple[str, int, int, int], str] = {}
        self._ffmpeg_status = ""
        self._ffmpeg_status_level = "warn"
        self._hf_token_status = ""
        self._hf_token_status_level = "warn"
        self._hf_token_preview = ""
        self._glm_ocr_model_status = ""
        self._glm_ocr_model_status_level = "warn"
        self._downloading = False

        self._window_x = int(self.settings.value("window/x", 100))
        self._window_y = int(self.settings.value("window/y", 100))
        self._window_w = int(self.settings.value("window/w", 900))
        self._window_h = int(self.settings.value("window/h", 560))

        self._apply_hf_token_environment_from_settings()
        self._refresh_dependency_statuses()

    def get_video_files(self) -> list[dict[str, object]]:
        return self._video_files

    videoFiles = Property("QVariantList", get_video_files, notify=videoFilesChanged)

    def get_all_files_checked(self) -> bool:
        return bool(self._video_files) and all(bool(item.get("checked")) for item in self._video_files)

    allFilesChecked = Property(bool, get_all_files_checked, notify=videoFilesChanged)

    def get_ffmpeg_status(self) -> str:
        return self._ffmpeg_status

    ffmpegStatus = Property(str, get_ffmpeg_status, notify=ffmpegStatusChanged)

    def get_ffmpeg_status_level(self) -> str:
        return self._ffmpeg_status_level

    ffmpegStatusLevel = Property(str, get_ffmpeg_status_level, notify=ffmpegStatusLevelChanged)

    def get_hf_token_status(self) -> str:
        return self._hf_token_status

    hfTokenStatus = Property(str, get_hf_token_status, notify=hfTokenStatusChanged)

    def get_hf_token_status_level(self) -> str:
        return self._hf_token_status_level

    hfTokenStatusLevel = Property(str, get_hf_token_status_level, notify=hfTokenStatusLevelChanged)

    def get_hf_token_preview(self) -> str:
        return self._hf_token_preview

    hfTokenPreview = Property(str, get_hf_token_preview, notify=hfTokenPreviewChanged)

    def get_glm_ocr_model_status(self) -> str:
        return self._glm_ocr_model_status

    glmOcrModelStatus = Property(str, get_glm_ocr_model_status, notify=glmOcrModelStatusChanged)

    def get_glm_ocr_model_status_level(self) -> str:
        return self._glm_ocr_model_status_level

    glmOcrModelStatusLevel = Property(
        str,
        get_glm_ocr_model_status_level,
        notify=glmOcrModelStatusLevelChanged,
    )

    def get_downloading(self) -> bool:
        return self._downloading

    isDownloading = Property(bool, get_downloading, notify=downloadingChanged)

    windowX = Property(int, lambda self: self._window_x, notify=windowGeometryChanged)
    windowY = Property(int, lambda self: self._window_y, notify=windowGeometryChanged)
    windowW = Property(int, lambda self: self._window_w, notify=windowGeometryChanged)
    windowH = Property(int, lambda self: self._window_h, notify=windowGeometryChanged)

    @Slot()
    def addVideos(self) -> None:
        self._refresh_ffmpeg_status()
        if not self.ffmpeg_path or not self.ffprobe_path:
            self._show_critical(
                "Missing Dependencies",
                "Cannot add videos because ffmpeg/ffprobe is not configured.\n\n"
                "Open Dependencies and configure ffmpeg/ffprobe first.",
            )
            return
        if self._add_videos_dialog is not None and self._add_videos_dialog.isVisible():
            self._add_videos_dialog.raise_()
            self._add_videos_dialog.activateWindow()
            return

        dialog = QFileDialog()
        dialog.setWindowTitle("Select Video Files")
        dialog.setNameFilter(VIDEO_FILTER)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.filesSelected.connect(self._on_add_videos_selected)
        dialog.finished.connect(self._cleanup_add_videos_dialog)
        self._add_videos_dialog = dialog
        dialog.open()

    @Slot("QStringList")
    def _on_add_videos_selected(self, files: list[str]) -> None:
        for path in files:
            absolute_path = os.path.abspath(path)
            if absolute_path in self._selected_paths:
                continue
            self._selected_paths.add(absolute_path)
            audio_languages, subtitle_languages = self._inspect_stream_languages(absolute_path)
            self._video_files.append(
                {
                    "name": os.path.basename(absolute_path),
                    "path": absolute_path,
                    "checked": False,
                    "audio_language_items": audio_languages,
                    "subtitle_language_items": subtitle_languages,
                }
            )
        self.videoFilesChanged.emit()

    @Slot(int)
    def _cleanup_add_videos_dialog(self, _result: int) -> None:
        self._add_videos_dialog = None

    @Slot(int, bool)
    def setFileChecked(self, index: int, checked: bool) -> None:
        if 0 <= index < len(self._video_files):
            self._video_files[index]["checked"] = checked
            self.videoFilesChanged.emit()

    @Slot(bool)
    def setAllFilesChecked(self, checked: bool) -> None:
        for item in self._video_files:
            item["checked"] = checked
        self.videoFilesChanged.emit()

    @Slot()
    def refreshDependencyStatuses(self) -> None:
        self._refresh_dependency_statuses()

    def _start_glm_ocr_model_download(
        self,
        after_download_action: Callable[[], None] | None = None,
    ) -> None:
        if self._glm_download_thread is not None:
            if self._glm_download_thread.isRunning():
                if after_download_action is not None:
                    self._glm_download_followup_action = after_download_action
                # If user previously dismissed progress, show it again.
                self._glm_download_progress_dismissed = False
                self._show_glm_download_progress("GLM-OCR download is already running...")
                return
            # Cleanup stale thread references so a new download can start.
            if self._glm_download_worker is not None:
                self._glm_download_worker = None
            self._glm_download_thread = None

        if after_download_action is not None:
            self._glm_download_followup_action = after_download_action
        self._glm_download_started_at = time.monotonic()
        self._glm_download_diagnostics = []
        self._append_glm_download_diagnostic("GLM download requested by user.")
        self._append_glm_download_diagnostic(
            f"HF token configured: {'yes' if self._hf_token_configured() else 'no'}."
        )
        self._glm_download_progress_dismissed = False
        self._glm_download_cancel_requested = False
        self._show_glm_download_progress("Preparing GLM-OCR download...")
        hf_token = self._current_hf_token()
        self._glm_download_thread = QThread(self)
        self._glm_download_worker = GlmOcrModelDownloadWorker(GLM_OCR_MODEL_ID, hf_token=hf_token)
        self._glm_download_worker.moveToThread(self._glm_download_thread)
        self._glm_download_thread.started.connect(self._glm_download_worker.run)
        self._glm_download_worker.progress.connect(self._on_glm_download_progress)
        self._glm_download_worker.progressPercent.connect(self._on_glm_download_progress_percent)
        self._glm_download_worker.diagnostic.connect(self._on_glm_download_diagnostic)
        self._glm_download_worker.finished.connect(self._on_glm_download_finished)
        self._glm_download_worker.failed.connect(self._on_glm_download_failed)
        self._glm_download_worker.finished.connect(self._glm_download_thread.quit)
        self._glm_download_worker.failed.connect(self._glm_download_thread.quit)
        self._glm_download_thread.finished.connect(self._on_glm_download_thread_finished)
        self._glm_download_thread.start()

    def _append_glm_download_diagnostic(self, message: str) -> None:
        elapsed_text = ""
        if self._glm_download_started_at is not None:
            elapsed_text = f"[+{(time.monotonic() - self._glm_download_started_at):.1f}s] "
        line = f"{elapsed_text}{message}"
        self._glm_download_diagnostics.append(line)
        logging.info("GLM download: %s", line)

    def _glm_download_diagnostics_text(self) -> str:
        if not self._glm_download_diagnostics:
            return "No diagnostics captured yet."
        return "\n".join(self._glm_download_diagnostics)

    def _hf_token_configured(self) -> bool:
        if self._get_saved_hf_token():
            return True
        env_token = str(os.environ.get("HF_TOKEN", "") or "").strip()
        return bool(env_token)

    def _show_glm_download_setup(
        self,
        continue_action: Callable[[], None],
    ) -> None:
        if self._glm_setup_window is not None and self._glm_setup_window.isVisible():
            self._glm_download_followup_action = continue_action
            self._glm_setup_window.raise_()
            self._glm_setup_window.activateWindow()
            return
        self._glm_download_followup_action = continue_action
        window = GlmDownloadSetupWindow(
            on_download=self._on_glm_setup_download_clicked,
            on_cancel=self._on_glm_setup_cancelled,
        )
        existing_token = self._get_saved_hf_token()
        if existing_token:
            window.token_edit.setText(existing_token)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.destroyed.connect(self._on_glm_setup_window_destroyed)
        self._glm_setup_window = window
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_glm_setup_download_clicked(self, token: str) -> None:
        if token:
            self.setHfToken(token)
        followup = self._glm_download_followup_action
        self._glm_download_followup_action = None
        # Always surface immediate feedback when user clicks Download Now.
        self._glm_download_progress_dismissed = False
        self._show_glm_download_progress("Preparing GLM-OCR download...")
        self._start_glm_ocr_model_download(after_download_action=followup)

    def _current_hf_token(self) -> str | None:
        saved = self._get_saved_hf_token()
        if saved:
            return saved
        env_token = str(os.environ.get("HF_TOKEN", "") or "").strip()
        return env_token or None

    def _on_glm_setup_cancelled(self) -> None:
        self._glm_download_followup_action = None

    @Slot()
    def _on_glm_setup_window_destroyed(self) -> None:
        self._glm_setup_window = None

    def _show_glm_download_progress(self, text: str) -> None:
        if self._glm_download_progress_dismissed:
            return
        window = self._glm_download_progress_window
        if window is not None and window.isVisible():
            window.set_status(text)
            window.raise_()
            window.activateWindow()
            return
        window = GlmDownloadProgressWindow(
            on_cancel=lambda: self._cancel_glm_download("Cancelling GLM-OCR download...")
        )
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.destroyed.connect(self._on_glm_download_progress_window_destroyed)
        window.set_status(text)
        for line in self._glm_download_diagnostics:
            window.append_log(line)
        self._glm_download_progress_window = window
        window.show()
        window.raise_()
        window.activateWindow()

    @Slot(str)
    def _on_glm_download_diagnostic(self, message: str) -> None:
        self._append_glm_download_diagnostic(message)
        window = self._glm_download_progress_window
        if window is not None:
            window.append_log(self._glm_download_diagnostics[-1])

    @Slot(str)
    def _on_glm_download_progress(self, message: str) -> None:
        self._append_glm_download_diagnostic(f"Progress: {message}")
        self._show_glm_download_progress(message)

    @Slot(int)
    def _on_glm_download_progress_percent(self, percent: int) -> None:
        window = self._glm_download_progress_window
        if window is not None:
            window.set_progress(percent)

    @Slot(str)
    def _on_glm_download_finished(self, local_model_path: str) -> None:
        if self._glm_download_cancel_requested:
            return
        self._append_glm_download_diagnostic(f"Download finished. Local path: {local_model_path}")
        followup_action = self._glm_download_followup_action
        self._glm_download_followup_action = None
        self._close_glm_download_progress_box()
        self._refresh_glm_ocr_model_status()
        if followup_action is not None:
            followup_action()
            return
        self._show_info(
            "GLM-OCR Downloaded",
            "GLM-OCR model is ready.\n\n"
            f"Model path:\n{local_model_path}\n\n"
            "Use session.log for detailed diagnostics.",
        )

    @Slot(str)
    def _on_glm_download_failed(self, error_message: str) -> None:
        if self._glm_download_cancel_requested or error_message.strip() == "GLM-OCR download cancelled.":
            return
        self._append_glm_download_diagnostic(f"Failure: {error_message}")
        self._glm_download_followup_action = None
        self._close_glm_download_progress_box()
        self._refresh_glm_ocr_model_status()
        self._show_warning(
            "GLM-OCR Download Failed",
            "Could not download GLM-OCR model.\n\n"
            f"{error_message}\n\n"
            "Diagnostics:\n"
            f"{self._glm_download_diagnostics_text()}",
        )

    def _cancel_glm_download(self, status_text: str | None = None) -> None:
        if self._glm_download_thread is None or not self._glm_download_thread.isRunning():
            return
        self._glm_download_cancel_requested = True
        self._glm_download_followup_action = None
        self._append_glm_download_diagnostic("Cancellation requested by user.")
        window = self._glm_download_progress_window
        if status_text and window is not None and window.isVisible():
            window.set_status(status_text)
        if self._glm_download_worker is not None:
            try:
                self._glm_download_worker.cancel()
            except Exception:
                pass
        self._glm_download_thread.requestInterruption()
        self._glm_download_thread.quit()
        if not self._glm_download_thread.wait(1500):
            self._glm_download_thread.terminate()
            self._glm_download_thread.wait(1000)
        self._close_glm_download_progress_box()
        self._refresh_glm_ocr_model_status()
        self._show_info("GLM-OCR Download", "GLM-OCR download was cancelled.")

    def _close_glm_download_progress_box(self) -> None:
        if self._glm_download_progress_window is not None:
            self._glm_download_progress_window.close()

    @Slot()
    def _on_glm_download_progress_window_destroyed(self) -> None:
        self._glm_download_progress_window = None
        if self._glm_download_thread is not None and self._glm_download_thread.isRunning():
            self._glm_download_progress_dismissed = True

    def _on_glm_download_thread_finished(self) -> None:
        if self._glm_download_worker is not None:
            self._glm_download_worker.deleteLater()
            self._glm_download_worker = None
        if self._glm_download_thread is not None:
            self._glm_download_thread.deleteLater()
            self._glm_download_thread = None
        self._glm_download_progress_dismissed = False
        self._glm_download_cancel_requested = False
        self._glm_download_started_at = None
        self._glm_download_followup_action = None

    @Slot(str)
    def setHfToken(self, token: str) -> None:
        clean_token = token.strip()
        if clean_token:
            self.settings.setValue("hf/token", clean_token)
            os.environ["HF_TOKEN"] = clean_token
        else:
            self.settings.remove("hf/token")
            os.environ.pop("HF_TOKEN", None)
        self._refresh_hf_token_status()

    @Slot()
    def clearHfToken(self) -> None:
        self.settings.remove("hf/token")
        os.environ.pop("HF_TOKEN", None)
        self._refresh_hf_token_status()

    def _inspect_stream_languages(
        self, file_path: str
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        payload = self._run_ffprobe_json(file_path)
        if payload is None:
            return ([{"label": "N/A", "stream_index": -1}], [{"label": "N/A", "stream_index": -1}])
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return ([{"label": "N/A", "stream_index": -1}], [{"label": "N/A", "stream_index": -1}])

        audio_streams: list[dict[str, object]] = []
        subtitle_streams: list[dict[str, object]] = []

        for fallback_index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                continue
            codec_type = str(stream.get("codec_type", "")).lower()
            stream_index = self._stream_index(stream, fallback_index)
            language = self._stream_language(stream)
            if codec_type == "audio":
                audio_streams.append(
                    {
                        "label": f"{language} #{stream_index}",
                        "stream_index": stream_index,
                        "language": language,
                    }
                )
            elif codec_type == "subtitle":
                codec_name = str(stream.get("codec_name", "unknown")).strip().lower() or "unknown"
                subtitle_streams.append(
                    {
                        "label": f"{language} #{stream_index}",
                        "stream_index": stream_index,
                        "language": language,
                        "codec_name": codec_name,
                    }
                )

        if not audio_streams:
            audio_streams = [{"label": "-", "stream_index": -1}]
        if not subtitle_streams:
            subtitle_streams = [{"label": "-", "stream_index": -1}]
        return (audio_streams, subtitle_streams)

    def _run_ffprobe_json(self, file_path: str) -> dict[str, object] | None:
        ffprobe_exe = self.ffprobe_path or shutil.which("ffprobe")
        if not ffprobe_exe:
            return None
        command = [
            ffprobe_exe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            file_path,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return None
            payload = json.loads(result.stdout or "{}")
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception:
            return None

    @Slot(str, int)
    def editSubtitle(self, file_path: str, stream_index: int) -> None:
        if stream_index < 0:
            self._show_info("Subtitle Editor", "No editable subtitle stream selected.")
            return

        payload = self._run_ffprobe_json(file_path)
        if payload is None:
            self._show_warning("Subtitle Editor", "Could not read stream info with ffprobe.")
            return

        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            self._show_info("Subtitle Editor", "No stream data found.")
            return

        stream = self._find_subtitle_stream_by_index(streams, stream_index)
        if stream is None:
            self._show_info("Subtitle Editor", f"Subtitle stream #{stream_index} was not found.")
            return

        codec_name = str(stream.get("codec_name", "unknown")).strip().lower() or "unknown"
        editor_key = f"{os.path.abspath(file_path)}::{stream_index}"
        if not self._can_open_editor(editor_key):
            return

        if codec_name in TEXT_BASED_SUBTITLE_CODECS:
            subtitle_text = self._extract_subtitle_text(file_path, stream_index)
            if subtitle_text is None:
                self._show_warning(
                    "Subtitle Editor",
                    "Could not extract subtitle text with ffmpeg.",
                )
                return
            stream_lang = self._stream_language(stream)
            self._open_subtitle_editor(
                file_path=file_path,
                stream_index=stream_index,
                stream_lang=stream_lang,
                codec_name=codec_name,
                subtitle_text=subtitle_text,
                editor_key=editor_key,
            )
            return

        if self._is_image_subtitle_codec(codec_name):
            stream_lang = self._stream_language(stream)
            cache_key = self._ocr_cache_key(file_path, stream_index)
            if cache_key is not None:
                cached_srt = self._ocr_srt_cache.get(cache_key)
                if cached_srt:
                    self._open_subtitle_editor(
                        file_path=file_path,
                        stream_index=stream_index,
                        stream_lang=stream_lang,
                        codec_name=codec_name,
                        subtitle_text=cached_srt,
                        editor_key=editor_key,
                    )
                    return
            self._ensure_glm_ocr_consent_then_start(
                file_path=file_path,
                stream_index=stream_index,
                stream_lang=stream_lang,
                codec_name=codec_name,
                editor_key=editor_key,
            )
            return

        if codec_name not in TEXT_BASED_SUBTITLE_CODECS:
            self._show_warning(
                "Subtitle Editor",
                f"Stream #{stream_index} uses '{codec_name}', which is not supported for editing.",
            )
            return

    def _open_subtitle_editor(
        self,
        file_path: str,
        stream_index: int,
        stream_lang: str,
        codec_name: str,
        subtitle_text: str,
        editor_key: str,
    ) -> None:
        if self._subtitle_editor is not None and self._subtitle_editor.isVisible():
            if self._subtitle_editor_key == editor_key:
                self._subtitle_editor.raise_()
                self._subtitle_editor.activateWindow()
                return
            self._subtitle_editor.raise_()
            self._subtitle_editor.activateWindow()
            self._show_info(
                "Subtitle Editor",
                "An editor window is already open. Close it before opening another subtitle stream.",
            )
            return

        editor = SubtitleEditorWindow(
            file_path,
            stream_index,
            stream_lang,
            codec_name,
            subtitle_text,
            self._save_embedded_subtitle,
        )
        editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        editor.destroyed.connect(self._on_subtitle_editor_destroyed)
        self._subtitle_editor = editor
        self._subtitle_editor_key = editor_key
        editor.show()
        editor.raise_()
        editor.activateWindow()

    def _can_open_editor(self, editor_key: str) -> bool:
        if self._subtitle_editor is None or not self._subtitle_editor.isVisible():
            return True
        if self._subtitle_editor_key == editor_key:
            self._subtitle_editor.raise_()
            self._subtitle_editor.activateWindow()
            return False
        self._subtitle_editor.raise_()
        self._subtitle_editor.activateWindow()
        self._show_info(
            "Subtitle Editor",
            "An editor window is already open. Close it before opening another subtitle stream.",
        )
        return False

    def _is_image_subtitle_codec(self, codec_name: str) -> bool:
        return codec_name in IMAGE_BASED_SUBTITLE_CODECS

    def _ocr_cache_key(self, file_path: str, stream_index: int) -> tuple[str, int, int, int] | None:
        absolute = os.path.abspath(file_path)
        try:
            stat = os.stat(absolute)
        except OSError:
            return None
        return (absolute, stream_index, int(stat.st_mtime_ns), int(stat.st_size))

    def _ensure_glm_ocr_consent_then_start(
        self,
        file_path: str,
        stream_index: int,
        stream_lang: str,
        codec_name: str,
        editor_key: str,
    ) -> None:
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            self._show_info(
                "Subtitle OCR",
                "Image subtitle OCR is already running. Please wait for it to finish.",
            )
            return
        self._start_image_subtitle_ocr_with_download_if_needed(
            file_path=file_path,
            stream_index=stream_index,
            stream_lang=stream_lang,
            codec_name=codec_name,
            editor_key=editor_key,
        )

    def _start_image_subtitle_ocr(
        self,
        file_path: str,
        stream_index: int,
        stream_lang: str,
        codec_name: str,
        editor_key: str,
    ) -> None:
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            self._show_info(
                "Subtitle OCR",
                "Image subtitle OCR is already running. Please wait for it to finish.",
            )
            return
        self._ocr_target_key = editor_key
        self._ocr_target_file_path = file_path
        self._ocr_target_stream_index = stream_index
        self._ocr_target_stream_lang = stream_lang
        self._ocr_target_codec_name = codec_name
        self._ocr_target_cache_key = self._ocr_cache_key(file_path, stream_index)

        self._show_ocr_progress("Preparing GLM-OCR...")
        hf_token = self._current_hf_token()
        self._ocr_thread = QThread(self)
        self._ocr_worker = ImageSubtitleOcrWorker(
            file_path=file_path,
            stream_index=stream_index,
            codec_name=codec_name,
            ffmpeg_exe=self.ffmpeg_path,
            ffprobe_exe=self.ffprobe_path,
            hf_token=hf_token,
        )
        self._ocr_worker.moveToThread(self._ocr_thread)
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.finished.connect(self._on_ocr_thread_finished)
        self._ocr_thread.start()

    def _start_image_subtitle_ocr_with_download_if_needed(
        self,
        file_path: str,
        stream_index: int,
        stream_lang: str,
        codec_name: str,
        editor_key: str,
    ) -> None:
        if self._find_local_glm_ocr_model_directory() is None:
            self._show_glm_download_setup(
                continue_action=lambda: self._start_image_subtitle_ocr(
                    file_path=file_path,
                    stream_index=stream_index,
                    stream_lang=stream_lang,
                    codec_name=codec_name,
                    editor_key=editor_key,
                )
            )
            return
        self._start_image_subtitle_ocr(
            file_path=file_path,
            stream_index=stream_index,
            stream_lang=stream_lang,
            codec_name=codec_name,
            editor_key=editor_key,
        )

    def _show_ocr_progress(self, text: str) -> None:
        box = self._ocr_progress_box
        if box is not None and box.isVisible():
            box.setText(text)
            box.raise_()
            box.activateWindow()
            return
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Subtitle OCR")
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Cancel)
        box.setModal(False)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.buttonClicked.connect(self._on_ocr_progress_box_clicked)
        box.finished.connect(self._cleanup_ocr_progress_box)
        self._ocr_progress_box = box
        box.show()

    @Slot(str)
    def _on_ocr_progress(self, message: str) -> None:
        self._show_ocr_progress(message)

    @Slot(str)
    def _on_ocr_finished(self, srt_text: str) -> None:
        self._close_ocr_progress_box()
        self._refresh_glm_ocr_model_status()
        file_path = self._ocr_target_file_path
        stream_index = self._ocr_target_stream_index
        stream_lang = self._ocr_target_stream_lang
        codec_name = self._ocr_target_codec_name
        editor_key = self._ocr_target_key
        cache_key = self._ocr_target_cache_key
        if (
            file_path is None
            or stream_index is None
            or stream_lang is None
            or codec_name is None
            or editor_key is None
        ):
            self._show_warning("Subtitle OCR", "OCR completed but target subtitle context is unavailable.")
            return
        if cache_key is not None:
            self._ocr_srt_cache[cache_key] = srt_text
        if not self._can_open_editor(editor_key):
            return
        self._open_subtitle_editor(
            file_path=file_path,
            stream_index=stream_index,
            stream_lang=stream_lang,
            codec_name=codec_name,
            subtitle_text=srt_text,
            editor_key=editor_key,
        )

    @Slot(str)
    def _on_ocr_failed(self, error_message: str) -> None:
        self._close_ocr_progress_box()
        self._refresh_glm_ocr_model_status()
        if error_message.strip() == "OCR cancelled.":
            self._show_info("Subtitle OCR", "Image subtitle OCR was cancelled.")
            return
        self._show_warning(
            "Subtitle OCR Failed",
            f"Could not convert image subtitle stream to SRT with GLM-OCR.\n\n{error_message}",
        )

    def _on_ocr_progress_box_clicked(self, button: object) -> None:
        if self._ocr_progress_box is None:
            return
        standard_button = self._ocr_progress_box.standardButton(button)
        if standard_button == QMessageBox.StandardButton.Cancel:
            self._cancel_ocr_operation("Cancelling OCR...")

    def _cancel_ocr_operation(self, status_text: str | None = None) -> None:
        if status_text:
            self._show_ocr_progress(status_text)
        if self._ocr_worker is not None:
            try:
                self._ocr_worker.cancel()
            except Exception:
                pass
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            self._ocr_thread.requestInterruption()
            self._ocr_thread.quit()

    def _close_ocr_progress_box(self) -> None:
        if self._ocr_progress_box is not None:
            self._ocr_progress_box.close()

    @Slot(int)
    def _cleanup_ocr_progress_box(self, _result: int) -> None:
        self._ocr_progress_box = None

    def _on_ocr_thread_finished(self) -> None:
        if self._ocr_worker is not None:
            self._ocr_worker.deleteLater()
            self._ocr_worker = None
        if self._ocr_thread is not None:
            self._ocr_thread.deleteLater()
            self._ocr_thread = None
        self._ocr_target_key = None
        self._ocr_target_file_path = None
        self._ocr_target_stream_index = None
        self._ocr_target_stream_lang = None
        self._ocr_target_codec_name = None
        self._ocr_target_cache_key = None

    @Slot()
    def _on_subtitle_editor_destroyed(self) -> None:
        self._subtitle_editor = None
        self._subtitle_editor_key = None

    @Slot(str, int)
    def showSubtitleInfo(self, file_path: str, stream_index: int) -> None:
        if stream_index < 0:
            self._show_info("Subtitle Info", "No subtitle stream selected.")
            return

        payload = self._run_ffprobe_json(file_path)
        if payload is None:
            self._show_warning("Subtitle Info", "Could not read stream info with ffprobe.")
            return

        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            self._show_info("Subtitle Info", "No stream data found.")
            return

        stream = self._find_subtitle_stream_by_index(streams, stream_index)
        if stream is None:
            self._show_info("Subtitle Info", f"Subtitle stream #{stream_index} was not found.")
            return

        codec_name = stream.get("codec_name", "unknown")
        tags = stream.get("tags")
        disposition = stream.get("disposition")
        stream_lang = "und"
        stream_title = "-"
        if isinstance(tags, dict):
            tag_lang = tags.get("language")
            if isinstance(tag_lang, str) and tag_lang.strip():
                stream_lang = tag_lang.strip().lower()
            tag_title = tags.get("title")
            if isinstance(tag_title, str) and tag_title.strip():
                stream_title = tag_title.strip()
        default_flag = 0
        forced_flag = 0
        if isinstance(disposition, dict):
            default_flag = int(disposition.get("default", 0) or 0)
            forced_flag = int(disposition.get("forced", 0) or 0)
        self._show_info(
            "Subtitle Info",
            (
                f"File: {os.path.basename(file_path)}\n\n"
                f"Stream #{stream_index}\n"
                f"  codec: {codec_name}\n"
                f"  language: {stream_lang}\n"
                f"  title: {stream_title}\n"
                f"  default: {default_flag}, forced: {forced_flag}"
            ),
        )

    def _find_subtitle_stream_by_index(
        self, streams: list[object], stream_index: int
    ) -> dict[str, object] | None:
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            if str(stream.get("codec_type", "")).lower() != "subtitle":
                continue
            if self._stream_index(stream, -1) == stream_index:
                return stream
        return None

    def _stream_language(self, stream: dict[str, object]) -> str:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            tag_lang = tags.get("language")
            if isinstance(tag_lang, str) and tag_lang.strip():
                return tag_lang.strip().lower()
        return "und"

    def _stream_index(self, stream: dict[str, object], fallback: int) -> int:
        value = stream.get("index", fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _extract_subtitle_text(self, file_path: str, stream_index: int) -> str | None:
        ffmpeg_exe = self.ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg_exe:
            return None
        command = [
            ffmpeg_exe,
            "-v",
            "error",
            "-nostdin",
            "-i",
            file_path,
            "-map",
            f"0:{stream_index}",
            "-f",
            "srt",
            "-",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=False, check=False)
            if result.returncode != 0:
                return None
            raw_text = result.stdout or b""
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    return raw_text.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw_text.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _save_embedded_subtitle(
        self, file_path: str, stream_index: int, subtitle_text: str
    ) -> tuple[bool, str]:
        ffmpeg_exe = self.ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg_exe:
            return (False, "ffmpeg binary not found.")

        payload = self._run_ffprobe_json(file_path)
        if payload is None:
            return (False, "Could not inspect file streams with ffprobe.")
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return (False, "Invalid stream metadata from ffprobe.")

        target_stream = self._find_subtitle_stream_by_index(streams, stream_index)
        if target_stream is None:
            return (False, f"Subtitle stream #{stream_index} not found.")

        subtitle_stream_count = 0
        for stream in streams:
            if isinstance(stream, dict) and str(stream.get("codec_type", "")).lower() == "subtitle":
                subtitle_stream_count += 1
        if subtitle_stream_count == 0:
            return (False, "No subtitle streams found.")

        output_codec = self._output_subtitle_codec_for_container(file_path)
        if output_codec is None:
            return (False, "Container does not support automatic subtitle replacement.")

        language = self._stream_language(target_stream)
        title = self._stream_title(target_stream)
        new_subtitle_index = subtitle_stream_count - 1

        source_path = Path(file_path)
        with tempfile.TemporaryDirectory(prefix="sub_manager_") as tmp_dir:
            temp_subtitle_path = Path(tmp_dir) / "edited_subtitle.srt"
            try:
                temp_subtitle_path.write_text(subtitle_text, encoding="utf-8")
            except Exception as exc:
                return (False, f"Could not prepare temporary subtitle file: {exc}")

            temp_output_path = source_path.parent / (
                f".{source_path.stem}.submgr_tmp{source_path.suffix}"
            )
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except Exception as exc:
                    return (False, f"Could not clear temporary output file: {exc}")

            command = [
                ffmpeg_exe,
                "-y",
                "-v",
                "error",
                "-nostdin",
                "-i",
                file_path,
                "-f",
                "srt",
                "-i",
                str(temp_subtitle_path),
                "-map",
                "0",
                "-map",
                f"-0:{stream_index}",
                "-map",
                "1:0",
                "-c",
                "copy",
                f"-c:s:{new_subtitle_index}",
                output_codec,
                f"-metadata:s:s:{new_subtitle_index}",
                f"language={language}",
            ]
            if title:
                command.extend(
                    [
                        f"-metadata:s:s:{new_subtitle_index}",
                        f"title={title}",
                    ]
                )
            command.append(str(temp_output_path))

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
            except Exception as exc:
                return (False, f"Could not run ffmpeg: {exc}")
            if result.returncode != 0:
                try:
                    if temp_output_path.exists():
                        temp_output_path.unlink()
                except Exception:
                    pass
                error_message = (result.stderr or "Unknown ffmpeg error").strip()
                return (False, error_message)
            try:
                os.replace(temp_output_path, source_path)
            except Exception as exc:
                try:
                    if temp_output_path.exists():
                        temp_output_path.unlink()
                except Exception:
                    pass
                return (False, f"Could not overwrite original file: {exc}")
        self._invalidate_ocr_cache_for_file(str(source_path))
        return (True, str(source_path))

    def _output_subtitle_codec_for_container(self, file_path: str) -> str | None:
        ext = Path(file_path).suffix.lower()
        if ext in {".mp4", ".m4v", ".mov"}:
            return "mov_text"
        if ext in {".mkv", ".avi"}:
            return "srt"
        if ext == ".webm":
            return "webvtt"
        return None

    def _invalidate_ocr_cache_for_file(self, file_path: str) -> None:
        absolute = os.path.abspath(file_path)
        stale_keys = [key for key in self._ocr_srt_cache if key[0] == absolute]
        for key in stale_keys:
            self._ocr_srt_cache.pop(key, None)

    def _stream_title(self, stream: dict[str, object]) -> str:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            tag_title = tags.get("title")
            if isinstance(tag_title, str) and tag_title.strip():
                return tag_title.strip()
        return ""

    @Slot()
    def selectFfmpegDirectory(self) -> None:
        initial_dir = self.settings.value("ffmpeg/bin_dir")
        start_dir = (
            initial_dir
            if isinstance(initial_dir, str) and initial_dir
            else os.path.expanduser("~")
        )
        selected_dir = QFileDialog.getExistingDirectory(None, "Select FFmpeg Directory", start_dir)
        if not selected_dir:
            return
        ffmpeg_path = self._binary_from_directory(selected_dir, "ffmpeg")
        ffprobe_path = self._binary_from_directory(selected_dir, "ffprobe")
        if ffmpeg_path and ffprobe_path:
            self.settings.setValue("ffmpeg/bin_dir", selected_dir)
            self._refresh_ffmpeg_status()
            return
        self._show_warning(
            "Invalid Directory",
            "Selected directory must contain executable ffmpeg and ffprobe binaries.",
        )

    @Slot()
    def searchFfmpegInPath(self) -> None:
        previous_dir = self.settings.value("ffmpeg/bin_dir")
        self.settings.remove("ffmpeg/bin_dir")
        self._refresh_ffmpeg_status()
        if self.ffmpeg_path and self.ffprobe_path:
            return
        if isinstance(previous_dir, str) and previous_dir:
            self.settings.setValue("ffmpeg/bin_dir", previous_dir)
            self._refresh_ffmpeg_status()

    @Slot()
    def downloadFfmpeg(self) -> None:
        if self.download_thread is not None:
            return
        asset_info = self._resolve_btbn_asset()
        if asset_info is None:
            self._show_warning(
                "Unsupported OS",
                "Automatic FFmpeg download is currently supported on Linux and Windows only.",
            )
            return

        target, archive_ext = asset_info
        archive_name = f"ffmpeg-master-latest-{target}-gpl.{archive_ext}"
        download_url = (
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
            f"{archive_name}"
        )
        tools_dir = Path(__file__).resolve().parent / "tools" / "ffmpeg"
        if self._tools_ffmpeg_binaries_exist(tools_dir):
            self._confirm_ffmpeg_override(download_url, archive_ext, tools_dir)
            return
        self._start_ffmpeg_download(download_url, archive_ext, tools_dir)

    def _tools_ffmpeg_binaries_exist(self, tools_dir: Path) -> bool:
        bin_dir = tools_dir / "bin"
        ffmpeg_path = self._binary_from_directory(str(bin_dir), "ffmpeg")
        ffprobe_path = self._binary_from_directory(str(bin_dir), "ffprobe")
        return bool(ffmpeg_path and ffprobe_path)

    def _confirm_ffmpeg_override(self, download_url: str, archive_ext: str, tools_dir: Path) -> None:
        if self._ffmpeg_override_box is not None and self._ffmpeg_override_box.isVisible():
            self._ffmpeg_override_box.raise_()
            self._ffmpeg_override_box.activateWindow()
            return

        bin_dir = tools_dir / "bin"
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Overwrite Existing Binaries")
        box.setText(
            "Found existing ffmpeg/ffprobe binaries.\n\n"
            f"Directory:\n{bin_dir}\n\n"
            "Choose what to do with the existing binaries."
        )
        download_button = box.addButton("Download and Override", QMessageBox.ButtonRole.AcceptRole)
        use_existing_button = box.addButton("Use Existing", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(use_existing_button)
        box.setModal(False)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.buttonClicked.connect(
            lambda button: self._on_ffmpeg_override_button_clicked(
                button,
                download_url,
                archive_ext,
                str(tools_dir),
            )
        )
        box.finished.connect(
            lambda _result: self._on_ffmpeg_override_finished(
                download_url,
                archive_ext,
                str(tools_dir),
            )
        )
        box.finished.connect(self._cleanup_ffmpeg_override_box)
        self._ffmpeg_override_box = box
        self._ffmpeg_override_download_button = download_button
        self._ffmpeg_override_use_existing_button = use_existing_button
        self._ffmpeg_override_decided = False
        box.open()

    def _use_existing_tools_ffmpeg(self, tools_dir: str) -> None:
        bin_dir = str(Path(tools_dir) / "bin")
        self.settings.setValue("ffmpeg/bin_dir", bin_dir)
        self._refresh_ffmpeg_status()

    def _on_ffmpeg_override_button_clicked(
        self,
        button: object,
        download_url: str,
        archive_ext: str,
        tools_dir: str,
    ) -> None:
        self._ffmpeg_override_decided = True
        if button == self._ffmpeg_override_download_button:
            self._start_ffmpeg_download(download_url, archive_ext, Path(tools_dir))
            return
        self._use_existing_tools_ffmpeg(tools_dir)

    def _on_ffmpeg_override_finished(
        self,
        download_url: str,
        archive_ext: str,
        tools_dir: str,
    ) -> None:
        if self._ffmpeg_override_decided:
            return
        self._use_existing_tools_ffmpeg(tools_dir)

    @Slot(int)
    def _cleanup_ffmpeg_override_box(self, _result: int) -> None:
        self._ffmpeg_override_box = None
        self._ffmpeg_override_download_button = None
        self._ffmpeg_override_use_existing_button = None
        self._ffmpeg_override_decided = False

    def _start_ffmpeg_download(self, download_url: str, archive_ext: str, tools_dir: Path) -> None:
        if self.download_thread is not None:
            return

        self._set_downloading(True)
        self._set_ffmpeg_status("progress", "Starting FFmpeg download...")
        self.download_thread = QThread(self)
        self.download_worker = FFmpegDownloadWorker(download_url, archive_ext, tools_dir)
        self.download_worker.moveToThread(self.download_thread)
        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.failed.connect(self._on_download_failed)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.failed.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self._on_download_thread_finished)
        self.download_thread.start()

    @Slot(str)
    def _on_download_progress(self, message: str) -> None:
        self._set_ffmpeg_status("progress", message)

    @Slot(str)
    def _on_download_finished(self, install_bin_dir: str) -> None:
        self.settings.setValue("ffmpeg/bin_dir", install_bin_dir)
        self._refresh_ffmpeg_status()
        self._show_info(
            "FFmpeg Downloaded",
            f"FFmpeg was downloaded and installed to:\n{install_bin_dir}",
        )

    @Slot(str)
    def _on_download_failed(self, error_message: str) -> None:
        self._set_ffmpeg_status("error", "FFmpeg download failed.")
        self._show_critical(
            "Download Failed",
            f"Could not download/install FFmpeg.\n\n{error_message}",
        )

    def _on_download_thread_finished(self) -> None:
        self._set_downloading(False)
        if self.download_worker is not None:
            self.download_worker.deleteLater()
            self.download_worker = None
        if self.download_thread is not None:
            self.download_thread.deleteLater()
            self.download_thread = None

    def _set_downloading(self, value: bool) -> None:
        if self._downloading != value:
            self._downloading = value
            self.downloadingChanged.emit()

    @Slot(int, int, int, int)
    def saveWindowGeometry(self, x: int, y: int, w: int, h: int) -> None:
        self._window_x, self._window_y, self._window_w, self._window_h = x, y, w, h
        self.settings.setValue("window/x", x)
        self.settings.setValue("window/y", y)
        self.settings.setValue("window/w", w)
        self.settings.setValue("window/h", h)

    @Slot(result=bool)
    def requestClose(self) -> bool:
        if self.download_thread is not None and self.download_thread.isRunning():
            self._show_info(
                "Download in Progress",
                "Please wait for FFmpeg download to finish before closing the app.",
            )
            return False
        if self._glm_download_thread is not None and self._glm_download_thread.isRunning():
            self._cancel_glm_download("Stopping GLM-OCR download for shutdown...")
            if self._glm_download_thread is not None and self._glm_download_thread.isRunning():
                self._show_warning(
                    "GLM-OCR Shutdown",
                    "GLM-OCR download did not stop in time. Please try again in a moment.",
                )
                return False
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            self._cancel_ocr_operation("Stopping OCR for shutdown...")
            if not self._wait_for_ocr_shutdown(timeout_ms=2500):
                self._show_warning(
                    "OCR Shutdown",
                    "OCR did not stop in time and will be force-stopped.",
                )
                self._force_stop_ocr_thread()
        return True

    def _wait_for_ocr_shutdown(self, timeout_ms: int) -> bool:
        if self._ocr_thread is None:
            return True
        return self._ocr_thread.wait(timeout_ms)

    def _force_stop_ocr_thread(self) -> None:
        if self._ocr_thread is None:
            return
        try:
            self._ocr_thread.terminate()
            self._ocr_thread.wait(1000)
        except Exception:
            pass

    def clear_session_ocr_state(self) -> None:
        self._ocr_srt_cache.clear()
        ImageSubtitleOcrWorker.clear_model_cache()

    def _show_message(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
        log_message = f"{title}: {text}"
        if icon == QMessageBox.Icon.Critical:
            logging.error(log_message)
        elif icon == QMessageBox.Icon.Warning:
            logging.warning(log_message)
        else:
            logging.info(log_message)

        box = self._active_message_box
        if box is not None and box.isVisible():
            box.setIcon(icon)
            box.setWindowTitle(title)
            box.setText(text)
            box.raise_()
            box.activateWindow()
            return

        box = QMessageBox()
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.setModal(False)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._active_message_box = box
        box.finished.connect(self._cleanup_message_box)
        box.show()

    @Slot(int)
    def _cleanup_message_box(self, _result: int) -> None:
        self._active_message_box = None

    def _show_info(self, title: str, text: str) -> None:
        self._show_message(QMessageBox.Icon.Information, title, text)

    def _show_warning(self, title: str, text: str) -> None:
        self._show_message(QMessageBox.Icon.Warning, title, text)

    def _show_critical(self, title: str, text: str) -> None:
        self._show_message(QMessageBox.Icon.Critical, title, text)

    def _set_hf_token_status(self, level: str, text: str) -> None:
        if self._hf_token_status != text:
            self._hf_token_status = text
            self.hfTokenStatusChanged.emit()
        if self._hf_token_status_level != level:
            self._hf_token_status_level = level
            self.hfTokenStatusLevelChanged.emit()

    def _set_hf_token_preview(self, preview: str) -> None:
        if self._hf_token_preview != preview:
            self._hf_token_preview = preview
            self.hfTokenPreviewChanged.emit()

    def _set_glm_ocr_model_status(self, level: str, text: str) -> None:
        if self._glm_ocr_model_status != text:
            self._glm_ocr_model_status = text
            self.glmOcrModelStatusChanged.emit()
        if self._glm_ocr_model_status_level != level:
            self._glm_ocr_model_status_level = level
            self.glmOcrModelStatusLevelChanged.emit()

    def _set_ffmpeg_status(self, level: str, text: str) -> None:
        if self._ffmpeg_status != text:
            self._ffmpeg_status = text
            self.ffmpegStatusChanged.emit()
        if self._ffmpeg_status_level != level:
            self._ffmpeg_status_level = level
            self.ffmpegStatusLevelChanged.emit()

    def _get_saved_hf_token(self) -> str:
        return str(self.settings.value("hf/token", "") or "").strip()

    def _apply_hf_token_environment_from_settings(self) -> None:
        saved_token = self._get_saved_hf_token()
        if saved_token:
            os.environ["HF_TOKEN"] = saved_token

    def _binary_from_directory(self, directory: str, binary_name: str) -> str | None:
        file_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
        path = os.path.join(directory, file_name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    def _refresh_dependency_statuses(self) -> None:
        self._refresh_ffmpeg_status()
        self._refresh_hf_token_status()
        self._refresh_glm_ocr_model_status()

    def _refresh_hf_token_status(self) -> None:
        saved_token = self._get_saved_hf_token()
        if saved_token:
            self._set_hf_token_status("ok", "Configured (saved).")
            self._set_hf_token_preview(self._mask_token(saved_token))
            return
        env_token = str(os.environ.get("HF_TOKEN", "") or "").strip()
        if env_token:
            self._set_hf_token_status("ok", "Configured (environment).")
            self._set_hf_token_preview(self._mask_token(env_token))
            return
        self._set_hf_token_status("warn", "Not configured (optional).")
        self._set_hf_token_preview("")

    def _mask_token(self, token: str) -> str:
        clean_token = token.strip()
        if not clean_token:
            return ""
        if len(clean_token) <= 8:
            return "********"
        return f"{clean_token[:4]}********{clean_token[-4:]}"

    def _refresh_glm_ocr_model_status(self) -> None:
        model_directory = self._find_local_glm_ocr_model_directory()
        if model_directory is not None:
            self._set_glm_ocr_model_status("ok", f"Installed: {model_directory}")
            return
        self._set_glm_ocr_model_status("warn", "Missing (download on first OCR run).")

    def _find_local_glm_ocr_model_directory(self) -> str | None:
        hub_cache_env = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
        if hub_cache_env:
            hub_cache = Path(hub_cache_env)
        else:
            hf_home_env = str(os.environ.get("HF_HOME", "") or "").strip()
            hf_home = Path(hf_home_env) if hf_home_env else Path.home() / ".cache" / "huggingface"
            hub_cache = hf_home / "hub"
        model_root = hub_cache / "models--zai-org--GLM-OCR"
        snapshot_dir = model_root / "snapshots"
        if not snapshot_dir.is_dir():
            return None
        try:
            snapshot_candidates = sorted(
                (path for path in snapshot_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
        except OSError:
            return None
        if not snapshot_candidates:
            return None
        return str(model_root)

    def _refresh_ffmpeg_status(self) -> None:
        self.ffmpeg_path = None
        self.ffprobe_path = None
        configured_dir = self.settings.value("ffmpeg/bin_dir")
        if isinstance(configured_dir, str) and configured_dir:
            ffmpeg_path = self._binary_from_directory(configured_dir, "ffmpeg")
            ffprobe_path = self._binary_from_directory(configured_dir, "ffprobe")
            if ffmpeg_path and ffprobe_path:
                self.ffmpeg_path = ffmpeg_path
                self.ffprobe_path = ffprobe_path
                self._set_ffmpeg_status("ok", f"Ready (manual): {configured_dir}")
                return

        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path and ffprobe_path:
            self.ffmpeg_path = ffmpeg_path
            self.ffprobe_path = ffprobe_path
            self._set_ffmpeg_status("ok", "Ready (PATH)")
            return
        self._set_ffmpeg_status(
            "warn",
            "Missing ffmpeg/ffprobe. Select directory or add to PATH.",
        )

    def _resolve_btbn_asset(self) -> tuple[str, str] | None:
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "linux":
            if machine in {"x86_64", "amd64"}:
                return ("linux64", "tar.xz")
            if machine in {"aarch64", "arm64"}:
                return ("linuxarm64", "tar.xz")
            return None
        if system == "windows":
            if machine in {"x86_64", "amd64"}:
                return ("win64", "zip")
            if machine in {"aarch64", "arm64"}:
                return ("winarm64", "zip")
            return None
        return None


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
    app.setOrganizationName("sub-manager")
    app.setApplicationName("sub-manager")
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    backend = AppBackend()
    engine = QQmlApplicationEngine()
    engine.setInitialProperties({"backend": backend})
    qml_path = Path(__file__).resolve().parent / "qml" / "Main.qml"
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
