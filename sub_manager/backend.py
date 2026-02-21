from __future__ import annotations

import logging
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

from PySide6.QtCore import QObject, Property, QSettings, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton

from sub_manager.constants import (
    FFMPEG_TOOLS_DIR,
    GLM_OCR_MODEL_ID,
    IMAGE_BASED_SUBTITLE_CODECS,
    TEXT_BASED_SUBTITLE_CODECS,
    VIDEO_FILTER,
)
from sub_manager.state_types import (
    AudioStreamItem,
    SubtitleCodecStreamItem,
    SubtitleStreamItem,
    VideoFileItem,
)
from sub_manager.services.dependency_service import DependencyService
from sub_manager.services.subtitle_service import SubtitleService
from sub_manager.ui.windows import (
    GlmDownloadProgressWindow,
    GlmDownloadSetupWindow,
    SubtitleEditorWindow,
)
from sub_manager.workers.download_workers import FFmpegDownloadWorker, GlmOcrModelDownloadWorker
from sub_manager.workers.ocr_worker import ImageSubtitleOcrWorker

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
    subtitleInfoRequested = Signal(str)
    ocrProgressRequested = Signal(str)
    ocrProgressHidden = Signal()
    subWindowStateChanged = Signal()
    focusSubWindowRequested = Signal(str)

    def __init__(
        self,
        settings_factory: Callable[[str, str], QSettings] | None = None,
        os_module: ModuleType | None = None,
        shutil_module: ModuleType | None = None,
        platform_module: ModuleType | None = None,
    ) -> None:
        super().__init__()
        import os as _os
        import platform as _platform
        import shutil as _shutil

        if os_module is None:
            os_module = _os
        if shutil_module is None:
            shutil_module = _shutil
        if platform_module is None:
            platform_module = _platform
        self._os = os_module
        self._shutil = shutil_module
        self._platform = platform_module
        factory = settings_factory or QSettings
        self.settings = factory("sub-manager", "sub-manager")
        self._dependency_service = DependencyService(
            settings=self.settings,
            os_module=self._os,
            shutil_module=self._shutil,
            platform_module=self._platform,
        )
        self._subtitle_service = SubtitleService(
            os_module=self._os,
            shutil_module=self._shutil,
        )
        self._selected_paths: set[str] = set()
        self._video_files: list[VideoFileItem] = []
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
        self._ocr_thread: QThread | None = None
        self._ocr_worker: ImageSubtitleOcrWorker | None = None
        self._ocr_target_key: str | None = None
        self._ocr_target_file_path: str | None = None
        self._ocr_target_stream_index: int | None = None
        self._ocr_target_stream_lang: str | None = None
        self._ocr_target_codec_name: str | None = None
        self._ocr_target_cache_key: tuple[str, int, int, int] | None = None
        self._ocr_progress_dismissed = False
        self._ocr_cancel_requested = False
        self._ocr_srt_cache: dict[tuple[str, int, int, int], str] = {}
        self._active_subwindow_id: str = ""
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

    def get_video_files(self) -> list[VideoFileItem]:
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
    activeSubWindowId = Property(str, lambda self: self._active_subwindow_id, notify=subWindowStateChanged)
    isSubWindowActive = Property(bool, lambda self: bool(self._active_subwindow_id), notify=subWindowStateChanged)

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
        if not self._try_activate_subwindow("add_videos"):
            return
        if self._add_videos_dialog is not None and self._add_videos_dialog.isVisible():
            self._add_videos_dialog.raise_()
            self._add_videos_dialog.activateWindow()
            return

        dialog = QFileDialog()
        dialog.setWindowTitle("Select Video Files")
        dialog.setNameFilter(VIDEO_FILTER)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setModal(True)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        dialog.filesSelected.connect(self._on_add_videos_selected)
        dialog.finished.connect(self._cleanup_add_videos_dialog)
        self._add_videos_dialog = dialog
        dialog.open()

    @Slot("QStringList")
    def _on_add_videos_selected(self, files: list[str]) -> None:
        for path in files:
            absolute_path = self._os.path.abspath(path)
            if absolute_path in self._selected_paths:
                continue
            self._selected_paths.add(absolute_path)
            audio_languages, subtitle_languages = self._inspect_stream_languages(absolute_path)
            self._video_files.append(
                {
                    "name": self._os.path.basename(absolute_path),
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
        self._release_subwindow("add_videos")

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

    def _set_active_subwindow(self, window_id: str) -> None:
        normalized = window_id.strip()
        if self._active_subwindow_id != normalized:
            self._active_subwindow_id = normalized
            self.subWindowStateChanged.emit()

    def _try_activate_subwindow(self, window_id: str) -> bool:
        normalized = window_id.strip()
        if not normalized:
            return True
        active = self._active_subwindow_id
        if active and active != normalized:
            self._focus_subwindow(active)
            return False
        self._set_active_subwindow(normalized)
        return True

    def _release_subwindow(self, window_id: str) -> None:
        if self._active_subwindow_id == window_id:
            self._set_active_subwindow("")

    def _focus_subwindow(self, window_id: str) -> None:
        if window_id == "subtitle_editor" and self._subtitle_editor is not None and self._subtitle_editor.isVisible():
            self._subtitle_editor.raise_()
            self._subtitle_editor.activateWindow()
            return
        if window_id == "glm_setup" and self._glm_setup_window is not None and self._glm_setup_window.isVisible():
            self._glm_setup_window.raise_()
            self._glm_setup_window.activateWindow()
            return
        if (
            window_id == "glm_download_progress"
            and self._glm_download_progress_window is not None
            and self._glm_download_progress_window.isVisible()
        ):
            self._glm_download_progress_window.raise_()
            self._glm_download_progress_window.activateWindow()
            return
        if window_id == "add_videos" and self._add_videos_dialog is not None and self._add_videos_dialog.isVisible():
            self._add_videos_dialog.raise_()
            self._add_videos_dialog.activateWindow()
            return
        if (
            window_id == "ffmpeg_override"
            and self._ffmpeg_override_box is not None
            and self._ffmpeg_override_box.isVisible()
        ):
            self._ffmpeg_override_box.raise_()
            self._ffmpeg_override_box.activateWindow()
            return
        if window_id == "message_box" and self._active_message_box is not None and self._active_message_box.isVisible():
            self._active_message_box.raise_()
            self._active_message_box.activateWindow()
            return
        self.focusSubWindowRequested.emit(window_id)

    @Slot()
    def focusActiveSubWindow(self) -> None:
        if self._active_subwindow_id:
            self._focus_subwindow(self._active_subwindow_id)

    @Slot(str)
    def notifySubWindowOpened(self, window_id: str) -> None:
        self._set_active_subwindow(window_id.strip())

    @Slot(str)
    def notifySubWindowClosed(self, window_id: str) -> None:
        self._release_subwindow(window_id.strip())

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
        enable_xet = bool(self.settings.value("glm/enable_xet", False, type=bool))
        self._glm_download_thread = QThread(self)
        self._glm_download_worker = GlmOcrModelDownloadWorker(
            GLM_OCR_MODEL_ID,
            hf_token=hf_token,
            enable_xet=enable_xet,
        )
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
        return self._dependency_service.hf_token_configured()

    def _show_glm_download_setup(
        self,
        continue_action: Callable[[], None],
    ) -> None:
        if not self._try_activate_subwindow("glm_setup"):
            return
        if self._glm_setup_window is not None and self._glm_setup_window.isVisible():
            self._glm_download_followup_action = continue_action
            self._glm_setup_window.raise_()
            self._glm_setup_window.activateWindow()
            return
        self._glm_download_followup_action = continue_action
        window = GlmDownloadSetupWindow(
            on_download=self._on_glm_setup_download_clicked,
            on_cancel=self._on_glm_setup_cancelled,
            enable_xet=bool(self.settings.value("glm/enable_xet", False, type=bool)),
        )
        existing_token = self._get_saved_hf_token()
        if existing_token:
            window.token_edit.setText(existing_token)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        window.setWindowModality(Qt.ApplicationModal)
        window.destroyed.connect(self._on_glm_setup_window_destroyed)
        self._glm_setup_window = window
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_glm_setup_download_clicked(self, token: str, enable_xet: bool) -> None:
        if token:
            self.setHfToken(token)
        self.settings.setValue("glm/enable_xet", bool(enable_xet))
        followup = self._glm_download_followup_action
        self._glm_download_followup_action = None
        # Always surface immediate feedback when user clicks Download Now.
        self._glm_download_progress_dismissed = False
        self._show_glm_download_progress("Preparing GLM-OCR download...")
        self._start_glm_ocr_model_download(after_download_action=followup)

    def _current_hf_token(self) -> str | None:
        return self._dependency_service.current_hf_token()

    def _on_glm_setup_cancelled(self) -> None:
        self._glm_download_followup_action = None

    @Slot()
    def _on_glm_setup_window_destroyed(self) -> None:
        self._glm_setup_window = None
        self._release_subwindow("glm_setup")

    def _show_glm_download_progress(self, text: str) -> None:
        if not self._try_activate_subwindow("glm_download_progress"):
            return
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
        window.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        window.setWindowModality(Qt.ApplicationModal)
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
        self._stop_thread(self._glm_download_thread)
        self._close_glm_download_progress_box()
        self._refresh_glm_ocr_model_status()
        self._show_info("GLM-OCR Download", "GLM-OCR download was cancelled.")

    def _close_glm_download_progress_box(self) -> None:
        if self._glm_download_progress_window is not None:
            self._glm_download_progress_window.close()

    @Slot()
    def _on_glm_download_progress_window_destroyed(self) -> None:
        self._glm_download_progress_window = None
        self._release_subwindow("glm_download_progress")
        if self._glm_download_thread is not None and self._glm_download_thread.isRunning():
            self._glm_download_progress_dismissed = True

    def _on_glm_download_thread_finished(self) -> None:
        self._cleanup_worker_thread("_glm_download_thread", "_glm_download_worker")
        self._glm_download_progress_dismissed = False
        self._glm_download_cancel_requested = False
        self._glm_download_started_at = None
        self._glm_download_followup_action = None

    @Slot(str)
    def setHfToken(self, token: str) -> None:
        clean_token = token.strip()
        if clean_token:
            self.settings.setValue("hf/token", clean_token)
            self._os.environ["HF_TOKEN"] = clean_token
        else:
            self.settings.remove("hf/token")
            self._os.environ.pop("HF_TOKEN", None)
        self._refresh_hf_token_status()

    @Slot()
    def clearHfToken(self) -> None:
        self.settings.remove("hf/token")
        self._os.environ.pop("HF_TOKEN", None)
        self._refresh_hf_token_status()

    def _inspect_stream_languages(
        self, file_path: str
    ) -> tuple[list[AudioStreamItem], list[SubtitleStreamItem | SubtitleCodecStreamItem]]:
        return self._subtitle_service.inspect_stream_languages(file_path, self.ffprobe_path)

    def _run_ffprobe_json(self, file_path: str) -> dict[str, object] | None:
        return self._subtitle_service.run_ffprobe_json(file_path, self.ffprobe_path)

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
        editor_key = f"{self._os.path.abspath(file_path)}::{stream_index}"
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
        editor.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        editor.setWindowModality(Qt.ApplicationModal)
        editor.destroyed.connect(self._on_subtitle_editor_destroyed)
        self._subtitle_editor = editor
        self._subtitle_editor_key = editor_key
        self._set_active_subwindow("subtitle_editor")
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
        absolute = self._os.path.abspath(file_path)
        try:
            stat = self._os.stat(absolute)
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
        self._ocr_progress_dismissed = False
        self._ocr_cancel_requested = False

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
        if not self._try_activate_subwindow("ocr_progress"):
            return
        if self._ocr_progress_dismissed:
            return
        self.ocrProgressRequested.emit(text)

    @Slot(str)
    def _on_ocr_progress(self, message: str) -> None:
        if self._ocr_cancel_requested:
            return
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
            return
        self._show_warning(
            "Subtitle OCR Failed",
            f"Could not convert image subtitle stream to SRT with GLM-OCR.\n\n{error_message}",
        )

    @Slot()
    def cancelOcrOperation(self) -> None:
        self._ocr_cancel_requested = True
        self._cancel_ocr_operation("Cancelling OCR...")

    @Slot()
    def dismissOcrProgressWindow(self) -> None:
        self._ocr_progress_dismissed = True
        self._release_subwindow("ocr_progress")
        self.ocrProgressHidden.emit()

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
        self._release_subwindow("ocr_progress")
        self.ocrProgressHidden.emit()

    def _on_ocr_thread_finished(self) -> None:
        self._cleanup_worker_thread("_ocr_thread", "_ocr_worker")
        self._ocr_target_key = None
        self._ocr_target_file_path = None
        self._ocr_target_stream_index = None
        self._ocr_target_stream_lang = None
        self._ocr_target_codec_name = None
        self._ocr_target_cache_key = None
        self._ocr_progress_dismissed = False
        self._ocr_cancel_requested = False

    @Slot()
    def _on_subtitle_editor_destroyed(self) -> None:
        self._subtitle_editor = None
        self._subtitle_editor_key = None
        self._release_subwindow("subtitle_editor")

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
        details_text = (
            f"File: {self._os.path.basename(file_path)}\n\n"
            f"Stream #{stream_index}\n"
            f"  codec: {codec_name}\n"
            f"  language: {stream_lang}\n"
            f"  title: {stream_title}\n"
            f"  default: {default_flag}, forced: {forced_flag}"
        )
        if not self._try_activate_subwindow("subtitle_info"):
            return
        self.subtitleInfoRequested.emit(details_text)

    def _find_subtitle_stream_by_index(
        self, streams: list[object], stream_index: int
    ) -> dict[str, object] | None:
        return self._subtitle_service.find_subtitle_stream_by_index(streams, stream_index)

    def _stream_language(self, stream: dict[str, object]) -> str:
        return self._subtitle_service.stream_language(stream)

    def _stream_index(self, stream: dict[str, object], fallback: int) -> int:
        return self._subtitle_service.stream_index(stream, fallback)

    def _extract_subtitle_text(self, file_path: str, stream_index: int) -> str | None:
        return self._subtitle_service.extract_subtitle_text(
            file_path=file_path,
            stream_index=stream_index,
            ffmpeg_path=self.ffmpeg_path,
        )

    def _save_embedded_subtitle(
        self, file_path: str, stream_index: int, subtitle_text: str
    ) -> tuple[bool, str]:
        ok, message = self._subtitle_service.save_embedded_subtitle(
            file_path=file_path,
            stream_index=stream_index,
            subtitle_text=subtitle_text,
            ffmpeg_path=self.ffmpeg_path,
            ffprobe_path=self.ffprobe_path,
        )
        if ok:
            self._invalidate_ocr_cache_for_file(file_path)
        return ok, message

    def _output_subtitle_codec_for_container(self, file_path: str) -> str | None:
        return self._subtitle_service.output_subtitle_codec_for_container(file_path)

    def _invalidate_ocr_cache_for_file(self, file_path: str) -> None:
        absolute = self._os.path.abspath(file_path)
        stale_keys = [key for key in self._ocr_srt_cache if key[0] == absolute]
        for key in stale_keys:
            self._ocr_srt_cache.pop(key, None)

    def _stream_title(self, stream: dict[str, object]) -> str:
        return self._subtitle_service.stream_title(stream)

    @Slot()
    def selectFfmpegDirectory(self) -> None:
        if not self._try_activate_subwindow("ffmpeg_directory_select"):
            return
        initial_dir = self.settings.value("ffmpeg/bin_dir")
        start_dir = (
            initial_dir
            if isinstance(initial_dir, str) and initial_dir
            else self._os.path.expanduser("~")
        )
        selected_dir = QFileDialog.getExistingDirectory(None, "Select FFmpeg Directory", start_dir)
        self._release_subwindow("ffmpeg_directory_select")
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
        tools_dir = FFMPEG_TOOLS_DIR
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
        if not self._try_activate_subwindow("ffmpeg_override"):
            return
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
        box.setModal(True)
        box.setWindowModality(Qt.ApplicationModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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
        self._release_subwindow("ffmpeg_override")

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
        self._cleanup_worker_thread("download_thread", "download_worker")
        # Final refresh after thread teardown so "Installing..." can transition
        # to Ready/Missing without requiring the Dependencies window to reopen.
        self._refresh_ffmpeg_status()

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
        if self._active_subwindow_id:
            self._focus_subwindow(self._active_subwindow_id)
            return False
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
        return self._wait_for_thread(self._ocr_thread, timeout_ms)

    def _force_stop_ocr_thread(self) -> None:
        self._stop_thread(self._ocr_thread, wait_ms=1, force_wait_ms=1000)

    def _wait_for_thread(self, thread: QThread | None, timeout_ms: int) -> bool:
        if thread is None:
            return True
        return thread.wait(timeout_ms)

    def _stop_thread(self, thread: QThread | None, wait_ms: int = 1500, force_wait_ms: int = 1000) -> bool:
        if thread is None:
            return True
        if not thread.isRunning():
            return True
        thread.requestInterruption()
        thread.quit()
        if thread.wait(wait_ms):
            return True
        thread.terminate()
        return thread.wait(force_wait_ms)

    def _cleanup_worker_thread(self, thread_attr: str, worker_attr: str) -> None:
        worker = getattr(self, worker_attr)
        if worker is not None:
            worker.deleteLater()
            setattr(self, worker_attr, None)
        thread = getattr(self, thread_attr)
        if thread is not None:
            thread.deleteLater()
            setattr(self, thread_attr, None)

    def clear_session_ocr_state(self) -> None:
        self._ocr_srt_cache.clear()
        ImageSubtitleOcrWorker.clear_model_cache()

    def _show_message(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
        if not self._try_activate_subwindow("message_box"):
            return
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
        box.setModal(True)
        box.setWindowModality(Qt.ApplicationModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._active_message_box = box
        box.finished.connect(self._cleanup_message_box)
        box.show()

    @Slot(int)
    def _cleanup_message_box(self, _result: int) -> None:
        self._active_message_box = None
        self._release_subwindow("message_box")

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
        return self._dependency_service.get_saved_hf_token()

    def _apply_hf_token_environment_from_settings(self) -> None:
        self._dependency_service.apply_hf_token_environment_from_settings()

    def _binary_from_directory(self, directory: str, binary_name: str) -> str | None:
        return self._dependency_service.binary_from_directory(directory, binary_name)

    def _refresh_dependency_statuses(self) -> None:
        self._refresh_ffmpeg_status()
        self._refresh_hf_token_status()
        self._refresh_glm_ocr_model_status()

    def _refresh_hf_token_status(self) -> None:
        level, text, preview = self._dependency_service.refresh_hf_token_status()
        self._set_hf_token_status(level, text)
        self._set_hf_token_preview(preview)

    def _mask_token(self, token: str) -> str:
        return self._dependency_service.mask_token(token)

    def _refresh_glm_ocr_model_status(self) -> None:
        level, text = self._dependency_service.refresh_glm_ocr_model_status()
        self._set_glm_ocr_model_status(level, text)

    def _find_local_glm_ocr_model_directory(self) -> str | None:
        return self._dependency_service.find_local_glm_ocr_model_directory()

    def _refresh_ffmpeg_status(self) -> None:
        # Preserve active FFmpeg download status text; do not overwrite it with
        # PATH/manual checks while the worker thread is running.
        if self._downloading or (self.download_thread is not None and self.download_thread.isRunning()):
            return
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

        ffmpeg_path = self._shutil.which("ffmpeg")
        ffprobe_path = self._shutil.which("ffprobe")
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
        return self._dependency_service.resolve_btbn_asset()
