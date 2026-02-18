from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Property, QSettings, QThread, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
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


class AppBackend(QObject):
    videoFilesChanged = Signal()
    ffmpegStatusChanged = Signal()
    ffmpegStatusLevelChanged = Signal()
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
        self._add_videos_dialog: QFileDialog | None = None
        self._active_message_box: QMessageBox | None = None
        self._subtitle_editor: SubtitleEditorWindow | None = None
        self._subtitle_editor_key: str | None = None
        self._ffmpeg_status = ""
        self._ffmpeg_status_level = "warn"
        self._downloading = False

        self._window_x = int(self.settings.value("window/x", 100))
        self._window_y = int(self.settings.value("window/y", 100))
        self._window_w = int(self.settings.value("window/w", 900))
        self._window_h = int(self.settings.value("window/h", 560))

        self._refresh_ffmpeg_status()

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

    def get_downloading(self) -> bool:
        return self._downloading

    isDownloading = Property(bool, get_downloading, notify=downloadingChanged)

    windowX = Property(int, lambda self: self._window_x, notify=windowGeometryChanged)
    windowY = Property(int, lambda self: self._window_y, notify=windowGeometryChanged)
    windowW = Property(int, lambda self: self._window_w, notify=windowGeometryChanged)
    windowH = Property(int, lambda self: self._window_h, notify=windowGeometryChanged)

    @Slot()
    def addVideos(self) -> None:
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
        if codec_name not in TEXT_BASED_SUBTITLE_CODECS:
            self._show_warning(
                "Subtitle Editor",
                f"Stream #{stream_index} uses '{codec_name}', which is not a text-based subtitle codec.",
            )
            return

        subtitle_text = self._extract_subtitle_text(file_path, stream_index)
        if subtitle_text is None:
            self._show_warning(
                "Subtitle Editor",
                "Could not extract subtitle text with ffmpeg.",
            )
            return

        editor_key = f"{os.path.abspath(file_path)}::{stream_index}"
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

        stream_lang = self._stream_language(stream)
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
        return True

    def _show_message(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
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

    def _set_ffmpeg_status(self, level: str, text: str) -> None:
        if self._ffmpeg_status != text:
            self._ffmpeg_status = text
            self.ffmpegStatusChanged.emit()
        if self._ffmpeg_status_level != level:
            self._ffmpeg_status_level = level
            self.ffmpegStatusLevelChanged.emit()

    def _binary_from_directory(self, directory: str, binary_name: str) -> str | None:
        file_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
        path = os.path.join(directory, file_name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

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


def main() -> int:
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
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
