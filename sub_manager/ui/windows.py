from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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
        box.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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
        dialog.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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
        self,
        on_download: Callable[[str, bool], None],
        on_cancel: Callable[[], None],
        enable_xet: bool = False,
    ) -> None:
        super().__init__(None, Qt.Window)
        self._on_download = on_download
        self._on_cancel = on_cancel
        self._download_started = False
        self.setWindowTitle("GLM-OCR Setup")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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

        self.xet_checkbox = QCheckBox("Enable Xet (requires administrator)")
        self.xet_checkbox.setChecked(enable_xet)
        root_layout.addWidget(self.xet_checkbox)

        xet_hint = QLabel("If not elevated on Windows, download will automatically fall back to standard transfer.")
        xet_hint.setWordWrap(True)
        root_layout.addWidget(xet_hint)

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
        enable_xet = self.xet_checkbox.isChecked()
        self._download_started = True
        self._on_download(token, enable_xet)
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
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
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
