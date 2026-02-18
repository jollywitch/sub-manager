from __future__ import annotations

from pathlib import Path

import pytest

import main


class FakeSettings:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def value(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)

    def setValue(self, key: str, value: object) -> None:
        self._data[key] = value

    def remove(self, key: str) -> None:
        self._data.pop(key, None)


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> FakeSettings:
    settings = FakeSettings()
    monkeypatch.setattr(main, "QSettings", lambda *_args, **_kwargs: settings)
    return settings


def _new_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> main.AppBackend:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(main.shutil, "which", lambda _name: None)
    return main.AppBackend()


def test_hf_token_status_warn_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    assert backend.hfTokenStatusLevel == "warn"
    assert backend.hfTokenStatus == "Not configured (optional)."


def test_hf_token_status_configured_when_saved(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    backend.setHfToken("hf_abc123")

    assert backend.hfTokenStatusLevel == "ok"
    assert backend.hfTokenStatus == "Configured (saved)."
    assert backend.hfTokenPreview.startswith("hf_a")
    assert backend.hfTokenPreview.endswith("c123")
    assert "*" in backend.hfTokenPreview


def test_set_hf_token_persists_and_sets_env(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    backend.setHfToken("hf_secret")

    assert fake_settings.value("hf/token") == "hf_secret"
    assert "HF_TOKEN" in main.os.environ
    assert main.os.environ["HF_TOKEN"] == "hf_secret"


def test_clear_hf_token_removes_setting_and_env(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    backend.setHfToken("hf_secret")
    backend.clearHfToken()

    assert fake_settings.value("hf/token") is None
    assert "HF_TOKEN" not in main.os.environ
    assert backend.hfTokenStatusLevel == "warn"


def test_glm_model_status_ok_when_snapshot_exists(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "hub" / "models--zai-org--GLM-OCR" / "snapshots" / "abc123"
    snapshots.mkdir(parents=True)
    backend = _new_backend(monkeypatch, tmp_path)

    backend.refreshDependencyStatuses()

    assert backend.glmOcrModelStatusLevel == "ok"
    assert backend.glmOcrModelStatus == f"Installed: {snapshots.parent.parent}"


def test_glm_model_status_warn_when_snapshot_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    assert backend.glmOcrModelStatusLevel == "warn"
    assert backend.glmOcrModelStatus == "Missing (download on first OCR run)."


def test_ffmpeg_ffprobe_pair_status_still_requires_both(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    def _which(binary: str) -> str | None:
        if binary == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.setattr(main.shutil, "which", _which)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    backend = main.AppBackend()

    assert backend.ffmpegStatusLevel == "warn"
    assert backend.ffmpegStatus == "Missing ffmpeg/ffprobe. Select directory or add to PATH."


def test_search_ffmpeg_in_path_clears_manual_directory_and_uses_path(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    fake_settings.setValue("ffmpeg/bin_dir", "/custom/ffmpeg")

    def _which(binary: str) -> str | None:
        if binary == "ffmpeg":
            return "/usr/bin/ffmpeg"
        if binary == "ffprobe":
            return "/usr/bin/ffprobe"
        return None

    monkeypatch.setattr(main.shutil, "which", _which)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    backend = main.AppBackend()

    backend.searchFfmpegInPath()

    assert fake_settings.value("ffmpeg/bin_dir") is None
    assert backend.ffmpegStatusLevel == "ok"
    assert backend.ffmpegStatus == "Ready (PATH)"


def test_search_ffmpeg_in_path_restores_previous_manual_directory_when_path_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    fake_settings.setValue("ffmpeg/bin_dir", "/custom/ffmpeg")

    def _binary_from_directory(directory: str, binary_name: str) -> str | None:
        if directory == "/custom/ffmpeg":
            return f"/custom/ffmpeg/{binary_name}"
        return None

    monkeypatch.setattr(backend, "_binary_from_directory", _binary_from_directory)
    monkeypatch.setattr(main.shutil, "which", lambda _binary: None)
    backend.refreshDependencyStatuses()
    assert backend.ffmpegStatus == "Ready (manual): /custom/ffmpeg"

    backend.searchFfmpegInPath()

    assert fake_settings.value("ffmpeg/bin_dir") == "/custom/ffmpeg"
    assert backend.ffmpegStatusLevel == "ok"
    assert backend.ffmpegStatus == "Ready (manual): /custom/ffmpeg"


def test_download_ffmpeg_asks_confirmation_before_overwrite_when_tools_binaries_exist(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_resolve_btbn_asset", lambda: ("linux64", "tar.xz"))
    monkeypatch.setattr(backend, "_tools_ffmpeg_binaries_exist", lambda _tools_dir: True)

    confirmations: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        backend,
        "_confirm_ffmpeg_override",
        lambda url, archive_ext, tools_dir: confirmations.append((url, archive_ext, tools_dir)),
    )

    backend.downloadFfmpeg()

    assert len(confirmations) == 1


def test_download_ffmpeg_starts_immediately_when_no_existing_tools_binaries(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_resolve_btbn_asset", lambda: ("linux64", "tar.xz"))
    monkeypatch.setattr(backend, "_tools_ffmpeg_binaries_exist", lambda _tools_dir: False)

    started: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        backend,
        "_start_ffmpeg_download",
        lambda url, archive_ext, tools_dir: started.append((url, archive_ext, tools_dir)),
    )

    backend.downloadFfmpeg()

    assert len(started) == 1
    assert started[0][1] == "tar.xz"


def test_download_ffmpeg_override_confirmation_yes_starts_download(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    started: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        backend,
        "_start_ffmpeg_download",
        lambda url, archive_ext, tools_dir: started.append((url, archive_ext, tools_dir)),
    )

    yes_button = object()
    backend._ffmpeg_override_download_button = yes_button  # type: ignore[assignment]
    backend._on_ffmpeg_override_button_clicked(
        yes_button,
        "https://example.test/ffmpeg.tar.xz",
        "tar.xz",
        "/tmp/tools/ffmpeg",
    )

    assert len(started) == 1
    assert started[0][2] == Path("/tmp/tools/ffmpeg")


def test_download_ffmpeg_override_confirmation_no_uses_existing_tools_binaries(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    started: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        backend,
        "_start_ffmpeg_download",
        lambda url, archive_ext, tools_dir: started.append((url, archive_ext, tools_dir)),
    )

    expected_bin_dir = str(tmp_path / "tools" / "ffmpeg" / "bin")

    def _binary_from_directory(directory: str, binary_name: str) -> str | None:
        if directory == expected_bin_dir:
            return f"{expected_bin_dir}/{binary_name}"
        return None

    monkeypatch.setattr(backend, "_binary_from_directory", _binary_from_directory)
    monkeypatch.setattr(main.shutil, "which", lambda _binary: None)

    backend._ffmpeg_override_download_button = object()  # type: ignore[assignment]
    backend._on_ffmpeg_override_button_clicked(
        object(),
        "https://example.test/ffmpeg.tar.xz",
        "tar.xz",
        str(tmp_path / "tools" / "ffmpeg"),
    )

    assert started == []
    assert fake_settings.value("ffmpeg/bin_dir") == expected_bin_dir
    assert backend.ffmpegStatus == f"Ready (manual): {expected_bin_dir}"


def test_download_ffmpeg_override_close_uses_existing_tools_binaries(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    started: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        backend,
        "_start_ffmpeg_download",
        lambda url, archive_ext, tools_dir: started.append((url, archive_ext, tools_dir)),
    )

    expected_bin_dir = str(tmp_path / "tools" / "ffmpeg" / "bin")

    def _binary_from_directory(directory: str, binary_name: str) -> str | None:
        if directory == expected_bin_dir:
            return f"{expected_bin_dir}/{binary_name}"
        return None

    monkeypatch.setattr(backend, "_binary_from_directory", _binary_from_directory)
    monkeypatch.setattr(main.shutil, "which", lambda _binary: None)

    backend._on_ffmpeg_override_finished(
        "https://example.test/ffmpeg.tar.xz",
        "tar.xz",
        str(tmp_path / "tools" / "ffmpeg"),
    )

    assert started == []
    assert fake_settings.value("ffmpeg/bin_dir") == expected_bin_dir
    assert backend.ffmpegStatus == f"Ready (manual): {expected_bin_dir}"


def test_add_videos_shows_error_when_ffmpeg_not_configured(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(backend, "_show_critical", lambda title, text: shown.append((title, text)))

    backend.addVideos()

    assert len(shown) == 1
    assert shown[0][0] == "Missing Dependencies"
    assert "ffmpeg/ffprobe is not configured" in shown[0][1]
    assert backend._add_videos_dialog is None


def test_image_ocr_missing_glm_model_opens_setup_window(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_find_local_glm_ocr_model_directory", lambda: None)
    events: list[str] = []
    monkeypatch.setattr(
        backend,
        "_show_glm_download_setup",
        lambda continue_action: events.append("setup_opened"),
    )
    monkeypatch.setattr(backend, "_start_image_subtitle_ocr", lambda **_kwargs: events.append("ocr_started"))

    backend._start_image_subtitle_ocr_with_download_if_needed(
        file_path="/tmp/video.mkv",
        stream_index=2,
        stream_lang="eng",
        codec_name="hdmv_pgs_subtitle",
        editor_key="video::2",
    )

    assert events == ["setup_opened"]


def test_glm_setup_download_saves_token_and_starts_download(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    backend._glm_download_followup_action = lambda: None
    started: list[object] = []
    monkeypatch.setattr(backend, "_show_glm_download_progress", lambda _text: None)
    monkeypatch.setattr(
        backend,
        "_start_glm_ocr_model_download",
        lambda after_download_action=None: started.append(after_download_action),
    )

    backend._on_glm_setup_download_clicked("hf_test_token")

    assert fake_settings.value("hf/token") == "hf_test_token"
    assert len(started) == 1


def test_glm_setup_cancel_clears_followup_action(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    backend._glm_download_followup_action = lambda: None

    backend._on_glm_setup_cancelled()

    assert backend._glm_download_followup_action is None


def test_start_glm_download_when_already_running_shows_progress_again(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    class FakeThread:
        def isRunning(self) -> bool:
            return True

    backend._glm_download_thread = FakeThread()  # type: ignore[assignment]
    events: list[str] = []
    backend._glm_download_progress_dismissed = True
    monkeypatch.setattr(backend, "_show_glm_download_progress", lambda text: events.append(text))

    backend._start_glm_ocr_model_download()

    assert backend._glm_download_progress_dismissed is False
    assert events == ["GLM-OCR download is already running..."]


def test_cancel_glm_download_stops_thread_and_notifies_user(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)

    class FakeThread:
        def __init__(self) -> None:
            self.interrupted = False
            self.quit_called = False
            self.terminated = False

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def quit(self) -> None:
            self.quit_called = True

        def wait(self, _timeout_ms: int) -> bool:
            return True

        def terminate(self) -> None:
            self.terminated = True

    class FakeWorker:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    fake_thread = FakeThread()
    fake_worker = FakeWorker()
    backend._glm_download_thread = fake_thread  # type: ignore[assignment]
    backend._glm_download_worker = fake_worker  # type: ignore[assignment]
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(backend, "_close_glm_download_progress_box", lambda: None)
    monkeypatch.setattr(backend, "_refresh_glm_ocr_model_status", lambda: None)
    monkeypatch.setattr(backend, "_show_info", lambda title, text: infos.append((title, text)))

    backend._cancel_glm_download("Cancelling GLM-OCR download...")

    assert fake_worker.cancel_called is True
    assert fake_thread.interrupted is True
    assert fake_thread.quit_called is True
    assert infos == [("GLM-OCR Download", "GLM-OCR download was cancelled.")]


def test_image_ocr_setup_followup_starts_ocr_after_glm_download(
    monkeypatch: pytest.MonkeyPatch,
    fake_settings: FakeSettings,
    tmp_path: Path,
) -> None:
    backend = _new_backend(monkeypatch, tmp_path)
    events: list[str] = []
    followup = lambda: events.append("ocr_started")
    backend._glm_download_followup_action = followup
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "_show_glm_download_progress", lambda _text: None)
    monkeypatch.setattr(
        backend,
        "_start_glm_ocr_model_download",
        lambda after_download_action=None: captured.setdefault("action", after_download_action),
    )

    backend._on_glm_setup_download_clicked("")

    assert events == []
    assert captured.get("action") is followup
    # Simulate download complete and ensure followup action executes.
    monkeypatch.setattr(backend, "_close_glm_download_progress_box", lambda: None)
    monkeypatch.setattr(backend, "_refresh_glm_ocr_model_status", lambda: None)
    monkeypatch.setattr(backend, "_show_info", lambda *_args: None)
    backend._glm_download_cancel_requested = False
    backend._glm_download_followup_action = followup
    backend._on_glm_download_finished("/tmp/model")

    assert events == ["ocr_started"]


def test_glm_download_worker_uses_xet_high_performance_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    worker = main.GlmOcrModelDownloadWorker("zai-org/GLM-OCR", hf_token="hf_test")

    env, xet_enabled = worker._build_download_env()

    assert xet_enabled is True
    assert env["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert env["HF_TOKEN"] == "hf_test"
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in env


def test_glm_download_worker_respects_explicit_xet_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    worker = main.GlmOcrModelDownloadWorker("zai-org/GLM-OCR", hf_token=None)

    env, xet_enabled = worker._build_download_env()

    assert xet_enabled is False
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
