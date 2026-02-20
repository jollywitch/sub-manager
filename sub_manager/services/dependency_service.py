from __future__ import annotations

from pathlib import Path
from types import ModuleType

from PySide6.QtCore import QSettings


class DependencyService:
    def __init__(
        self,
        settings: QSettings,
        os_module: ModuleType,
        shutil_module: ModuleType,
        platform_module: ModuleType,
    ) -> None:
        self.settings = settings
        self._os = os_module
        self._shutil = shutil_module
        self._platform = platform_module

    def get_saved_hf_token(self) -> str:
        return str(self.settings.value("hf/token", "") or "").strip()

    def apply_hf_token_environment_from_settings(self) -> None:
        saved_token = self.get_saved_hf_token()
        if saved_token:
            self._os.environ["HF_TOKEN"] = saved_token

    def current_hf_token(self) -> str | None:
        saved = self.get_saved_hf_token()
        if saved:
            return saved
        env_token = str(self._os.environ.get("HF_TOKEN", "") or "").strip()
        return env_token or None

    def hf_token_configured(self) -> bool:
        return bool(self.current_hf_token())

    def mask_token(self, token: str) -> str:
        clean_token = token.strip()
        if not clean_token:
            return ""
        if len(clean_token) <= 8:
            return "********"
        return f"{clean_token[:4]}********{clean_token[-4:]}"

    def refresh_hf_token_status(self) -> tuple[str, str, str]:
        saved_token = self.get_saved_hf_token()
        if saved_token:
            return ("ok", "Configured (saved).", self.mask_token(saved_token))
        env_token = str(self._os.environ.get("HF_TOKEN", "") or "").strip()
        if env_token:
            return ("ok", "Configured (environment).", self.mask_token(env_token))
        return ("warn", "Not configured (optional).", "")

    def find_local_glm_ocr_model_directory(self) -> str | None:
        hub_cache_env = str(self._os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
        if hub_cache_env:
            hub_cache = Path(hub_cache_env)
        else:
            hf_home_env = str(self._os.environ.get("HF_HOME", "") or "").strip()
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

    def refresh_glm_ocr_model_status(self) -> tuple[str, str]:
        model_directory = self.find_local_glm_ocr_model_directory()
        if model_directory is not None:
            return ("ok", f"Installed: {model_directory}")
        return ("warn", "Missing (download on first OCR run).")

    def binary_from_directory(self, directory: str, binary_name: str) -> str | None:
        file_name = f"{binary_name}.exe" if self._os.name == "nt" else binary_name
        path = self._os.path.join(directory, file_name)
        if self._os.path.isfile(path) and self._os.access(path, self._os.X_OK):
            return path
        return None

    def refresh_ffmpeg_status(self) -> tuple[str, str, str | None, str | None]:
        ffmpeg_path: str | None = None
        ffprobe_path: str | None = None
        configured_dir = self.settings.value("ffmpeg/bin_dir")
        if isinstance(configured_dir, str) and configured_dir:
            ffmpeg_candidate = self.binary_from_directory(configured_dir, "ffmpeg")
            ffprobe_candidate = self.binary_from_directory(configured_dir, "ffprobe")
            if ffmpeg_candidate and ffprobe_candidate:
                return ("ok", f"Ready (manual): {configured_dir}", ffmpeg_candidate, ffprobe_candidate)

        ffmpeg_path = self._shutil.which("ffmpeg")
        ffprobe_path = self._shutil.which("ffprobe")
        if ffmpeg_path and ffprobe_path:
            return ("ok", "Ready (PATH)", ffmpeg_path, ffprobe_path)
        return ("warn", "Missing ffmpeg/ffprobe. Select directory or add to PATH.", None, None)

    def resolve_btbn_asset(self) -> tuple[str, str] | None:
        system = self._platform.system().lower()
        machine = self._platform.machine().lower()
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
