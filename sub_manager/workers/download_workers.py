from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

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
            self._prepare_download_directories(temp_dir)
            self._download_archive(archive_path)
            extract_dir = self._extract_archive(temp_dir, archive_path)
            ffmpeg_path, ffprobe_path = self._find_extracted_binaries(extract_dir)
            if ffmpeg_path is None or ffprobe_path is None:
                raise RuntimeError("Downloaded archive does not contain ffmpeg and ffprobe.")
            self._install_binaries(install_bin_dir, ffmpeg_path, ffprobe_path)
            self.finished.emit(str(install_bin_dir))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _prepare_download_directories(self, temp_dir: Path) -> None:
        self.progress.emit("Preparing download...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def _download_archive(self, archive_path: Path) -> None:
        self.progress.emit("Downloading FFmpeg archive...")
        with urllib.request.urlopen(self.download_url) as response:
            archive_path.write_bytes(response.read())

    def _extract_archive(self, temp_dir: Path, archive_path: Path) -> Path:
        self.progress.emit("Extracting archive...")
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_ext == "zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            return extract_dir
        with tarfile.open(archive_path, "r:xz") as tar_ref:
            try:
                tar_ref.extractall(extract_dir, filter="data")
            except TypeError:
                tar_ref.extractall(extract_dir)
        return extract_dir

    def _install_binaries(self, install_bin_dir: Path, ffmpeg_path: Path, ffprobe_path: Path) -> None:
        self.progress.emit("Installing ffmpeg/ffprobe...")
        install_bin_dir.mkdir(parents=True, exist_ok=True)
        target_ffmpeg = install_bin_dir / ffmpeg_path.name
        target_ffprobe = install_bin_dir / ffprobe_path.name
        shutil.copy2(ffmpeg_path, target_ffmpeg)
        shutil.copy2(ffprobe_path, target_ffprobe)
        if os.name != "nt":
            os.chmod(target_ffmpeg, 0o755)
            os.chmod(target_ffprobe, 0o755)

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

    def _snapshot_download_script(self) -> str:
        return """
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

    def _start_snapshot_subprocess(
        self, env: dict[str, str]
    ) -> tuple[queue.Queue[dict[str, object]], queue.Queue[str], threading.Thread, threading.Thread]:
        self._process = subprocess.Popen(
            [sys.executable, "-c", self._snapshot_download_script()],
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
        return event_queue, stderr_queue, stdout_thread, stderr_thread

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
            event_queue, stderr_queue, stdout_thread, stderr_thread = self._start_snapshot_subprocess(env)
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
