from __future__ import annotations

import json
import ctypes
import importlib
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

from sub_manager.constants import APP_DATA_DIR
from sub_manager.process_utils import windows_hidden_subprocess_kwargs

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

    def __init__(self, model_id: str, hf_token: str | None = None, enable_xet: bool = False) -> None:
        super().__init__()
        self.model_id = model_id
        self.hf_token = (hf_token or "").strip() or None
        self.enable_xet = bool(enable_xet)
        self._cancel_requested = False
        self._process: subprocess.Popen[str] | None = None
        self._runtime_site_packages = APP_DATA_DIR / "runtime-packages"
        self._runtime_python_dir = APP_DATA_DIR / "runtime-python"
        self._python_command: list[str] | None = None
        self._snapshot_script_path: Path | None = None
        self._snapshot_event_path: Path | None = None
        self._snapshot_stderr_path: Path | None = None

    def _is_windows_admin(self) -> bool:
        if os.name != "nt":
            return True
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

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
        runtime_path = str(self._runtime_site_packages)
        command = self._python_command or []
        if command and not self._is_embedded_python_command(command):
            existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
            if existing_pythonpath:
                env["PYTHONPATH"] = f"{runtime_path}{os.pathsep}{existing_pythonpath}"
            else:
                env["PYTHONPATH"] = runtime_path
        env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        if os.name == "nt":
            if self.enable_xet:
                env["HF_HUB_DISABLE_XET"] = "0"
                env["HF_XET_HIGH_PERFORMANCE"] = "1"
                return env, True
            env["HF_HUB_DISABLE_XET"] = "1"
            env["HF_XET_HIGH_PERFORMANCE"] = "0"
            return env, False
        xet_raw = str(env.get("HF_XET_HIGH_PERFORMANCE", "") or "").strip().lower()
        if xet_raw in {"0", "false", "no", "off"}:
            return env, False
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        return env, True

    def _candidate_python_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []
        seen: set[str] = set()

        def add(cmd: list[str]) -> None:
            key = " ".join(cmd)
            if key in seen:
                return
            seen.add(key)
            commands.append(cmd)

        # In frozen builds, sys.executable points to the app .exe (not python),
        # so using it with "-m pip" would recurse into the app and hang.
        sys_exe = str(sys.executable or "").strip()
        if sys_exe:
            sys_exe_name = Path(sys_exe).name.lower()
            if "python" in sys_exe_name:
                add([sys_exe])
        if os.name == "nt":
            add(["py", "-3"])
        add(["python"])
        return commands

    def _run_python_probe(self, command: list[str]) -> bool:
        probe_cmd = command + ["-c", "import sys;print(sys.executable)"]
        try:
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception:
            return False
        return result.returncode == 0

    def _embedded_python_url_candidates(self) -> list[str]:
        configured = str(os.environ.get("SUB_MANAGER_EMBEDDED_PYTHON_URL", "") or "").strip()
        if configured:
            return [configured]
        # Try newest-to-oldest 3.13 patch versions to avoid hard failure when one URL changes.
        versions = ["3.13.8", "3.13.7", "3.13.6", "3.13.5", "3.13.4", "3.13.3", "3.13.2", "3.13.1", "3.13.0"]
        return [f"https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip" for v in versions]

    def _ensure_embedded_python(self) -> list[str]:
        python_exe = self._runtime_python_dir / "python.exe"
        if python_exe.exists():
            return [str(python_exe)]

        self.progress.emit("Preparing embedded Python runtime...")
        self.diagnostic.emit("No usable system Python runtime was found. Bootstrapping embedded Python.")
        self._runtime_python_dir.mkdir(parents=True, exist_ok=True)

        errors: list[str] = []
        downloaded = False
        for url in self._embedded_python_url_candidates():
            if self._cancel_requested:
                raise RuntimeError("GLM-OCR download cancelled.")
            archive_path = self._runtime_python_dir / "python-embed.zip"
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    archive_path.write_bytes(response.read())
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(self._runtime_python_dir)
                archive_path.unlink(missing_ok=True)
                downloaded = True
                self.diagnostic.emit(f"Embedded Python downloaded from: {url}")
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                try:
                    archive_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if not downloaded or not python_exe.exists():
            raise RuntimeError(
                "Could not bootstrap embedded Python runtime. "
                f"Errors: {' | '.join(errors) if errors else 'unknown error'}"
            )

        pth_files = sorted(self._runtime_python_dir.glob("python*._pth"))
        for pth_file in pth_files:
            try:
                lines = pth_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            cleaned = [line for line in lines if line.strip() and not line.strip().startswith("#")]
            if "Lib\\site-packages" not in cleaned:
                cleaned.append("Lib\\site-packages")
            if "import site" not in cleaned:
                cleaned.append("import site")
            pth_file.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

        # Install pip into embedded Python.
        get_pip_path = self._runtime_python_dir / "get-pip.py"
        try:
            with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=30) as response:
                get_pip_path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Could not download get-pip.py for embedded Python: {exc}") from exc
        try:
            result = subprocess.run(
                [str(python_exe), str(get_pip_path), "--disable-pip-version-check"],
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
                **windows_hidden_subprocess_kwargs(),
            )
        finally:
            try:
                get_pip_path.unlink(missing_ok=True)
            except Exception:
                pass
        if result.returncode != 0:
            stderr_text = str(result.stderr or "").strip()
            raise RuntimeError(
                "Embedded Python bootstrap failed while installing pip. "
                f"{stderr_text or 'Unknown pip error'}"
            )
        self.diagnostic.emit(f"Embedded Python runtime ready: {python_exe}")
        return [str(python_exe)]

    def _resolve_python_command(self) -> list[str]:
        if self._python_command is not None:
            return self._python_command
        for command in self._candidate_python_commands():
            if self._run_python_probe(command):
                self._python_command = command
                self.diagnostic.emit(f"Using Python runtime command: {' '.join(command)}")
                return command
        command = self._ensure_embedded_python()
        self._python_command = command
        return command

    def _is_embedded_python_command(self, command: list[str]) -> bool:
        if not command:
            return False
        exe = Path(command[0]).resolve()
        root = self._runtime_python_dir.resolve()
        try:
            exe.relative_to(root)
            return True
        except ValueError:
            return False

    def _python_can_import(self, command: list[str], module_name: str) -> bool:
        env = os.environ.copy()
        if not self._is_embedded_python_command(command):
            runtime_path = str(self._runtime_site_packages)
            existing_pythonpath = str(env.get("PYTHONPATH", "") or "").strip()
            env["PYTHONPATH"] = f"{runtime_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else runtime_path
        probe_cmd = command + ["-c", f"import {module_name}"]
        try:
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
                env=env,
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception:
            return False
        return result.returncode == 0

    def _ensure_runtime_packages_available(self) -> None:
        package_dir = self._runtime_site_packages
        package_dir.mkdir(parents=True, exist_ok=True)

        package_dir_str = str(package_dir)
        importlib.invalidate_caches()
        primary_command = self._resolve_python_command()
        if self._python_can_import(primary_command, "huggingface_hub"):
            return

        self.progress.emit("Preparing GLM runtime dependencies...")
        self.diagnostic.emit(
            "Missing Python runtime package 'huggingface_hub'. Attempting automatic install."
        )

        packages = ["huggingface_hub>=0.28.0", "hf_transfer>=0.1.9"]
        install_succeeded = False
        last_error = ""
        fallback_commands = [cmd for cmd in self._candidate_python_commands() if cmd != primary_command]
        for cmd in [primary_command, *fallback_commands]:
            if self._cancel_requested:
                raise RuntimeError("GLM-OCR download cancelled.")
            use_target = not self._is_embedded_python_command(cmd)
            pip_args = [
                "-m",
                "pip",
                "install",
                "--upgrade",
            ]
            if use_target:
                pip_args.extend(["--target", package_dir_str])
            full_cmd = cmd + pip_args + packages
            self.diagnostic.emit(f"Trying dependency install via: {' '.join(cmd)} -m pip ...")
            try:
                result = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                    **windows_hidden_subprocess_kwargs(),
                )
            except subprocess.TimeoutExpired:
                self.diagnostic.emit(
                    "Dependency install command timed out after 120s."
                )
                last_error = "Dependency install command timed out."
                continue
            except Exception as exc:
                last_error = str(exc)
                self.diagnostic.emit(f"Dependency install command failed to start: {exc}")
                continue
            if result.returncode == 0:
                if self._python_can_import(cmd, "huggingface_hub"):
                    install_succeeded = True
                    stdout_text = str(result.stdout or "").strip()
                    if stdout_text:
                        self.diagnostic.emit(stdout_text.splitlines()[-1])
                    self._python_command = cmd
                    break
                last_error = "Package install completed but module import check failed."
            stderr_text = str(result.stderr or "").strip()
            if stderr_text:
                self.diagnostic.emit(stderr_text.splitlines()[-1])
                last_error = stderr_text

        if not install_succeeded:
            raise RuntimeError(
                "Could not install required GLM runtime dependencies automatically. "
                f"Last error: {last_error or 'unknown error'}"
            )

        install_location = (
            self._runtime_python_dir / "Lib" / "site-packages"
            if self._is_embedded_python_command(self._python_command or [])
            else package_dir
        )
        self.diagnostic.emit(f"Runtime dependencies installed to: {install_location}")

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
            # `st_blocks` is typically 0 on Windows, so fall back to file size.
            block_bytes = int(getattr(stat_result, "st_blocks", 0) or 0) * 512
            if block_bytes > 0:
                total += block_bytes
            else:
                total += int(getattr(stat_result, "st_size", 0) or 0)
        return max(0, total)

    def _snapshot_download_script(self) -> str:
        return """
import json
import os
import sys
import traceback

repo = os.environ["GLM_OCR_MODEL_ID"]
token = os.environ.get("HF_TOKEN") or None
event_file = os.environ.get("GLM_OCR_EVENT_FILE") or ""
stderr_file = os.environ.get("GLM_OCR_STDERR_FILE") or ""

def emit(obj):
    payload = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(payload + "\\n")
    sys.stdout.flush()
    if event_file:
        with open(event_file, "a", encoding="utf-8") as f:
            f.write(payload + "\\n")

def emit_stderr(text):
    if not stderr_file:
        return
    with open(stderr_file, "a", encoding="utf-8") as f:
        f.write(str(text) + "\\n")

try:
    from huggingface_hub import HfApi, snapshot_download
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
    emit_stderr(traceback.format_exc())
    emit({"type": "error", "message": f"{type(exc).__name__}: {exc}\\n{traceback.format_exc()}"})
    raise
"""

    def _single_quote_for_powershell(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _start_elevated_snapshot_subprocess(
        self, env: dict[str, str]
    ) -> tuple[queue.Queue[dict[str, object]], queue.Queue[str], threading.Thread, threading.Thread]:
        python_command = self._resolve_python_command()
        python_exe = str(Path(python_command[0]).resolve())
        temp_dir = APP_DATA_DIR / "runtime-tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{os.getpid()}-{int(time.time() * 1000)}"
        self._snapshot_script_path = temp_dir / f"glm_snapshot_{suffix}.py"
        self._snapshot_event_path = temp_dir / f"glm_snapshot_{suffix}.jsonl"
        self._snapshot_stderr_path = temp_dir / f"glm_snapshot_{suffix}.stderr.log"
        self._snapshot_script_path.write_text(self._snapshot_download_script(), encoding="utf-8")
        self._snapshot_event_path.write_text("", encoding="utf-8")
        self._snapshot_stderr_path.write_text("", encoding="utf-8")
        env["GLM_OCR_EVENT_FILE"] = str(self._snapshot_event_path)
        env["GLM_OCR_STDERR_FILE"] = str(self._snapshot_stderr_path)

        ps_script = (
            "$ErrorActionPreference='Stop';"
            f"Start-Process -FilePath {self._single_quote_for_powershell(python_exe)} "
            f"-ArgumentList @({self._single_quote_for_powershell(str(self._snapshot_script_path))}) "
            "-Verb RunAs -WindowStyle Hidden -Wait"
        )
        self._process = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
            **windows_hidden_subprocess_kwargs(),
        )
        event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        stderr_queue: queue.Queue[str] = queue.Queue()

        def consume_event_file() -> None:
            offset = 0
            while True:
                path = self._snapshot_event_path
                if path and path.exists():
                    try:
                        with path.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(offset)
                            chunk = handle.read()
                            offset = handle.tell()
                    except Exception:
                        chunk = ""
                    for raw_line in chunk.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            stderr_queue.put(f"[event] {line}")
                            continue
                        if isinstance(payload, dict):
                            event_queue.put(payload)
                process = self._process
                if process is None or process.poll() is not None:
                    break
                time.sleep(0.1)
            path = self._snapshot_event_path
            if path and path.exists():
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                except Exception:
                    chunk = ""
                for raw_line in chunk.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        stderr_queue.put(f"[event] {line}")
                        continue
                    if isinstance(payload, dict):
                        event_queue.put(payload)

        def consume_stderr() -> None:
            offset = 0
            while True:
                path = self._snapshot_stderr_path
                if path and path.exists():
                    try:
                        with path.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(offset)
                            chunk = handle.read()
                            offset = handle.tell()
                    except Exception:
                        chunk = ""
                    for raw_line in chunk.splitlines():
                        line = raw_line.strip()
                        if line:
                            stderr_queue.put(line)
                process = self._process
                if process is None or process.poll() is not None:
                    break
                time.sleep(0.1)
            path = self._snapshot_stderr_path
            if path and path.exists():
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                except Exception:
                    chunk = ""
                for raw_line in chunk.splitlines():
                    line = raw_line.strip()
                    if line:
                        stderr_queue.put(line)
            process = self._process
            if process is not None and process.stderr is not None:
                for raw_line in process.stderr:
                    line = raw_line.strip()
                    if line:
                        stderr_queue.put(f"[powershell] {line}")
            if process is not None and process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if line:
                        stderr_queue.put(f"[powershell] {line}")

        stdout_thread = threading.Thread(target=consume_event_file, daemon=True)
        stderr_thread = threading.Thread(target=consume_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return event_queue, stderr_queue, stdout_thread, stderr_thread

    def _start_snapshot_subprocess(
        self, env: dict[str, str]
    ) -> tuple[queue.Queue[dict[str, object]], queue.Queue[str], threading.Thread, threading.Thread]:
        if os.name == "nt" and self.enable_xet and not self._is_windows_admin():
            return self._start_elevated_snapshot_subprocess(env)
        python_command = self._resolve_python_command()
        self._process = subprocess.Popen(
            [*python_command, "-c", self._snapshot_download_script()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
            **windows_hidden_subprocess_kwargs(),
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
            self._ensure_runtime_packages_available()
            self.progress.emit("Downloading or reusing GLM-OCR model files...")
            env, xet_enabled = self._build_download_env()
            if os.name == "nt" and self.enable_xet and not self._is_windows_admin():
                self.diagnostic.emit(
                    "Xet was requested. Triggering UAC elevation for GLM download process."
                )
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
            last_heartbeat_at = 0.0
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
                now = time.monotonic()
                if now - last_heartbeat_at >= 5.0:
                    self.diagnostic.emit(
                        "Download still running: "
                        f"{self._format_bytes(current_allocated)} downloaded so far."
                    )
                    last_heartbeat_at = now
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
            for temp_path in (self._snapshot_script_path, self._snapshot_event_path, self._snapshot_stderr_path):
                try:
                    if temp_path:
                        temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._snapshot_script_path = None
            self._snapshot_event_path = None
            self._snapshot_stderr_path = None
