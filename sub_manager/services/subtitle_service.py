from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

from sub_manager.process_utils import windows_hidden_subprocess_kwargs
from sub_manager.state_types import AudioStreamItem, SubtitleCodecStreamItem, SubtitleStreamItem


class SubtitleService:
    def __init__(self, os_module: ModuleType, shutil_module: ModuleType) -> None:
        self._os = os_module
        self._shutil = shutil_module

    def run_ffprobe_json(self, file_path: str, ffprobe_path: str | None) -> dict[str, object] | None:
        ffprobe_exe = ffprobe_path or self._shutil.which("ffprobe")
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
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                **windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                return None
            payload = json.loads(result.stdout or "{}")
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception:
            return None

    def inspect_stream_languages(
        self, file_path: str, ffprobe_path: str | None
    ) -> tuple[list[AudioStreamItem], list[SubtitleStreamItem | SubtitleCodecStreamItem]]:
        payload = self.run_ffprobe_json(file_path, ffprobe_path)
        fallback_audio: AudioStreamItem = {"label": "N/A", "stream_index": -1, "language": "und"}
        fallback_subtitle: SubtitleStreamItem = {
            "label": "N/A",
            "stream_index": -1,
            "language": "und",
        }
        if payload is None:
            return ([fallback_audio], [fallback_subtitle])
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return ([fallback_audio], [fallback_subtitle])

        audio_streams: list[AudioStreamItem] = []
        subtitle_streams: list[SubtitleStreamItem | SubtitleCodecStreamItem] = []

        for fallback_index, stream in enumerate(streams):
            if not isinstance(stream, dict):
                continue
            codec_type = str(stream.get("codec_type", "")).lower()
            stream_index = self.stream_index(stream, fallback_index)
            language = self.stream_language(stream)
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
                subtitle_stream: SubtitleCodecStreamItem = {
                    "label": f"{language} #{stream_index}",
                    "stream_index": stream_index,
                    "language": language,
                    "codec_name": codec_name,
                }
                subtitle_streams.append(subtitle_stream)

        if not audio_streams:
            audio_streams = [{"label": "-", "stream_index": -1, "language": "und"}]
        if not subtitle_streams:
            subtitle_streams = [{"label": "-", "stream_index": -1, "language": "und"}]
        return (audio_streams, subtitle_streams)

    def find_subtitle_stream_by_index(
        self, streams: list[object], stream_index: int
    ) -> dict[str, object] | None:
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            if str(stream.get("codec_type", "")).lower() != "subtitle":
                continue
            if self.stream_index(stream, -1) == stream_index:
                return stream
        return None

    def stream_language(self, stream: dict[str, object]) -> str:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            tag_lang = tags.get("language")
            if isinstance(tag_lang, str) and tag_lang.strip():
                return tag_lang.strip().lower()
        return "und"

    def stream_title(self, stream: dict[str, object]) -> str:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            tag_title = tags.get("title")
            if isinstance(tag_title, str) and tag_title.strip():
                return tag_title.strip()
        return ""

    def stream_index(self, stream: dict[str, object], fallback: int) -> int:
        value = stream.get("index", fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def extract_subtitle_text(
        self,
        file_path: str,
        stream_index: int,
        ffmpeg_path: str | None,
    ) -> str | None:
        ffmpeg_exe = ffmpeg_path or self._shutil.which("ffmpeg")
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
            result = subprocess.run(
                command,
                capture_output=True,
                text=False,
                check=False,
                **windows_hidden_subprocess_kwargs(),
            )
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

    def output_subtitle_codec_for_container(self, file_path: str) -> str | None:
        ext = Path(file_path).suffix.lower()
        if ext in {".mp4", ".m4v", ".mov"}:
            return "mov_text"
        if ext in {".mkv", ".avi"}:
            return "srt"
        if ext == ".webm":
            return "webvtt"
        return None

    def save_embedded_subtitle(
        self,
        file_path: str,
        stream_index: int,
        subtitle_text: str,
        ffmpeg_path: str | None,
        ffprobe_path: str | None,
    ) -> tuple[bool, str]:
        ffmpeg_exe = ffmpeg_path or self._shutil.which("ffmpeg")
        if not ffmpeg_exe:
            return (False, "ffmpeg binary not found.")

        payload = self.run_ffprobe_json(file_path, ffprobe_path)
        if payload is None:
            return (False, "Could not inspect file streams with ffprobe.")
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            return (False, "Invalid stream metadata from ffprobe.")

        target_stream = self.find_subtitle_stream_by_index(streams, stream_index)
        if target_stream is None:
            return (False, f"Subtitle stream #{stream_index} not found.")

        subtitle_stream_count = 0
        for stream in streams:
            if isinstance(stream, dict) and str(stream.get("codec_type", "")).lower() == "subtitle":
                subtitle_stream_count += 1
        if subtitle_stream_count == 0:
            return (False, "No subtitle streams found.")

        output_codec = self.output_subtitle_codec_for_container(file_path)
        if output_codec is None:
            return (False, "Container does not support automatic subtitle replacement.")

        language = self.stream_language(target_stream)
        title = self.stream_title(target_stream)
        new_subtitle_index = subtitle_stream_count - 1

        source_path = Path(file_path)
        with tempfile.TemporaryDirectory(prefix="sub_manager_") as tmp_dir:
            temp_subtitle_path = Path(tmp_dir) / "edited_subtitle.srt"
            try:
                temp_subtitle_path.write_text(subtitle_text, encoding="utf-8")
            except Exception as exc:
                return (False, f"Could not prepare temporary subtitle file: {exc}")

            temp_output_path = source_path.parent / (f".{source_path.stem}.submgr_tmp{source_path.suffix}")
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
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    **windows_hidden_subprocess_kwargs(),
                )
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
                self._os.replace(temp_output_path, source_path)
            except Exception as exc:
                try:
                    if temp_output_path.exists():
                        temp_output_path.unlink()
                except Exception:
                    pass
                return (False, f"Could not overwrite original file: {exc}")
        return (True, str(source_path))
