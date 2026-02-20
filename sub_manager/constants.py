from __future__ import annotations

from pathlib import Path

from sub_manager.paths import app_data_root

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
APP_DATA_DIR = app_data_root()
LOG_DIR = APP_DATA_DIR / "logs"
SESSION_LOG_PATH = LOG_DIR / "session.log"
FFMPEG_TOOLS_DIR = APP_DATA_DIR / "tools" / "ffmpeg"
