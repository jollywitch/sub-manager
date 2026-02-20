from __future__ import annotations

from typing import TypedDict


class AudioStreamItem(TypedDict):
    label: str
    stream_index: int
    language: str


class SubtitleStreamItem(TypedDict):
    label: str
    stream_index: int
    language: str


class SubtitleCodecStreamItem(SubtitleStreamItem, total=False):
    codec_name: str


class VideoFileItem(TypedDict):
    name: str
    path: str
    checked: bool
    audio_language_items: list[AudioStreamItem]
    subtitle_language_items: list[SubtitleStreamItem | SubtitleCodecStreamItem]
