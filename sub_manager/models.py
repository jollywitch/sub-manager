from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PgsObjectData:
    width: int
    height: int
    expected_rle_length: int
    rle_data: bytearray


@dataclass
class PgsCompositionObject:
    object_id: int
    window_id: int
    x: int
    y: int


@dataclass
class PgsCue:
    start: float
    end: float
    image_path: Path
