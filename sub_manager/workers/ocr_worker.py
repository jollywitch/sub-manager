from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from sub_manager.constants import GLM_OCR_MODEL_ID
from sub_manager.models import PgsCompositionObject, PgsCue, PgsObjectData
from sub_manager.process_utils import windows_hidden_subprocess_kwargs

class ImageSubtitleOcrWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)
    OCR_BATCH_SIZE_CUDA = 4
    OCR_BATCH_SIZE_MPS = 2
    OCR_BATCH_SIZE_CPU = 1
    OCR_HASH_HAMMING_THRESHOLD = 2
    OCR_MAX_NEW_TOKENS = 256
    _model_cache_lock = threading.Lock()
    _cached_model_bundle: tuple[object, object, object] | None = None
    _cached_model_id: str | None = None

    def __init__(
        self,
        file_path: str,
        stream_index: int,
        codec_name: str,
        ffmpeg_exe: str | None,
        ffprobe_exe: str | None,
        hf_token: str | None,
        model_id: str = GLM_OCR_MODEL_ID,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.stream_index = stream_index
        self.codec_name = codec_name
        self.ffmpeg_exe = ffmpeg_exe or shutil.which("ffmpeg")
        self.ffprobe_exe = ffprobe_exe or shutil.which("ffprobe")
        self.hf_token = (hf_token or "").strip() or None
        self.model_id = model_id
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self._validate_runtime_dependencies()
            self.progress.emit("Checking GLM-OCR runtime...")
            torch, processor, model = self._load_model()
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")

            self.progress.emit("Extracting subtitle bitmaps...")
            with tempfile.TemporaryDirectory(prefix="sub_manager_ocr_") as temp_dir:
                cues = self._extract_image_subtitle_cues(Path(temp_dir))
                cue_count = len(cues)
                if cue_count == 0:
                    raise RuntimeError("Could not find image subtitle cues.")
                recognized_texts, reused_count = self._recognize_cues(
                    torch_module=torch,
                    processor=processor,
                    model=model,
                    cues=cues,
                )
                if reused_count > 0:
                    self.progress.emit(f"Reused OCR text for {reused_count} duplicate subtitle images.")

                recognized_cues = self._build_recognized_cues(cues, recognized_texts)
                merged_cues = self._merge_adjacent_cues(recognized_cues)
                srt_text = self._build_srt(merged_cues)
                if not srt_text.strip():
                    raise RuntimeError("OCR completed but no subtitle text was recognized.")
                self.finished.emit(srt_text)
        except Exception as exc:
            logging.error("Image subtitle OCR worker failed", exc_info=True)
            self.failed.emit(str(exc))

    def _validate_runtime_dependencies(self) -> None:
        if self._cancel_requested:
            raise RuntimeError("OCR cancelled.")
        if not self.ffmpeg_exe:
            raise RuntimeError("ffmpeg binary not found.")
        if not self.ffprobe_exe:
            raise RuntimeError("ffprobe binary not found.")

    def _recognize_cues(
        self,
        torch_module: object,
        processor: object,
        model: object,
        cues: list[PgsCue],
    ) -> tuple[list[str], int]:
        recognized_texts: list[str] = [""] * len(cues)
        pending_indices: list[int] = []
        pending_frame_paths: list[Path] = []
        cue_hashes: list[int] = [0] * len(cues)
        hash_text_cache: dict[int, str] = {}
        hash_index: list[tuple[int, str]] = []
        reused_count = 0

        self.progress.emit(f"Running OCR on {len(cues)} subtitle images...")
        for idx, cue in enumerate(cues):
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")
            hash_value = self._compute_frame_hash(cue.image_path)
            cue_hashes[idx] = hash_value
            reused = self._lookup_cached_text_for_hash(hash_value, hash_text_cache, hash_index)
            if reused is not None:
                recognized_texts[idx] = reused
                reused_count += 1
                continue
            pending_indices.append(idx)
            pending_frame_paths.append(cue.image_path)

        batch_size = self._suggest_batch_size(model)
        total_batches = (len(pending_frame_paths) + batch_size - 1) // batch_size if pending_frame_paths else 0
        for batch_idx, start in enumerate(range(0, len(pending_frame_paths), batch_size), start=1):
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")
            batch_paths = pending_frame_paths[start : start + batch_size]
            self.progress.emit(f"Running OCR batch {batch_idx}/{total_batches} ({len(batch_paths)} frames)...")
            batch_texts = self._ocr_image_frames_batched(torch_module, processor, model, batch_paths)
            for offset, text in enumerate(batch_texts):
                original_idx = pending_indices[start + offset]
                clean_text = self._normalize_ocr_text(text)
                recognized_texts[original_idx] = clean_text
                self._store_hash_text(cue_hashes[original_idx], clean_text, hash_text_cache, hash_index)

        return recognized_texts, reused_count

    def _build_recognized_cues(
        self, cues: list[PgsCue], recognized_texts: list[str]
    ) -> list[tuple[float, float, str]]:
        recognized_cues: list[tuple[float, float, str]] = []
        for idx, text in enumerate(recognized_texts):
            if not text:
                continue
            recognized_cues.append((cues[idx].start, cues[idx].end, text))
        return recognized_cues

    def _load_model(self) -> tuple[object, object, object]:
        with self._model_cache_lock:
            if self._cached_model_bundle is not None and self._cached_model_id == self.model_id:
                self.progress.emit("Reusing loaded GLM-OCR model...")
                return self._cached_model_bundle

        try:
            import torch  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "GLM-OCR dependencies are missing. Install: transformers, torch, pillow, huggingface_hub."
            ) from exc

        xet_raw = str(os.environ.get("HF_XET_HIGH_PERFORMANCE", "") or "").strip().lower()
        if xet_raw not in {"0", "false", "no", "off"}:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

        has_cuda = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        has_mps = bool(mps_backend) and bool(getattr(mps_backend, "is_available", lambda: False)())
        if not (has_cuda or has_mps):
            raise RuntimeError("GLM-OCR requires a GPU runtime (CUDA or MPS).")
        target_device = "cuda" if has_cuda else "mps"

        self.progress.emit("Downloading or reusing GLM-OCR model files...")
        local_model_path = snapshot_download(repo_id=self.model_id, token=self.hf_token)

        self.progress.emit("Loading GLM-OCR model...")
        try:
            loaded_processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load GLM-OCR processor. Install the latest transformers from GitHub main and run `uv sync`."
            ) from exc
        try:
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception:
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            if hasattr(loaded_model, "to"):
                loaded_model = loaded_model.to(target_device)
        try:
            first_param = next(loaded_model.parameters())
            device_type = str(getattr(first_param, "device", "cpu"))
            if device_type.startswith("cpu"):
                raise RuntimeError("GLM-OCR loaded on CPU. GPU runtime is required.")
        except StopIteration:
            pass
        bundle = (torch, loaded_processor, loaded_model)
        with self._model_cache_lock:
            self._cached_model_bundle = bundle
            self._cached_model_id = self.model_id
        return bundle

    @classmethod
    def clear_model_cache(cls) -> None:
        with cls._model_cache_lock:
            cls._cached_model_bundle = None
            cls._cached_model_id = None

    def _extract_image_subtitle_cues(self, temp_root: Path) -> list[PgsCue]:
        if self.codec_name == "hdmv_pgs_subtitle":
            return self._extract_pgs_cues(temp_root)
        if self.codec_name == "dvd_subtitle":
            raise RuntimeError("DVD subtitle OCR is not implemented yet. Only PGS is supported.")
        raise RuntimeError(f"Unsupported image subtitle codec: {self.codec_name}")

    def _extract_pgs_cues(self, temp_root: Path) -> list[PgsCue]:
        if self._cancel_requested:
            raise RuntimeError("OCR cancelled.")
        sup_path = temp_root / "stream.sup"
        command = [
            self.ffmpeg_exe,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            self.file_path,
            "-map",
            f"0:{self.stream_index}",
            "-c:s",
            "copy",
            str(sup_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not run ffmpeg for PGS extraction: {exc}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "Unknown ffmpeg error").strip()
            raise RuntimeError(f"Could not extract PGS subtitle stream: {stderr}")
        if not sup_path.exists():
            raise RuntimeError("PGS extraction did not produce a .sup file.")

        frames_dir = temp_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        return self._parse_pgs_sup_to_cues(sup_path, frames_dir)

    def _parse_pgs_sup_to_cues(self, sup_path: Path, frames_dir: Path) -> list[PgsCue]:
        raw = sup_path.read_bytes()
        palettes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
        objects: dict[int, PgsObjectData] = {}
        current_palette_id = 0
        active_composition: list[PgsCompositionObject] = []
        active_video_size = (0, 0)
        cues: list[PgsCue] = []
        pending_clear_time: float | None = None
        i = 0
        cue_index = 0

        while i + 13 <= len(raw):
            if self._cancel_requested:
                raise RuntimeError("OCR cancelled.")
            if raw[i : i + 2] != b"PG":
                i += 1
                continue
            pts_90k = int.from_bytes(raw[i + 2 : i + 6], "big", signed=False)
            segment_type = raw[i + 10]
            payload_size = int.from_bytes(raw[i + 11 : i + 13], "big", signed=False)
            payload_start = i + 13
            payload_end = payload_start + payload_size
            if payload_end > len(raw):
                break
            payload = raw[payload_start:payload_end]
            pts_seconds = pts_90k / 90000.0

            if segment_type == 0x14:
                palette_id, palette = self._parse_pds(payload)
                palettes[palette_id] = palette
            elif segment_type == 0x15:
                object_id, parsed = self._parse_ods(payload, objects.get)
                objects[object_id] = parsed
            elif segment_type == 0x16:
                (
                    video_width,
                    video_height,
                    palette_id,
                    composition_objects,
                ) = self._parse_pcs(payload)
                active_video_size = (video_width, video_height)
                current_palette_id = palette_id
                active_composition = composition_objects
                pending_clear_time = pts_seconds if not composition_objects else None
            elif segment_type == 0x80:
                if pending_clear_time is not None:
                    if cues and cues[-1].end <= cues[-1].start:
                        cues[-1].end = max(cues[-1].start + 0.05, pending_clear_time)
                    pending_clear_time = None
                    i = payload_end
                    continue
                if not active_composition:
                    i = payload_end
                    continue
                cue_path = frames_dir / f"cue_{cue_index:08d}.png"
                rendered = self._render_pgs_composition_image(
                    cue_path=cue_path,
                    composition_objects=active_composition,
                    objects=objects,
                    palette=palettes.get(current_palette_id, {}),
                    video_size=active_video_size,
                )
                if rendered:
                    cue_start = pts_seconds
                    if cues and cues[-1].end <= cues[-1].start:
                        cues[-1].end = max(cues[-1].start + 0.05, cue_start)
                    cues.append(PgsCue(start=cue_start, end=cue_start, image_path=cue_path))
                    cue_index += 1
            i = payload_end

        if cues:
            for idx in range(len(cues) - 1):
                if cues[idx].end <= cues[idx].start:
                    cues[idx].end = max(cues[idx].start + 0.05, cues[idx + 1].start)
            if cues[-1].end <= cues[-1].start:
                cues[-1].end = cues[-1].start + 2.0
        if not cues:
            raise RuntimeError("No PGS cues could be extracted from the subtitle stream.")
        return cues

    def _parse_pds(self, payload: bytes) -> tuple[int, dict[int, tuple[int, int, int, int]]]:
        if len(payload) < 2:
            return (0, {})
        palette_id = payload[0]
        palette: dict[int, tuple[int, int, int, int]] = {}
        pos = 2
        while pos + 5 <= len(payload):
            index = payload[pos]
            y = payload[pos + 1]
            cr = payload[pos + 2]
            cb = payload[pos + 3]
            alpha = payload[pos + 4]
            r, g, b = self._ycbcr_to_rgb(y, cb, cr)
            palette[index] = (r, g, b, alpha)
            pos += 5
        return (palette_id, palette)

    def _parse_ods(
        self,
        payload: bytes,
        object_lookup: Callable[[int], PgsObjectData | None],
    ) -> tuple[int, PgsObjectData]:
        if len(payload) < 4:
            raise RuntimeError("Invalid PGS ODS payload.")
        object_id = int.from_bytes(payload[0:2], "big", signed=False)
        sequence_flag = payload[3]
        first_in_sequence = bool(sequence_flag & 0x80)
        data_start = 4
        if first_in_sequence:
            if len(payload) < 11:
                raise RuntimeError("Invalid first PGS ODS payload.")
            object_data_length = int.from_bytes(payload[4:7], "big", signed=False)
            width = int.from_bytes(payload[7:9], "big", signed=False)
            height = int.from_bytes(payload[9:11], "big", signed=False)
            expected_rle_length = max(object_data_length - 4, 0)
            rle_data = bytearray(payload[11:])
            return (
                object_id,
                PgsObjectData(
                    width=width,
                    height=height,
                    expected_rle_length=expected_rle_length,
                    rle_data=rle_data[:expected_rle_length],
                ),
            )
        existing = object_lookup(object_id)
        if existing is None:
            raise RuntimeError("PGS object continuation encountered without prior object data.")
        existing.rle_data.extend(payload[data_start:])
        if existing.expected_rle_length > 0 and len(existing.rle_data) > existing.expected_rle_length:
            existing.rle_data = existing.rle_data[: existing.expected_rle_length]
        return (object_id, existing)

    def _parse_pcs(
        self, payload: bytes
    ) -> tuple[int, int, int, list[PgsCompositionObject]]:
        if len(payload) < 11:
            raise RuntimeError("Invalid PGS PCS payload.")
        video_width = int.from_bytes(payload[0:2], "big", signed=False)
        video_height = int.from_bytes(payload[2:4], "big", signed=False)
        palette_id = payload[9]
        composition_object_count = payload[10]
        composition_objects: list[PgsCompositionObject] = []
        pos = 11
        for _ in range(composition_object_count):
            if pos + 8 > len(payload):
                break
            object_id = int.from_bytes(payload[pos : pos + 2], "big", signed=False)
            window_id = payload[pos + 2]
            crop_flag = payload[pos + 3]
            x = int.from_bytes(payload[pos + 4 : pos + 6], "big", signed=False)
            y = int.from_bytes(payload[pos + 6 : pos + 8], "big", signed=False)
            pos += 8
            if crop_flag & 0x40:
                if pos + 8 > len(payload):
                    break
                pos += 8
            composition_objects.append(
                PgsCompositionObject(object_id=object_id, window_id=window_id, x=x, y=y)
            )
        return (video_width, video_height, palette_id, composition_objects)

    def _render_pgs_composition_image(
        self,
        cue_path: Path,
        composition_objects: list[PgsCompositionObject],
        objects: dict[int, PgsObjectData],
        palette: dict[int, tuple[int, int, int, int]],
        video_size: tuple[int, int],
    ) -> bool:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required for PGS image rendering.") from exc

        if not composition_objects:
            return False
        available: list[tuple[PgsCompositionObject, PgsObjectData]] = []
        for comp in composition_objects:
            obj = objects.get(comp.object_id)
            if obj is None or obj.width <= 0 or obj.height <= 0 or not obj.rle_data:
                continue
            available.append((comp, obj))
        if not available:
            return False

        min_x = min(comp.x for comp, _ in available)
        min_y = min(comp.y for comp, _ in available)
        max_x = max(comp.x + obj.width for comp, obj in available)
        max_y = max(comp.y + obj.height for comp, obj in available)
        if video_size[0] > 0 and video_size[1] > 0:
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, video_size[0])
            max_y = min(max_y, video_size[1])
        canvas_w = max(1, max_x - min_x)
        canvas_h = max(1, max_y - min_y)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for comp, obj in available:
            indices = self._decode_pgs_rle(obj.rle_data, obj.width, obj.height)
            rgba_data = [palette.get(index, (255, 255, 255, 255)) for index in indices]
            sprite = Image.new("RGBA", (obj.width, obj.height))
            sprite.putdata(rgba_data)
            canvas.alpha_composite(sprite, (max(comp.x - min_x, 0), max(comp.y - min_y, 0)))

        bbox = canvas.getbbox()
        if bbox is None:
            return False
        cropped = canvas.crop(bbox)
        cropped.save(cue_path)
        return True

    def _decode_pgs_rle(self, rle_data: bytes | bytearray, width: int, height: int) -> list[int]:
        pixels: list[int] = []
        row_pixels = 0
        i = 0
        target = max(width * height, 0)
        while i < len(rle_data) and len(pixels) < target:
            value = rle_data[i]
            i += 1
            if value != 0:
                pixels.append(value)
                row_pixels += 1
                continue
            if i >= len(rle_data):
                break
            code = rle_data[i]
            i += 1
            if code == 0:
                if width > 0 and row_pixels < width:
                    pixels.extend([0] * (width - row_pixels))
                row_pixels = 0
                continue
            if code & 0x40:
                if i >= len(rle_data):
                    break
                run_length = ((code & 0x3F) << 8) | rle_data[i]
                i += 1
            else:
                run_length = code & 0x3F
            if code & 0x80:
                if i >= len(rle_data):
                    break
                color = rle_data[i]
                i += 1
            else:
                color = 0
            if run_length <= 0:
                continue
            pixels.extend([color] * run_length)
            row_pixels += run_length

        if len(pixels) < target:
            pixels.extend([0] * (target - len(pixels)))
        return pixels[:target]

    def _compute_frame_hash(self, frame_path: Path) -> int:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required for OCR frame hashing.") from exc
        with Image.open(frame_path) as image:
            grayscale = image.convert("L").resize((9, 8))
            pixels = list(grayscale.get_flattened_data())
        hash_bits = 0
        for row in range(8):
            for col in range(8):
                left = pixels[(row * 9) + col]
                right = pixels[(row * 9) + col + 1]
                hash_bits = (hash_bits << 1) | (1 if left > right else 0)
        return hash_bits

    def _lookup_cached_text_for_hash(
        self,
        hash_value: int,
        hash_text_cache: dict[int, str],
        hash_index: list[tuple[int, str]],
    ) -> str | None:
        exact = hash_text_cache.get(hash_value)
        if exact is not None:
            return exact
        best_text: str | None = None
        best_distance = self.OCR_HASH_HAMMING_THRESHOLD + 1
        for previous_hash, previous_text in hash_index:
            distance = self._hamming_distance(hash_value, previous_hash)
            if distance < best_distance:
                best_distance = distance
                best_text = previous_text
                if distance == 0:
                    break
        if best_text is None or best_distance > self.OCR_HASH_HAMMING_THRESHOLD:
            return None
        return best_text

    def _store_hash_text(
        self,
        hash_value: int,
        text: str,
        hash_text_cache: dict[int, str],
        hash_index: list[tuple[int, str]],
    ) -> None:
        if not text:
            return
        if hash_value not in hash_text_cache:
            hash_index.append((hash_value, text))
        hash_text_cache[hash_value] = text

    def _hamming_distance(self, left: int, right: int) -> int:
        return int((left ^ right).bit_count())

    def _suggest_batch_size(self, model: object) -> int:
        model_device = str(self._model_device(model))
        if model_device.startswith("cuda"):
            return self.OCR_BATCH_SIZE_CUDA
        if model_device.startswith("mps"):
            return self.OCR_BATCH_SIZE_MPS
        return self.OCR_BATCH_SIZE_CPU

    def _ocr_image_frames_batched(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_paths: list[Path],
    ) -> list[str]:
        if not frame_paths:
            return []
        try:
            return self._ocr_image_frames_batched_impl(torch_module, processor, model, frame_paths)
        except Exception:
            results: list[str] = []
            for frame_path in frame_paths:
                results.append(self._ocr_image_frame(torch_module, processor, model, frame_path))
            return results

    def _ocr_image_frames_batched_impl(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_paths: list[Path],
    ) -> list[str]:
        prompt = "Text Recognition:"
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("GLM-OCR processor does not support chat templates.")
        conversations = []
        for frame_path in frame_paths:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": str(frame_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        model_device = self._model_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)

        with torch_module.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self.OCR_MAX_NEW_TOKENS)

        input_lengths: list[int] = []
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None and hasattr(attention_mask, "sum"):
                try:
                    mask_lengths = attention_mask.sum(dim=1).tolist()
                    input_lengths = [int(v) for v in mask_lengths]
                except Exception:
                    input_lengths = []
            if not input_lengths:
                input_ids = inputs.get("input_ids")
                if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
                    input_lengths = [int(input_ids.shape[1])] * int(input_ids.shape[0])
        if not input_lengths:
            input_lengths = [0] * len(frame_paths)

        decoded_texts: list[str] = []
        for row_idx, prompt_len in enumerate(input_lengths):
            sample_ids = output_ids[row_idx]
            if prompt_len > 0 and hasattr(sample_ids, "__getitem__"):
                sample_ids = sample_ids[prompt_len:]
            decoded = processor.decode(sample_ids, skip_special_tokens=True)
            decoded_texts.append(decoded if isinstance(decoded, str) else "")
        return decoded_texts

    def _ocr_image_frame(
        self,
        torch_module: object,
        processor: object,
        model: object,
        frame_path: Path,
    ) -> str:
        prompt = "Text Recognition:"
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("GLM-OCR processor does not support chat templates.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(frame_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = self._model_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)

        with torch_module.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self.OCR_MAX_NEW_TOKENS)

        prompt_tokens = 0
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
                prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > 0 and hasattr(output_ids, "__getitem__"):
            output_ids = output_ids[:, prompt_tokens:]

        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        return self._normalize_ocr_text(decoded[0])

    def _model_device(self, model: object) -> object:
        try:
            first_param = next(model.parameters())
            return getattr(first_param, "device")
        except StopIteration:
            return "cpu"

    def _merge_adjacent_cues(
        self, cues: list[tuple[float, float, str]], max_gap_seconds: float = 0.15
    ) -> list[tuple[float, float, str]]:
        merged: list[tuple[float, float, str]] = []
        for start, end, text in cues:
            if not merged:
                merged.append((start, end, text))
                continue
            prev_start, prev_end, prev_text = merged[-1]
            if text == prev_text and (start - prev_end) <= max_gap_seconds:
                merged[-1] = (prev_start, max(prev_end, end), prev_text)
                continue
            merged.append((start, end, text))
        return merged

    def _build_srt(self, cues: list[tuple[float, float, str]]) -> str:
        lines: list[str] = []
        cue_index = 1
        for start, end, text in cues:
            clean = self._normalize_ocr_text(text)
            if not clean:
                continue
            if end <= start:
                end = start + 0.05
            lines.append(str(cue_index))
            lines.append(f"{self._srt_timestamp(start)} --> {self._srt_timestamp(end)}")
            lines.append(clean)
            lines.append("")
            cue_index += 1
        return "\n".join(lines).strip() + ("\n" if lines else "")

    def _srt_timestamp(self, seconds: float) -> str:
        safe_seconds = max(seconds, 0.0)
        total_ms = int(round(safe_seconds * 1000))
        hours = total_ms // 3_600_000
        total_ms -= hours * 3_600_000
        minutes = total_ms // 60_000
        total_ms -= minutes * 60_000
        secs = total_ms // 1000
        millis = total_ms - (secs * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _normalize_ocr_text(self, text: str) -> str:
        collapsed = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        # Remove multimodal special tokens leaked by model decode.
        collapsed = re.sub(r"<\|[^|>]+?\|>", "", collapsed)
        # Keep only actual OCR content after the prompt echo, if present.
        pivot = "Text Recognition:"
        if pivot in collapsed:
            collapsed = collapsed.rsplit(pivot, 1)[-1]
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    def _ycbcr_to_rgb(self, y: int, cb: int, cr: int) -> tuple[int, int, int]:
        yy = float(y)
        cbb = float(cb) - 128.0
        crr = float(cr) - 128.0
        r = int(round(yy + (1.402 * crr)))
        g = int(round(yy - (0.344136 * cbb) - (0.714136 * crr)))
        b = int(round(yy + (1.772 * cbb)))
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b)),
        )
