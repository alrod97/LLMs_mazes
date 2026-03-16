"""Maze image discovery and encoding helpers."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_OPENAI_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MazeImage:
    name: str
    path: Path
    mime_type: str
    size_bytes: int


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_maze_images(image_dir: Path) -> list[MazeImage]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {image_dir}")

    images: list[MazeImage] = []
    for path in image_dir.iterdir():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        images.append(
            MazeImage(
                name=path.stem,
                path=path,
                mime_type=detect_image_mime_type(path),
                size_bytes=path.stat().st_size,
            )
        )

    return sorted(images, key=lambda item: _natural_key(item.path.name))


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def encode_image_data_url(image_path: Path, mime_type: str) -> str:
    encoded = encode_image_base64(image_path)
    return f"data:{mime_type};base64,{encoded}"


def detect_image_mime_type(image_path: Path) -> str:
    header = image_path.read_bytes()[:16]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    mime_type, _ = mimetypes.guess_type(image_path.name)
    return mime_type or "application/octet-stream"
