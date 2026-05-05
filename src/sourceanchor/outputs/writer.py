from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def image_to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def numpy_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(image_to_numpy(array))


def save_image(path: str | Path, image: Image.Image | np.ndarray) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    numpy_to_image(image).save(resolved)
    return resolved


def compose_simple_overview(items: list[tuple[str, Image.Image | np.ndarray]], columns: int = 3) -> np.ndarray:
    if not items:
        raise ValueError("items must not be empty")
    font = ImageFont.load_default()
    tile_width = 512
    tile_height = 512
    label_height = 28
    gap = 12
    padding = 16
    columns = max(1, columns)
    rows = (len(items) + columns - 1) // columns
    canvas_width = padding * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = padding * 2 + rows * (tile_height + label_height) + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    drawer = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        row = index // columns
        col = index % columns
        x = padding + col * (tile_width + gap)
        y = padding + row * (tile_height + label_height + gap)
        drawer.rectangle((x, y, x + tile_width - 1, y + tile_height + label_height - 1), outline="black", width=1)
        drawer.text((x + 8, y + 6), label, fill="black", font=font)
        fitted = ImageOps.contain(numpy_to_image(image), (tile_width - 12, tile_height - 12))
        tile = Image.new("RGB", (tile_width - 12, tile_height - 12), color="white")
        tile.paste(fitted, ((tile.width - fitted.width) // 2, (tile.height - fitted.height) // 2))
        canvas.paste(tile, (x + 6, y + label_height + 6))
    return np.asarray(canvas)
