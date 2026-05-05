from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(Path(path).expanduser().resolve()).convert("RGB"))


def load_rgb_pil(path: str | Path) -> Image.Image:
    return Image.open(Path(path).expanduser().resolve()).convert("RGB")
