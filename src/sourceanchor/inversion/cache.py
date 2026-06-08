from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sourceanchor.inversion.ddim import InversionOutput
from sourceanchor.outputs.writer import save_image


ZT_FILENAME = "zt_src.pt"
LATENTS_FILENAME = "src_latents.pt"
RECONSTRUCTION_FILENAME = "reconstruction.png"
META_FILENAME = "inversion_meta.json"


def save_inversion_output(cache_dir: str | Path, inversion: InversionOutput) -> Path:
    target_dir = Path(cache_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(inversion.zt_src.detach().cpu(), target_dir / ZT_FILENAME)
    torch.save([latent.detach().cpu() for latent in inversion.src_latents], target_dir / LATENTS_FILENAME)
    save_image(target_dir / RECONSTRUCTION_FILENAME, inversion.reconstruction_image)
    (target_dir / META_FILENAME).write_text(
        json.dumps(inversion.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_dir


def _load_tensor_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_inversion_output(cache_dir: str | Path, *, device: torch.device | str, dtype: torch.dtype) -> InversionOutput:
    target_dir = Path(cache_dir).expanduser().resolve()
    zt_path = target_dir / ZT_FILENAME
    latents_path = target_dir / LATENTS_FILENAME
    reconstruction_path = target_dir / RECONSTRUCTION_FILENAME
    meta_path = target_dir / META_FILENAME
    for path in (zt_path, latents_path, reconstruction_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Inversion cache file not found: {path}")

    zt_src = _load_tensor_payload(zt_path).to(device=device, dtype=dtype)
    src_latents = [
        latent.to(device=device, dtype=dtype)
        for latent in _load_tensor_payload(latents_path)
    ]
    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    reconstruction_image = np.asarray(Image.open(reconstruction_path).convert("RGB"))
    metadata = {
        **metadata,
        "cache_dir": str(target_dir),
        "loaded_from_cache": True,
    }
    return InversionOutput(
        zt_src=zt_src,
        src_latents=src_latents,
        reconstruction_image=reconstruction_image,
        metadata=metadata,
    )
