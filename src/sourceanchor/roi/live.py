from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sourceanchor.config import MethodConfig, RoiConfig
from sourceanchor.roi.base import RoiPayload


def describe_live_roi_behavior() -> str:
    return "ROI is generated on the fly and does not read historical cache."


def _normalize_mask_batch(mask_output) -> list[np.ndarray]:
    if isinstance(mask_output, np.ndarray):
        if mask_output.ndim == 2:
            return [np.asarray(mask_output, dtype=np.float32)]
        if mask_output.ndim == 3:
            return [np.asarray(mask_output[index], dtype=np.float32) for index in range(mask_output.shape[0])]
    if isinstance(mask_output, list):
        normalized: list[np.ndarray] = []
        for item in mask_output:
            if isinstance(item, Image.Image):
                normalized.append(np.asarray(item.convert("L"), dtype=np.float32) / 255.0)
            else:
                normalized.append(np.asarray(item, dtype=np.float32))
        return normalized
    raise ValueError(f"Unexpected DiffEdit mask output type: {type(mask_output)}")


def load_sample_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def generate_live_roi(diffedit_pipe, sample, method: MethodConfig, roi: RoiConfig) -> RoiPayload:
    source_pils = [load_sample_pil(sample.source_image_path)]
    mask_output = diffedit_pipe.generate_mask(
        image=source_pils,
        source_prompt=[sample.source_prompt],
        target_prompt=[sample.target_prompt],
        num_maps_per_mask=roi.num_maps_per_mask,
        mask_encode_strength=roi.mask_encode_strength,
        mask_thresholding_ratio=roi.mask_thresholding_ratio,
        num_inference_steps=method.num_edit_steps,
        guidance_scale=method.guidance_scale,
        output_type="np",
    )
    mask_batch = _normalize_mask_batch(mask_output)
    if len(mask_batch) != 1:
        raise ValueError(f"Expected one ROI mask, got {len(mask_batch)}")
    soft_mask = np.clip(np.asarray(mask_batch[0], dtype=np.float32), 0.0, 1.0)
    return RoiPayload(
        sample_id=sample.sample_id,
        mask=soft_mask,
        source="live",
        metadata={
            "threshold": float(roi.threshold),
            "num_maps_per_mask": int(roi.num_maps_per_mask),
            "mask_encode_strength": float(roi.mask_encode_strength),
            "mask_thresholding_ratio": float(roi.mask_thresholding_ratio),
        },
    )
