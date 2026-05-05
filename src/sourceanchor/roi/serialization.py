from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sourceanchor.roi.base import RoiPayload


ROI_MASK_FILENAME = "roi_mask.npy"
ROI_META_FILENAME = "roi_meta.json"


def save_roi_payload(cache_dir: str | Path, payload: RoiPayload) -> Path:
    target_dir = Path(cache_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / ROI_MASK_FILENAME, np.asarray(payload.mask, dtype=np.float32))
    meta: dict[str, Any] = {
        "sample_id": payload.sample_id,
        "source": payload.source,
        **payload.metadata,
    }
    (target_dir / ROI_META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_dir


def load_roi_payload(cache_dir: str | Path) -> RoiPayload:
    target_dir = Path(cache_dir).expanduser().resolve()
    mask_path = target_dir / ROI_MASK_FILENAME
    meta_path = target_dir / ROI_META_FILENAME
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"ROI metadata not found: {meta_path}")
    mask = np.load(mask_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"ROI metadata must be an object: {meta_path}")
    sample_id = str(meta.pop("sample_id"))
    source = str(meta.pop("source"))
    return RoiPayload(
        sample_id=sample_id,
        mask=np.asarray(mask, dtype=np.float32),
        source=source,
        cache_dir=target_dir,
        metadata=meta,
    )
