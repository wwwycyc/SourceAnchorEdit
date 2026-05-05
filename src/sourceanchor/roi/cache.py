from __future__ import annotations

from pathlib import Path

from sourceanchor.roi.base import RoiPayload
from sourceanchor.roi.serialization import load_roi_payload


def load_cached_roi(cache_root: str | Path, sample_id: str) -> RoiPayload:
    cache_dir = Path(cache_root).expanduser().resolve() / sample_id
    return load_roi_payload(cache_dir)
