from __future__ import annotations

from pathlib import Path

from sourceanchor.config import ExperimentConfig
from sourceanchor.outputs.artifacts import roi_cache_dir
from sourceanchor.roi.base import RoiBuildResult
from sourceanchor.roi.serialization import save_roi_payload


def build_roi_cache_for_samples(editor, config: ExperimentConfig, samples) -> list[RoiBuildResult]:
    if config.roi.cache_root is None:
        raise ValueError("roi.cache_root must be configured to build ROI cache.")

    results: list[RoiBuildResult] = []
    for sample in samples:
        editor._restore_default_attention_processors()
        roi_payload = editor._generate_live_roi_payload(sample)
        cache_dir = roi_cache_dir(config.roi.cache_root, sample.sample_id)
        save_roi_payload(cache_dir, roi_payload)
        editor._register_attention_store()
        results.append(
            RoiBuildResult(
                sample_id=sample.sample_id,
                cache_dir=Path(cache_dir).resolve(),
                source=roi_payload.source,
            )
        )
    return results
