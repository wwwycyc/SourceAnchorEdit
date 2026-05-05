from __future__ import annotations

from sourceanchor.config import SaveConfig


def should_save_roi_cache(config: SaveConfig) -> bool:
    return bool(config.roi_cache)


def should_save_inversion_tensors(config: SaveConfig) -> bool:
    return bool(config.inversion_tensors)


def should_save_debug_json(config: SaveConfig) -> bool:
    return bool(config.debug_json)


def should_save_step_visualizations(config: SaveConfig) -> bool:
    return bool(config.step_visualizations)


def should_save_overview(config: SaveConfig) -> bool:
    return bool(config.overview)
