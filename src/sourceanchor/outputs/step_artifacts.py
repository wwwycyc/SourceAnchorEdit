from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sourceanchor.config import SaveConfig
from sourceanchor.outputs.writer import save_image, write_json


def should_record_step(config: SaveConfig, step_idx: int, total_steps: int) -> bool:
    if not config.step_visualizations:
        return False
    if step_idx in config.step_indices:
        return True
    if config.step_include_last and step_idx == total_steps - 1:
        return True
    return step_idx % config.step_interval == 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _clean_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return str(tuple(value.shape))
        value = value.detach().cpu().item()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_scalar(row.get(key)) for key in fieldnames})
    return path.resolve()


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def extract_sample_map(value: torch.Tensor | np.ndarray, batch_index: int) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim >= 4:
        array = array[batch_index]
    elif array.ndim == 3:
        if array.shape[0] > 1 and batch_index < array.shape[0]:
            array = array[batch_index]
        else:
            array = array[0]
    while array.ndim > 2:
        array = array.mean(axis=0)
    return np.asarray(array, dtype=np.float32)


def _normalize_for_image(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if min_value < 0.0 or max_value > 1.0:
        denom = max(max_value - min_value, 1e-6)
        values = (values - min_value) / denom
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def map_to_rgb(array: np.ndarray) -> np.ndarray:
    normalized = _normalize_for_image(array)
    gray = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def compute_map_stats(map_array: np.ndarray) -> dict[str, float]:
    flat = np.asarray(map_array, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "top10_mean": 0.0,
            "bottom10_mean": 0.0,
            "top_bottom_gap": 0.0,
            "active_ratio": 0.0,
            "entropy": 0.0,
        }

    flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
    p05, p10, p50, p90, p95 = np.quantile(flat, [0.05, 0.10, 0.50, 0.90, 0.95]).tolist()
    top_values = flat[flat >= p90]
    bottom_values = flat[flat <= p10]
    top10_mean = float(top_values.mean()) if top_values.size else float(flat.mean())
    bottom10_mean = float(bottom_values.mean()) if bottom_values.size else float(flat.mean())

    normalized = _normalize_for_image(flat)
    histogram, _ = np.histogram(normalized, bins=32, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities = probabilities / max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log2(probabilities))) / math.log2(32.0)

    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p05": float(p05),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "top10_mean": top10_mean,
        "bottom10_mean": bottom10_mean,
        "top_bottom_gap": float(top10_mean - bottom10_mean),
        "active_ratio": float((normalized > 0.5).mean()),
        "entropy": entropy,
    }


@dataclass
class StepArtifactWriter:
    config: SaveConfig
    sample_id: str
    sample_output_dir: Path
    total_steps: int
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_rows: list[dict[str, Any]] = field(default_factory=list)
    selected_outputs: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def steps_dir(self) -> Path:
        return self.sample_output_dir / "steps"

    def is_selected(self, step_idx: int) -> bool:
        return should_record_step(self.config, step_idx, self.total_steps)

    def record_step(
        self,
        *,
        step_idx: int,
        timestep: int | float,
        batch_index: int,
        trace_row: dict[str, Any],
        maps: dict[str, torch.Tensor | np.ndarray],
        latent_preview: np.ndarray | None = None,
    ) -> None:
        if not self.is_selected(step_idx):
            return

        base_row = {str(key): _clean_scalar(value) for key, value in trace_row.items()}
        base_row["step_idx"] = int(step_idx)
        base_row["timestep"] = _clean_scalar(timestep)
        if self.config.step_save_trace:
            self.trace_rows.append(base_row)

        extracted_maps = {
            key: extract_sample_map(value, batch_index)
            for key, value in maps.items()
            if value is not None
        }
        if self.config.step_save_diagnostics:
            diagnostic_row = dict(base_row)
            for key, map_array in extracted_maps.items():
                for stat_name, stat_value in compute_map_stats(map_array).items():
                    diagnostic_row[f"{key}_{stat_name}"] = stat_value
            self.diagnostic_rows.append(diagnostic_row)

        step_label = f"step_{step_idx:03d}"
        step_dir = self.steps_dir / step_label
        saved: dict[str, str] = {}
        for key, map_array in extracted_maps.items():
            path = save_image(step_dir / f"{key}.png", map_to_rgb(map_array))
            saved[key] = str(path)
        if latent_preview is not None:
            path = save_image(step_dir / "latent_preview.png", latent_preview)
            saved["latent_preview"] = str(path)
        if saved:
            self.selected_outputs[step_label] = saved

    def _diagnostic_summary(self) -> dict[str, dict[str, float | int | bool]]:
        summary: dict[str, dict[str, float | int | bool]] = {}
        required_suffixes = (
            "mean",
            "std",
            "min",
            "max",
            "top_bottom_gap",
            "active_ratio",
            "entropy",
        )
        prefixes = sorted(
            {
                key[: -len("_std")]
                for row in self.diagnostic_rows
                for key in row
                if key.endswith("_std")
            }
        )
        for prefix in prefixes:
            rows = [
                row
                for row in self.diagnostic_rows
                if all(f"{prefix}_{suffix}" in row for suffix in required_suffixes)
            ]
            if not rows:
                continue
            summary[prefix] = {
                "step_count": len(rows),
                "mean_of_mean": float(np.mean([float(row[f"{prefix}_mean"]) for row in rows])),
                "mean_of_std": float(np.mean([float(row[f"{prefix}_std"]) for row in rows])),
                "mean_top_bottom_gap": float(np.mean([float(row[f"{prefix}_top_bottom_gap"]) for row in rows])),
                "mean_active_ratio": float(np.mean([float(row[f"{prefix}_active_ratio"]) for row in rows])),
                "mean_entropy": float(np.mean([float(row[f"{prefix}_entropy"]) for row in rows])),
                "min_value": float(np.min([float(row[f"{prefix}_min"]) for row in rows])),
                "max_value": float(np.max([float(row[f"{prefix}_max"]) for row in rows])),
            }
        mask_summary = summary.get("dynamic_mask")
        if mask_summary is not None:
            mask_summary["heuristic_soft_global_like"] = (
                float(mask_summary["mean_of_std"]) < 0.05
                and float(mask_summary["mean_top_bottom_gap"]) < 0.25
            )
        return summary

    def finish(self) -> dict[str, Any]:
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        trace_csv_path = _write_csv(self.steps_dir / "step_trace.csv", self.trace_rows) if self.config.step_save_trace else None
        trace_json_path = (
            write_json(
                self.steps_dir / "step_trace.json",
                {
                    "sample_id": self.sample_id,
                    "total_steps": self.total_steps,
                    "recorded_steps": [int(row["step_idx"]) for row in self.trace_rows],
                    "trace": _jsonable(self.trace_rows),
                },
            )
            if self.config.step_save_trace and self.trace_rows
            else None
        )

        diagnostics_summary = self._diagnostic_summary() if self.config.step_save_diagnostics else {}
        diagnostics_csv_path = (
            _write_csv(self.steps_dir / "step_diagnostics.csv", self.diagnostic_rows)
            if self.config.step_save_diagnostics
            else None
        )
        diagnostics_json_path = (
            write_json(
                self.steps_dir / "step_diagnostics.json",
                {
                    "sample_id": self.sample_id,
                    "total_steps": self.total_steps,
                    "recorded_steps": [int(row["step_idx"]) for row in self.diagnostic_rows],
                    "steps": _jsonable(self.diagnostic_rows),
                    "summary": _jsonable(diagnostics_summary),
                },
            )
            if self.config.step_save_diagnostics and self.diagnostic_rows
            else None
        )

        manifest = {
            "sample_id": self.sample_id,
            "enabled": True,
            "total_steps": self.total_steps,
            "step_interval": self.config.step_interval,
            "step_indices": list(self.config.step_indices),
            "step_include_last": self.config.step_include_last,
            "recorded_steps": sorted(
                {
                    int(row["step_idx"])
                    for row in [*self.trace_rows, *self.diagnostic_rows]
                }
                | {
                    int(label.removeprefix("step_"))
                    for label in self.selected_outputs
                }
            ),
            "outputs": {
                "trace_csv_path": str(trace_csv_path) if trace_csv_path is not None else None,
                "trace_json_path": str(trace_json_path) if trace_json_path is not None else None,
                "diagnostics_csv_path": str(diagnostics_csv_path) if diagnostics_csv_path is not None else None,
                "diagnostics_json_path": str(diagnostics_json_path) if diagnostics_json_path is not None else None,
                "selected_steps": self.selected_outputs,
            },
            "diagnostics_summary": _jsonable(diagnostics_summary),
        }
        manifest_path = (self.steps_dir / "step_manifest.json").expanduser().resolve()
        manifest["manifest_path"] = str(manifest_path)
        write_json(manifest_path, _jsonable(manifest))
        return manifest
