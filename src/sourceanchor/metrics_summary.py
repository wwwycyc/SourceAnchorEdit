from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sourceanchor.outputs.writer import write_json


METRICS_SUMMARY_JSON = "metrics_summary.json"
METRICS_SUMMARY_CSV = "metrics_summary.csv"
MAX_ERROR_EXAMPLES = 20


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _collect_debug_json_paths(run_dir: Path) -> list[Path]:
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return []
    return sorted(samples_dir.glob("*/debug.json"))


def _load_debug_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"debug.json root must be an object: {path}")
    return payload


def _metric_stats(values: list[float], *, total_count: int) -> dict[str, Any]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "missing": max(total_count - count, 0),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def build_metrics_summary_payload(
    run_dir: str | Path,
    debug_json_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    paths = [Path(path).expanduser().resolve() for path in debug_json_paths] if debug_json_paths is not None else _collect_debug_json_paths(resolved_run_dir)

    numeric_values: dict[str, list[float]] = {}
    error_records: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for debug_path in paths:
        payload = _load_debug_payload(debug_path)
        sample_id = str(payload.get("sample_id") or debug_path.parent.name)
        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {"error": f"metrics must be an object, got {type(metrics).__name__}"}

        numeric_metric_count = 0
        for key, value in metrics.items():
            if _is_finite_number(value):
                numeric_values.setdefault(str(key), []).append(float(value))
                numeric_metric_count += 1
                continue
            if key == "error" or str(key).endswith("_error"):
                record = error_records.setdefault(str(key), {"count": 0, "examples": []})
                record["count"] += 1
                if len(record["examples"]) < MAX_ERROR_EXAMPLES:
                    record["examples"].append(
                        {
                            "sample_id": sample_id,
                            "debug_json_path": str(debug_path),
                            "message": str(value),
                        }
                    )

        samples.append(
            {
                "sample_id": sample_id,
                "debug_json_path": str(debug_path),
                "numeric_metric_count": numeric_metric_count,
                "has_metrics_error": "error" in metrics or any(str(key).endswith("_error") for key in metrics),
            }
        )

    sample_count = len(samples)
    metric_stats = {
        metric_name: _metric_stats(values, total_count=sample_count)
        for metric_name, values in sorted(numeric_values.items())
        if values
    }
    return {
        "run_dir": str(resolved_run_dir),
        "sample_count": sample_count,
        "sample_with_numeric_metrics_count": sum(1 for sample in samples if sample["numeric_metric_count"] > 0),
        "metrics": metric_stats,
        "errors": error_records,
        "samples": samples,
    }


def write_metrics_summary(
    run_dir: str | Path,
    debug_json_paths: Iterable[str | Path] | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    payload = build_metrics_summary_payload(resolved_run_dir, debug_json_paths)
    json_path = write_json(resolved_run_dir / METRICS_SUMMARY_JSON, payload)
    csv_path = resolved_run_dir / METRICS_SUMMARY_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "count", "missing", "mean", "std", "min", "max"])
        writer.writeheader()
        for metric, stats in payload["metrics"].items():
            writer.writerow(
                {
                    "metric": metric,
                    "count": stats["count"],
                    "missing": stats["missing"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )
    return json_path, csv_path.resolve()
