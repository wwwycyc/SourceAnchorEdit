from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sourceanchor.input.sample import load_standard_sample, sample_from_mapping
from sourceanchor.schemas import StandardSample


def _load_mapping_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest file not found: {resolved}")
    suffix = resolved.suffix.lower()
    raw_text = resolved.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported manifest format: {resolved}")
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be an object: {resolved}")
    return payload


def load_samples_from_path(path: str | Path) -> list[StandardSample]:
    resolved = Path(path).expanduser().resolve()
    payload = _load_mapping_file(resolved)
    if "sample_id" in payload:
        return [sample_from_mapping(payload, base_dir=resolved.parent, sample_json_path=resolved)]

    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list):
        raise ValueError("Manifest must contain either a single sample object or a top-level 'samples' list.")

    samples: list[StandardSample] = []
    for item in samples_payload:
        if isinstance(item, str):
            samples.append(load_standard_sample((resolved.parent / item).resolve()))
            continue
        if isinstance(item, dict):
            samples.append(sample_from_mapping(item, base_dir=resolved.parent, sample_json_path=resolved))
            continue
        raise ValueError(f"Unsupported manifest entry type: {type(item)!r}")
    return samples
