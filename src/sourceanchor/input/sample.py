from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sourceanchor.schemas import SampleMetadata, StandardSample


def _load_mapping_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Sample file not found: {resolved}")
    suffix = resolved.suffix.lower()
    raw_text = resolved.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported sample format: {resolved}")
    if not isinstance(payload, dict):
        raise ValueError(f"Sample root must be an object: {resolved}")
    return payload


def _resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def sample_from_mapping(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    sample_json_path: Path | None = None,
) -> StandardSample:
    sample_id = str(mapping.get("sample_id") or "").strip()
    source_prompt = str(mapping.get("source_prompt") or "").strip()
    target_prompt = str(mapping.get("target_prompt") or "").strip()
    source_image_raw = mapping.get("source_image_path")
    if not sample_id:
        raise ValueError("sample_id is required")
    if source_image_raw is None:
        raise ValueError("source_image_path is required")
    if not source_prompt:
        raise ValueError("source_prompt is required")
    if not target_prompt:
        raise ValueError("target_prompt is required")

    metadata_payload = mapping.get("metadata") or {}
    if not isinstance(metadata_payload, dict):
        raise ValueError("metadata must be an object when provided")
    extras = {
        key: value
        for key, value in metadata_payload.items()
        if key not in {"dataset", "record_id", "edit_instruction"}
    }
    metadata = SampleMetadata(
        dataset=metadata_payload.get("dataset"),
        record_id=metadata_payload.get("record_id"),
        edit_instruction=metadata_payload.get("edit_instruction"),
        extras=extras,
    )
    source_image_path = _resolve_path(source_image_raw, base_dir=base_dir) or Path()
    if not source_image_path.exists():
        raise FileNotFoundError(f"source_image_path not found: {source_image_path}")

    target_reference_path = _resolve_path(mapping.get("target_reference_path"), base_dir=base_dir)
    if target_reference_path is not None and not target_reference_path.exists():
        raise FileNotFoundError(f"target_reference_path not found: {target_reference_path}")

    mask_path = _resolve_path(mapping.get("mask_path"), base_dir=base_dir)
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"mask_path not found: {mask_path}")

    return StandardSample(
        sample_id=sample_id,
        source_image_path=source_image_path,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        metadata=metadata,
        target_reference_path=target_reference_path,
        mask_path=mask_path,
        sample_json_path=sample_json_path.resolve() if sample_json_path is not None else None,
    )


def load_standard_sample(path: str | Path) -> StandardSample:
    sample_path = Path(path).expanduser().resolve()
    payload = _load_mapping_file(sample_path)
    return sample_from_mapping(payload, base_dir=sample_path.parent, sample_json_path=sample_path)
