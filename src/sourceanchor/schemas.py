from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleMetadata:
    dataset: str | None = None
    record_id: str | None = None
    edit_instruction: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.dataset is not None:
            payload["dataset"] = self.dataset
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        if self.edit_instruction is not None:
            payload["edit_instruction"] = self.edit_instruction
        payload.update(self.extras)
        return payload


@dataclass
class StandardSample:
    sample_id: str
    source_image_path: Path
    source_prompt: str
    target_prompt: str
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    target_reference_path: Path | None = None
    mask_path: Path | None = None
    sample_json_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_image_path": str(self.source_image_path),
            "source_prompt": self.source_prompt,
            "target_prompt": self.target_prompt,
            "metadata": self.metadata.to_dict(),
            "target_reference_path": str(self.target_reference_path) if self.target_reference_path is not None else None,
            "mask_path": str(self.mask_path) if self.mask_path is not None else None,
            "sample_json_path": str(self.sample_json_path) if self.sample_json_path is not None else None,
        }
