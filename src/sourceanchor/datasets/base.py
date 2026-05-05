from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sourceanchor.schemas import StandardSample
from sourceanchor.outputs.writer import ensure_directory, write_json


class DatasetAdapter(ABC):
    """Convert a raw dataset into the standard sample format."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def export(self, output_dir: Path) -> list[StandardSample]:
        raise NotImplementedError

    @staticmethod
    def write_standard_samples(samples: list[StandardSample], output_dir: Path) -> list[Path]:
        output_dir = ensure_directory(output_dir)
        sample_paths: list[Path] = []
        for sample in samples:
            sample_dir = ensure_directory(output_dir / sample.sample_id)
            sample_json_path = sample_dir / "sample.json"
            write_json(sample_json_path, sample.to_dict())
            sample_paths.append(sample_json_path)
        return sample_paths
